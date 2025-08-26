"""分子生成器模块

整合扩散模型和强化学习进行分子生成和优化。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import time
from torch.cuda.amp import autocast
import gc

from ..models.diffusion_model import DiffusionModel
from ..models.reinforcement_learning import RLAgent
from ..features.molecule_features import MoleculeFeatureExtractor

class MoleculeGenerator:
    """分子生成器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 调试：打印完整配置
        self.logger.info(f"MoleculeGenerator初始化配置: {config}")
        
        # 生成参数
        self.max_attempts = config.get('max_attempts', 1000)
        self.diversity_threshold = config.get('diversity_threshold', 0.7)
        self.validity_threshold = config.get('validity_threshold', 0.8)
        
        # 性能优化参数
        self.batch_size = config.get('generation_batch_size', 32)
        self.max_parallel_workers = config.get('max_parallel_workers', min(8, mp.cpu_count()))
        self.use_mixed_precision = config.get('use_mixed_precision', True)
        self.enable_caching = config.get('enable_caching', True)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.diffusion_model = None
        
        # 初始化RL智能体
        rl_config = config.get('reinforcement_learning', {})
        if rl_config:
            self.logger.info(f"初始化RL智能体，配置: {rl_config}")
            try:
                from src.models.reinforcement_learning import RLAgent
                self.rl_agent = RLAgent(rl_config)
                self.logger.info("RL智能体初始化成功")
            except Exception as e:
                self.logger.error(f"RL智能体初始化失败: {e}")
                self.rl_agent = None
        else:
            self.logger.warning("未提供RL配置，RL智能体未初始化")
            self.rl_agent = None
        # 使用features配置而不是molecule_features
        features_config = config.get('features', {})
        self.logger.info(f"特征提取器配置: {features_config}")
        self.feature_extractor = MoleculeFeatureExtractor(features_config)
        
        # 计算特征维度并初始化状态投影层
        # 从features配置中获取实际的特征维度
        fingerprint_size = features_config.get('fingerprint_size', 2048)
        descriptor_count = 14  # 固定的描述符数量
        
        # 分子特征维度：descriptors + morgan_fp
        mol_feature_dim = descriptor_count + fingerprint_size
        
        # 蛋白质特征维度：在generate_submission.py中使用的是512维
        protein_feature_dim = 512
        
        # 总特征维度
        total_feature_dim = mol_feature_dim + protein_feature_dim
        target_state_dim = 512  # 与RLAgent的state_dim匹配
        
        self.logger.info(f"特征维度计算: 分子特征={mol_feature_dim}(描述符{descriptor_count}+指纹{fingerprint_size}), 蛋白质特征={protein_feature_dim}, 总计={total_feature_dim}")
        
        # 创建状态投影层
        self.state_projection = nn.Linear(total_feature_dim, target_state_dim).to(self.device)
        self.logger.info(f"状态投影层创建: {total_feature_dim} -> {target_state_dim}")
        
        # 缓存系统
        self.feature_cache = {} if self.enable_caching else None
        self.validity_cache = {} if self.enable_caching else None
        
        # 优化的线程池和进程池
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.max_parallel_workers,
            thread_name_prefix="MolGen_Thread"
        )
        self.process_pool = ProcessPoolExecutor(
            max_workers=min(4, mp.cpu_count()),
            mp_context=mp.get_context('spawn')  # 使用spawn方法避免内存泄漏
        )
        
        # 内存监控
        self.memory_threshold = config.get('memory_threshold', 0.8)  # 80%内存使用率阈值
        self.gc_frequency = config.get('gc_frequency', 100)  # 每100次操作进行一次垃圾回收
        self.operation_count = 0
        
        # 分子库
        self.generated_molecules = []
        self.molecule_scores = {}
        
        self.logger.info(f"分子生成器初始化完成，特征维度: {total_feature_dim} -> {target_state_dim}, 批处理大小: {self.batch_size}, 并行工作者: {self.max_parallel_workers}")
    
    def load_models(self, diffusion_path: Optional[str] = None, rl_path: Optional[str] = None):
        """加载预训练模型
        
        Args:
            diffusion_path: 扩散模型路径
            rl_path: 强化学习模型路径
        """
        if diffusion_path:
            self.diffusion_model = DiffusionModel(self.config.get('diffusion', {}))
            self.diffusion_model.load_model(diffusion_path)
            self.logger.info(f"扩散模型已加载: {diffusion_path}")
            
        if rl_path:
            # 确保正确读取reinforcement_learning配置
            rl_config = self.config.get('reinforcement_learning', {})
            self.logger.info(f"RL配置: {rl_config}")
            self.rl_agent = RLAgent(rl_config)
            self.rl_agent.load_model(rl_path)
            self.logger.info(f"强化学习模型已加载: {rl_path}")
            
    def set_rl_agent(self, rl_agent: RLAgent):
        """设置强化学习智能体
        
        Args:
            rl_agent: 强化学习智能体实例
        """
        self.rl_agent = rl_agent
        self.logger.info("强化学习智能体已设置")
        
    def set_diffusion_model(self, diffusion_model: DiffusionModel):
        """设置扩散模型
        
        Args:
            diffusion_model: 扩散模型实例
        """
        self.diffusion_model = diffusion_model
        self.logger.info("扩散模型已设置")
    
    def generate_initial_molecules(self, num_molecules: int, protein_features: Optional[torch.Tensor] = None) -> List[str]:
        """
        使用扩散模型生成初始分子（优化版本，支持批处理和并行处理）
        
        Args:
            num_molecules: 生成分子数量
            protein_features: 蛋白质特征
            
        Returns:
            生成的SMILES列表
        """
        if self.diffusion_model is None:
            raise ValueError("扩散模型未加载")
        
        self.logger.info(f"开始生成 {num_molecules} 个初始分子（批处理大小: {self.batch_size}）")
        
        generated_smiles = []
        attempts = 0
        start_time = time.time()
        
        # 计算需要的批次数
        num_batches = (num_molecules + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            if len(generated_smiles) >= num_molecules:
                break
                
            current_batch_size = min(self.batch_size, num_molecules - len(generated_smiles))
            batch_attempts = 0
            max_batch_attempts = self.max_attempts // num_batches
            
            while len(generated_smiles) < num_molecules and batch_attempts < max_batch_attempts:
                try:
                    # 生成分子特征（批处理）
                    batch_smiles = self._generate_batch_molecules(
                        current_batch_size, protein_features
                    )
                    
                    # 并行验证分子
                    valid_smiles = self._parallel_validate_molecules(batch_smiles)
                    
                    # 添加有效分子
                    for smiles in valid_smiles:
                        if len(generated_smiles) < num_molecules:
                            generated_smiles.append(smiles)
                    
                    batch_attempts += 1
                    attempts += 1
                    
                    # 定期清理GPU内存
                    if batch_attempts % 5 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    if batch_attempts % 10 == 0:
                        elapsed_time = time.time() - start_time
                        self.logger.info(
                            f"批次 {batch_idx+1}/{num_batches}, 尝试 {batch_attempts}, "
                            f"已生成 {len(generated_smiles)}/{num_molecules} 个分子, "
                            f"耗时 {elapsed_time:.2f}s"
                        )
                        
                except Exception as e:
                    self.logger.warning(f"批次生成失败: {e}")
                    batch_attempts += 1
                    continue
        
        elapsed_time = time.time() - start_time
        success_rate = len(generated_smiles) / max(attempts * self.batch_size, 1) * 100
        
        self.logger.info(
            f"生成完成，共生成 {len(generated_smiles)} 个分子，"
            f"成功率: {success_rate:.2f}%, 总耗时: {elapsed_time:.2f}s"
        )
        
        return generated_smiles
    
    def _generate_batch_molecules(self, batch_size: int, protein_features: Optional[torch.Tensor] = None) -> List[str]:
        """
        批量生成分子
        """
        with torch.no_grad():
            # 准备条件信息
            condition = None
            if protein_features is not None:
                condition = protein_features.to(self.device)
                # 扩展到批处理大小
                if condition.dim() == 1:
                    condition = condition.unsqueeze(0).repeat(batch_size, 1)
                elif condition.shape[0] == 1:
                    condition = condition.repeat(batch_size, 1)
            
            # 使用混合精度生成
            if self.use_mixed_precision:
                with autocast():
                    generated_features = self.diffusion_model.sample(
                        batch_size=batch_size,
                        condition=condition,
                        use_ddim=True,  # 使用DDIM加速采样
                        ddim_steps=50
                    )
            else:
                generated_features = self.diffusion_model.sample(
                    batch_size=batch_size,
                    condition=condition,
                    use_ddim=True,
                    ddim_steps=50
                )
        
        # 将特征转换为SMILES
        batch_smiles = []
        for features in generated_features:
            smiles = self._features_to_smiles(features)
            if smiles:
                batch_smiles.append(smiles)
        
        return batch_smiles
    
    def _parallel_validate_molecules(self, smiles_list: List[str]) -> List[str]:
        """
        并行验证分子有效性
        """
        if not smiles_list:
            return []
        
        # 使用缓存加速验证
        if self.enable_caching:
            cached_results = []
            uncached_smiles = []
            uncached_indices = []
            
            for i, smiles in enumerate(smiles_list):
                if smiles in self.validity_cache:
                    if self.validity_cache[smiles]:
                        cached_results.append(smiles)
                else:
                    uncached_smiles.append(smiles)
                    uncached_indices.append(i)
            
            # 并行验证未缓存的分子
            if uncached_smiles:
                validation_func = partial(self._validate_single_molecule)
                with self.thread_pool as executor:
                    validation_results = list(executor.map(validation_func, uncached_smiles))
                
                # 更新缓存并收集结果
                for smiles, is_valid in zip(uncached_smiles, validation_results):
                    self.validity_cache[smiles] = is_valid
                    if is_valid:
                        cached_results.append(smiles)
            
            return cached_results
        else:
            # 不使用缓存的并行验证
            validation_func = partial(self._validate_single_molecule)
            with self.thread_pool as executor:
                validation_results = list(executor.map(validation_func, smiles_list))
            
            return [smiles for smiles, is_valid in zip(smiles_list, validation_results) if is_valid]
    
    def _validate_single_molecule(self, smiles: str) -> bool:
        """
        验证单个分子的有效性
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # 基本有效性检查
            if mol.GetNumAtoms() < 3 or mol.GetNumAtoms() > 100:
                return False
            
            # 检查分子量
            mw = Descriptors.MolWt(mol)
            if mw < 100 or mw > 1000:
                return False
            
            # 检查LogP
            logp = Descriptors.MolLogP(mol)
            if logp < -5 or logp > 8:
                return False
            
            return True
            
        except Exception:
            return False
    
    def optimize_molecules(self, initial_molecules: List[str], 
                          protein_features: torch.Tensor,
                          target_properties: Dict[str, float] = None,
                          num_iterations: int = 100) -> List[str]:
        """使用强化学习优化分子（优化版本，支持并行处理和智能筛选）
        
        Args:
            initial_molecules: 初始分子SMILES列表
            protein_features: 蛋白质特征
            target_properties: 目标属性
            num_iterations: 优化迭代次数
            
        Returns:
            优化后的分子SMILES列表
        """
        if self.rl_agent is None:
            self.logger.warning("强化学习代理未加载，返回初始分子")
            return initial_molecules
            
        self.logger.info(f"开始优化 {len(initial_molecules)} 个分子（并行处理）")
        start_time = time.time()
        
        # 预筛选分子，移除明显不合适的分子
        filtered_molecules = self._prefilter_molecules(initial_molecules)
        self.logger.info(f"预筛选后保留 {len(filtered_molecules)}/{len(initial_molecules)} 个分子")
        
        # 并行优化分子
        optimized_molecules = self._parallel_optimize_molecules(
            filtered_molecules, protein_features, target_properties, num_iterations
        )
        
        # 后处理：去重和质量筛选
        final_molecules = self._postprocess_molecules(optimized_molecules)
        
        elapsed_time = time.time() - start_time
        improvement_rate = self._calculate_improvement_rate(initial_molecules, final_molecules, protein_features)
        
        self.logger.info(
            f"分子优化完成，共优化 {len(final_molecules)} 个分子，"
            f"改进率: {improvement_rate:.2f}%, 耗时: {elapsed_time:.2f}s"
        )
        
        return final_molecules
    
    def _prefilter_molecules(self, molecules: List[str]) -> List[str]:
        """
        预筛选分子，移除明显不合适的分子
        """
        filtered = []
        
        for smiles in molecules:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                # 基本属性检查
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                
                # Lipinski规则和其他药物相似性规则
                if (150 <= mw <= 800 and 
                    -3 <= logp <= 6 and 
                    tpsa <= 200 and 
                    hbd <= 10 and 
                    hba <= 15):
                    filtered.append(smiles)
                    
            except Exception:
                continue
        
        return filtered
    
    def _parallel_optimize_molecules(self, molecules: List[str], 
                                   protein_features: torch.Tensor,
                                   target_properties: Dict[str, float],
                                   num_iterations: int) -> List[str]:
        """
        并行优化分子（增强版，支持内存监控和自适应批处理）
        """
        if not molecules:
            return []
        
        # 内存监控
        self._check_memory_usage()
        
        # 自适应批处理大小
        batch_size = self._calculate_optimal_batch_size(len(molecules))
        
        # 准备优化参数
        optimization_func = partial(
            self._optimize_single_molecule_enhanced,
            protein_features=protein_features,
            target_properties=target_properties,
            num_iterations=num_iterations
        )
        
        optimized_molecules = []
        
        try:
            # 使用进程池进行并行优化（CPU密集型任务）
            for i in range(0, len(molecules), batch_size):
                batch = molecules[i:i + batch_size]
                
                # 提交任务到进程池
                futures = []
                for mol in batch:
                    future = self.process_pool.submit(optimization_func, mol)
                    futures.append(future)
                
                # 收集结果
                batch_results = []
                for future in futures:
                    try:
                        result = future.result(timeout=300)  # 5分钟超时
                        batch_results.append(result)
                    except Exception as e:
                        self.logger.warning(f"分子优化超时或失败: {e}")
                        batch_results.append(batch[len(batch_results)])  # 使用原始分子
                
                optimized_molecules.extend(batch_results)
                
                # 进度报告
                progress = min(i + batch_size, len(molecules))
                self.logger.info(f"已优化 {progress}/{len(molecules)} 个分子")
                
                # 内存管理
                self._manage_memory()
                
        except Exception as e:
            self.logger.error(f"并行优化过程中出现错误: {e}")
            # 降级到串行处理
            optimized_molecules = self._serial_optimize_molecules(
                molecules, protein_features, target_properties, num_iterations
            )
        
        return optimized_molecules
    
    def _optimize_single_molecule_enhanced(self, smiles: str, 
                                         protein_features: torch.Tensor,
                                         target_properties: Dict[str, float],
                                         num_iterations: int) -> str:
        """
        增强版单分子优化
        """
        try:
            current_smiles = smiles
            best_smiles = smiles
            best_score = self._calculate_molecule_score(smiles, protein_features, target_properties)
            
            # 自适应学习率
            learning_rate = 0.01
            patience = 10
            no_improvement_count = 0
            
            for iteration in range(num_iterations):
                # 生成候选分子
                candidates = self._generate_molecule_variants(current_smiles, num_variants=5)
                
                # 评估候选分子
                best_candidate = None
                best_candidate_score = best_score
                
                for candidate in candidates:
                    if self._validate_single_molecule(candidate):
                        score = self._calculate_molecule_score(candidate, protein_features, target_properties)
                        if score > best_candidate_score:
                            best_candidate = candidate
                            best_candidate_score = score
                
                # 更新最佳分子
                if best_candidate and best_candidate_score > best_score:
                    best_smiles = best_candidate
                    best_score = best_candidate_score
                    current_smiles = best_candidate
                    no_improvement_count = 0
                    
                    # 动态调整学习率
                    learning_rate = min(0.02, learning_rate * 1.1)
                else:
                    no_improvement_count += 1
                    learning_rate *= 0.95
                    
                    # 早停机制
                    if no_improvement_count >= patience:
                        break
                
                # 偶尔进行随机探索
                if iteration % 20 == 0 and iteration > 0:
                    current_smiles = smiles  # 回到原始分子进行探索
            
            return best_smiles
            
        except Exception as e:
            self.logger.warning(f"优化分子 {smiles} 时出错: {e}")
            return smiles
    
    def _generate_molecule_variants(self, smiles: str, num_variants: int = 5) -> List[str]:
        """
        生成分子变体
        """
        variants = []
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [smiles]
            
            # 方法1：随机SMILES
            for _ in range(num_variants // 2):
                try:
                    random_smiles = Chem.MolToSmiles(mol, doRandom=True)
                    if random_smiles != smiles:
                        variants.append(random_smiles)
                except:
                    continue
            
            # 方法2：简单的分子修改（如果有可用的化学转换库）
            # 这里可以添加更复杂的分子修改策略
            
            # 确保返回足够的变体
            while len(variants) < num_variants:
                variants.append(smiles)
            
            return variants[:num_variants]
            
        except Exception:
            return [smiles] * num_variants
    
    def _calculate_molecule_score(self, smiles: str, 
                                protein_features: torch.Tensor,
                                target_properties: Dict[str, float]) -> float:
        """
        计算分子综合评分
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            score = 0.0
            
            # 基础药物相似性评分
            qed_score = self._calculate_qed(mol)
            score += qed_score * 0.3
            
            # 合成可达性评分（简化版）
            sa_score = 1.0 / (1.0 + abs(Descriptors.MolLogP(mol) - 2.5))  # 简化的SA评分
            score += sa_score * 0.2
            
            # 目标属性匹配评分
            if target_properties:
                property_score = self._calculate_property_match_score(mol, target_properties)
                score += property_score * 0.3
            
            # 分子多样性评分（与已知分子的差异）
            diversity_score = self._calculate_diversity_score(smiles)
            score += diversity_score * 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.0
    
    def _calculate_property_match_score(self, mol, target_properties: Dict[str, float]) -> float:
        """
        计算属性匹配评分
        """
        if not target_properties:
            return 0.5
        
        score = 0.0
        count = 0
        
        for prop_name, target_value in target_properties.items():
            try:
                if prop_name == 'mw':
                    actual_value = Descriptors.MolWt(mol)
                    score += 1.0 / (1.0 + abs(actual_value - target_value) / target_value)
                elif prop_name == 'logp':
                    actual_value = Descriptors.MolLogP(mol)
                    score += 1.0 / (1.0 + abs(actual_value - target_value) / max(abs(target_value), 1.0))
                elif prop_name == 'tpsa':
                    actual_value = Descriptors.TPSA(mol)
                    score += 1.0 / (1.0 + abs(actual_value - target_value) / target_value)
                
                count += 1
            except:
                continue
        
        return score / max(count, 1)
    
    def _calculate_diversity_score(self, smiles: str) -> float:
        """
        计算多样性评分（简化版）
        """
        # 这里可以实现与已知分子库的相似性比较
        # 暂时返回基于分子复杂度的简单评分
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            # 基于分子复杂度的简单多样性评分
            num_rings = Descriptors.RingCount(mol)
            num_heteroatoms = mol.GetNumHeteroatoms()
            num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            
            complexity_score = (num_rings * 0.3 + num_heteroatoms * 0.4 + num_rotatable_bonds * 0.3) / 10.0
            return min(1.0, complexity_score)
            
        except:
            return 0.0
    
    def _postprocess_molecules(self, molecules: List[str]) -> List[str]:
        """
        后处理分子：去重和质量筛选
        """
        if not molecules:
            return []
        
        # 去重
        unique_molecules = list(set(molecules))
        
        # 质量筛选
        high_quality_molecules = []
        for smiles in unique_molecules:
            if self._is_high_quality_molecule(smiles):
                high_quality_molecules.append(smiles)
        
        # 如果高质量分子太少，保留一些中等质量的分子
        if len(high_quality_molecules) < len(molecules) * 0.5:
            medium_quality = [smiles for smiles in unique_molecules 
                            if smiles not in high_quality_molecules and self._validate_single_molecule(smiles)]
            high_quality_molecules.extend(medium_quality[:len(molecules) - len(high_quality_molecules)])
        
        return high_quality_molecules
    
    def _is_high_quality_molecule(self, smiles: str) -> bool:
        """
        判断是否为高质量分子
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # 更严格的质量标准
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            qed_score = self._calculate_qed(mol)
            
            return (200 <= mw <= 600 and 
                   -2 <= logp <= 5 and 
                   20 <= tpsa <= 150 and 
                   qed_score >= 0.5)
                   
        except:
            return False
    
    def _calculate_improvement_rate(self, initial_molecules: List[str], 
                                  final_molecules: List[str],
                                  protein_features: torch.Tensor) -> float:
        """
        计算改进率
        """
        try:
            if not initial_molecules or not final_molecules:
                return 0.0
            
            # 计算初始分子的平均评分
            initial_scores = [self._calculate_molecule_score(smiles, protein_features, {}) 
                            for smiles in initial_molecules[:10]]  # 采样计算
            initial_avg = sum(initial_scores) / len(initial_scores) if initial_scores else 0.0
            
            # 计算最终分子的平均评分
            final_scores = [self._calculate_molecule_score(smiles, protein_features, {}) 
                          for smiles in final_molecules[:10]]  # 采样计算
            final_avg = sum(final_scores) / len(final_scores) if final_scores else 0.0
            
            if initial_avg == 0:
                return 0.0
            
            improvement = ((final_avg - initial_avg) / initial_avg) * 100
            return max(0.0, improvement)
            
        except:
            return 0.0
    
    def _optimize_single_molecule(self, smiles: str, protein_features: torch.Tensor,
                                target_properties: Dict[str, float], 
                                num_iterations: int) -> str:
        """优化单个分子
        
        Args:
            smiles: 分子SMILES
            protein_features: 蛋白质特征
            target_properties: 目标属性
            num_iterations: 迭代次数
            
        Returns:
            优化后的SMILES
        """
        current_smiles = smiles
        best_smiles = smiles
        best_score = self._calculate_reward(smiles, target_properties)
        
        for iteration in range(num_iterations):
            # 获取当前状态
            state = self._get_molecule_state(current_smiles, protein_features)
            
            # 选择动作
            action, log_prob = self.rl_agent.select_action(state)
            
            # 执行动作（修改分子）
            new_smiles = self._apply_action(current_smiles, action.cpu().numpy())
            
            if new_smiles and self._is_valid_molecule(new_smiles):
                # 计算奖励
                reward = self._calculate_reward(new_smiles, target_properties)
                
                # 更新最佳分子
                if reward > best_score:
                    best_score = reward
                    best_smiles = new_smiles
                    
                current_smiles = new_smiles
            
        return best_smiles
    
    def _get_molecule_state(self, smiles: str, protein_features: torch.Tensor) -> torch.Tensor:
        """获取分子状态表示
        
        Args:
            smiles: 分子SMILES
            protein_features: 蛋白质特征
            
        Returns:
            状态向量
        """
        # 提取分子特征
        try:
            mol_features_dict = self.feature_extractor.extract_features([smiles])
            self.logger.debug(f"提取的分子特征键: {list(mol_features_dict.keys())}")
            for key, value in mol_features_dict.items():
                self.logger.debug(f"特征 {key} 形状: {value.shape}")
        except Exception as e:
            self.logger.error(f"提取分子特征失败: {e}")
            # 使用默认特征
            mol_features_dict = {
                'descriptors': np.zeros((1, 14), dtype=np.float32),
                'morgan_fp': np.zeros((1, 2048), dtype=np.float32)
            }
        
        # 正确处理字典格式的特征
        feature_vectors = []
        
        # 提取描述符特征
        if 'descriptors' in mol_features_dict and mol_features_dict['descriptors'].size > 0:
            descriptors = mol_features_dict['descriptors']
            if len(descriptors.shape) == 2:
                feature_vectors.append(descriptors[0])  # 取第一个分子的描述符
            else:
                feature_vectors.append(descriptors)
        else:
            # 默认14维描述符
            feature_vectors.append(np.zeros(14, dtype=np.float32))
            
        # 提取指纹特征
        if 'morgan_fp' in mol_features_dict and mol_features_dict['morgan_fp'].size > 0:
            fingerprints = mol_features_dict['morgan_fp']
            if len(fingerprints.shape) == 2:
                feature_vectors.append(fingerprints[0])  # 取第一个分子的指纹
            else:
                feature_vectors.append(fingerprints)
        else:
            # 默认2048维Morgan指纹
            feature_vectors.append(np.zeros(2048, dtype=np.float32))
            
        # 拼接所有特征
        mol_features_combined = np.concatenate(feature_vectors)
        mol_tensor = torch.FloatTensor(mol_features_combined).to(self.device)
        
        # 确保蛋白质特征维度正确 (应该是512维)
        if protein_features.dim() == 1:
            protein_features = protein_features.unsqueeze(0)
        
        # 确保分子特征维度匹配
        if mol_tensor.dim() == 1:
            mol_tensor = mol_tensor.unsqueeze(0)
            
        # 验证维度
        expected_mol_dim = 2062  # 14 + 2048
        expected_protein_dim = 512
        
        self.logger.debug(f"连接后的分子特征形状: {mol_tensor.shape}")
        if mol_tensor.shape[1] != expected_mol_dim:
            self.logger.warning(f"分子特征维度不匹配: 期望{expected_mol_dim}, 实际{mol_tensor.shape[1]}")
            self.logger.warning(f"特征向量详情: {[f'{i}: {v.shape}' for i, v in enumerate(feature_vectors)]}")
            # 调整维度
            if mol_tensor.shape[1] < expected_mol_dim:
                padding_size = expected_mol_dim - mol_tensor.shape[1]
                padding = torch.zeros((mol_tensor.shape[0], padding_size), dtype=mol_tensor.dtype, device=mol_tensor.device)
                mol_tensor = torch.cat([mol_tensor, padding], dim=1)
                self.logger.debug(f"填充后的分子特征形状: {mol_tensor.shape}")
            else:
                mol_tensor = mol_tensor[:, :expected_mol_dim]
                self.logger.debug(f"截断后的分子特征形状: {mol_tensor.shape}")
        
        if protein_features.shape[1] != expected_protein_dim:
            self.logger.warning(f"蛋白质特征维度不匹配: 期望{expected_protein_dim}, 实际{protein_features.shape[1]}")
            
        # 检查维度兼容性
        if protein_features.size(0) != mol_tensor.size(0):
            # 调整batch维度
            if protein_features.size(0) == 1:
                protein_features = protein_features.repeat(mol_tensor.size(0), 1)
            elif mol_tensor.size(0) == 1:
                mol_tensor = mol_tensor.repeat(protein_features.size(0), 1)
            else:
                self.logger.error(f"蛋白质特征和分子特征的batch维度不匹配: {protein_features.shape} vs {mol_tensor.shape}")
                raise ValueError(f"特征维度不匹配: {protein_features.shape} vs {mol_tensor.shape}")
        
        # 拼接蛋白质和分子特征
        combined_features = torch.cat([protein_features, mol_tensor], dim=1)
        self.logger.debug(f"合并后的特征形状: {combined_features.shape}")
        
        # 验证合并后的特征维度
        expected_combined_dim = 2574  # 2062 + 512
        if combined_features.shape[1] != expected_combined_dim:
            self.logger.error(f"合并特征维度错误: 期望{expected_combined_dim}, 实际{combined_features.shape[1]}")
            self.logger.error(f"分子特征维度: {mol_tensor.shape[1]}, 蛋白质特征维度: {protein_features.shape[1]}")
            # 强制调整到正确维度
            if combined_features.shape[1] < expected_combined_dim:
                padding_size = expected_combined_dim - combined_features.shape[1]
                padding = torch.zeros((combined_features.shape[0], padding_size), 
                                    dtype=combined_features.dtype, device=combined_features.device)
                combined_features = torch.cat([combined_features, padding], dim=1)
            else:
                combined_features = combined_features[:, :expected_combined_dim]
            self.logger.debug(f"调整后的合并特征形状: {combined_features.shape}")
        
        # 使用预先创建的状态投影层将特征投影到目标维度
        self.logger.debug(f"state_projection输入维度: {combined_features.shape}")
        state = self.state_projection(combined_features)
        self.logger.debug(f"state_projection输出维度: {state.shape}")
            
        return state
    
    def _apply_action(self, smiles: str, action: np.ndarray) -> Optional[str]:
        """应用动作修改分子
        
        Args:
            smiles: 原始SMILES
            action: 动作向量
            
        Returns:
            修改后的SMILES
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            # 根据动作向量选择修改策略
            action_type = np.argmax(action[:4])  # 前4个元素决定动作类型
            
            if action_type == 0:  # 添加原子
                return self._add_atom(mol, action[4:])
            elif action_type == 1:  # 删除原子
                return self._remove_atom(mol, action[4:])
            elif action_type == 2:  # 修改键
                return self._modify_bond(mol, action[4:])
            else:  # 添加官能团
                return self._add_functional_group(mol, action[4:])
                
        except Exception:
            return None
    
    def _add_atom(self, mol: Chem.Mol, action_params: np.ndarray) -> Optional[str]:
        """添加原子
        
        Args:
            mol: 分子对象
            action_params: 动作参数
            
        Returns:
            修改后的SMILES
        """
        # 简化实现：随机添加碳原子
        try:
            editable_mol = Chem.EditableMol(mol)
            atom_idx = editable_mol.AddAtom(Chem.Atom(6))  # 添加碳原子
            
            # 随机连接到现有原子
            if mol.GetNumAtoms() > 0:
                connect_to = random.randint(0, mol.GetNumAtoms() - 1)
                editable_mol.AddBond(atom_idx, connect_to, Chem.BondType.SINGLE)
                
            new_mol = editable_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            return Chem.MolToSmiles(new_mol)
            
        except Exception:
            return None
    
    def _remove_atom(self, mol: Chem.Mol, action_params: np.ndarray) -> Optional[str]:
        """删除原子
        
        Args:
            mol: 分子对象
            action_params: 动作参数
            
        Returns:
            修改后的SMILES
        """
        try:
            if mol.GetNumAtoms() <= 2:  # 保持最小分子大小
                return None
                
            editable_mol = Chem.EditableMol(mol)
            atom_to_remove = random.randint(0, mol.GetNumAtoms() - 1)
            editable_mol.RemoveAtom(atom_to_remove)
            
            new_mol = editable_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            return Chem.MolToSmiles(new_mol)
            
        except Exception:
            return None
    
    def _modify_bond(self, mol: Chem.Mol, action_params: np.ndarray) -> Optional[str]:
        """修改键
        
        Args:
            mol: 分子对象
            action_params: 动作参数
            
        Returns:
            修改后的SMILES
        """
        try:
            if mol.GetNumBonds() == 0:
                return None
                
            editable_mol = Chem.EditableMol(mol)
            bond_idx = random.randint(0, mol.GetNumBonds() - 1)
            bond = mol.GetBondWithIdx(bond_idx)
            
            # 改变键类型
            new_bond_type = Chem.BondType.DOUBLE if bond.GetBondType() == Chem.BondType.SINGLE else Chem.BondType.SINGLE
            editable_mol.ReplaceBond(bond_idx, Chem.Bond(new_bond_type))
            
            new_mol = editable_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            return Chem.MolToSmiles(new_mol)
            
        except Exception:
            return None
    
    def _add_functional_group(self, mol: Chem.Mol, action_params: np.ndarray) -> Optional[str]:
        """添加官能团
        
        Args:
            mol: 分子对象
            action_params: 动作参数
            
        Returns:
            修改后的SMILES
        """
        # 简化实现：添加甲基
        try:
            if mol.GetNumAtoms() == 0:
                return None
                
            editable_mol = Chem.EditableMol(mol)
            
            # 添加甲基 (-CH3)
            carbon_idx = editable_mol.AddAtom(Chem.Atom(6))
            connect_to = random.randint(0, mol.GetNumAtoms() - 1)
            editable_mol.AddBond(carbon_idx, connect_to, Chem.BondType.SINGLE)
            
            new_mol = editable_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            return Chem.MolToSmiles(new_mol)
            
        except Exception:
            return None
    
    def _features_to_smiles(self, features: np.ndarray) -> Optional[str]:
        """将特征向量转换为SMILES
        
        Args:
            features: 特征向量
            
        Returns:
            SMILES字符串
        """
        try:
            # 将特征向量转换为numpy数组
            if isinstance(features, torch.Tensor):
                features_np = features.cpu().numpy()
            else:
                features_np = features
            
            # 使用特征提取器的逆向转换（如果可用）
            if hasattr(self.feature_extractor, 'features_to_smiles'):
                return self.feature_extractor.features_to_smiles(features_np)
            
            # 简化的特征到SMILES转换
            # 这里实现一个基于特征向量的SMILES生成逻辑
            smiles = self._decode_features_to_smiles(features_np)
            
            # 验证生成的SMILES
            if smiles and self._validate_single_molecule(smiles):
                return smiles
            else:
                # 如果生成失败，返回一个基于特征的简单分子
                return self._generate_fallback_smiles(features_np)
            
        except Exception as e:
            self.logger.warning(f"特征转换为SMILES失败: {e}")
            return self._generate_fallback_smiles(None)
    
    def _decode_features_to_smiles(self, features: np.ndarray) -> str:
        """
        解码特征向量为SMILES
        """
        try:
            # 基于特征向量的简化SMILES生成
            # 这里可以实现更复杂的解码逻辑
            
            # 使用特征向量的统计信息生成SMILES
            feature_sum = np.sum(features)
            feature_mean = np.mean(features)
            feature_std = np.std(features)
            
            # 基于特征统计选择分子骨架
            if feature_mean > 0.5:
                base_smiles = "c1ccccc1"  # 苯环
            elif feature_mean > 0.3:
                base_smiles = "CCCCCC"   # 链状烷烃
            elif feature_mean > 0.1:
                base_smiles = "CCO"      # 简单醇
            else:
                base_smiles = "CC"       # 乙烷
            
            # 基于特征添加官能团
            modified_smiles = self._add_functional_groups(base_smiles, features)
            
            return modified_smiles
            
        except Exception:
            return "CCO"  # 默认分子
    
    def _add_functional_groups(self, base_smiles: str, features: np.ndarray) -> str:
        """
        基于特征向量添加官能团
        """
        try:
            mol = Chem.MolFromSmiles(base_smiles)
            if mol is None:
                return base_smiles
            
            # 基于特征向量的不同部分决定添加的官能团
            if len(features) > 10:
                if features[5] > 0.7:  # 添加羟基
                    if "c1ccccc1" in base_smiles:
                        return "c1ccc(O)cc1"  # 苯酚
                    else:
                        return base_smiles + "O"  # 添加羟基
                
                if features[8] > 0.6:  # 添加氨基
                    if "c1ccccc1" in base_smiles:
                        return "c1ccc(N)cc1"  # 苯胺
                    else:
                        return base_smiles + "N"  # 添加氨基
                
                if features[3] > 0.8:  # 添加羧基
                    return base_smiles + "C(=O)O"
            
            return base_smiles
            
        except Exception:
            return base_smiles
    
    def _generate_fallback_smiles(self, features: Optional[np.ndarray]) -> str:
        """
        生成备用SMILES
        """
        import random
        
        # 药物相似的分子库
        drug_like_smiles = [
            "CCO",  # 乙醇
            "CC(=O)O",  # 乙酸
            "c1ccccc1",  # 苯
            "c1ccc(O)cc1",  # 苯酚
            "c1ccc(N)cc1",  # 苯胺
            "CCN",  # 乙胺
            "CC(C)O",  # 异丙醇
            "CCCC",  # 丁烷
            "c1ccc(C)cc1",  # 甲苯
            "CC(=O)N",  # 乙酰胺
            "c1ccc(Cl)cc1",  # 氯苯
            "CCc1ccccc1",  # 乙苯
            "CC(C)(C)O",  # 叔丁醇
            "c1ccc2ccccc2c1",  # 萘
            "CC(=O)c1ccccc1"  # 苯乙酮
        ]
        
        if features is not None and len(features) > 0:
            # 基于特征选择
            idx = int(abs(np.sum(features)) * len(drug_like_smiles)) % len(drug_like_smiles)
            return drug_like_smiles[idx]
        else:
            return random.choice(drug_like_smiles)
    
    def _is_valid_molecule(self, smiles: str) -> bool:
        """检查分子是否有效
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            是否有效
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
                
            # 检查基本属性
            if mol.GetNumAtoms() < 1 or mol.GetNumAtoms() > 100:
                return False
                
            # 检查分子量
            mw = Descriptors.MolWt(mol)
            if mw < 50 or mw > 1000:
                return False
                
            return True
            
        except Exception:
            return False
    
    def _calculate_reward(self, smiles: str, target_properties: Dict[str, float]) -> float:
        """计算增强的多维度分子奖励
        
        Args:
            smiles: SMILES字符串
            target_properties: 目标属性
            
        Returns:
            奖励值 (0-1)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return -1.0
            
            # 多维度奖励计算
            rewards = {
                'physicochemical': self._calculate_physicochemical_reward(mol, target_properties),
                'drug_likeness': self._calculate_drug_likeness_reward(mol),
                'synthetic': self._calculate_synthetic_reward(mol),
                'dar_prediction': self._calculate_dar_reward(smiles, target_properties),
                'admet': self._calculate_admet_reward(mol),
                'structural': self._calculate_structural_reward(mol, target_properties)
            }
            
            # 权重配置
            weights = {
                'physicochemical': 0.25,
                'drug_likeness': 0.20,
                'synthetic': 0.15,
                'dar_prediction': 0.25,
                'admet': 0.10,
                'structural': 0.05
            }
            
            # 加权平均
            total_reward = sum(rewards[key] * weights[key] for key in rewards)
            
            # 奖励调制
            modulated_reward = self._apply_reward_modulation(total_reward, rewards, mol)
            
            return max(0.0, min(1.0, modulated_reward))
            
        except Exception as e:
            self.logger.warning(f"奖励计算失败 {smiles}: {e}")
            return -1.0
    
    def _calculate_physicochemical_reward(self, mol: Chem.Mol, target_properties: Dict[str, float]) -> float:
        """计算理化性质奖励"""
        try:
            reward = 0.0
            
            # 分子量奖励 (使用高斯函数)
            target_mw = target_properties.get('molecular_weight', 400)
            actual_mw = Descriptors.MolWt(mol)
            mw_reward = self._gaussian_score(actual_mw, target_mw, sigma=100)
            reward += mw_reward * 0.3
            
            # LogP奖励
            target_logp = target_properties.get('logp', 2.5)
            actual_logp = Descriptors.MolLogP(mol)
            logp_reward = self._gaussian_score(actual_logp, target_logp, sigma=1.5)
            reward += logp_reward * 0.3
            
            # TPSA奖励
            target_tpsa = target_properties.get('tpsa', 80)
            actual_tpsa = Descriptors.TPSA(mol)
            tpsa_reward = self._gaussian_score(actual_tpsa, target_tpsa, sigma=40)
            reward += tpsa_reward * 0.2
            
            # 氢键供体/受体奖励
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            hb_reward = (self._gaussian_score(hbd, 2, sigma=1) + self._gaussian_score(hba, 6, sigma=2)) / 2
            reward += hb_reward * 0.2
            
            return reward
            
        except Exception:
            return 0.0
    
    def _calculate_drug_likeness_reward(self, mol: Chem.Mol) -> float:
        """计算药物相似性奖励"""
        try:
            # QED评分
            qed_score = self._calculate_qed(mol)
            
            # Lipinski规则评分
            lipinski_score = self._calculate_lipinski_score(mol)
            
            # Veber规则评分
            veber_score = self._calculate_veber_score(mol)
            
            # 综合评分
            drug_likeness = (qed_score * 0.5 + lipinski_score * 0.3 + veber_score * 0.2)
            
            return drug_likeness
            
        except Exception:
            return 0.0
    
    def _calculate_synthetic_reward(self, mol: Chem.Mol) -> float:
        """计算合成可达性奖励"""
        try:
            sa_score = self._calculate_sa_score(mol)
            # SA评分越低越好，转换为奖励
            synthetic_reward = max(0, (10 - sa_score) / 10)
            
            # 考虑分子复杂度
            complexity = Descriptors.BertzCT(mol)
            complexity_penalty = min(1.0, complexity / 1000)  # 归一化复杂度
            
            return synthetic_reward * (1 - complexity_penalty * 0.3)
            
        except Exception:
            return 0.0
    
    def _calculate_dar_reward(self, smiles: str, target_properties: Dict[str, float]) -> float:
        """计算DAR预测奖励"""
        try:
            # 提取DAR相关特征
            dar_features = self._extract_dar_features(smiles)
            
            # 使用DAR预测器或简化模型
            if hasattr(self, 'dar_predictor') and self.dar_predictor:
                dar_score = self.dar_predictor.predict(smiles)
            else:
                dar_score = self._estimate_dar_from_features(dar_features)
            
            return max(0.0, min(1.0, dar_score))
            
        except Exception:
            return 0.0
    
    def _calculate_admet_reward(self, mol: Chem.Mol) -> float:
        """计算ADMET性质奖励"""
        try:
            admet_score = 0.0
            
            # 口服生物利用度估计
            bioavailability = self._calculate_oral_bioavailability(mol)
            admet_score += bioavailability * 0.4
            
            # 代谢稳定性估计
            metabolic_stability = self._estimate_metabolic_stability(mol)
            admet_score += metabolic_stability * 0.3
            
            # 血脑屏障透过性
            bbb_permeability = self._estimate_bbb_permeability(mol)
            admet_score += bbb_permeability * 0.3
            
            return admet_score
            
        except Exception:
            return 0.0
    
    def _calculate_structural_reward(self, mol: Chem.Mol, target_properties: Dict[str, float]) -> float:
        """计算结构特征奖励"""
        try:
            structural_score = 0.0
            
            # 环系统评分
            ring_count = Descriptors.RingCount(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            ring_score = self._gaussian_score(ring_count, 3, sigma=1) * 0.5 + \
                        self._gaussian_score(aromatic_rings, 2, sigma=1) * 0.5
            structural_score += ring_score * 0.4
            
            # 可旋转键评分
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            flexibility_score = self._gaussian_score(rotatable_bonds, 6, sigma=3)
            structural_score += flexibility_score * 0.3
            
            # 立体化学复杂度
            stereo_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            stereo_score = self._gaussian_score(stereo_centers, 2, sigma=1)
            structural_score += stereo_score * 0.3
            
            return structural_score
            
        except Exception:
            return 0.0
    
    def _gaussian_score(self, value: float, target: float, sigma: float) -> float:
        """高斯评分函数"""
        return np.exp(-0.5 * ((value - target) / sigma) ** 2)
    
    def _calculate_lipinski_score(self, mol: Chem.Mol) -> float:
        """计算Lipinski规则评分"""
        try:
            violations = 0
            
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            if mw > 500: violations += 1
            if logp > 5: violations += 1
            if hbd > 5: violations += 1
            if hba > 10: violations += 1
            
            return max(0, (4 - violations) / 4)
            
        except Exception:
            return 0.0
    
    def _calculate_veber_score(self, mol: Chem.Mol) -> float:
        """计算Veber规则评分"""
        try:
            violations = 0
            
            tpsa = Descriptors.TPSA(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            
            if tpsa > 140: violations += 1
            if rotatable_bonds > 10: violations += 1
            
            return max(0, (2 - violations) / 2)
            
        except Exception:
            return 0.0
    
    def _extract_dar_features(self, smiles: str) -> Dict[str, float]:
        """提取DAR相关特征"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            features = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'heteroatoms': Descriptors.NumHeteroatoms(mol)
            }
            
            return features
            
        except Exception:
            return {}
    
    def _estimate_dar_from_features(self, features: Dict[str, float]) -> float:
        """基于特征估计DAR评分"""
        try:
            if not features:
                return 0.0
            
            # 简化的DAR评分模型
            score = 0.5  # 基础分
            
            # 分子量影响
            mw = features.get('molecular_weight', 400)
            if 200 <= mw <= 600:
                score += 0.2
            elif 150 <= mw < 200 or 600 < mw <= 800:
                score += 0.1
            
            # LogP影响
            logp = features.get('logp', 2)
            if 1 <= logp <= 4:
                score += 0.2
            elif 0 <= logp < 1 or 4 < logp <= 6:
                score += 0.1
            
            # TPSA影响
            tpsa = features.get('tpsa', 80)
            if 40 <= tpsa <= 120:
                score += 0.1
            
            return min(1.0, score)
            
        except Exception:
            return 0.0
    
    def _calculate_oral_bioavailability(self, mol: Chem.Mol) -> float:
        """估计口服生物利用度"""
        try:
            # 基于Lipinski和Veber规则的简化模型
            lipinski = self._calculate_lipinski_score(mol)
            veber = self._calculate_veber_score(mol)
            
            # 额外考虑因素
            tpsa = Descriptors.TPSA(mol)
            mw = Descriptors.MolWt(mol)
            
            bioavailability = (lipinski + veber) / 2
            
            # TPSA调整
            if tpsa > 140:
                bioavailability *= 0.7
            elif tpsa < 60:
                bioavailability *= 0.9
            
            # 分子量调整
            if mw > 500:
                bioavailability *= 0.8
            
            return bioavailability
            
        except Exception:
            return 0.0
    
    def _estimate_metabolic_stability(self, mol: Chem.Mol) -> float:
        """估计代谢稳定性"""
        try:
            # 基于结构特征的简化模型
            stability = 0.5
            
            # 芳香环有助于稳定性
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            stability += min(0.3, aromatic_rings * 0.1)
            
            # 过多的可旋转键降低稳定性
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            if rotatable_bonds > 8:
                stability -= 0.2
            
            # 杂原子影响
            heteroatoms = Descriptors.NumHeteroatoms(mol)
            if heteroatoms > 8:
                stability -= 0.1
            
            return max(0.0, min(1.0, stability))
            
        except Exception:
            return 0.0
    
    def _estimate_bbb_permeability(self, mol: Chem.Mol) -> float:
        """估计血脑屏障透过性"""
        try:
            # 基于分子描述符的简化模型
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            
            # BBB透过性规则
            permeability = 1.0
            
            if mw > 450: permeability *= 0.5
            if logp < 1 or logp > 3: permeability *= 0.7
            if tpsa > 90: permeability *= 0.6
            if hbd > 3: permeability *= 0.5
            
            return permeability
            
        except Exception:
            return 0.0
    
    def _apply_reward_modulation(self, base_reward: float, rewards: Dict[str, float], mol: Chem.Mol) -> float:
        """应用奖励调制机制"""
        try:
            modulated_reward = base_reward
            
            # 平衡性奖励：各维度评分不能过于极端
            reward_values = list(rewards.values())
            reward_std = np.std(reward_values)
            balance_bonus = max(0, 0.1 - reward_std)  # 标准差越小，平衡性越好
            modulated_reward += balance_bonus
            
            # 分子有效性检查
            if mol is None:
                return -1.0
            
            # 基本化学有效性
            try:
                Chem.SanitizeMol(mol)
                validity_bonus = 0.05
            except:
                validity_bonus = -0.2
            
            modulated_reward += validity_bonus
            
            # 新颖性奖励（基于分子复杂度）
            complexity = Descriptors.BertzCT(mol)
            novelty_bonus = min(0.1, complexity / 2000)  # 适度的复杂度奖励
            modulated_reward += novelty_bonus
            
            return modulated_reward
            
        except Exception:
            return base_reward
    
    def _calculate_qed(self, mol: Chem.Mol) -> float:
        """计算QED（药物相似性）
        
        Args:
            mol: 分子对象
            
        Returns:
            QED值
        """
        try:
            from rdkit.Chem import QED
            return QED.qed(mol)
        except ImportError:
            # 如果QED模块不可用，使用简化的药物相似性评分
            return self._calculate_simple_drug_likeness(mol)
        except Exception:
            return 0.0
    
    def _calculate_simple_drug_likeness(self, mol) -> float:
        """
        计算简化的药物相似性评分
        """
        try:
            # Lipinski规则的简化实现
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            score = 1.0
            
            # 分子量评分 (150-500 Da)
            if 150 <= mw <= 500:
                score *= 1.0
            elif 100 <= mw < 150 or 500 < mw <= 600:
                score *= 0.8
            else:
                score *= 0.3
            
            # LogP评分 (-0.4 to 5.6)
            if -0.4 <= logp <= 5.6:
                score *= 1.0
            elif -2 <= logp < -0.4 or 5.6 < logp <= 7:
                score *= 0.7
            else:
                score *= 0.2
            
            # 氢键供体评分 (≤5)
            if hbd <= 5:
                score *= 1.0
            else:
                score *= 0.5
            
            # 氢键受体评分 (≤10)
            if hba <= 10:
                score *= 1.0
            else:
                score *= 0.5
            
            return min(1.0, score)
            
        except Exception:
            return 0.5  # 默认中等评分
    
    def _calculate_sa_score(self, mol: Chem.Mol) -> float:
        """
        计算合成可达性评分 (Synthetic Accessibility Score)
        
        Args:
            mol: RDKit分子对象
            
        Returns:
            合成可达性评分 (1-10, 1表示最容易合成)
        """
        try:
            # 尝试使用RDKit的SA_Score模块
            from rdkit.Contrib.SA_Score import sascorer
            return sascorer.calculateScore(mol)
        except ImportError:
            # 如果SA_Score模块不可用，使用简化的合成可达性评分
            return self._calculate_simple_sa_score(mol)
        except Exception:
            return 5.0  # 默认中等评分
    
    def _calculate_simple_sa_score(self, mol: Chem.Mol) -> float:
        """
        计算简化的合成可达性评分
        """
        try:
            # 基于分子复杂度的简化评分
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            num_rings = Descriptors.RingCount(mol)
            num_aromatic_rings = Descriptors.NumAromaticRings(mol)
            num_heteroatoms = Descriptors.NumHeteroatoms(mol)
            
            # 基础评分
            score = 1.0
            
            # 原子数量影响
            if num_atoms > 50:
                score += 2.0
            elif num_atoms > 30:
                score += 1.0
            elif num_atoms > 20:
                score += 0.5
            
            # 环系统复杂度
            if num_rings > 4:
                score += 2.0
            elif num_rings > 2:
                score += 1.0
            
            # 芳香环影响
            if num_aromatic_rings > 3:
                score += 1.5
            elif num_aromatic_rings > 1:
                score += 0.5
            
            # 杂原子影响
            if num_heteroatoms > 8:
                score += 1.0
            elif num_heteroatoms > 4:
                score += 0.5
            
            # 立体化学复杂度
            num_stereo_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            if num_stereo_centers > 4:
                score += 2.0
            elif num_stereo_centers > 2:
                score += 1.0
            
            return min(10.0, score)
            
        except Exception:
            return 5.0  # 默认中等评分
    
    def _calculate_diversity_score(self, mol: Chem.Mol, reference_mols: List[Chem.Mol]) -> float:
        """
        计算分子与参考分子集的多样性评分
        
        Args:
            mol: 目标分子
            reference_mols: 参考分子列表
            
        Returns:
            多样性评分 (0-1, 1表示最多样)
        """
        try:
            if not reference_mols:
                return 1.0
            
            # 计算Morgan指纹
            target_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            
            # 计算与所有参考分子的Tanimoto相似性
            similarities = []
            for ref_mol in reference_mols:
                try:
                    ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
                    similarity = DataStructs.TanimotoSimilarity(target_fp, ref_fp)
                    similarities.append(similarity)
                except Exception:
                    continue
            
            if not similarities:
                return 1.0
            
            # 多样性 = 1 - 最大相似性
            max_similarity = max(similarities)
            diversity = 1.0 - max_similarity
            
            return diversity
            
        except Exception:
            return 0.5  # 默认中等多样性
    
    def _check_memory_usage(self):
        """
        检查内存使用情况
        """
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent / 100.0
            
            if memory_percent > self.memory_threshold:
                self.logger.warning(f"内存使用率过高: {memory_percent:.1%}，触发垃圾回收")
                self._force_garbage_collection()
                
            # GPU内存检查
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if gpu_memory > self.memory_threshold:
                    self.logger.warning(f"GPU内存使用率过高: {gpu_memory:.1%}，清理缓存")
                    torch.cuda.empty_cache()
                    
        except ImportError:
            # 如果没有psutil，使用简单的计数器
            self.operation_count += 1
            if self.operation_count % self.gc_frequency == 0:
                self._force_garbage_collection()
    
    def _calculate_optimal_batch_size(self, total_molecules: int) -> int:
        """
        计算最优批处理大小
        """
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # 根据可用内存动态调整批处理大小
            if available_memory_gb > 8:
                base_batch_size = self.max_parallel_workers
            elif available_memory_gb > 4:
                base_batch_size = max(2, self.max_parallel_workers // 2)
            else:
                base_batch_size = 1
                
            # 确保不超过总分子数
            return min(base_batch_size, total_molecules, self.max_parallel_workers)
            
        except ImportError:
            return min(self.max_parallel_workers, total_molecules)
    
    def _manage_memory(self):
        """
        内存管理
        """
        self.operation_count += 1
        
        # 定期垃圾回收
        if self.operation_count % self.gc_frequency == 0:
            self._force_garbage_collection()
            
        # 清理缓存（如果内存紧张）
        try:
            import psutil
            if psutil.virtual_memory().percent > 85:  # 85%内存使用率
                self._clear_caches()
        except ImportError:
            pass
    
    def _force_garbage_collection(self):
        """
        强制垃圾回收
        """
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _clear_caches(self):
        """
        清理缓存
        """
        if hasattr(self, 'feature_cache') and self.feature_cache:
            cache_size = len(self.feature_cache)
            # 只保留最近使用的50%
            if cache_size > 100:
                items = list(self.feature_cache.items())
                self.feature_cache.clear()
                self.feature_cache.update(items[-cache_size//2:])
                
        if hasattr(self, 'validity_cache') and self.validity_cache:
            cache_size = len(self.validity_cache)
            if cache_size > 100:
                items = list(self.validity_cache.items())
                self.validity_cache.clear()
                self.validity_cache.update(items[-cache_size//2:])
    
    def _serial_optimize_molecules(self, molecules: List[str], 
                                 protein_features: torch.Tensor,
                                 target_properties: Dict[str, float],
                                 num_iterations: int) -> List[str]:
        """
        串行优化分子（降级处理）
        """
        self.logger.info("使用串行处理模式优化分子")
        optimized_molecules = []
        
        for i, smiles in enumerate(molecules):
            try:
                optimized = self._optimize_single_molecule_enhanced(
                    smiles, protein_features, target_properties, num_iterations
                )
                optimized_molecules.append(optimized)
                
                # 进度报告
                if (i + 1) % 10 == 0:
                    self.logger.info(f"串行优化进度: {i + 1}/{len(molecules)}")
                    
                # 内存管理
                if (i + 1) % 50 == 0:
                    self._manage_memory()
                    
            except Exception as e:
                self.logger.warning(f"串行优化分子 {smiles} 失败: {e}")
                optimized_molecules.append(smiles)
        
        return optimized_molecules
    
    def _cleanup_resources(self):
        """
        清理资源（增强版）
        """
        try:
            # 清理缓存
            if hasattr(self, 'feature_cache'):
                self.feature_cache.clear()
            if hasattr(self, 'validity_cache'):
                self.validity_cache.clear()
            
            # 关闭线程池
            if hasattr(self, 'thread_pool') and self.thread_pool:
                self.thread_pool.shutdown(wait=True, timeout=30)
            if hasattr(self, 'process_pool') and self.process_pool:
                self.process_pool.shutdown(wait=True, timeout=30)
            
            # 强制垃圾回收
            self._force_garbage_collection()
            
            self.logger.info("资源清理完成")
            
        except Exception as e:
            self.logger.warning(f"资源清理时出现错误: {e}")
    
    def __del__(self):
        """
        析构函数
        """
        try:
            self._cleanup_resources()
        except Exception:
            pass
    
    def generate_diverse_molecules(self, protein_features: torch.Tensor,
                                 target_properties: Dict[str, float],
                                 num_molecules: int = 100,
                                 diversity_weight: float = 0.3) -> List[str]:
        """生成多样化分子
        
        Args:
            protein_features: 蛋白质特征
            target_properties: 目标属性
            num_molecules: 生成数量
            diversity_weight: 多样性权重
            
        Returns:
            多样化分子SMILES列表
        """
        self.logger.info(f"开始生成 {num_molecules} 个多样化分子")
        
        # 生成初始分子
        initial_molecules = self.generate_initial_molecules(protein_features, num_molecules * 2)
        
        # 优化分子
        if self.rl_agent:
            optimized_molecules = self.optimize_molecules(initial_molecules, protein_features, target_properties)
        else:
            optimized_molecules = initial_molecules
            
        # 选择多样化分子
        diverse_molecules = self._select_diverse_molecules(
            optimized_molecules, target_properties, num_molecules, diversity_weight
        )
        
        self.logger.info(f"成功生成 {len(diverse_molecules)} 个多样化分子")
        return diverse_molecules
    
    def _select_diverse_molecules(self, molecules: List[str], 
                                target_properties: Dict[str, float],
                                num_select: int,
                                diversity_weight: float) -> List[str]:
        """选择多样化分子（增强版，支持多种相似性度量和聚类算法）
        
        Args:
            molecules: 候选分子
            target_properties: 目标属性
            num_select: 选择数量
            diversity_weight: 多样性权重
            
        Returns:
            选择的分子
        """
        if len(molecules) <= num_select:
            return molecules
            
        try:
            # 计算多种分子特征
            molecule_data = []
            for smiles in molecules:
                mol_data = self._compute_enhanced_molecular_features(smiles, target_properties)
                if mol_data is not None:
                    molecule_data.append(mol_data)
            
            if not molecule_data:
                return molecules[:num_select]
            
            # 使用聚类算法进行多样性选择
            if len(molecule_data) > num_select * 2:
                selected = self._cluster_based_selection(molecule_data, num_select, diversity_weight)
            else:
                # 使用改进的贪心算法
                selected = self._enhanced_greedy_selection(molecule_data, num_select, diversity_weight)
            
            return [data['smiles'] for data in selected]
            
        except Exception as e:
            self.logger.warning(f"多样性选择失败，使用简单选择: {e}")
            return molecules[:num_select]
    
    def _get_fingerprint(self, smiles: str, fp_type: str = 'morgan') -> np.ndarray:
        """获取分子指纹（支持多种指纹类型）
        
        Args:
            smiles: SMILES字符串
            fp_type: 指纹类型 ('morgan', 'rdkit', 'maccs', 'topological')
            
        Returns:
            分子指纹
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(2048)
                
            if fp_type == 'morgan':
                from rdkit.Chem import rdMolDescriptors
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            elif fp_type == 'rdkit':
                from rdkit.Chem import rdMolDescriptors
                fp = rdMolDescriptors.RDKFingerprint(mol, fpSize=2048)
            elif fp_type == 'maccs':
                from rdkit.Chem import MACCSkeys
                fp = MACCSkeys.GenMACCSKeys(mol)
                # 扩展到2048位
                fp_array = np.array(fp)
                fp = np.pad(fp_array, (0, 2048 - len(fp_array)), 'constant')
                return fp
            elif fp_type == 'topological':
                from rdkit.Chem import rdMolDescriptors
                fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)
            else:
                # 默认使用Morgan指纹
                from rdkit.Chem import rdMolDescriptors
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                
            return np.array(fp)
            
        except Exception:
            return np.zeros(2048)
    
    def _tanimoto_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """计算Tanimoto相似性
        
        Args:
            fp1: 指纹1
            fp2: 指纹2
            
        Returns:
            相似性值
        """
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def _compute_enhanced_molecular_features(self, smiles: str, target_properties: Dict[str, float]) -> Optional[Dict]:
        """计算增强的分子特征
        
        Args:
            smiles: SMILES字符串
            target_properties: 目标属性
            
        Returns:
            分子特征字典
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 计算多种指纹
            morgan_fp = self._get_fingerprint(smiles, 'morgan')
            rdkit_fp = self._get_fingerprint(smiles, 'rdkit')
            maccs_fp = self._get_fingerprint(smiles, 'maccs')
            
            # 计算分子描述符
            descriptors = self._compute_molecular_descriptors(mol)
            
            # 计算奖励得分
            reward = self._calculate_reward(smiles, target_properties)
            
            return {
                'smiles': smiles,
                'mol': mol,
                'morgan_fp': morgan_fp,
                'rdkit_fp': rdkit_fp,
                'maccs_fp': maccs_fp,
                'descriptors': descriptors,
                'reward': reward
            }
            
        except Exception as e:
            self.logger.warning(f"计算分子特征失败 {smiles}: {e}")
            return None
    
    def _compute_molecular_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """计算分子描述符
        
        Args:
            mol: RDKit分子对象
            
        Returns:
            描述符向量
        """
        try:
            descriptors = []
            
            # 基本描述符
            descriptors.append(Descriptors.MolWt(mol))  # 分子量
            descriptors.append(Descriptors.MolLogP(mol))  # LogP
            descriptors.append(Descriptors.NumHDonors(mol))  # 氢键供体
            descriptors.append(Descriptors.NumHAcceptors(mol))  # 氢键受体
            descriptors.append(Descriptors.TPSA(mol))  # 拓扑极性表面积
            descriptors.append(Descriptors.NumRotatableBonds(mol))  # 可旋转键
            descriptors.append(Descriptors.NumAromaticRings(mol))  # 芳香环数
            descriptors.append(Descriptors.NumSaturatedRings(mol))  # 饱和环数
            descriptors.append(Descriptors.FractionCsp3(mol))  # sp3碳比例
            descriptors.append(Descriptors.BertzCT(mol))  # 分子复杂度
            
            # 归一化描述符
            descriptors = np.array(descriptors, dtype=np.float32)
            descriptors = np.nan_to_num(descriptors, 0.0)
            
            return descriptors
            
        except Exception:
            return np.zeros(10, dtype=np.float32)
    
    def _cluster_based_selection(self, molecule_data: List[Dict], num_select: int, diversity_weight: float) -> List[Dict]:
        """基于聚类的多样性选择
        
        Args:
            molecule_data: 分子数据列表
            num_select: 选择数量
            diversity_weight: 多样性权重
            
        Returns:
            选择的分子数据
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # 组合特征矩阵
            features = []
            for data in molecule_data:
                # 组合多种特征
                combined_features = np.concatenate([
                    data['morgan_fp'][:512],  # 压缩Morgan指纹
                    data['rdkit_fp'][:512],   # 压缩RDKit指纹
                    data['maccs_fp'][:167],   # MACCS指纹
                    data['descriptors']       # 分子描述符
                ])
                features.append(combined_features)
            
            features = np.array(features)
            
            # 标准化特征
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # K-means聚类
            n_clusters = min(num_select, len(molecule_data) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # 从每个聚类中选择最佳分子
            selected = []
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_indices) == 0:
                    continue
                
                # 在聚类内选择奖励最高的分子
                cluster_molecules = [molecule_data[i] for i in cluster_indices]
                best_mol = max(cluster_molecules, key=lambda x: x['reward'])
                selected.append(best_mol)
            
            # 如果选择的分子不够，用贪心算法补充
            if len(selected) < num_select:
                remaining = [mol for mol in molecule_data if mol not in selected]
                additional = self._enhanced_greedy_selection(
                    remaining, num_select - len(selected), diversity_weight, selected
                )
                selected.extend(additional)
            
            return selected[:num_select]
            
        except Exception as e:
            self.logger.warning(f"聚类选择失败，使用贪心算法: {e}")
            return self._enhanced_greedy_selection(molecule_data, num_select, diversity_weight)
    
    def _enhanced_greedy_selection(self, molecule_data: List[Dict], num_select: int, 
                                 diversity_weight: float, initial_selected: List[Dict] = None) -> List[Dict]:
        """增强的贪心多样性选择
        
        Args:
            molecule_data: 分子数据列表
            num_select: 选择数量
            diversity_weight: 多样性权重
            initial_selected: 初始已选择的分子
            
        Returns:
            选择的分子数据
        """
        if initial_selected is None:
            selected = []
        else:
            selected = initial_selected.copy()
        
        available = [mol for mol in molecule_data if mol not in selected]
        
        # 如果没有初始选择，选择奖励最高的分子作为起始
        if not selected and available:
            best_mol = max(available, key=lambda x: x['reward'])
            selected.append(best_mol)
            available.remove(best_mol)
        
        # 迭代选择剩余分子
        while len(selected) < num_select and available:
            best_score = -float('inf')
            best_mol = None
            
            for mol in available:
                # 计算多维度多样性得分
                diversity_scores = []
                
                for sel_mol in selected:
                    # Morgan指纹相似性
                    morgan_sim = self._tanimoto_similarity(mol['morgan_fp'], sel_mol['morgan_fp'])
                    # RDKit指纹相似性
                    rdkit_sim = self._tanimoto_similarity(mol['rdkit_fp'], sel_mol['rdkit_fp'])
                    # MACCS指纹相似性
                    maccs_sim = self._tanimoto_similarity(mol['maccs_fp'], sel_mol['maccs_fp'])
                    # 描述符相似性
                    desc_sim = self._cosine_similarity(mol['descriptors'], sel_mol['descriptors'])
                    
                    # 综合相似性
                    combined_sim = (morgan_sim + rdkit_sim + maccs_sim + desc_sim) / 4
                    diversity_scores.append(1 - combined_sim)
                
                # 最小多样性得分（与最相似分子的多样性）
                min_diversity = min(diversity_scores) if diversity_scores else 1.0
                
                # 综合得分
                total_score = (1 - diversity_weight) * mol['reward'] + diversity_weight * min_diversity
                
                if total_score > best_score:
                    best_score = total_score
                    best_mol = mol
            
            if best_mol:
                selected.append(best_mol)
                available.remove(best_mol)
            else:
                break
        
        return selected[len(initial_selected or []):]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似性
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            余弦相似性
        """
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception:
            return 0.0