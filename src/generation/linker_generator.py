"""Linker生成器模块

专门用于生成连接蛋白质和药物分子的Linker结构。
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
import random

class LinkerGenerator:
    """Linker生成器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Linker参数
        self.min_length = config.get('min_length', 2)
        self.max_length = config.get('max_length', 20)
        self.flexibility_weight = config.get('flexibility_weight', 0.3)
        self.stability_weight = config.get('stability_weight', 0.4)
        self.cleavability_weight = config.get('cleavability_weight', 0.3)
        
        # 常用Linker片段
        self.linker_fragments = {
            'peg': ['CCOCCO', 'CCOCCOCCOC', 'CCOCCOCCOCCO'],  # PEG片段
            'alkyl': ['CC', 'CCC', 'CCCC', 'CCCCC'],  # 烷基链
            'peptide': ['CC(=O)N', 'CNC(=O)', 'CC(=O)NC'],  # 肽键
            'ester': ['CC(=O)O', 'COC(=O)', 'OC(=O)C'],  # 酯键
            'amide': ['CC(=O)N', 'NC(=O)C', 'CNC(=O)C'],  # 酰胺键
            'disulfide': ['CSSC', 'CC(S)SC'],  # 二硫键
            'hydrazone': ['CNN=C', 'C=NNC'],  # 腙键
            'triazole': ['c1cn[nH]c1', 'c1cnn[nH]1']  # 三唑环
        }
        
        # 可切割位点
        self.cleavable_sites = {
            'protease': ['CC(=O)NC(C)C(=O)', 'CC(=O)NC(CC)C(=O)'],  # 蛋白酶切割
            'esterase': ['CC(=O)OC', 'COC(=O)C'],  # 酯酶切割
            'glycosidase': ['COC1OC(CO)C(O)C(O)C1O'],  # 糖苷酶切割
            'reductase': ['CSSC', 'C(=O)C(=O)'],  # 还原酶切割
            'acid_labile': ['COC(C)(C)OC(=O)', 'CC(C)(C)OC(=O)']  # 酸不稳定
        }
        
        self.logger.info("Linker生成器初始化完成")
    
    def generate_linkers(self, protein_features: torch.Tensor,
                        drug_features: torch.Tensor,
                        target_properties: Dict[str, float],
                        num_linkers: int = 50) -> List[str]:
        """生成Linker分子
        
        Args:
            protein_features: 蛋白质特征
            drug_features: 药物特征
            target_properties: 目标属性
            num_linkers: 生成数量
            
        Returns:
            Linker SMILES列表
        """
        self.logger.info(f"开始生成 {num_linkers} 个Linker")
        
        linkers = []
        
        # 基于规则的生成
        rule_based_linkers = self._generate_rule_based_linkers(
            protein_features, drug_features, target_properties, num_linkers // 2
        )
        linkers.extend(rule_based_linkers)
        
        # 基于片段的组合生成
        fragment_based_linkers = self._generate_fragment_based_linkers(
            protein_features, drug_features, target_properties, num_linkers - len(linkers)
        )
        linkers.extend(fragment_based_linkers)
        
        # 过滤和验证
        valid_linkers = []
        for linker in linkers:
            if self._validate_linker(linker, target_properties):
                valid_linkers.append(linker)
                
        self.logger.info(f"成功生成 {len(valid_linkers)} 个有效Linker")
        return valid_linkers[:num_linkers]
    
    def _generate_rule_based_linkers(self, protein_features: torch.Tensor,
                                   drug_features: torch.Tensor,
                                   target_properties: Dict[str, float],
                                   num_linkers: int) -> List[str]:
        """基于规则生成Linker
        
        Args:
            protein_features: 蛋白质特征
            drug_features: 药物特征
            target_properties: 目标属性
            num_linkers: 生成数量
            
        Returns:
            Linker SMILES列表
        """
        linkers = []
        
        # 根据目标属性选择Linker类型
        target_length = target_properties.get('linker_length', 10)
        cleavability = target_properties.get('cleavability', 0.5)
        flexibility = target_properties.get('flexibility', 0.5)
        
        for _ in range(num_linkers):
            try:
                # 选择Linker骨架类型
                if cleavability > 0.7:
                    # 高可切割性：选择含有可切割位点的Linker
                    linker = self._build_cleavable_linker(target_length)
                elif flexibility > 0.7:
                    # 高柔性：选择PEG或烷基链
                    linker = self._build_flexible_linker(target_length)
                else:
                    # 平衡型：混合不同片段
                    linker = self._build_balanced_linker(target_length)
                    
                if linker:
                    linkers.append(linker)
                    
            except Exception as e:
                self.logger.warning(f"生成规则Linker时出错: {e}")
                
        return linkers
    
    def _build_cleavable_linker(self, target_length: int) -> Optional[str]:
        """构建可切割Linker
        
        Args:
            target_length: 目标长度
            
        Returns:
            Linker SMILES
        """
        try:
            # 选择可切割位点
            cleavage_type = random.choice(list(self.cleavable_sites.keys()))
            cleavage_site = random.choice(self.cleavable_sites[cleavage_type])
            
            # 添加连接片段
            remaining_length = max(0, target_length - len(cleavage_site))
            
            if remaining_length > 0:
                # 在两端添加连接片段
                left_length = remaining_length // 2
                right_length = remaining_length - left_length
                
                left_fragment = self._get_connecting_fragment(left_length)
                right_fragment = self._get_connecting_fragment(right_length)
                
                linker = left_fragment + cleavage_site + right_fragment
            else:
                linker = cleavage_site
                
            return linker
            
        except Exception:
            return None
    
    def _build_flexible_linker(self, target_length: int) -> Optional[str]:
        """构建柔性Linker
        
        Args:
            target_length: 目标长度
            
        Returns:
            Linker SMILES
        """
        try:
            # 优先选择PEG片段
            if random.random() < 0.7:
                fragments = self.linker_fragments['peg']
            else:
                fragments = self.linker_fragments['alkyl']
                
            # 组合片段达到目标长度
            linker = ""
            current_length = 0
            
            while current_length < target_length:
                fragment = random.choice(fragments)
                if current_length + len(fragment) <= target_length + 2:  # 允许小幅超出
                    linker += fragment
                    current_length += len(fragment)
                else:
                    break
                    
            return linker if linker else random.choice(fragments)
            
        except Exception:
            return None
    
    def _build_balanced_linker(self, target_length: int) -> Optional[str]:
        """构建平衡型Linker
        
        Args:
            target_length: 目标长度
            
        Returns:
            Linker SMILES
        """
        try:
            # 混合不同类型的片段
            fragment_types = ['peg', 'alkyl', 'peptide', 'ester']
            
            linker = ""
            current_length = 0
            
            while current_length < target_length:
                fragment_type = random.choice(fragment_types)
                fragment = random.choice(self.linker_fragments[fragment_type])
                
                if current_length + len(fragment) <= target_length + 3:
                    linker += fragment
                    current_length += len(fragment)
                else:
                    break
                    
            return linker if linker else "CCOCCOC"  # 默认PEG片段
            
        except Exception:
            return None
    
    def _get_connecting_fragment(self, length: int) -> str:
        """获取连接片段
        
        Args:
            length: 片段长度
            
        Returns:
            连接片段SMILES
        """
        if length <= 0:
            return ""
        elif length <= 2:
            return "C" * length
        elif length <= 5:
            return random.choice(self.linker_fragments['alkyl'])
        else:
            return random.choice(self.linker_fragments['peg'])
    
    def _generate_fragment_based_linkers(self, protein_features: torch.Tensor,
                                       drug_features: torch.Tensor,
                                       target_properties: Dict[str, float],
                                       num_linkers: int) -> List[str]:
        """基于片段组合生成Linker
        
        Args:
            protein_features: 蛋白质特征
            drug_features: 药物特征
            target_properties: 目标属性
            num_linkers: 生成数量
            
        Returns:
            Linker SMILES列表
        """
        linkers = []
        
        for _ in range(num_linkers):
            try:
                # 随机选择2-4个片段进行组合
                num_fragments = random.randint(2, 4)
                selected_fragments = []
                
                for _ in range(num_fragments):
                    fragment_type = random.choice(list(self.linker_fragments.keys()))
                    fragment = random.choice(self.linker_fragments[fragment_type])
                    selected_fragments.append(fragment)
                    
                # 组合片段
                linker = "".join(selected_fragments)
                
                # 检查长度
                if self.min_length <= len(linker) <= self.max_length:
                    linkers.append(linker)
                    
            except Exception as e:
                self.logger.warning(f"生成片段Linker时出错: {e}")
                
        return linkers
    
    def _validate_linker(self, smiles: str, target_properties: Dict[str, float]) -> bool:
        """验证Linker有效性
        
        Args:
            smiles: Linker SMILES
            target_properties: 目标属性
            
        Returns:
            是否有效
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
                
            # 检查基本属性
            num_atoms = mol.GetNumAtoms()
            if num_atoms < self.min_length or num_atoms > self.max_length:
                return False
                
            # 检查分子量
            mw = Descriptors.MolWt(mol)
            if mw < 50 or mw > 2000:
                return False
                
            # 检查LogP（亲脂性）
            logp = Descriptors.MolLogP(mol)
            if logp < -5 or logp > 10:
                return False
                
            # 检查旋转键数量（柔性指标）
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            max_rotatable = target_properties.get('max_rotatable_bonds', 20)
            if rotatable_bonds > max_rotatable:
                return False
                
            return True
            
        except Exception:
            return False
    
    def evaluate_linker_properties(self, smiles: str) -> Dict[str, float]:
        """评估Linker属性
        
        Args:
            smiles: Linker SMILES
            
        Returns:
            属性字典
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
                
            properties = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'flexibility': self._calculate_flexibility(mol),
                'stability': self._calculate_stability(mol),
                'cleavability': self._calculate_cleavability(mol)
            }
            
            return properties
            
        except Exception as e:
            self.logger.warning(f"评估Linker属性时出错: {e}")
            return {}
    
    def _calculate_flexibility(self, mol: Chem.Mol) -> float:
        """计算Linker柔性
        
        Args:
            mol: 分子对象
            
        Returns:
            柔性评分 (0-1)
        """
        try:
            # 基于旋转键数量和分子长度
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            num_atoms = mol.GetNumAtoms()
            
            if num_atoms == 0:
                return 0.0
                
            # 归一化柔性评分
            flexibility = min(rotatable_bonds / num_atoms * 2, 1.0)
            return flexibility
            
        except Exception:
            return 0.0
    
    def _calculate_stability(self, mol: Chem.Mol) -> float:
        """计算Linker稳定性
        
        Args:
            mol: 分子对象
            
        Returns:
            稳定性评分 (0-1)
        """
        try:
            # 基于键的类型和强度
            stability = 1.0
            
            # 检查不稳定键
            for bond in mol.GetBonds():
                bond_type = bond.GetBondType()
                
                # 酯键相对不稳定
                if self._is_ester_bond(bond, mol):
                    stability -= 0.1
                    
                # 二硫键在还原环境下不稳定
                elif self._is_disulfide_bond(bond, mol):
                    stability -= 0.2
                    
                # 酰胺键相对稳定
                elif self._is_amide_bond(bond, mol):
                    stability += 0.1
                    
            return max(0.0, min(stability, 1.0))
            
        except Exception:
            return 0.5
    
    def _calculate_cleavability(self, mol: Chem.Mol) -> float:
        """计算Linker可切割性
        
        Args:
            mol: 分子对象
            
        Returns:
            可切割性评分 (0-1)
        """
        try:
            cleavability = 0.0
            smiles = Chem.MolToSmiles(mol)
            
            # 检查已知的可切割位点
            for cleavage_type, sites in self.cleavable_sites.items():
                for site in sites:
                    if site in smiles:
                        if cleavage_type == 'protease':
                            cleavability += 0.3
                        elif cleavage_type == 'esterase':
                            cleavability += 0.2
                        elif cleavage_type == 'reductase':
                            cleavability += 0.4
                        else:
                            cleavability += 0.1
                            
            return min(cleavability, 1.0)
            
        except Exception:
            return 0.0
    
    def _is_ester_bond(self, bond: Chem.Bond, mol: Chem.Mol) -> bool:
        """检查是否为酯键
        
        Args:
            bond: 键对象
            mol: 分子对象
            
        Returns:
            是否为酯键
        """
        try:
            atom1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
            
            # C-O键，且其中一个碳连接双键氧
            if ((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'O') or
                (atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'C')):
                
                carbon_atom = atom1 if atom1.GetSymbol() == 'C' else atom2
                
                # 检查碳是否连接双键氧
                for neighbor in carbon_atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'O':
                        bond_to_o = mol.GetBondBetweenAtoms(carbon_atom.GetIdx(), neighbor.GetIdx())
                        if bond_to_o.GetBondType() == Chem.BondType.DOUBLE:
                            return True
                            
            return False
            
        except Exception:
            return False
    
    def _is_disulfide_bond(self, bond: Chem.Bond, mol: Chem.Mol) -> bool:
        """检查是否为二硫键
        
        Args:
            bond: 键对象
            mol: 分子对象
            
        Returns:
            是否为二硫键
        """
        try:
            atom1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
            
            return (atom1.GetSymbol() == 'S' and atom2.GetSymbol() == 'S')
            
        except Exception:
            return False
    
    def _is_amide_bond(self, bond: Chem.Bond, mol: Chem.Mol) -> bool:
        """检查是否为酰胺键
        
        Args:
            bond: 键对象
            mol: 分子对象
            
        Returns:
            是否为酰胺键
        """
        try:
            atom1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
            
            # C-N键，且碳连接双键氧
            if ((atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'N') or
                (atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'C')):
                
                carbon_atom = atom1 if atom1.GetSymbol() == 'C' else atom2
                
                # 检查碳是否连接双键氧
                for neighbor in carbon_atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'O':
                        bond_to_o = mol.GetBondBetweenAtoms(carbon_atom.GetIdx(), neighbor.GetIdx())
                        if bond_to_o.GetBondType() == Chem.BondType.DOUBLE:
                            return True
                            
            return False
            
        except Exception:
            return False
    
    def optimize_linker_for_target(self, initial_linker: str,
                                 protein_features: torch.Tensor,
                                 drug_features: torch.Tensor,
                                 target_properties: Dict[str, float],
                                 num_iterations: int = 20) -> str:
        """针对目标优化Linker
        
        Args:
            initial_linker: 初始Linker SMILES
            protein_features: 蛋白质特征
            drug_features: 药物特征
            target_properties: 目标属性
            num_iterations: 优化迭代次数
            
        Returns:
            优化后的Linker SMILES
        """
        current_linker = initial_linker
        best_linker = initial_linker
        best_score = self._score_linker(initial_linker, target_properties)
        
        for iteration in range(num_iterations):
            try:
                # 生成变体
                variants = self._generate_linker_variants(current_linker)
                
                for variant in variants:
                    if self._validate_linker(variant, target_properties):
                        score = self._score_linker(variant, target_properties)
                        
                        if score > best_score:
                            best_score = score
                            best_linker = variant
                            current_linker = variant
                            break
                            
            except Exception as e:
                self.logger.warning(f"优化Linker时出错: {e}")
                
        return best_linker
    
    def _generate_linker_variants(self, smiles: str) -> List[str]:
        """生成Linker变体
        
        Args:
            smiles: 原始SMILES
            
        Returns:
            变体SMILES列表
        """
        variants = []
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return variants
                
            # 添加片段
            for fragment_type in ['peg', 'alkyl']:
                for fragment in self.linker_fragments[fragment_type][:2]:  # 只取前2个
                    new_smiles = smiles + fragment
                    variants.append(new_smiles)
                    
                    new_smiles = fragment + smiles
                    variants.append(new_smiles)
                    
            # 替换片段（简化实现）
            if len(smiles) > 4:
                for i in range(0, len(smiles) - 2, 2):
                    for fragment in self.linker_fragments['peg'][:2]:
                        new_smiles = smiles[:i] + fragment + smiles[i+2:]
                        variants.append(new_smiles)
                        
        except Exception:
            pass
            
        return variants[:10]  # 限制变体数量
    
    def _score_linker(self, smiles: str, target_properties: Dict[str, float]) -> float:
        """评分Linker
        
        Args:
            smiles: Linker SMILES
            target_properties: 目标属性
            
        Returns:
            评分
        """
        try:
            properties = self.evaluate_linker_properties(smiles)
            if not properties:
                return 0.0
                
            score = 0.0
            
            # 长度评分
            target_length = target_properties.get('linker_length', 10)
            length_diff = abs(properties['num_atoms'] - target_length) / target_length
            score += max(0, 1 - length_diff) * 0.3
            
            # 柔性评分
            target_flexibility = target_properties.get('flexibility', 0.5)
            flexibility_diff = abs(properties['flexibility'] - target_flexibility)
            score += max(0, 1 - flexibility_diff) * self.flexibility_weight
            
            # 稳定性评分
            score += properties['stability'] * self.stability_weight
            
            # 可切割性评分
            target_cleavability = target_properties.get('cleavability', 0.5)
            cleavability_diff = abs(properties['cleavability'] - target_cleavability)
            score += max(0, 1 - cleavability_diff) * self.cleavability_weight
            
            return score
            
        except Exception:
            return 0.0