"""有效性指标计算模块

用于评估生成分子的有效性和药物相似性。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import logging

class ValidityMetrics:
    """有效性指标计算器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 初始化过滤器
        self._init_filters()
        
        # 药物相似性参数
        self.lipinski_strict = self.config.get('lipinski_strict', True)
        self.qed_threshold = self.config.get('qed_threshold', 0.5)
        
        self.logger.info("有效性指标计算器初始化完成")
    
    def _init_filters(self):
        """初始化分子过滤器"""
        try:
            # PAINS过滤器
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            self.pains_filter = FilterCatalog(params)
            
            # BRENK过滤器
            params_brenk = FilterCatalogParams()
            params_brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
            self.brenk_filter = FilterCatalog(params_brenk)
            
        except Exception as e:
            self.logger.warning(f"过滤器初始化失败: {e}")
            self.pains_filter = None
            self.brenk_filter = None
    
    def calculate_validity_rate(self, smiles_list: List[str]) -> Dict[str, float]:
        """计算分子有效性比率
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            有效性指标
        """
        self.logger.info(f"计算 {len(smiles_list)} 个分子的有效性")
        
        total_count = len(smiles_list)
        valid_count = 0
        unique_count = 0
        
        valid_smiles = set()
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_count += 1
                canonical_smiles = Chem.MolToSmiles(mol)
                valid_smiles.add(canonical_smiles)
        
        unique_count = len(valid_smiles)
        
        validity_rate = valid_count / total_count if total_count > 0 else 0.0
        uniqueness_rate = unique_count / valid_count if valid_count > 0 else 0.0
        novelty_rate = unique_count / total_count if total_count > 0 else 0.0
        
        return {
            'total_molecules': total_count,
            'valid_molecules': valid_count,
            'unique_molecules': unique_count,
            'validity_rate': float(validity_rate),
            'uniqueness_rate': float(uniqueness_rate),
            'novelty_rate': float(novelty_rate)
        }
    
    def calculate_lipinski_compliance(self, smiles_list: List[str]) -> Dict[str, any]:
        """计算Lipinski五规则符合性
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            Lipinski符合性指标
        """
        self.logger.info(f"计算 {len(smiles_list)} 个分子的Lipinski符合性")
        
        lipinski_results = []
        valid_count = 0
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_count += 1
                
                # 计算Lipinski描述符
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                
                # 检查Lipinski规则
                mw_pass = mw <= 500
                logp_pass = logp <= 5
                hbd_pass = hbd <= 5
                hba_pass = hba <= 10
                
                if self.lipinski_strict:
                    # 严格模式：所有规则都必须满足
                    lipinski_pass = mw_pass and logp_pass and hbd_pass and hba_pass
                else:
                    # 宽松模式：至少满足3个规则
                    lipinski_pass = sum([mw_pass, logp_pass, hbd_pass, hba_pass]) >= 3
                
                lipinski_results.append({
                    'smiles': smiles,
                    'molecular_weight': mw,
                    'logp': logp,
                    'hbd': hbd,
                    'hba': hba,
                    'mw_pass': mw_pass,
                    'logp_pass': logp_pass,
                    'hbd_pass': hbd_pass,
                    'hba_pass': hba_pass,
                    'lipinski_pass': lipinski_pass
                })
        
        if not lipinski_results:
            return {
                'lipinski_compliance_rate': 0.0,
                'mean_molecular_weight': 0.0,
                'mean_logp': 0.0,
                'mean_hbd': 0.0,
                'mean_hba': 0.0,
                'detailed_results': []
            }
        
        # 计算统计指标
        compliance_rate = sum(1 for r in lipinski_results if r['lipinski_pass']) / len(lipinski_results)
        mean_mw = np.mean([r['molecular_weight'] for r in lipinski_results])
        mean_logp = np.mean([r['logp'] for r in lipinski_results])
        mean_hbd = np.mean([r['hbd'] for r in lipinski_results])
        mean_hba = np.mean([r['hba'] for r in lipinski_results])
        
        return {
            'lipinski_compliance_rate': float(compliance_rate),
            'mean_molecular_weight': float(mean_mw),
            'mean_logp': float(mean_logp),
            'mean_hbd': float(mean_hbd),
            'mean_hba': float(mean_hba),
            'detailed_results': lipinski_results
        }
    
    def calculate_qed_scores(self, smiles_list: List[str]) -> Dict[str, float]:
        """计算QED（定量药物相似性）分数
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            QED分数统计
        """
        self.logger.info(f"计算 {len(smiles_list)} 个分子的QED分数")
        
        qed_scores = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    qed_score = QED.qed(mol)
                    qed_scores.append(qed_score)
                except:
                    continue
        
        if not qed_scores:
            return {
                'mean_qed': 0.0,
                'median_qed': 0.0,
                'std_qed': 0.0,
                'min_qed': 0.0,
                'max_qed': 0.0,
                'qed_above_threshold': 0.0
            }
        
        qed_scores = np.array(qed_scores)
        
        return {
            'mean_qed': float(np.mean(qed_scores)),
            'median_qed': float(np.median(qed_scores)),
            'std_qed': float(np.std(qed_scores)),
            'min_qed': float(np.min(qed_scores)),
            'max_qed': float(np.max(qed_scores)),
            'qed_above_threshold': float(np.sum(qed_scores >= self.qed_threshold) / len(qed_scores))
        }
    
    def calculate_sa_scores(self, smiles_list: List[str]) -> Dict[str, float]:
        """计算合成可达性（SA）分数
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            SA分数统计
        """
        try:
            from rdkit.Chem import rdMolDescriptors
        except ImportError:
            self.logger.warning("无法导入SA分数计算模块")
            return {
                'mean_sa': 0.0,
                'median_sa': 0.0,
                'std_sa': 0.0,
                'min_sa': 0.0,
                'max_sa': 0.0
            }
        
        self.logger.info(f"计算 {len(smiles_list)} 个分子的SA分数")
        
        sa_scores = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    # 使用RDKit的合成可达性分数
                    # 注意：这需要额外的数据文件，可能不可用
                    sa_score = rdMolDescriptors.BertzCT(mol)  # 使用Bertz复杂度作为替代
                    sa_scores.append(sa_score)
                except:
                    continue
        
        if not sa_scores:
            return {
                'mean_sa': 0.0,
                'median_sa': 0.0,
                'std_sa': 0.0,
                'min_sa': 0.0,
                'max_sa': 0.0
            }
        
        sa_scores = np.array(sa_scores)
        
        return {
            'mean_sa': float(np.mean(sa_scores)),
            'median_sa': float(np.median(sa_scores)),
            'std_sa': float(np.std(sa_scores)),
            'min_sa': float(np.min(sa_scores)),
            'max_sa': float(np.max(sa_scores))
        }
    
    def check_pains_alerts(self, smiles_list: List[str]) -> Dict[str, any]:
        """检查PAINS（泛测定干扰化合物）警报
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            PAINS检查结果
        """
        if self.pains_filter is None:
            self.logger.warning("PAINS过滤器不可用")
            return {
                'pains_free_rate': 1.0,
                'pains_alerts': 0,
                'total_checked': 0
            }
        
        self.logger.info(f"检查 {len(smiles_list)} 个分子的PAINS警报")
        
        pains_free_count = 0
        total_checked = 0
        pains_alerts = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                total_checked += 1
                
                # 检查PAINS
                if self.pains_filter.HasMatch(mol):
                    matches = self.pains_filter.GetMatches(mol)
                    pains_alerts.append({
                        'smiles': smiles,
                        'alerts': [match.GetDescription() for match in matches]
                    })
                else:
                    pains_free_count += 1
        
        pains_free_rate = pains_free_count / total_checked if total_checked > 0 else 0.0
        
        return {
            'pains_free_rate': float(pains_free_rate),
            'pains_alerts': len(pains_alerts),
            'total_checked': total_checked,
            'alert_details': pains_alerts
        }
    
    def calculate_admet_properties(self, smiles_list: List[str]) -> Dict[str, any]:
        """计算ADMET（吸收、分布、代谢、排泄、毒性）相关属性
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            ADMET属性统计
        """
        self.logger.info(f"计算 {len(smiles_list)} 个分子的ADMET属性")
        
        admet_results = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    # 计算ADMET相关描述符
                    tpsa = Descriptors.TPSA(mol)  # 拓扑极性表面积
                    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
                    aromatic_rings = Descriptors.NumAromaticRings(mol)
                    heteroatoms = Descriptors.NumHeteroatoms(mol)
                    
                    # 血脑屏障渗透性预测（简化规则）
                    bbb_penetration = (tpsa <= 90 and Descriptors.MolWt(mol) <= 450)
                    
                    # 口服生物利用度预测（简化规则）
                    oral_bioavailability = (
                        Descriptors.MolWt(mol) <= 500 and
                        Descriptors.MolLogP(mol) <= 5 and
                        tpsa <= 140 and
                        rotatable_bonds <= 10
                    )
                    
                    admet_results.append({
                        'smiles': smiles,
                        'tpsa': tpsa,
                        'rotatable_bonds': rotatable_bonds,
                        'aromatic_rings': aromatic_rings,
                        'heteroatoms': heteroatoms,
                        'bbb_penetration': bbb_penetration,
                        'oral_bioavailability': oral_bioavailability
                    })
                    
                except Exception as e:
                    self.logger.debug(f"计算ADMET属性失败: {smiles}, {e}")
                    continue
        
        if not admet_results:
            return {
                'mean_tpsa': 0.0,
                'mean_rotatable_bonds': 0.0,
                'bbb_penetration_rate': 0.0,
                'oral_bioavailability_rate': 0.0,
                'detailed_results': []
            }
        
        # 计算统计指标
        mean_tpsa = np.mean([r['tpsa'] for r in admet_results])
        mean_rotatable_bonds = np.mean([r['rotatable_bonds'] for r in admet_results])
        bbb_rate = sum(1 for r in admet_results if r['bbb_penetration']) / len(admet_results)
        oral_rate = sum(1 for r in admet_results if r['oral_bioavailability']) / len(admet_results)
        
        return {
            'mean_tpsa': float(mean_tpsa),
            'mean_rotatable_bonds': float(mean_rotatable_bonds),
            'bbb_penetration_rate': float(bbb_rate),
            'oral_bioavailability_rate': float(oral_rate),
            'detailed_results': admet_results
        }
    
    def calculate_comprehensive_validity(self, smiles_list: List[str]) -> Dict[str, any]:
        """计算综合有效性指标
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            综合有效性指标
        """
        self.logger.info(f"计算 {len(smiles_list)} 个分子的综合有效性")
        
        results = {
            'basic_validity': self.calculate_validity_rate(smiles_list),
            'lipinski_compliance': self.calculate_lipinski_compliance(smiles_list),
            'qed_scores': self.calculate_qed_scores(smiles_list),
            'sa_scores': self.calculate_sa_scores(smiles_list),
            'pains_check': self.check_pains_alerts(smiles_list),
            'admet_properties': self.calculate_admet_properties(smiles_list)
        }
        
        # 计算综合有效性分数
        validity_components = [
            results['basic_validity']['validity_rate'],
            results['basic_validity']['uniqueness_rate'],
            results['lipinski_compliance']['lipinski_compliance_rate'],
            results['qed_scores']['qed_above_threshold'],
            results['pains_check']['pains_free_rate'],
            results['admet_properties']['oral_bioavailability_rate']
        ]
        
        # 过滤有效分数
        valid_components = [score for score in validity_components if not np.isnan(score)]
        
        if valid_components:
            comprehensive_validity = np.mean(valid_components)
        else:
            comprehensive_validity = 0.0
        
        results['comprehensive_validity_score'] = float(comprehensive_validity)
        
        return results
    
    def compare_validity(self, smiles_list1: List[str], smiles_list2: List[str],
                       labels: List[str] = None) -> Dict[str, any]:
        """比较两组分子的有效性
        
        Args:
            smiles_list1: 第一组SMILES
            smiles_list2: 第二组SMILES
            labels: 组标签
            
        Returns:
            有效性比较结果
        """
        if labels is None:
            labels = ['Group 1', 'Group 2']
            
        self.logger.info(f"比较两组分子的有效性: {labels[0]} vs {labels[1]}")
        
        # 计算各组有效性
        validity1 = self.calculate_comprehensive_validity(smiles_list1)
        validity2 = self.calculate_comprehensive_validity(smiles_list2)
        
        # 组织比较结果
        comparison = {
            labels[0]: validity1,
            labels[1]: validity2,
            'comparison': {
                'validity_rate_diff': 
                    validity2['basic_validity']['validity_rate'] - 
                    validity1['basic_validity']['validity_rate'],
                'lipinski_compliance_diff': 
                    validity2['lipinski_compliance']['lipinski_compliance_rate'] - 
                    validity1['lipinski_compliance']['lipinski_compliance_rate'],
                'qed_score_diff': 
                    validity2['qed_scores']['mean_qed'] - 
                    validity1['qed_scores']['mean_qed'],
                'comprehensive_validity_diff': 
                    validity2['comprehensive_validity_score'] - 
                    validity1['comprehensive_validity_score']
            }
        }
        
        return comparison
    
    def save_validity_report(self, validity_results: Dict, filepath: str):
        """保存有效性报告
        
        Args:
            validity_results: 有效性结果
            filepath: 保存路径
        """
        import json
        
        # 处理numpy类型以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 递归转换
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        converted_results = recursive_convert(validity_results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"有效性报告已保存到: {filepath}")