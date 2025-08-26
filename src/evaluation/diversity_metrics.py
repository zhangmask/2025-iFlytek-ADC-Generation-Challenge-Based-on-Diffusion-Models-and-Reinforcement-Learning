"""多样性指标计算模块

用于评估生成分子的多样性。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import DataStructs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import logging

class DiversityMetrics:
    """多样性指标计算器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 指纹参数
        self.fp_radius = self.config.get('fp_radius', 2)
        self.fp_nbits = self.config.get('fp_nbits', 2048)
        
        # 聚类参数
        self.n_clusters = self.config.get('n_clusters', 10)
        
        self.logger.info("多样性指标计算器初始化完成")
    
    def calculate_tanimoto_diversity(self, smiles_list: List[str]) -> Dict[str, float]:
        """计算Tanimoto多样性
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            多样性指标字典
        """
        self.logger.info(f"计算 {len(smiles_list)} 个分子的Tanimoto多样性")
        
        # 生成分子指纹
        fingerprints = []
        valid_smiles = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_nbits
                )
                fingerprints.append(fp)
                valid_smiles.append(smiles)
        
        if len(fingerprints) < 2:
            return {
                'mean_tanimoto_distance': 0.0,
                'median_tanimoto_distance': 0.0,
                'min_tanimoto_distance': 0.0,
                'max_tanimoto_distance': 0.0,
                'tanimoto_diversity_index': 0.0,
                'valid_molecules': len(fingerprints)
            }
        
        # 计算成对Tanimoto相似性
        n_mols = len(fingerprints)
        similarities = []
        
        for i in range(n_mols):
            for j in range(i + 1, n_mols):
                similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                similarities.append(similarity)
        
        similarities = np.array(similarities)
        distances = 1.0 - similarities  # 距离 = 1 - 相似性
        
        # 计算统计指标
        mean_distance = np.mean(distances)
        median_distance = np.median(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        # 多样性指数（平均距离）
        diversity_index = mean_distance
        
        return {
            'mean_tanimoto_distance': float(mean_distance),
            'median_tanimoto_distance': float(median_distance),
            'min_tanimoto_distance': float(min_distance),
            'max_tanimoto_distance': float(max_distance),
            'tanimoto_diversity_index': float(diversity_index),
            'valid_molecules': len(fingerprints)
        }
    
    def calculate_scaffold_diversity(self, smiles_list: List[str]) -> Dict[str, float]:
        """计算骨架多样性
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            骨架多样性指标
        """
        from rdkit.Chem.Scaffolds import MurckoScaffold
        
        self.logger.info(f"计算 {len(smiles_list)} 个分子的骨架多样性")
        
        scaffolds = set()
        valid_count = 0
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_count += 1
                try:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    if scaffold is not None:
                        scaffold_smiles = Chem.MolToSmiles(scaffold)
                        scaffolds.add(scaffold_smiles)
                except:
                    continue
        
        n_unique_scaffolds = len(scaffolds)
        scaffold_diversity = n_unique_scaffolds / valid_count if valid_count > 0 else 0.0
        
        return {
            'unique_scaffolds': n_unique_scaffolds,
            'total_valid_molecules': valid_count,
            'scaffold_diversity_ratio': float(scaffold_diversity)
        }
    
    def calculate_descriptor_diversity(self, smiles_list: List[str]) -> Dict[str, float]:
        """计算分子描述符多样性
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            描述符多样性指标
        """
        self.logger.info(f"计算 {len(smiles_list)} 个分子的描述符多样性")
        
        # 计算分子描述符
        descriptors = []
        descriptor_names = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'NumRotatableBonds']
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                desc_values = [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumRotatableBonds(mol)
                ]
                descriptors.append(desc_values)
        
        if len(descriptors) < 2:
            return {
                'descriptor_variance_sum': 0.0,
                'descriptor_range_sum': 0.0,
                'descriptor_diversity_index': 0.0
            }
        
        descriptors = np.array(descriptors)
        
        # 计算每个描述符的方差和范围
        variances = np.var(descriptors, axis=0)
        ranges = np.ptp(descriptors, axis=0)  # peak-to-peak (max - min)
        
        # 标准化方差和范围
        variance_sum = np.sum(variances)
        range_sum = np.sum(ranges)
        
        # 多样性指数（标准化方差和范围的组合）
        diversity_index = (variance_sum + range_sum) / (2 * len(descriptor_names))
        
        return {
            'descriptor_variance_sum': float(variance_sum),
            'descriptor_range_sum': float(range_sum),
            'descriptor_diversity_index': float(diversity_index)
        }
    
    def calculate_cluster_diversity(self, smiles_list: List[str]) -> Dict[str, float]:
        """计算聚类多样性
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            聚类多样性指标
        """
        self.logger.info(f"计算 {len(smiles_list)} 个分子的聚类多样性")
        
        # 生成分子指纹
        fingerprints = []
        valid_smiles = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_nbits
                )
                # 转换为numpy数组
                fp_array = np.zeros((self.fp_nbits,))
                DataStructs.ConvertToNumpyArray(fp, fp_array)
                fingerprints.append(fp_array)
                valid_smiles.append(smiles)
        
        if len(fingerprints) < self.n_clusters:
            return {
                'silhouette_score': 0.0,
                'cluster_balance': 0.0,
                'intra_cluster_distance': 0.0,
                'inter_cluster_distance': 0.0
            }
        
        fingerprints = np.array(fingerprints)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(fingerprints)
        
        # 计算轮廓系数
        silhouette_avg = silhouette_score(fingerprints, cluster_labels)
        
        # 计算聚类平衡性
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        cluster_balance = 1.0 - np.std(counts) / np.mean(counts)  # 越接近1越平衡
        
        # 计算类内和类间距离
        intra_distances = []
        inter_distances = []
        
        for i in range(self.n_clusters):
            cluster_mask = cluster_labels == i
            if np.sum(cluster_mask) > 1:
                cluster_fps = fingerprints[cluster_mask]
                # 类内距离
                intra_dist = np.mean(pdist(cluster_fps, metric='jaccard'))
                intra_distances.append(intra_dist)
                
                # 类间距离（与其他聚类中心的距离）
                center = kmeans.cluster_centers_[i]
                other_centers = np.delete(kmeans.cluster_centers_, i, axis=0)
                inter_dist = np.mean([np.linalg.norm(center - other_center) 
                                    for other_center in other_centers])
                inter_distances.append(inter_dist)
        
        avg_intra_distance = np.mean(intra_distances) if intra_distances else 0.0
        avg_inter_distance = np.mean(inter_distances) if inter_distances else 0.0
        
        return {
            'silhouette_score': float(silhouette_avg),
            'cluster_balance': float(cluster_balance),
            'intra_cluster_distance': float(avg_intra_distance),
            'inter_cluster_distance': float(avg_inter_distance)
        }
    
    def calculate_functional_group_diversity(self, smiles_list: List[str]) -> Dict[str, float]:
        """计算官能团多样性
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            官能团多样性指标
        """
        self.logger.info(f"计算 {len(smiles_list)} 个分子的官能团多样性")
        
        # 定义常见官能团的SMARTS模式
        functional_groups = {
            'alcohol': '[OH]',
            'aldehyde': '[CX3H1](=O)',
            'ketone': '[CX3](=O)[CX4]',
            'carboxylic_acid': '[CX3](=O)[OX2H1]',
            'ester': '[CX3](=O)[OX2H0]',
            'ether': '[OD2]([#6])[#6]',
            'amine': '[NX3;H2,H1;!$(NC=O)]',
            'amide': '[NX3][CX3](=[OX1])',
            'aromatic_ring': 'c1ccccc1',
            'halogen': '[F,Cl,Br,I]',
            'nitro': '[N+](=O)[O-]',
            'sulfur': '[#16]'
        }
        
        # 统计每个官能团的出现次数
        fg_counts = {fg: 0 for fg in functional_groups}
        total_valid = 0
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                total_valid += 1
                for fg_name, smarts in functional_groups.items():
                    pattern = Chem.MolFromSmarts(smarts)
                    if pattern is not None:
                        matches = mol.GetSubstructMatches(pattern)
                        if matches:
                            fg_counts[fg_name] += 1
        
        # 计算多样性指标
        if total_valid == 0:
            return {
                'functional_group_richness': 0.0,
                'functional_group_evenness': 0.0,
                'functional_group_diversity_index': 0.0
            }
        
        # 官能团丰富度（出现的官能团种类数）
        richness = sum(1 for count in fg_counts.values() if count > 0)
        
        # 官能团均匀度（Shannon均匀度）
        proportions = [count / total_valid for count in fg_counts.values() if count > 0]
        if proportions:
            evenness = -sum(p * np.log(p) for p in proportions) / np.log(len(proportions))
        else:
            evenness = 0.0
        
        # 综合多样性指数
        diversity_index = richness * evenness
        
        return {
            'functional_group_richness': float(richness),
            'functional_group_evenness': float(evenness),
            'functional_group_diversity_index': float(diversity_index),
            'functional_group_counts': fg_counts
        }
    
    def calculate_comprehensive_diversity(self, smiles_list: List[str]) -> Dict[str, any]:
        """计算综合多样性指标
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            综合多样性指标
        """
        self.logger.info(f"计算 {len(smiles_list)} 个分子的综合多样性")
        
        results = {
            'total_molecules': len(smiles_list),
            'tanimoto_diversity': self.calculate_tanimoto_diversity(smiles_list),
            'scaffold_diversity': self.calculate_scaffold_diversity(smiles_list),
            'descriptor_diversity': self.calculate_descriptor_diversity(smiles_list),
            'cluster_diversity': self.calculate_cluster_diversity(smiles_list),
            'functional_group_diversity': self.calculate_functional_group_diversity(smiles_list)
        }
        
        # 计算综合多样性分数
        diversity_scores = [
            results['tanimoto_diversity']['tanimoto_diversity_index'],
            results['scaffold_diversity']['scaffold_diversity_ratio'],
            results['descriptor_diversity']['descriptor_diversity_index'] / 1000,  # 标准化
            results['cluster_diversity']['silhouette_score'],
            results['functional_group_diversity']['functional_group_diversity_index'] / 10  # 标准化
        ]
        
        # 过滤有效分数
        valid_scores = [score for score in diversity_scores if not np.isnan(score) and score > 0]
        
        if valid_scores:
            comprehensive_diversity = np.mean(valid_scores)
        else:
            comprehensive_diversity = 0.0
        
        results['comprehensive_diversity_score'] = float(comprehensive_diversity)
        
        return results
    
    def compare_diversity(self, smiles_list1: List[str], smiles_list2: List[str], 
                         labels: List[str] = None) -> Dict[str, any]:
        """比较两组分子的多样性
        
        Args:
            smiles_list1: 第一组SMILES
            smiles_list2: 第二组SMILES
            labels: 组标签
            
        Returns:
            多样性比较结果
        """
        if labels is None:
            labels = ['Group 1', 'Group 2']
            
        self.logger.info(f"比较两组分子的多样性: {labels[0]} vs {labels[1]}")
        
        # 计算各组多样性
        diversity1 = self.calculate_comprehensive_diversity(smiles_list1)
        diversity2 = self.calculate_comprehensive_diversity(smiles_list2)
        
        # 组织比较结果
        comparison = {
            labels[0]: diversity1,
            labels[1]: diversity2,
            'comparison': {
                'tanimoto_diversity_diff': 
                    diversity2['tanimoto_diversity']['tanimoto_diversity_index'] - 
                    diversity1['tanimoto_diversity']['tanimoto_diversity_index'],
                'scaffold_diversity_diff': 
                    diversity2['scaffold_diversity']['scaffold_diversity_ratio'] - 
                    diversity1['scaffold_diversity']['scaffold_diversity_ratio'],
                'comprehensive_diversity_diff': 
                    diversity2['comprehensive_diversity_score'] - 
                    diversity1['comprehensive_diversity_score']
            }
        }
        
        return comparison
    
    def save_diversity_report(self, diversity_results: Dict, filepath: str):
        """保存多样性报告
        
        Args:
            diversity_results: 多样性结果
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
        
        converted_results = recursive_convert(diversity_results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"多样性报告已保存到: {filepath}")