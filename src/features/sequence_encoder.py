"""蛋白质序列编码器

实现多种蛋白质序列编码方法，包括one-hot编码、氨基酸属性编码和预训练模型编码。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import logging

class SequenceEncoder:
    """蛋白质序列编码器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 标准氨基酸字母表
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        self.idx_to_aa = {idx: aa for aa, idx in self.aa_to_idx.items()}
        
        # 氨基酸物理化学属性
        self.aa_properties = self._load_aa_properties()
        
        # 编码器组件
        self.scalers = {}
        
    def _load_aa_properties(self) -> Dict[str, Dict[str, float]]:
        """加载氨基酸物理化学属性"""
        properties = {
            'A': {'hydrophobicity': 1.8, 'volume': 88.6, 'polarity': 8.1, 'charge': 0},
            'C': {'hydrophobicity': 2.5, 'volume': 108.5, 'polarity': 5.5, 'charge': 0},
            'D': {'hydrophobicity': -3.5, 'volume': 111.1, 'polarity': 13.0, 'charge': -1},
            'E': {'hydrophobicity': -3.5, 'volume': 138.4, 'polarity': 12.3, 'charge': -1},
            'F': {'hydrophobicity': 2.8, 'volume': 189.9, 'polarity': 5.2, 'charge': 0},
            'G': {'hydrophobicity': -0.4, 'volume': 60.1, 'polarity': 9.0, 'charge': 0},
            'H': {'hydrophobicity': -3.2, 'volume': 153.2, 'polarity': 10.4, 'charge': 1},
            'I': {'hydrophobicity': 4.5, 'volume': 166.7, 'polarity': 5.2, 'charge': 0},
            'K': {'hydrophobicity': -3.9, 'volume': 168.6, 'polarity': 11.3, 'charge': 1},
            'L': {'hydrophobicity': 3.8, 'volume': 166.7, 'polarity': 4.9, 'charge': 0},
            'M': {'hydrophobicity': 1.9, 'volume': 162.9, 'polarity': 5.7, 'charge': 0},
            'N': {'hydrophobicity': -3.5, 'volume': 114.1, 'polarity': 11.6, 'charge': 0},
            'P': {'hydrophobicity': -1.6, 'volume': 112.7, 'polarity': 8.0, 'charge': 0},
            'Q': {'hydrophobicity': -3.5, 'volume': 143.8, 'polarity': 10.5, 'charge': 0},
            'R': {'hydrophobicity': -4.5, 'volume': 173.4, 'polarity': 10.5, 'charge': 1},
            'S': {'hydrophobicity': -0.8, 'volume': 89.0, 'polarity': 9.2, 'charge': 0},
            'T': {'hydrophobicity': -0.7, 'volume': 116.1, 'polarity': 8.6, 'charge': 0},
            'V': {'hydrophobicity': 4.2, 'volume': 140.0, 'polarity': 5.9, 'charge': 0},
            'W': {'hydrophobicity': -0.9, 'volume': 227.8, 'polarity': 5.4, 'charge': 0},
            'Y': {'hydrophobicity': -1.3, 'volume': 193.6, 'polarity': 6.2, 'charge': 0}
        }
        return properties
    
    def encode_onehot(self, sequences: List[str], max_length: Optional[int] = None) -> np.ndarray:
        """One-hot编码
        
        Args:
            sequences: 蛋白质序列列表
            max_length: 最大序列长度，如果为None则使用最长序列长度
            
        Returns:
            编码后的数组，形状为 (n_sequences, max_length, 20)
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
            
        encoded = np.zeros((len(sequences), max_length, len(self.amino_acids)))
        
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq[:max_length]):
                if aa in self.aa_to_idx:
                    encoded[i, j, self.aa_to_idx[aa]] = 1
                    
        self.logger.info(f"One-hot编码完成，形状: {encoded.shape}")
        return encoded
    
    def encode_properties(self, sequences: List[str], max_length: Optional[int] = None) -> np.ndarray:
        """氨基酸属性编码
        
        Args:
            sequences: 蛋白质序列列表
            max_length: 最大序列长度
            
        Returns:
            编码后的数组，形状为 (n_sequences, max_length, 4)
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
            
        property_names = ['hydrophobicity', 'volume', 'polarity', 'charge']
        encoded = np.zeros((len(sequences), max_length, len(property_names)))
        
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq[:max_length]):
                if aa in self.aa_properties:
                    for k, prop in enumerate(property_names):
                        encoded[i, j, k] = self.aa_properties[aa][prop]
                        
        self.logger.info(f"属性编码完成，形状: {encoded.shape}")
        return encoded
    
    def encode_kmer(self, sequences: List[str], k: int = 3) -> np.ndarray:
        """K-mer编码
        
        Args:
            sequences: 蛋白质序列列表
            k: k-mer长度
            
        Returns:
            编码后的数组
        """
        # 生成所有可能的k-mer
        from itertools import product
        kmers = [''.join(p) for p in product(self.amino_acids, repeat=k)]
        kmer_to_idx = {kmer: idx for idx, kmer in enumerate(kmers)}
        
        encoded = np.zeros((len(sequences), len(kmers)))
        
        for i, seq in enumerate(sequences):
            kmer_counts = {}
            for j in range(len(seq) - k + 1):
                kmer = seq[j:j+k]
                if all(aa in self.amino_acids for aa in kmer):
                    kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
                    
            # 归一化
            total_kmers = sum(kmer_counts.values())
            if total_kmers > 0:
                for kmer, count in kmer_counts.items():
                    if kmer in kmer_to_idx:
                        encoded[i, kmer_to_idx[kmer]] = count / total_kmers
                        
        self.logger.info(f"K-mer编码完成，k={k}，形状: {encoded.shape}")
        return encoded
    
    def encode_statistical(self, sequences: List[str]) -> np.ndarray:
        """统计特征编码
        
        Args:
            sequences: 蛋白质序列列表
            
        Returns:
            统计特征数组
        """
        features = []
        
        for seq in sequences:
            seq_features = []
            
            # 基本统计
            seq_features.append(len(seq))  # 序列长度
            
            # 氨基酸组成
            aa_counts = {aa: seq.count(aa) / len(seq) for aa in self.amino_acids}
            seq_features.extend([aa_counts.get(aa, 0) for aa in self.amino_acids])
            
            # 物理化学属性统计
            if len(seq) > 0:
                hydrophobicity = [self.aa_properties.get(aa, {}).get('hydrophobicity', 0) for aa in seq]
                volume = [self.aa_properties.get(aa, {}).get('volume', 0) for aa in seq]
                polarity = [self.aa_properties.get(aa, {}).get('polarity', 0) for aa in seq]
                charge = [self.aa_properties.get(aa, {}).get('charge', 0) for aa in seq]
                
                # 统计量
                for prop_values in [hydrophobicity, volume, polarity, charge]:
                    seq_features.extend([
                        np.mean(prop_values),
                        np.std(prop_values),
                        np.min(prop_values),
                        np.max(prop_values)
                    ])
            else:
                seq_features.extend([0] * 16)  # 4个属性 * 4个统计量
                
            features.append(seq_features)
            
        encoded = np.array(features)
        self.logger.info(f"统计特征编码完成，形状: {encoded.shape}")
        return encoded
    
    def encode_sequences(self, sequences: List[str], method: str = 'onehot', **kwargs) -> np.ndarray:
        """编码蛋白质序列
        
        Args:
            sequences: 蛋白质序列列表
            method: 编码方法 ('onehot', 'properties', 'kmer', 'statistical')
            **kwargs: 编码方法的额外参数
            
        Returns:
            编码后的特征数组
        """
        self.logger.info(f"开始编码 {len(sequences)} 个序列，方法: {method}")
        
        if method == 'onehot':
            return self.encode_onehot(sequences, **kwargs)
        elif method == 'properties':
            return self.encode_properties(sequences, **kwargs)
        elif method == 'kmer':
            return self.encode_kmer(sequences, **kwargs)
        elif method == 'statistical':
            return self.encode_statistical(sequences, **kwargs)
        else:
            raise ValueError(f"不支持的编码方法: {method}")
    
    def fit_transform(self, sequences: List[str], method: str = 'onehot', normalize: bool = True, **kwargs) -> np.ndarray:
        """拟合并转换序列
        
        Args:
            sequences: 蛋白质序列列表
            method: 编码方法
            normalize: 是否标准化
            **kwargs: 编码方法的额外参数
            
        Returns:
            编码并可能标准化后的特征数组
        """
        encoded = self.encode_sequences(sequences, method, **kwargs)
        
        if normalize and method in ['properties', 'statistical']:
            # 对需要标准化的方法进行标准化
            scaler_key = f"{method}_{hash(str(kwargs))}"
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
                
            if encoded.ndim == 3:  # 序列级特征
                original_shape = encoded.shape
                encoded_flat = encoded.reshape(-1, encoded.shape[-1])
                encoded_flat = self.scalers[scaler_key].fit_transform(encoded_flat)
                encoded = encoded_flat.reshape(original_shape)
            else:  # 全局特征
                encoded = self.scalers[scaler_key].fit_transform(encoded)
                
            self.logger.info(f"特征标准化完成")
            
        return encoded
    
    def transform(self, sequences: List[str], method: str = 'onehot', normalize: bool = True, **kwargs) -> np.ndarray:
        """转换序列（使用已拟合的标准化器）
        
        Args:
            sequences: 蛋白质序列列表
            method: 编码方法
            normalize: 是否标准化
            **kwargs: 编码方法的额外参数
            
        Returns:
            编码并可能标准化后的特征数组
        """
        encoded = self.encode_sequences(sequences, method, **kwargs)
        
        if normalize and method in ['properties', 'statistical']:
            scaler_key = f"{method}_{hash(str(kwargs))}"
            if scaler_key in self.scalers:
                if encoded.ndim == 3:  # 序列级特征
                    original_shape = encoded.shape
                    encoded_flat = encoded.reshape(-1, encoded.shape[-1])
                    encoded_flat = self.scalers[scaler_key].transform(encoded_flat)
                    encoded = encoded_flat.reshape(original_shape)
                else:  # 全局特征
                    encoded = self.scalers[scaler_key].transform(encoded)
            else:
                self.logger.warning(f"标准化器 {scaler_key} 未找到，跳过标准化")
                
        return encoded