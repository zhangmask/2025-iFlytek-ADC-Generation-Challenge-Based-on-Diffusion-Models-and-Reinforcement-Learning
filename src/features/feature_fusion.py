"""特征融合模块

实现蛋白质序列特征和分子特征的多种融合方法。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FeatureFusion:
    """特征融合器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 融合方法配置
        self.fusion_method = getattr(config, 'fusion_method', 'concatenation')
        self.output_dim = getattr(config, 'fusion_dim', 512)
        
        # 组件
        self.pca_components = {}
        self.scalers = {}
        self.fusion_networks = {}
        
    def concatenate_features(self, protein_features: np.ndarray, 
                           molecule_features: np.ndarray) -> np.ndarray:
        """简单拼接特征
        
        Args:
            protein_features: 蛋白质特征 (n_samples, protein_dim)
            molecule_features: 分子特征 (n_samples, molecule_dim)
            
        Returns:
            拼接后的特征
        """
        # 确保特征是2D的
        if protein_features.ndim > 2:
            protein_features = protein_features.reshape(protein_features.shape[0], -1)
        if molecule_features.ndim > 2:
            molecule_features = molecule_features.reshape(molecule_features.shape[0], -1)
            
        fused = np.concatenate([protein_features, molecule_features], axis=1)
        self.logger.info(f"特征拼接完成，形状: {fused.shape}")
        return fused
    
    def weighted_fusion(self, protein_features: np.ndarray, 
                       molecule_features: np.ndarray,
                       protein_weight: float = 0.5,
                       molecule_weight: float = 0.5) -> np.ndarray:
        """加权融合特征
        
        Args:
            protein_features: 蛋白质特征
            molecule_features: 分子特征
            protein_weight: 蛋白质特征权重
            molecule_weight: 分子特征权重
            
        Returns:
            加权融合后的特征
        """
        # 标准化特征维度
        if protein_features.ndim > 2:
            protein_features = protein_features.reshape(protein_features.shape[0], -1)
        if molecule_features.ndim > 2:
            molecule_features = molecule_features.reshape(molecule_features.shape[0], -1)
            
        # 标准化特征
        protein_norm = StandardScaler().fit_transform(protein_features)
        molecule_norm = StandardScaler().fit_transform(molecule_features)
        
        # 确保维度匹配
        min_dim = min(protein_norm.shape[1], molecule_norm.shape[1])
        protein_norm = protein_norm[:, :min_dim]
        molecule_norm = molecule_norm[:, :min_dim]
        
        # 加权融合
        fused = protein_weight * protein_norm + molecule_weight * molecule_norm
        self.logger.info(f"加权融合完成，形状: {fused.shape}")
        return fused
    
    def pca_fusion(self, protein_features: np.ndarray, 
                   molecule_features: np.ndarray,
                   n_components: int = 100) -> np.ndarray:
        """PCA降维融合
        
        Args:
            protein_features: 蛋白质特征
            molecule_features: 分子特征
            n_components: PCA组件数
            
        Returns:
            PCA融合后的特征
        """
        # 先拼接特征
        concatenated = self.concatenate_features(protein_features, molecule_features)
        
        # PCA降维
        if 'pca' not in self.pca_components:
            self.pca_components['pca'] = PCA(n_components=n_components)
            fused = self.pca_components['pca'].fit_transform(concatenated)
        else:
            fused = self.pca_components['pca'].transform(concatenated)
            
        self.logger.info(f"PCA融合完成，形状: {fused.shape}")
        return fused
    
    def attention_fusion(self, protein_features: np.ndarray, 
                        molecule_features: np.ndarray) -> np.ndarray:
        """注意力机制融合
        
        Args:
            protein_features: 蛋白质特征
            molecule_features: 分子特征
            
        Returns:
            注意力融合后的特征
        """
        # 转换为torch张量
        protein_tensor = torch.FloatTensor(protein_features)
        molecule_tensor = torch.FloatTensor(molecule_features)
        
        # 确保特征是2D的
        if protein_tensor.dim() > 2:
            protein_tensor = protein_tensor.view(protein_tensor.size(0), -1)
        if molecule_tensor.dim() > 2:
            molecule_tensor = molecule_tensor.view(molecule_tensor.size(0), -1)
            
        # 创建注意力网络
        if 'attention' not in self.fusion_networks:
            protein_dim = protein_tensor.size(1)
            molecule_dim = molecule_tensor.size(1)
            
            self.fusion_networks['attention'] = AttentionFusionNetwork(
                protein_dim, molecule_dim, self.output_dim
            )
            
        # 注意力融合
        with torch.no_grad():
            fused = self.fusion_networks['attention'](protein_tensor, molecule_tensor)
            
        result = fused.numpy()
        self.logger.info(f"注意力融合完成，形状: {result.shape}")
        return result
    
    def multimodal_fusion(self, protein_features: np.ndarray, 
                         molecule_features: np.ndarray) -> np.ndarray:
        """多模态融合
        
        Args:
            protein_features: 蛋白质特征
            molecule_features: 分子特征
            
        Returns:
            多模态融合后的特征
        """
        # 转换为torch张量
        protein_tensor = torch.FloatTensor(protein_features)
        molecule_tensor = torch.FloatTensor(molecule_features)
        
        # 确保特征是2D的
        if protein_tensor.dim() > 2:
            protein_tensor = protein_tensor.view(protein_tensor.size(0), -1)
        if molecule_tensor.dim() > 2:
            molecule_tensor = molecule_tensor.view(molecule_tensor.size(0), -1)
            
        # 创建多模态融合网络
        if 'multimodal' not in self.fusion_networks:
            protein_dim = protein_tensor.size(1)
            molecule_dim = molecule_tensor.size(1)
            
            self.fusion_networks['multimodal'] = MultimodalFusionNetwork(
                protein_dim, molecule_dim, self.output_dim
            )
            
        # 多模态融合
        with torch.no_grad():
            fused = self.fusion_networks['multimodal'](protein_tensor, molecule_tensor)
            
        result = fused.numpy()
        self.logger.info(f"多模态融合完成，形状: {result.shape}")
        return result
    
    def fuse_features(self, protein_features: np.ndarray, 
                     molecule_features: np.ndarray,
                     method: Optional[str] = None,
                     **kwargs) -> np.ndarray:
        """融合特征
        
        Args:
            protein_features: 蛋白质特征
            molecule_features: 分子特征
            method: 融合方法
            **kwargs: 额外参数
            
        Returns:
            融合后的特征
        """
        if method is None:
            method = self.fusion_method
            
        self.logger.info(f"开始特征融合，方法: {method}")
        
        if method == 'concatenation':
            return self.concatenate_features(protein_features, molecule_features)
        elif method == 'weighted':
            return self.weighted_fusion(protein_features, molecule_features, **kwargs)
        elif method == 'pca':
            return self.pca_fusion(protein_features, molecule_features, **kwargs)
        elif method == 'attention':
            return self.attention_fusion(protein_features, molecule_features)
        elif method == 'multimodal':
            return self.multimodal_fusion(protein_features, molecule_features)
        else:
            raise ValueError(f"不支持的融合方法: {method}")
    
    def fit_transform(self, protein_features: np.ndarray, 
                     molecule_features: np.ndarray,
                     method: Optional[str] = None,
                     **kwargs) -> np.ndarray:
        """拟合并转换特征
        
        Args:
            protein_features: 蛋白质特征
            molecule_features: 分子特征
            method: 融合方法
            **kwargs: 额外参数
            
        Returns:
            融合后的特征
        """
        return self.fuse_features(protein_features, molecule_features, method, **kwargs)
    
    def transform(self, protein_features: np.ndarray, 
                 molecule_features: np.ndarray,
                 method: Optional[str] = None,
                 **kwargs) -> np.ndarray:
        """转换特征（使用已拟合的组件）
        
        Args:
            protein_features: 蛋白质特征
            molecule_features: 分子特征
            method: 融合方法
            **kwargs: 额外参数
            
        Returns:
            融合后的特征
        """
        return self.fuse_features(protein_features, molecule_features, method, **kwargs)


class AttentionFusionNetwork(nn.Module):
    """注意力融合网络"""
    
    def __init__(self, protein_dim: int, molecule_dim: int, output_dim: int):
        super().__init__()
        
        # 特征投影层
        self.protein_proj = nn.Linear(protein_dim, output_dim)
        self.molecule_proj = nn.Linear(molecule_dim, output_dim)
        
        # 注意力层
        self.attention = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, protein_features: torch.Tensor, molecule_features: torch.Tensor) -> torch.Tensor:
        # 特征投影
        protein_proj = self.protein_proj(protein_features)  # (batch, output_dim)
        molecule_proj = self.molecule_proj(molecule_features)  # (batch, output_dim)
        
        # 添加序列维度用于注意力计算
        protein_seq = protein_proj.unsqueeze(1)  # (batch, 1, output_dim)
        molecule_seq = molecule_proj.unsqueeze(1)  # (batch, 1, output_dim)
        
        # 拼接序列
        combined_seq = torch.cat([protein_seq, molecule_seq], dim=1)  # (batch, 2, output_dim)
        
        # 自注意力
        attended, _ = self.attention(combined_seq, combined_seq, combined_seq)
        
        # 展平并输出
        attended_flat = attended.view(attended.size(0), -1)  # (batch, 2*output_dim)
        output = self.output_layer(attended_flat)
        
        return output


class MultimodalFusionNetwork(nn.Module):
    """多模态融合网络"""
    
    def __init__(self, protein_dim: int, molecule_dim: int, output_dim: int):
        super().__init__()
        
        # 模态特定编码器
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
        self.molecule_encoder = nn.Sequential(
            nn.Linear(molecule_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(output_dim, num_heads=4, batch_first=True)
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, protein_features: torch.Tensor, molecule_features: torch.Tensor) -> torch.Tensor:
        # 模态编码
        protein_encoded = self.protein_encoder(protein_features)
        molecule_encoded = self.molecule_encoder(molecule_features)
        
        # 添加序列维度
        protein_seq = protein_encoded.unsqueeze(1)
        molecule_seq = molecule_encoded.unsqueeze(1)
        
        # 交叉注意力
        protein_attended, _ = self.cross_attention(protein_seq, molecule_seq, molecule_seq)
        molecule_attended, _ = self.cross_attention(molecule_seq, protein_seq, protein_seq)
        
        # 移除序列维度
        protein_attended = protein_attended.squeeze(1)
        molecule_attended = molecule_attended.squeeze(1)
        
        # 元素级乘积
        interaction = protein_encoded * molecule_encoded
        
        # 融合所有表示
        combined = torch.cat([protein_attended, molecule_attended, interaction], dim=1)
        output = self.fusion_layer(combined)
        
        return output