"""特征工程模块

该模块包含蛋白质序列编码、分子特征提取和特征融合功能。
"""

from .sequence_encoder import SequenceEncoder
from .molecule_features import MoleculeFeatureExtractor
from .feature_fusion import FeatureFusion

__all__ = [
    'SequenceEncoder',
    'MoleculeFeatureExtractor', 
    'FeatureFusion'
]