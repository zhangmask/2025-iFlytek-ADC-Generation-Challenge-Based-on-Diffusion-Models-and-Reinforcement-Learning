"""分子特征提取器

实现分子描述符计算、SMILES编码和分子指纹生成。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from sklearn.preprocessing import StandardScaler
import torch

class MoleculeFeatureExtractor:
    """分子特征提取器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # SMILES字符集
        self.smiles_chars = self._get_smiles_charset()
        self.char_to_idx = {char: idx for idx, char in enumerate(self.smiles_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # 标准化器
        self.scalers = {}
        
    def _get_smiles_charset(self) -> List[str]:
        """获取SMILES字符集"""
        # 常用SMILES字符
        chars = [
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',  # 原子
            'c', 'n', 'o', 's', 'p',  # 芳香原子
            '(', ')', '[', ']', '=', '#', '-', '+',  # 键和括号
            '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 环编号
            '@', '@@', '/', '\\',  # 立体化学
            '.', '%',  # 其他
            '<PAD>', '<UNK>', '<START>', '<END>'  # 特殊标记
        ]
        return chars
    
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """将SMILES转换为分子对象
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            RDKit分子对象，如果无效则返回None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                Chem.SanitizeMol(mol)
            return mol
        except Exception as e:
            self.logger.warning(f"无法解析SMILES: {smiles}, 错误: {e}")
            return None
    
    def calculate_descriptors(self, smiles_list: List[str]) -> np.ndarray:
        """计算分子描述符
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            分子描述符数组
        """
        descriptors = []
        descriptor_names = [
            'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA',
            'NumRotatableBonds', 'NumAromaticRings', 'NumSaturatedRings',
            'NumAliphaticRings', 'RingCount',
            'NumHeteroatoms', 'BertzCT', 'BalabanJ', 'Ipc'
        ]
        
        for smiles in smiles_list:
            mol = self.smiles_to_mol(smiles)
            if mol is not None:
                desc = [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.NumSaturatedRings(mol),
                    Descriptors.NumAliphaticRings(mol),
                    Descriptors.RingCount(mol),
                    Descriptors.NumHeteroatoms(mol),
                    Descriptors.BertzCT(mol),
                    Descriptors.BalabanJ(mol),
                    Descriptors.Ipc(mol)
                ]
            else:
                desc = [0.0] * len(descriptor_names)
                
            descriptors.append(desc)
            
        result = np.array(descriptors)
        self.logger.info(f"计算了 {len(descriptor_names)} 个分子描述符，形状: {result.shape}")
        return result
    
    def generate_fingerprints(self, smiles_list: List[str], 
                            fp_type: str = 'morgan', 
                            radius: int = 2, 
                            n_bits: Optional[int] = None) -> np.ndarray:
        """生成分子指纹
        
        Args:
            smiles_list: SMILES字符串列表
            fp_type: 指纹类型 ('morgan', 'rdkit', 'maccs')
            radius: Morgan指纹半径
            n_bits: 指纹位数
            
        Returns:
            分子指纹数组
        """
        # 从配置中获取指纹大小，默认为2048
        if n_bits is None:
            n_bits = getattr(self.config, 'fingerprint_size', 2048)
        
        fingerprints = []
        
        for smiles in smiles_list:
            mol = self.smiles_to_mol(smiles)
            if mol is not None:
                if fp_type == 'morgan':
                    fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                elif fp_type == 'rdkit':
                    fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
                elif fp_type == 'maccs':
                    fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
                    n_bits = 167  # MACCS固定长度
                else:
                    raise ValueError(f"不支持的指纹类型: {fp_type}")
                    
                fp_array = np.array(fp)
            else:
                fp_array = np.zeros(n_bits)
                
            fingerprints.append(fp_array)
            
        result = np.array(fingerprints)
        self.logger.info(f"生成了 {fp_type} 指纹，形状: {result.shape}")
        return result
    
    def encode_smiles_onehot(self, smiles_list: List[str], max_length: Optional[int] = None) -> np.ndarray:
        """SMILES one-hot编码
        
        Args:
            smiles_list: SMILES字符串列表
            max_length: 最大长度
            
        Returns:
            One-hot编码数组
        """
        if max_length is None:
            max_length = max(len(smiles) for smiles in smiles_list)
            
        encoded = np.zeros((len(smiles_list), max_length, len(self.smiles_chars)))
        
        for i, smiles in enumerate(smiles_list):
            for j, char in enumerate(smiles[:max_length]):
                if char in self.char_to_idx:
                    encoded[i, j, self.char_to_idx[char]] = 1
                else:
                    # 未知字符用<UNK>表示
                    encoded[i, j, self.char_to_idx['<UNK>']] = 1
                    
        self.logger.info(f"SMILES one-hot编码完成，形状: {encoded.shape}")
        return encoded
    
    def encode_smiles_integer(self, smiles_list: List[str], max_length: Optional[int] = None) -> np.ndarray:
        """SMILES整数编码
        
        Args:
            smiles_list: SMILES字符串列表
            max_length: 最大长度
            
        Returns:
            整数编码数组
        """
        if max_length is None:
            max_length = max(len(smiles) for smiles in smiles_list)
            
        encoded = np.zeros((len(smiles_list), max_length), dtype=int)
        
        for i, smiles in enumerate(smiles_list):
            for j, char in enumerate(smiles[:max_length]):
                if char in self.char_to_idx:
                    encoded[i, j] = self.char_to_idx[char]
                else:
                    encoded[i, j] = self.char_to_idx['<UNK>']
                    
        self.logger.info(f"SMILES整数编码完成，形状: {encoded.shape}")
        return encoded
    
    def calculate_molecular_properties(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """计算分子性质
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            分子性质字典
        """
        properties = {
            'valid_molecules': [],
            'molecular_weight': [],
            'logp': [],
            'tpsa': [],
            'qed': [],
            'sa_score': [],
            'num_rings': [],
            'num_aromatic_rings': [],
            'num_rotatable_bonds': []
        }
        
        for smiles in smiles_list:
            mol = self.smiles_to_mol(smiles)
            if mol is not None:
                properties['valid_molecules'].append(1)
                properties['molecular_weight'].append(Descriptors.MolWt(mol))
                properties['logp'].append(Descriptors.MolLogP(mol))
                properties['tpsa'].append(Descriptors.TPSA(mol))
                
                # QED (药物相似性)
                try:
                    from rdkit.Chem import QED
                    properties['qed'].append(QED.qed(mol))
                except:
                    properties['qed'].append(0.0)
                
                # SA Score (合成可达性)
                try:
                    from rdkit.Contrib.SA_Score import sascorer
                    properties['sa_score'].append(sascorer.calculateScore(mol))
                except:
                    properties['sa_score'].append(5.0)  # 中等难度
                
                properties['num_rings'].append(Descriptors.RingCount(mol))
                properties['num_aromatic_rings'].append(Descriptors.NumAromaticRings(mol))
                properties['num_rotatable_bonds'].append(Descriptors.NumRotatableBonds(mol))
            else:
                properties['valid_molecules'].append(0)
                for key in properties.keys():
                    if key != 'valid_molecules':
                        properties[key].append(0.0)
        
        # 转换为numpy数组
        for key in properties:
            properties[key] = np.array(properties[key])
            
        self.logger.info(f"计算了 {len(properties)} 种分子性质")
        return properties
    
    def extract_features(self, smiles_list: List[str], 
                        feature_types: List[str] = ['descriptors', 'fingerprints']) -> Dict[str, np.ndarray]:
        """提取分子特征
        
        Args:
            smiles_list: SMILES字符串列表
            feature_types: 特征类型列表
            
        Returns:
            特征字典
        """
        features = {}
        
        if 'descriptors' in feature_types:
            features['descriptors'] = self.calculate_descriptors(smiles_list)
            
        if 'fingerprints' in feature_types:
            features['morgan_fp'] = self.generate_fingerprints(smiles_list, 'morgan')
            
        if 'maccs' in feature_types:
            features['maccs_fp'] = self.generate_fingerprints(smiles_list, 'maccs')
            
        if 'smiles_onehot' in feature_types:
            features['smiles_onehot'] = self.encode_smiles_onehot(smiles_list)
            
        if 'smiles_integer' in feature_types:
            features['smiles_integer'] = self.encode_smiles_integer(smiles_list)
            
        if 'properties' in feature_types:
            properties = self.calculate_molecular_properties(smiles_list)
            features.update(properties)
            
        self.logger.info(f"提取了 {len(features)} 种分子特征")
        return features
    
    def fit_transform(self, smiles_list: List[str], 
                     feature_types: List[str] = ['descriptors', 'fingerprints'],
                     normalize: bool = True) -> Dict[str, np.ndarray]:
        """拟合并转换分子特征
        
        Args:
            smiles_list: SMILES字符串列表
            feature_types: 特征类型列表
            normalize: 是否标准化
            
        Returns:
            特征字典
        """
        features = self.extract_features(smiles_list, feature_types)
        
        if normalize:
            # 对描述符进行标准化
            if 'descriptors' in features:
                if 'descriptors' not in self.scalers:
                    self.scalers['descriptors'] = StandardScaler()
                features['descriptors'] = self.scalers['descriptors'].fit_transform(features['descriptors'])
                
        return features
    
    def transform(self, smiles_list: List[str], 
                 feature_types: List[str] = ['descriptors', 'fingerprints'],
                 normalize: bool = True) -> Dict[str, np.ndarray]:
        """转换分子特征（使用已拟合的标准化器）
        
        Args:
            smiles_list: SMILES字符串列表
            feature_types: 特征类型列表
            normalize: 是否标准化
            
        Returns:
            特征字典
        """
        features = self.extract_features(smiles_list, feature_types)
        
        if normalize:
            # 使用已拟合的标准化器
            if 'descriptors' in features and 'descriptors' in self.scalers:
                features['descriptors'] = self.scalers['descriptors'].transform(features['descriptors'])
                
        return features