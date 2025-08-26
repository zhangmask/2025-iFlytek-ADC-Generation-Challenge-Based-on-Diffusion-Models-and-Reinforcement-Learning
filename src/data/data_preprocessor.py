"""数据预处理模块

提供数据清洗、转换、标准化等预处理功能。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, random_state: int = 42):
        """
        初始化数据预处理器
        
        Args:
            random_state: 随机种子
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        # 存储预处理参数
        self.scalers = {}
        self.encoders = {}
        self.preprocessing_params = {}
        
        self.logger.info("数据预处理器初始化完成")
    
    def clean_data(self, data: pd.DataFrame, 
                   remove_duplicates: bool = True,
                   handle_missing: str = 'drop',
                   missing_threshold: float = 0.5) -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            data: 输入数据
            remove_duplicates: 是否移除重复行
            handle_missing: 处理缺失值的方法 ('drop', 'fill', 'interpolate')
            missing_threshold: 缺失值阈值，超过此比例的列将被删除
            
        Returns:
            清洗后的数据
        """
        self.logger.info("开始数据清洗")
        
        cleaned_data = data.copy()
        original_shape = cleaned_data.shape
        
        # 移除重复行
        if remove_duplicates:
            duplicates_before = cleaned_data.duplicated().sum()
            cleaned_data = cleaned_data.drop_duplicates()
            duplicates_removed = duplicates_before - cleaned_data.duplicated().sum()
            self.logger.info(f"移除重复行: {duplicates_removed}")
        
        # 处理缺失值过多的列
        missing_ratios = cleaned_data.isnull().sum() / len(cleaned_data)
        columns_to_drop = missing_ratios[missing_ratios > missing_threshold].index.tolist()
        
        if columns_to_drop:
            cleaned_data = cleaned_data.drop(columns=columns_to_drop)
            self.logger.info(f"删除缺失值过多的列: {columns_to_drop}")
        
        # 处理剩余缺失值
        if handle_missing == 'drop':
            cleaned_data = cleaned_data.dropna()
        elif handle_missing == 'fill':
            # 数值列用均值填充，文本列用众数填充
            for column in cleaned_data.columns:
                if cleaned_data[column].dtype in ['int64', 'float64']:
                    cleaned_data[column].fillna(cleaned_data[column].mean(), inplace=True)
                else:
                    mode_value = cleaned_data[column].mode()
                    if not mode_value.empty:
                        cleaned_data[column].fillna(mode_value[0], inplace=True)
        elif handle_missing == 'interpolate':
            # 仅对数值列进行插值
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
            cleaned_data[numeric_columns] = cleaned_data[numeric_columns].interpolate()
        
        final_shape = cleaned_data.shape
        self.logger.info(f"数据清洗完成: {original_shape} -> {final_shape}")
        
        return cleaned_data
    
    def validate_sequences(self, data: pd.DataFrame, 
                          sequence_column: str = 'protein_sequence') -> pd.DataFrame:
        """
        验证和清洗蛋白质序列
        
        Args:
            data: 输入数据
            sequence_column: 蛋白质序列列名
            
        Returns:
            验证后的数据
        """
        if sequence_column not in data.columns:
            self.logger.warning(f"列 '{sequence_column}' 不存在，跳过序列验证")
            return data
        
        self.logger.info("开始蛋白质序列验证")
        
        validated_data = data.copy()
        original_count = len(validated_data)
        
        # 标准氨基酸字符
        valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        
        def is_valid_sequence(sequence):
            if pd.isna(sequence) or not isinstance(sequence, str):
                return False
            
            # 检查是否只包含标准氨基酸
            sequence_chars = set(sequence.upper())
            return sequence_chars.issubset(valid_amino_acids) and len(sequence) > 0
        
        # 过滤无效序列
        valid_mask = validated_data[sequence_column].apply(is_valid_sequence)
        validated_data = validated_data[valid_mask]
        
        # 标准化序列（转为大写）
        validated_data[sequence_column] = validated_data[sequence_column].str.upper()
        
        # 移除过短或过长的序列
        sequence_lengths = validated_data[sequence_column].str.len()
        length_q1 = sequence_lengths.quantile(0.01)
        length_q99 = sequence_lengths.quantile(0.99)
        
        length_mask = (sequence_lengths >= length_q1) & (sequence_lengths <= length_q99)
        validated_data = validated_data[length_mask]
        
        final_count = len(validated_data)
        removed_count = original_count - final_count
        
        self.logger.info(f"序列验证完成，移除无效序列: {removed_count}")
        
        return validated_data
    
    def validate_smiles(self, data: pd.DataFrame, 
                       smiles_column: str = 'smiles') -> pd.DataFrame:
        """
        验证和清洗SMILES字符串
        
        Args:
            data: 输入数据
            smiles_column: SMILES列名
            
        Returns:
            验证后的数据
        """
        if smiles_column not in data.columns:
            self.logger.warning(f"列 '{smiles_column}' 不存在，跳过SMILES验证")
            return data
        
        self.logger.info("开始SMILES验证")
        
        validated_data = data.copy()
        original_count = len(validated_data)
        
        def is_valid_smiles(smiles):
            if pd.isna(smiles) or not isinstance(smiles, str):
                return False
            
            # 基本SMILES字符检查
            valid_chars = set('ABCDEFGHIKLMNOPRSTUVWXYZabcdefghiklmnoprstuvwxyz0123456789[]()=#+\\-@/.%')
            smiles_chars = set(smiles)
            
            if not smiles_chars.issubset(valid_chars):
                return False
            
            # 检查括号匹配
            if smiles.count('(') != smiles.count(')'):
                return False
            if smiles.count('[') != smiles.count(']'):
                return False
            
            # 长度检查
            if len(smiles) < 3 or len(smiles) > 1000:
                return False
            
            return True
        
        # 过滤无效SMILES
        valid_mask = validated_data[smiles_column].apply(is_valid_smiles)
        validated_data = validated_data[valid_mask]
        
        # 清理SMILES（移除多余空格）
        validated_data[smiles_column] = validated_data[smiles_column].str.strip()
        
        final_count = len(validated_data)
        removed_count = original_count - final_count
        
        self.logger.info(f"SMILES验证完成，移除无效SMILES: {removed_count}")
        
        return validated_data
    
    def normalize_features(self, data: pd.DataFrame, 
                          columns: Optional[List[str]] = None,
                          method: str = 'standard') -> pd.DataFrame:
        """
        特征标准化
        
        Args:
            data: 输入数据
            columns: 要标准化的列，None表示所有数值列
            method: 标准化方法 ('standard', 'minmax')
            
        Returns:
            标准化后的数据
        """
        self.logger.info(f"开始特征标准化，方法: {method}")
        
        normalized_data = data.copy()
        
        if columns is None:
            columns = normalized_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 过滤存在的列
        columns = [col for col in columns if col in normalized_data.columns]
        
        if not columns:
            self.logger.warning("没有找到需要标准化的数值列")
            return normalized_data
        
        # 选择标准化器
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
        
        # 拟合和转换
        normalized_data[columns] = scaler.fit_transform(normalized_data[columns])
        
        # 保存标准化器
        scaler_key = f"{method}_scaler"
        self.scalers[scaler_key] = {
            'scaler': scaler,
            'columns': columns,
            'method': method
        }
        
        self.logger.info(f"特征标准化完成，处理列: {columns}")
        
        return normalized_data
    
    def encode_categorical(self, data: pd.DataFrame, 
                          columns: Optional[List[str]] = None,
                          method: str = 'label') -> pd.DataFrame:
        """
        分类变量编码
        
        Args:
            data: 输入数据
            columns: 要编码的列，None表示所有文本列
            method: 编码方法 ('label', 'onehot')
            
        Returns:
            编码后的数据
        """
        self.logger.info(f"开始分类变量编码，方法: {method}")
        
        encoded_data = data.copy()
        
        if columns is None:
            columns = encoded_data.select_dtypes(include=['object']).columns.tolist()
            # 排除序列列
            columns = [col for col in columns if 'sequence' not in col.lower() and 'smiles' not in col.lower()]
        
        # 过滤存在的列
        columns = [col for col in columns if col in encoded_data.columns]
        
        if not columns:
            self.logger.warning("没有找到需要编码的分类列")
            return encoded_data
        
        for column in columns:
            if method == 'label':
                encoder = LabelEncoder()
                encoded_data[column] = encoder.fit_transform(encoded_data[column].astype(str))
                
                # 保存编码器
                self.encoders[column] = {
                    'encoder': encoder,
                    'method': method
                }
                
            elif method == 'onehot':
                # 独热编码
                dummies = pd.get_dummies(encoded_data[column], prefix=column)
                encoded_data = pd.concat([encoded_data.drop(column, axis=1), dummies], axis=1)
                
                # 保存编码信息
                self.encoders[column] = {
                    'method': method,
                    'columns': dummies.columns.tolist()
                }
        
        self.logger.info(f"分类变量编码完成，处理列: {columns}")
        
        return encoded_data
    
    def create_splits(self, data: pd.DataFrame, 
                     target_column: Optional[str] = None,
                     test_size: float = 0.2,
                     val_size: float = 0.1,
                     stratify: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        创建训练/验证/测试集划分
        
        Args:
            data: 输入数据
            target_column: 目标列名（用于分层抽样）
            test_size: 测试集比例
            val_size: 验证集比例
            stratify: 是否进行分层抽样
            
        Returns:
            (训练集, 验证集, 测试集)
        """
        self.logger.info("开始创建数据集划分")
        
        # 分层抽样的目标
        stratify_column = None
        if stratify and target_column and target_column in data.columns:
            stratify_column = data[target_column]
        
        # 首先划分训练集和测试集
        train_val_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_column
        )
        
        # 再从训练集中划分验证集
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)  # 调整验证集比例
            
            stratify_train_val = None
            if stratify and target_column and target_column in train_val_data.columns:
                stratify_train_val = train_val_data[target_column]
            
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=val_size_adjusted,
                random_state=self.random_state,
                stratify=stratify_train_val
            )
        else:
            train_data = train_val_data
            val_data = pd.DataFrame()  # 空的验证集
        
        self.logger.info(f"数据集划分完成 - 训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def remove_outliers(self, data: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        移除异常值
        
        Args:
            data: 输入数据
            columns: 要处理的列，None表示所有数值列
            method: 异常值检测方法 ('iqr', 'zscore')
            threshold: 阈值
            
        Returns:
            移除异常值后的数据
        """
        self.logger.info(f"开始移除异常值，方法: {method}")
        
        cleaned_data = data.copy()
        original_count = len(cleaned_data)
        
        if columns is None:
            columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 过滤存在的列
        columns = [col for col in columns if col in cleaned_data.columns]
        
        if not columns:
            self.logger.warning("没有找到需要处理异常值的数值列")
            return cleaned_data
        
        outlier_mask = pd.Series([True] * len(cleaned_data), index=cleaned_data.index)
        
        for column in columns:
            if method == 'iqr':
                Q1 = cleaned_data[column].quantile(0.25)
                Q3 = cleaned_data[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                column_mask = (cleaned_data[column] >= lower_bound) & (cleaned_data[column] <= upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((cleaned_data[column] - cleaned_data[column].mean()) / cleaned_data[column].std())
                column_mask = z_scores <= threshold
            
            else:
                raise ValueError(f"不支持的异常值检测方法: {method}")
            
            outlier_mask = outlier_mask & column_mask
        
        cleaned_data = cleaned_data[outlier_mask]
        
        final_count = len(cleaned_data)
        removed_count = original_count - final_count
        
        self.logger.info(f"异常值移除完成，移除样本: {removed_count}")
        
        return cleaned_data
    
    def balance_dataset(self, data: pd.DataFrame, 
                       target_column: str,
                       method: str = 'undersample') -> pd.DataFrame:
        """
        数据集平衡
        
        Args:
            data: 输入数据
            target_column: 目标列名
            method: 平衡方法 ('undersample', 'oversample')
            
        Returns:
            平衡后的数据
        """
        if target_column not in data.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在")
        
        self.logger.info(f"开始数据集平衡，方法: {method}")
        
        balanced_data = data.copy()
        
        # 计算各类别样本数
        class_counts = balanced_data[target_column].value_counts()
        self.logger.info(f"原始类别分布: {class_counts.to_dict()}")
        
        if method == 'undersample':
            # 下采样到最小类别的样本数
            min_count = class_counts.min()
            
            balanced_samples = []
            for class_value in class_counts.index:
                class_data = balanced_data[balanced_data[target_column] == class_value]
                sampled_data = class_data.sample(n=min_count, random_state=self.random_state)
                balanced_samples.append(sampled_data)
            
            balanced_data = pd.concat(balanced_samples, ignore_index=True)
            
        elif method == 'oversample':
            # 上采样到最大类别的样本数
            max_count = class_counts.max()
            
            balanced_samples = []
            for class_value in class_counts.index:
                class_data = balanced_data[balanced_data[target_column] == class_value]
                
                if len(class_data) < max_count:
                    # 需要上采样
                    additional_samples = max_count - len(class_data)
                    oversampled_data = class_data.sample(n=additional_samples, 
                                                       replace=True, 
                                                       random_state=self.random_state)
                    class_data = pd.concat([class_data, oversampled_data], ignore_index=True)
                
                balanced_samples.append(class_data)
            
            balanced_data = pd.concat(balanced_samples, ignore_index=True)
        
        else:
            raise ValueError(f"不支持的平衡方法: {method}")
        
        # 打乱数据
        balanced_data = balanced_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # 计算平衡后的类别分布
        new_class_counts = balanced_data[target_column].value_counts()
        self.logger.info(f"平衡后类别分布: {new_class_counts.to_dict()}")
        
        return balanced_data
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        使用已保存的预处理参数转换新数据
        
        Args:
            data: 新数据
            
        Returns:
            转换后的数据
        """
        self.logger.info("开始转换新数据")
        
        transformed_data = data.copy()
        
        # 应用标准化器
        for scaler_key, scaler_info in self.scalers.items():
            scaler = scaler_info['scaler']
            columns = scaler_info['columns']
            
            # 检查列是否存在
            available_columns = [col for col in columns if col in transformed_data.columns]
            if available_columns:
                transformed_data[available_columns] = scaler.transform(transformed_data[available_columns])
                self.logger.info(f"应用标准化器 {scaler_key} 到列: {available_columns}")
        
        # 应用编码器
        for column, encoder_info in self.encoders.items():
            if column in transformed_data.columns:
                if encoder_info['method'] == 'label':
                    encoder = encoder_info['encoder']
                    # 处理未见过的类别
                    try:
                        transformed_data[column] = encoder.transform(transformed_data[column].astype(str))
                    except ValueError:
                        # 如果有未见过的类别，用最常见的类别替换
                        known_classes = set(encoder.classes_)
                        transformed_data[column] = transformed_data[column].apply(
                            lambda x: x if x in known_classes else encoder.classes_[0]
                        )
                        transformed_data[column] = encoder.transform(transformed_data[column])
                    
                    self.logger.info(f"应用标签编码器到列: {column}")
                
                elif encoder_info['method'] == 'onehot':
                    # 独热编码需要特殊处理
                    dummies = pd.get_dummies(transformed_data[column], prefix=column)
                    
                    # 确保所有原始列都存在
                    for orig_col in encoder_info['columns']:
                        if orig_col not in dummies.columns:
                            dummies[orig_col] = 0
                    
                    # 只保留原始训练时的列
                    dummies = dummies[encoder_info['columns']]
                    
                    transformed_data = pd.concat([transformed_data.drop(column, axis=1), dummies], axis=1)
                    self.logger.info(f"应用独热编码器到列: {column}")
        
        self.logger.info("新数据转换完成")
        
        return transformed_data
    
    def save_preprocessing_params(self, filepath: str) -> None:
        """
        保存预处理参数
        
        Args:
            filepath: 保存路径
        """
        import pickle
        
        params = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'preprocessing_params': self.preprocessing_params,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        
        self.logger.info(f"预处理参数已保存到: {filepath}")
    
    def load_preprocessing_params(self, filepath: str) -> None:
        """
        加载预处理参数
        
        Args:
            filepath: 参数文件路径
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        self.scalers = params.get('scalers', {})
        self.encoders = params.get('encoders', {})
        self.preprocessing_params = params.get('preprocessing_params', {})
        self.random_state = params.get('random_state', 42)
        
        self.logger.info(f"预处理参数已从 {filepath} 加载")
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        获取预处理摘要
        
        Returns:
            预处理摘要信息
        """
        summary = {
            'scalers': {
                name: {
                    'method': info['method'],
                    'columns': info['columns']
                } for name, info in self.scalers.items()
            },
            'encoders': {
                name: {
                    'method': info['method'],
                    'columns': info.get('columns', [name])
                } for name, info in self.encoders.items()
            },
            'random_state': self.random_state
        }
        
        return summary