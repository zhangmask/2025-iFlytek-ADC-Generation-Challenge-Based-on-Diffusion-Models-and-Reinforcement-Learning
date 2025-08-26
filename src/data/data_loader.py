"""数据加载模块

负责加载和管理训练数据、测试数据等。
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        
        # 数据缓存
        self._train_data = None
        self._test_data = None
        self._validation_data = None
        
        self.logger.info(f"数据加载器初始化完成，数据目录: {self.data_dir}")
    
    def load_train_data(self, filename: str = "train.csv", 
                       reload: bool = False) -> pd.DataFrame:
        """
        加载训练数据
        
        Args:
            filename: 训练数据文件名
            reload: 是否重新加载数据
            
        Returns:
            训练数据DataFrame
        """
        if self._train_data is not None and not reload:
            self.logger.info("返回缓存的训练数据")
            return self._train_data
        
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"训练数据文件不存在: {filepath}")
        
        self.logger.info(f"加载训练数据: {filepath}")
        
        try:
            self._train_data = pd.read_csv(filepath)
            self.logger.info(f"训练数据加载成功，形状: {self._train_data.shape}")
            
            # 基本数据验证
            self._validate_data(self._train_data, "训练数据")
            
            return self._train_data
            
        except Exception as e:
            self.logger.error(f"加载训练数据失败: {e}")
            raise
    
    def load_test_data(self, filename: str = "test.csv", 
                      reload: bool = False) -> pd.DataFrame:
        """
        加载测试数据
        
        Args:
            filename: 测试数据文件名
            reload: 是否重新加载数据
            
        Returns:
            测试数据DataFrame
        """
        if self._test_data is not None and not reload:
            self.logger.info("返回缓存的测试数据")
            return self._test_data
        
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"测试数据文件不存在: {filepath}")
        
        self.logger.info(f"加载测试数据: {filepath}")
        
        try:
            self._test_data = pd.read_csv(filepath)
            self.logger.info(f"测试数据加载成功，形状: {self._test_data.shape}")
            
            # 基本数据验证
            self._validate_data(self._test_data, "测试数据")
            
            return self._test_data
            
        except Exception as e:
            self.logger.error(f"加载测试数据失败: {e}")
            raise
    
    def load_custom_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        加载自定义数据文件
        
        Args:
            filepath: 数据文件路径
            
        Returns:
            数据DataFrame
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        
        self.logger.info(f"加载自定义数据: {filepath}")
        
        try:
            # 根据文件扩展名选择加载方法
            if filepath.suffix.lower() == '.csv':
                data = pd.read_csv(filepath)
            elif filepath.suffix.lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(filepath)
            elif filepath.suffix.lower() == '.json':
                data = pd.read_json(filepath)
            elif filepath.suffix.lower() == '.parquet':
                data = pd.read_parquet(filepath)
            else:
                # 默认尝试CSV格式
                data = pd.read_csv(filepath)
            
            self.logger.info(f"自定义数据加载成功，形状: {data.shape}")
            return data
            
        except Exception as e:
            self.logger.error(f"加载自定义数据失败: {e}")
            raise
    
    def _validate_data(self, data: pd.DataFrame, data_name: str) -> None:
        """
        验证数据基本格式
        
        Args:
            data: 数据DataFrame
            data_name: 数据名称
        """
        if data.empty:
            raise ValueError(f"{data_name}为空")
        
        # 检查必要的列
        expected_columns = ['protein_sequence', 'smiles']
        missing_columns = [col for col in expected_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"{data_name}缺少预期列: {missing_columns}")
        
        # 检查空值
        null_counts = data.isnull().sum()
        if null_counts.sum() > 0:
            self.logger.warning(f"{data_name}包含空值:\n{null_counts[null_counts > 0]}")
        
        self.logger.info(f"{data_name}验证完成")
    
    def create_validation_split(self, train_data: pd.DataFrame = None, 
                              validation_ratio: float = 0.2, 
                              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        创建验证集分割
        
        Args:
            train_data: 训练数据，如果为None则使用已加载的训练数据
            validation_ratio: 验证集比例
            random_state: 随机种子
            
        Returns:
            (训练集, 验证集)
        """
        if train_data is None:
            if self._train_data is None:
                raise ValueError("请先加载训练数据或提供train_data参数")
            train_data = self._train_data
        
        self.logger.info(f"创建验证集分割，验证集比例: {validation_ratio}")
        
        # 随机分割
        shuffled_data = train_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        validation_size = int(len(shuffled_data) * validation_ratio)
        
        validation_data = shuffled_data[:validation_size]
        train_data_split = shuffled_data[validation_size:]
        
        self.logger.info(f"数据分割完成 - 训练集: {len(train_data_split)}, 验证集: {len(validation_data)}")
        
        # 缓存验证集
        self._validation_data = validation_data
        
        return train_data_split, validation_data
    
    def get_data_info(self, data: pd.DataFrame = None) -> Dict[str, any]:
        """
        获取数据基本信息
        
        Args:
            data: 数据DataFrame，如果为None则使用训练数据
            
        Returns:
            数据信息字典
        """
        if data is None:
            if self._train_data is None:
                raise ValueError("请先加载数据或提供data参数")
            data = self._train_data
        
        info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'null_counts': data.isnull().sum().to_dict(),
            'duplicate_rows': data.duplicated().sum()
        }
        
        # 数值列统计
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            info['numeric_stats'] = data[numeric_columns].describe().to_dict()
        
        # 文本列统计
        text_columns = data.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            text_stats = {}
            for col in text_columns:
                text_stats[col] = {
                    'unique_count': data[col].nunique(),
                    'most_common': data[col].value_counts().head(5).to_dict() if not data[col].empty else {},
                    'avg_length': data[col].astype(str).str.len().mean() if not data[col].empty else 0
                }
            info['text_stats'] = text_stats
        
        return info
    
    def filter_data(self, data: pd.DataFrame, 
                   filters: Dict[str, any]) -> pd.DataFrame:
        """
        根据条件过滤数据
        
        Args:
            data: 原始数据
            filters: 过滤条件字典
                    例: {'column_name': {'min': 0, 'max': 100}, 
                         'other_column': {'values': ['A', 'B']}}
        
        Returns:
            过滤后的数据
        """
        filtered_data = data.copy()
        
        for column, conditions in filters.items():
            if column not in filtered_data.columns:
                self.logger.warning(f"列 '{column}' 不存在，跳过过滤")
                continue
            
            # 数值范围过滤
            if 'min' in conditions:
                filtered_data = filtered_data[filtered_data[column] >= conditions['min']]
            
            if 'max' in conditions:
                filtered_data = filtered_data[filtered_data[column] <= conditions['max']]
            
            # 值列表过滤
            if 'values' in conditions:
                filtered_data = filtered_data[filtered_data[column].isin(conditions['values'])]
            
            # 非空过滤
            if conditions.get('not_null', False):
                filtered_data = filtered_data[filtered_data[column].notna()]
            
            # 字符串长度过滤
            if 'min_length' in conditions:
                filtered_data = filtered_data[filtered_data[column].astype(str).str.len() >= conditions['min_length']]
            
            if 'max_length' in conditions:
                filtered_data = filtered_data[filtered_data[column].astype(str).str.len() <= conditions['max_length']]
        
        self.logger.info(f"数据过滤完成: {len(data)} -> {len(filtered_data)}")
        return filtered_data
    
    def sample_data(self, data: pd.DataFrame, 
                   n_samples: int = None, 
                   frac: float = None, 
                   random_state: int = 42) -> pd.DataFrame:
        """
        数据采样
        
        Args:
            data: 原始数据
            n_samples: 采样数量
            frac: 采样比例
            random_state: 随机种子
            
        Returns:
            采样后的数据
        """
        if n_samples is not None and frac is not None:
            raise ValueError("n_samples和frac不能同时指定")
        
        if n_samples is not None:
            if n_samples > len(data):
                self.logger.warning(f"采样数量({n_samples})大于数据总量({len(data)})，返回全部数据")
                return data
            sampled_data = data.sample(n=n_samples, random_state=random_state)
        elif frac is not None:
            sampled_data = data.sample(frac=frac, random_state=random_state)
        else:
            raise ValueError("必须指定n_samples或frac")
        
        self.logger.info(f"数据采样完成: {len(data)} -> {len(sampled_data)}")
        return sampled_data
    
    def save_data(self, data: pd.DataFrame, 
                 filepath: Union[str, Path], 
                 format: str = 'csv') -> None:
        """
        保存数据
        
        Args:
            data: 要保存的数据
            filepath: 保存路径
            format: 保存格式 ('csv', 'excel', 'json', 'parquet')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"保存数据到: {filepath}")
        
        try:
            if format.lower() == 'csv':
                data.to_csv(filepath, index=False)
            elif format.lower() == 'excel':
                data.to_excel(filepath, index=False)
            elif format.lower() == 'json':
                data.to_json(filepath, orient='records', indent=2)
            elif format.lower() == 'parquet':
                data.to_parquet(filepath, index=False)
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            self.logger.info("数据保存成功")
            
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
            raise
    
    def get_cached_data(self) -> Dict[str, pd.DataFrame]:
        """
        获取所有缓存的数据
        
        Returns:
            缓存数据字典
        """
        cached_data = {}
        
        if self._train_data is not None:
            cached_data['train'] = self._train_data
        
        if self._test_data is not None:
            cached_data['test'] = self._test_data
        
        if self._validation_data is not None:
            cached_data['validation'] = self._validation_data
        
        return cached_data
    
    def clear_cache(self) -> None:
        """
        清除数据缓存
        """
        self._train_data = None
        self._test_data = None
        self._validation_data = None
        self.logger.info("数据缓存已清除")
    
    def list_data_files(self) -> List[str]:
        """
        列出数据目录中的所有文件
        
        Returns:
            文件名列表
        """
        if not self.data_dir.exists():
            self.logger.warning(f"数据目录不存在: {self.data_dir}")
            return []
        
        # 支持的文件格式
        supported_extensions = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
        
        files = []
        for file in self.data_dir.iterdir():
            if file.is_file() and file.suffix.lower() in supported_extensions:
                files.append(file.name)
        
        return sorted(files)