#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试工具模块

提供测试辅助函数、模拟对象和测试数据生成功能。

Author: AI Developer
Date: 2025
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
import yaml
import json
from rdkit import Chem
from rdkit.Chem import Descriptors

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class TestConfig:
    """测试配置类"""
    use_gpu: bool = False
    batch_size: int = 4
    num_samples: int = 10
    timeout: int = 30
    mock_models: bool = True
    temp_dir: Optional[str] = None


class MockDataGenerator:
    """模拟数据生成器"""
    
    def __init__(self, seed: int = 42):
        """
        初始化模拟数据生成器
        
        Args:
            seed: 随机种子
        """
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_smiles_list(self, count: int = 10) -> List[str]:
        """生成SMILES列表
        
        Args:
            count: 生成数量
            
        Returns:
            SMILES字符串列表
        """
        # 一些有效的SMILES示例
        base_smiles = [
            'CCO',  # 乙醇
            'CC(=O)O',  # 乙酸
            'c1ccccc1',  # 苯
            'CCN(CC)CC',  # 三乙胺
            'CC(C)O',  # 异丙醇
            'CCOCC',  # 乙醚
            'CC(C)(C)O',  # 叔丁醇
            'c1ccc(cc1)O',  # 苯酚
            'CC(=O)N',  # 乙酰胺
            'CCCCCCCCCCCCCCCCCC(=O)O'  # 硬脂酸
        ]
        
        # 随机选择和重复
        smiles_list = []
        for _ in range(count):
            smiles = np.random.choice(base_smiles)
            smiles_list.append(smiles)
        
        return smiles_list
    
    def generate_molecular_properties(self, smiles_list: List[str]) -> pd.DataFrame:
        """生成分子属性数据
        
        Args:
            smiles_list: SMILES列表
            
        Returns:
            属性数据框
        """
        data = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    properties = {
                        'smiles': smiles,
                        'molecular_weight': Descriptors.MolWt(mol),
                        'logp': Descriptors.MolLogP(mol),
                        'hbd': Descriptors.NumHDonors(mol),
                        'hba': Descriptors.NumHAcceptors(mol),
                        'tpsa': Descriptors.TPSA(mol),
                        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                        'dar_score': np.random.uniform(0.1, 0.9),
                        'diversity_score': np.random.uniform(0.1, 0.9)
                    }
                else:
                    # 无效分子的默认值
                    properties = {
                        'smiles': smiles,
                        'molecular_weight': 0,
                        'logp': 0,
                        'hbd': 0,
                        'hba': 0,
                        'tpsa': 0,
                        'rotatable_bonds': 0,
                        'dar_score': 0,
                        'diversity_score': 0
                    }
                
                data.append(properties)
                
            except Exception:
                # 异常情况的默认值
                properties = {
                    'smiles': smiles,
                    'molecular_weight': 0,
                    'logp': 0,
                    'hbd': 0,
                    'hba': 0,
                    'tpsa': 0,
                    'rotatable_bonds': 0,
                    'dar_score': 0,
                    'diversity_score': 0
                }
                data.append(properties)
        
        return pd.DataFrame(data)
    
    def generate_training_data(self, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成训练数据
        
        Args:
            num_samples: 样本数量
            
        Returns:
            输入张量和标签张量
        """
        # 生成随机输入数据
        input_dim = 128
        inputs = torch.randn(num_samples, input_dim)
        
        # 生成随机标签
        labels = torch.randint(0, 2, (num_samples,)).float()
        
        return inputs, labels
    
    def generate_config_dict(self) -> Dict[str, Any]:
        """生成配置字典
        
        Returns:
            配置字典
        """
        config = {
            'model': {
                'type': 'test_model',
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_epochs': 10,
                'early_stopping': True
            },
            'data': {
                'train_path': 'data/train.csv',
                'test_path': 'data/test.csv',
                'validation_split': 0.2
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        return config


class MockModel:
    """模拟模型类"""
    
    def __init__(self, input_dim: int = 128, output_dim: int = 1):
        """
        初始化模拟模型
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_trained = False
        self.training_history = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量
        """
        batch_size = x.shape[0]
        return torch.randn(batch_size, self.output_dim)
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """训练步骤
        
        Args:
            x: 输入张量
            y: 标签张量
            
        Returns:
            训练指标
        """
        loss = np.random.uniform(0.1, 1.0)
        accuracy = np.random.uniform(0.5, 0.95)
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy
        }
        
        self.training_history.append(metrics)
        return metrics
    
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """评估模型
        
        Args:
            x: 输入张量
            y: 标签张量
            
        Returns:
            评估指标
        """
        loss = np.random.uniform(0.1, 1.0)
        accuracy = np.random.uniform(0.5, 0.95)
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    
    def save(self, path: str) -> None:
        """保存模型
        
        Args:
            path: 保存路径
        """
        # 模拟保存操作
        pass
    
    def load(self, path: str) -> None:
        """加载模型
        
        Args:
            path: 模型路径
        """
        # 模拟加载操作
        self.is_trained = True


class TestEnvironment:
    """测试环境管理器"""
    
    def __init__(self, config: Optional[TestConfig] = None):
        """
        初始化测试环境
        
        Args:
            config: 测试配置
        """
        self.config = config or TestConfig()
        self.temp_dir = None
        self.original_cwd = None
        self.mock_patches = []
    
    def __enter__(self):
        """进入测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.config.temp_dir = self.temp_dir
        
        # 保存当前工作目录
        self.original_cwd = os.getcwd()
        
        # 切换到临时目录
        os.chdir(self.temp_dir)
        
        # 创建必要的子目录
        subdirs = ['data', 'models', 'logs', 'output', 'config']
        for subdir in subdirs:
            os.makedirs(subdir, exist_ok=True)
        
        # 设置环境变量
        os.environ['TESTING'] = 'true'
        os.environ['TEST_TEMP_DIR'] = self.temp_dir
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出测试环境"""
        # 恢复工作目录
        if self.original_cwd:
            os.chdir(self.original_cwd)
        
        # 清理临时目录
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # 清理环境变量
        os.environ.pop('TESTING', None)
        os.environ.pop('TEST_TEMP_DIR', None)
        
        # 清理模拟补丁
        for patch_obj in self.mock_patches:
            patch_obj.stop()
        self.mock_patches.clear()
    
    def create_test_file(self, filename: str, content: str) -> str:
        """创建测试文件
        
        Args:
            filename: 文件名
            content: 文件内容
            
        Returns:
            文件路径
        """
        file_path = os.path.join(self.temp_dir, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return file_path
    
    def create_test_config(self, config_dict: Dict[str, Any], filename: str = 'config.yaml') -> str:
        """创建测试配置文件
        
        Args:
            config_dict: 配置字典
            filename: 配置文件名
            
        Returns:
            配置文件路径
        """
        config_path = os.path.join(self.temp_dir, 'config', filename)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        return config_path
    
    def create_test_data(self, data: Union[pd.DataFrame, Dict, List], 
                        filename: str, format: str = 'csv') -> str:
        """创建测试数据文件
        
        Args:
            data: 数据
            filename: 文件名
            format: 文件格式 ('csv', 'json', 'yaml')
            
        Returns:
            数据文件路径
        """
        data_path = os.path.join(self.temp_dir, 'data', filename)
        
        if format == 'csv' and isinstance(data, pd.DataFrame):
            data.to_csv(data_path, index=False)
        elif format == 'json':
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == 'yaml':
            with open(data_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        return data_path
    
    def add_mock_patch(self, target: str, **kwargs) -> Mock:
        """添加模拟补丁
        
        Args:
            target: 目标对象
            **kwargs: 补丁参数
            
        Returns:
            模拟对象
        """
        patch_obj = patch(target, **kwargs)
        mock_obj = patch_obj.start()
        self.mock_patches.append(patch_obj)
        return mock_obj


def assert_tensor_equal(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                       rtol: float = 1e-5, atol: float = 1e-8) -> None:
    """断言张量相等
    
    Args:
        tensor1: 张量1
        tensor2: 张量2
        rtol: 相对容差
        atol: 绝对容差
    """
    assert tensor1.shape == tensor2.shape, f"张量形状不匹配: {tensor1.shape} vs {tensor2.shape}"
    assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), "张量值不相等"


def assert_dict_contains(actual: Dict[str, Any], expected: Dict[str, Any]) -> None:
    """断言字典包含期望的键值对
    
    Args:
        actual: 实际字典
        expected: 期望字典
    """
    for key, value in expected.items():
        assert key in actual, f"缺少键: {key}"
        if isinstance(value, dict):
            assert_dict_contains(actual[key], value)
        else:
            assert actual[key] == value, f"键 {key} 的值不匹配: {actual[key]} vs {value}"


def create_mock_logger() -> Mock:
    """创建模拟日志器
    
    Returns:
        模拟日志器
    """
    logger = Mock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    return logger


def skip_if_no_gpu(func):
    """如果没有GPU则跳过测试的装饰器
    
    Args:
        func: 测试函数
        
    Returns:
        装饰后的函数
    """
    import pytest
    
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            pytest.skip("需要GPU支持")
        return func(*args, **kwargs)
    
    return wrapper


def timeout_test(seconds: int):
    """测试超时装饰器
    
    Args:
        seconds: 超时秒数
        
    Returns:
        装饰器函数
    """
    import threading
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"测试超时: {seconds}秒")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    return decorator


# 常用的测试数据
TEST_SMILES = [
    'CCO',  # 乙醇
    'CC(=O)O',  # 乙酸
    'c1ccccc1',  # 苯
    'CCN(CC)CC',  # 三乙胺
    'CC(C)O',  # 异丙醇
]

TEST_CONFIG = {
    'model': {
        'type': 'test_model',
        'hidden_dim': 64,
        'num_layers': 2
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 10
    },
    'data': {
        'input_dim': 128,
        'output_dim': 64
    }
}