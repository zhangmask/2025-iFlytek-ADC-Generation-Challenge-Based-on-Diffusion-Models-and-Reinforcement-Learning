#!/usr/bin/env python3
"""调试维度问题的脚本"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
import logging
from src.generation.molecule_generator import MoleculeGenerator
from src.features.molecule_features import MoleculeFeatureExtractor

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_dimensions():
    """调试维度问题"""
    print("=== 调试维度问题 ===")
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建特征提取器
    features_config = config.get('features', {})
    extractor = MoleculeFeatureExtractor(features_config)
    
    # 测试分子特征提取
    test_smiles = "CCO"  # 乙醇
    print(f"\n测试SMILES: {test_smiles}")
    
    # 提取特征
    mol_features = extractor.extract_features([test_smiles])
    print(f"\n提取的特征键: {list(mol_features.keys())}")
    
    for key, value in mol_features.items():
        print(f"特征 {key} 形状: {value.shape}")
    
    # 计算总的分子特征维度
    descriptors = mol_features.get('descriptors', np.array([]))
    morgan_fp = mol_features.get('morgan_fp', np.array([]))
    
    print(f"\n描述符维度: {descriptors.shape}")
    print(f"Morgan指纹维度: {morgan_fp.shape}")
    
    if len(descriptors.shape) > 1 and len(morgan_fp.shape) > 1:
        total_mol_dim = descriptors.shape[1] + morgan_fp.shape[1]
        print(f"总分子特征维度: {total_mol_dim}")
    
    # 测试MoleculeGenerator初始化
    print("\n=== 测试MoleculeGenerator初始化 ===")
    
    generation_config = {
        'max_attempts': 100,
        'features': features_config,
        'reinforcement_learning': {
            'action_space_size': config['model']['rl_batch_size'],  # 这里可能有问题
            'state_dim': 512,
            'policy_hidden_dims': config['model']['policy_hidden_dims'],
            'value_hidden_dims': config['model']['value_hidden_dims']
        }
    }
    
    print(f"生成配置: {generation_config}")
    
    try:
        generator = MoleculeGenerator(generation_config)
        print("MoleculeGenerator初始化成功")
        
        # 测试状态投影
        protein_features = torch.randn(1, 512)
        print(f"\n蛋白质特征形状: {protein_features.shape}")
        
        state = generator._get_molecule_state(test_smiles, protein_features)
        print(f"状态投影输出形状: {state.shape}")
        
    except Exception as e:
        print(f"MoleculeGenerator初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import numpy as np
    debug_dimensions()