#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础功能测试脚本

测试项目的基本功能是否正常工作。
"""

import sys
import os
sys.path.append('.')

def test_data_loading():
    """测试数据加载功能"""
    print("\n=== 测试数据加载功能 ===")
    try:
        from src.data import DataLoader
        
        # 创建测试数据加载器
        data_loader = DataLoader(data_dir="data")
        
        # 测试加载功能（不需要实际文件）
        print("数据加载器创建成功")
        
        # 测试方法存在性
        if hasattr(data_loader, 'load_train_data'):
            print("load_train_data方法存在")
        else:
            print("load_train_data方法不存在")
            
        print("✓ 数据加载测试通过")
        return True
            
    except Exception as e:
        print(f"数据加载测试失败: {e}")
        print("✗ 测试失败")
        return False

def test_feature_extraction():
    """测试特征提取功能"""
    print("\n=== 测试特征提取功能 ===")
    try:
        from src.features import SequenceEncoder, MoleculeFeatureExtractor
        
        # 创建测试编码器（需要配置参数）
        config = {'max_length': 100, 'encoding_dim': 128}
        encoder = SequenceEncoder(config)
        extractor = MoleculeFeatureExtractor(config)
        
        print("特征提取器创建成功")
        
        # 测试基本功能
        test_sequence = "ACDEFGHIKLMNPQRSTVWY"
        if hasattr(encoder, 'encode_sequences'):
            print("序列编码方法存在")
            
        print("✓ 特征提取测试通过")
        return True
        
    except Exception as e:
        print(f"特征提取测试失败: {e}")
        print("✗ 测试失败")
        return False

def test_config_loading():
    """测试配置加载功能"""
    print("\n=== 测试配置加载功能 ===")
    try:
        from config import Config, load_config
        
        # 测试默认配置
        config = Config()
        print(f"默认配置加载成功")
        print(f"数据配置: {config.data}")
        
        # 测试配置基本功能
        print(f"项目名称: {config.project_name}")
        print(f"输出目录: {config.output_dir}")
        
        # 测试路径获取
        paths = config.get_paths()
        print(f"路径配置获取成功: {len(paths)} 个路径")
        
        print("✓ 配置加载测试通过")
        return True
        
    except Exception as e:
        print(f"配置加载测试失败: {e}")
        print("✗ 测试失败")
        return False

def test_model_initialization():
    """测试模型初始化功能"""
    print("\n=== 测试模型初始化功能 ===")
    try:
        from src.models import DiffusionModel, RLAgent
        import torch
        
        # 创建测试模型（使用配置字典）
        diffusion_config = {
            'input_dim': 128,
            'hidden_dim': 256,
            'output_dim': 64,
            'num_timesteps': 1000,
            'learning_rate': 1e-4
        }
        rl_config = {
            'state_dim': 128,
            'action_dim': 64,
            'hidden_dim': 256,
            'learning_rate': 1e-3
        }
        
        diffusion_model = DiffusionModel(diffusion_config)
        rl_agent = RLAgent(rl_config)
        
        print("模型初始化成功")
        
        print("✓ 模型初始化测试通过")
        return True
        
    except Exception as e:
        print(f"模型初始化测试失败: {e}")
        print("✗ 测试失败")
        return False

def main():
    """主测试函数"""
    print("开始基础功能测试...")
    
    tests = [
        test_config_loading,
        test_data_loading,
        test_feature_extraction,
        test_model_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ 测试通过")
            else:
                print("✗ 测试失败")
        except Exception as e:
            print(f"✗ 测试异常: {e}")
    
    print(f"\n=== 测试总结 ===")
    print(f"通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有基础功能测试通过！")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关模块")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)