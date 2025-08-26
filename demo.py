#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADC生成挑战赛 - 项目演示脚本

本脚本展示了项目的主要功能模块，包括：
1. 配置管理
2. 数据加载和探索
3. 特征工程
4. 模型初始化
5. 分子生成流程

运行方式：
    python demo.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from config import Config, setup_environment
from src.data import DataLoader
from src.features import SequenceEncoder, MoleculeFeatureExtractor
from src.models import DiffusionModel, RLAgent
from src.generation import MoleculeGenerator, LinkerGenerator, DARPredictor
from src.evaluation import DiversityMetrics, ValidityMetrics

def print_section(title):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def demo_configuration():
    """演示配置管理功能"""
    print_section("配置管理演示")
    
    # 加载默认配置
    config = Config()
    print(f"✓ 项目名称: {config.project_name}")
    print(f"✓ 输出目录: {config.output_dir}")
    print(f"✓ 数据目录: {config.data.data_dir}")
    
    # 设置环境
    setup_environment()
    print(f"✓ 环境设置完成")
    
    # 获取路径配置
    paths = config.get_paths()
    print(f"✓ 配置路径数量: {len(paths)}")
    
    return config

def demo_data_loading(config):
    """演示数据加载功能"""
    print_section("数据加载演示")
    
    # 创建数据加载器
    data_loader = DataLoader(str(config.data.data_dir))
    print(f"✓ 数据加载器创建成功")
    
    # 检查数据文件
    data_dir = str(config.data.data_dir)
    train_file = os.path.join(data_dir, config.data.train_file)
    test_file = os.path.join(data_dir, config.data.test_file)
    
    if os.path.exists(train_file):
        print(f"✓ 训练数据文件存在: {train_file}")
        # 这里可以添加实际的数据加载代码
        # train_data = data_loader.load_train_data()
    else:
        print(f"⚠ 训练数据文件不存在: {train_file}")
        print(f"  请将train.csv文件放置在 {config.data.data_dir} 目录下")
    
    if os.path.exists(test_file):
        print(f"✓ 测试数据文件存在: {test_file}")
    else:
        print(f"⚠ 测试数据文件不存在: {test_file}")
        print(f"  请将test.csv文件放置在 {config.data.data_dir} 目录下")
    
    return data_loader

def demo_feature_engineering(config):
    """演示特征工程功能"""
    print_section("特征工程演示")
    
    # 创建序列编码器
    sequence_config = {
        'embedding_dim': config.model.embedding_dim,
        'max_length': config.data.max_sequence_length,
        'vocab_size': 25  # 氨基酸词汇表大小
    }
    sequence_encoder = SequenceEncoder(sequence_config)
    print(f"✓ 蛋白质序列编码器创建成功")
    
    # 创建分子特征提取器
    molecule_config = {
        'feature_dim': 2048,
        'fingerprint_type': 'morgan',
        'radius': 2
    }
    molecule_extractor = MoleculeFeatureExtractor(molecule_config)
    print(f"✓ 分子特征提取器创建成功")
    
    # 演示编码功能
    sample_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    sample_smiles = "CCO"  # 乙醇的SMILES表示
    
    print(f"✓ 示例蛋白质序列长度: {len(sample_sequence)}")
    print(f"✓ 示例SMILES: {sample_smiles}")
    
    return sequence_encoder, molecule_extractor

def demo_model_initialization(config):
    """演示模型初始化功能"""
    print_section("模型初始化演示")
    
    # 扩散模型配置
    diffusion_config = {
        'input_dim': config.model.embedding_dim,
        'hidden_dim': config.model.hidden_dim,
        'output_dim': config.model.embedding_dim,
        'num_timesteps': 1000
    }
    
    # 强化学习配置
    rl_config = {
        'state_dim': config.model.embedding_dim,
        'action_dim': config.model.hidden_dim,
        'hidden_dim': config.model.hidden_dim,
        'lr': config.training.learning_rate
    }
    
    # 创建模型
    diffusion_model = DiffusionModel(diffusion_config)
    rl_agent = RLAgent(rl_config)
    
    print(f"✓ 扩散模型创建成功")
    print(f"✓ 强化学习智能体创建成功")
    
    # 显示模型参数数量
    diffusion_params = sum(p.numel() for p in diffusion_model.parameters())
    
    # RLAgent包含多个网络，分别计算参数数量
    policy_params = sum(p.numel() for p in rl_agent.policy_net.parameters())
    value_params = sum(p.numel() for p in rl_agent.value_net.parameters())
    rl_params = policy_params + value_params
    
    print(f"✓ 扩散模型参数数量: {diffusion_params:,}")
    print(f"✓ 强化学习模型参数数量: {rl_params:,}")
    print(f"  - 策略网络: {policy_params:,}")
    print(f"  - 价值网络: {value_params:,}")
    
    return diffusion_model, rl_agent

def demo_generation_pipeline(config, diffusion_model, rl_agent):
    """演示分子生成流程"""
    print_section("分子生成流程演示")
    
    # 创建生成器配置
    generator_config = {
        'max_attempts': 100,
        'diversity_threshold': 0.7,
        'validity_threshold': 0.8,
        'molecule_features': {
            'feature_dim': 2048,
            'fingerprint_type': 'morgan',
            'radius': 2
        }
    }
    
    # 创建生成器
    molecule_generator = MoleculeGenerator(generator_config)
    linker_generator = LinkerGenerator(generator_config)
    dar_predictor = DARPredictor(generator_config)
    
    print(f"✓ 分子生成器创建成功")
    print(f"✓ Linker生成器创建成功")
    print(f"✓ DAR预测器创建成功")
    
    # 演示生成流程（模拟）
    print(f"\n--- 模拟生成流程 ---")
    print(f"1. 输入抗体序列和药物分子")
    print(f"2. 提取特征表示")
    print(f"3. 使用扩散模型生成候选Linker")
    print(f"4. 使用强化学习优化生成策略")
    print(f"5. 预测DAR值")
    print(f"6. 评估生成质量")
    
    return molecule_generator, linker_generator, dar_predictor

def demo_evaluation_metrics():
    """演示评估指标功能"""
    print_section("评估指标演示")
    
    # 创建评估器
    diversity_metrics = DiversityMetrics()
    validity_metrics = ValidityMetrics()
    
    print(f"✓ 多样性评估器创建成功")
    print(f"✓ 有效性评估器创建成功")
    
    # 模拟评估数据
    sample_molecules = ["CCO", "CCN", "CCC", "C1CCCCC1", "c1ccccc1"]
    print(f"\n--- 模拟评估 ---")
    print(f"示例分子数量: {len(sample_molecules)}")
    print(f"示例分子: {sample_molecules}")
    
    return diversity_metrics, validity_metrics

def main():
    """主演示函数"""
    print("🚀 ADC生成挑战赛项目演示")
    print("本演示将展示项目的主要功能模块")
    
    try:
        # 1. 配置管理
        config = demo_configuration()
        
        # 2. 数据加载
        data_loader = demo_data_loading(config)
        
        # 3. 特征工程
        sequence_encoder, molecule_extractor = demo_feature_engineering(config)
        
        # 4. 模型初始化
        diffusion_model, rl_agent = demo_model_initialization(config)
        
        # 5. 分子生成流程
        molecule_generator, linker_generator, dar_predictor = demo_generation_pipeline(
            config, diffusion_model, rl_agent
        )
        
        # 6. 评估指标
        diversity_metrics, validity_metrics = demo_evaluation_metrics()
        
        # 总结
        print_section("演示总结")
        print("✅ 所有模块演示完成！")
        print("\n📋 项目功能概览:")
        print("  • 配置管理系统 ✓")
        print("  • 数据加载和预处理 ✓")
        print("  • 蛋白质序列编码 ✓")
        print("  • 分子特征提取 ✓")
        print("  • 扩散模型架构 ✓")
        print("  • 强化学习框架 ✓")
        print("  • 分子生成流程 ✓")
        print("  • 评估指标系统 ✓")
        
        print("\n🎯 下一步操作建议:")
        print("  1. 准备训练数据 (train.csv, test.csv)")
        print("  2. 运行数据探索分析: python -m src.data.data_explorer")
        print("  3. 开始模型训练: python main.py --mode train")
        print("  4. 进行分子生成: python main.py --mode generate")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("请检查项目依赖是否正确安装")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)