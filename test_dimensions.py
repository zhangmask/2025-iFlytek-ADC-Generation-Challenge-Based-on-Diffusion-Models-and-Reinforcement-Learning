#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
维度测试脚本

测试MoleculeGenerator和RLAgent的维度配置
"""

import sys
import torch
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from config import Config
from src.generation.molecule_generator import MoleculeGenerator
from src.models.reinforcement_learning import RLAgent

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_molecule_generator_config():
    """测试MoleculeGenerator配置"""
    logger.info("=== 测试MoleculeGenerator配置 ===")
    
    config = Config()
    
    # 分子生成器配置
    generation_config = {
        'max_attempts': config.generation.max_attempts,
        'diversity_threshold': config.generation.diversity_threshold,
        'validity_threshold': 0.8,
        'num_samples': config.generation.num_samples,
        'optimization_steps': config.generation.optimization_steps,
        'features': {
            'fingerprint_size': config.features.fingerprint_size,
            'fingerprint_type': 'morgan',
            'radius': 2
        },
        'reinforcement_learning': {
            'environment': {
                'action_space_size': 100,
                'state_dim': 512
            },
            'agent': {
                'policy_network': {
                    'hidden_dims': config.model.policy_hidden_dims
                },
                'value_network': {
                    'hidden_dims': config.model.value_hidden_dims
                }
            }
        }
    }
    
    logger.info(f"生成配置: {generation_config}")
    
    # 创建MoleculeGenerator
    molecule_generator = MoleculeGenerator(generation_config)
    
    return molecule_generator

def test_rl_agent_config():
    """测试RLAgent配置"""
    logger.info("=== 测试RLAgent配置 ===")
    
    rl_config = {
        'environment': {
            'action_space_size': 100,
            'state_dim': 512
        },
        'agent': {
            'policy_network': {
                'hidden_dims': [256, 256]
            },
            'value_network': {
                'hidden_dims': [256, 256]
            }
        }
    }
    
    logger.info(f"RL配置: {rl_config}")
    
    # 创建RLAgent
    rl_agent = RLAgent(rl_config)
    
    return rl_agent

def test_state_projection():
    """测试状态投影"""
    logger.info("=== 测试状态投影 ===")
    
    # 创建测试数据
    batch_size = 32
    protein_features = torch.randn(batch_size, 512)
    test_smiles = "CCO"  # 简单的乙醇分子
    
    # 创建MoleculeGenerator
    molecule_generator = test_molecule_generator_config()
    
    # 创建RLAgent
    rl_agent = test_rl_agent_config()
    
    # 设置RL智能体
    molecule_generator.set_rl_agent(rl_agent)
    
    try:
        # 测试状态获取
        logger.info(f"测试分子: {test_smiles}")
        logger.info(f"蛋白质特征形状: {protein_features.shape}")
        
        state = molecule_generator._get_molecule_state(test_smiles, protein_features)
        logger.info(f"状态形状: {state.shape}")
        
        # 测试动作选择
        action, log_prob = rl_agent.select_action(state)
        logger.info(f"动作形状: {action.shape}")
        logger.info(f"对数概率形状: {log_prob.shape}")
        
        logger.info("✅ 状态投影测试成功！")
        
    except Exception as e:
        logger.error(f"❌ 状态投影测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_state_projection()