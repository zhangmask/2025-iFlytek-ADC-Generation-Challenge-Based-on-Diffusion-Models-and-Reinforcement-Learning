#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试批次维度问题
"""

import torch
import numpy as np
import logging
from src.generation.molecule_generator import MoleculeGenerator
from src.models.reinforcement_learning import RLAgent
import yaml

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_batch_dimensions():
    """测试批次维度问题"""
    logger.info("=== 测试批次维度问题 ===")
    
    # 加载配置
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建MoleculeGenerator
    generator = MoleculeGenerator(config)
    
    # 测试单个分子的状态生成
    test_smiles = "CCO"  # 乙醇
    protein_features = torch.randn(1, 512)  # 模拟蛋白质特征
    
    logger.info(f"测试SMILES: {test_smiles}")
    logger.info(f"蛋白质特征形状: {protein_features.shape}")
    
    # 获取分子状态
    state = generator._get_molecule_state(test_smiles, protein_features)
    logger.info(f"生成的状态形状: {state.shape}")
    
    # 测试RL agent
    if generator.rl_agent is not None:
        logger.info("测试RL agent...")
        logger.info(f"RL agent state_dim: {generator.rl_agent.state_dim}")
        logger.info(f"RL agent action_dim: {generator.rl_agent.action_dim}")
        
        try:
            action, log_prob = generator.rl_agent.select_action(state)
            logger.info(f"动作形状: {action.shape}")
            logger.info(f"对数概率形状: {log_prob.shape if log_prob is not None else None}")
        except Exception as e:
            logger.error(f"RL agent选择动作失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.error("RL agent未初始化")
    
    # 测试批次处理
    logger.info("\n=== 测试批次处理 ===")
    batch_size = 32
    protein_features_batch = torch.randn(batch_size, 512)
    logger.info(f"批次蛋白质特征形状: {protein_features_batch.shape}")
    
    # 模拟批次分子状态
    batch_states = []
    for i in range(batch_size):
        state = generator._get_molecule_state(test_smiles, protein_features_batch[i:i+1])
        batch_states.append(state)
    
    batch_states_tensor = torch.cat(batch_states, dim=0)
    logger.info(f"批次状态形状: {batch_states_tensor.shape}")
    
    # 测试批次RL处理
    if generator.rl_agent is not None:
        try:
            batch_actions, batch_log_probs = generator.rl_agent.select_action(batch_states_tensor)
            logger.info(f"批次动作形状: {batch_actions.shape}")
            logger.info(f"批次对数概率形状: {batch_log_probs.shape if batch_log_probs is not None else None}")
        except Exception as e:
            logger.error(f"批次RL处理失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_batch_dimensions()