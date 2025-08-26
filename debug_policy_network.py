#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试PolicyNetwork维度问题
"""

import torch
import numpy as np
import logging
import yaml
from src.models.reinforcement_learning import RLAgent, PolicyNetwork

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_policy_network_dimensions():
    """测试PolicyNetwork维度问题"""
    logger.info("=== 测试PolicyNetwork维度问题 ===")
    
    # 加载配置
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    rl_config = config.get('reinforcement_learning', {})
    logger.info(f"强化学习配置: {rl_config}")
    
    # 测试直接创建PolicyNetwork
    logger.info("\n=== 直接创建PolicyNetwork ===")
    
    # 从配置中获取参数
    env_config = rl_config.get('environment', {})
    agent_config = rl_config.get('agent', {})
    policy_net_config = agent_config.get('policy_network', {})
    
    state_dim = env_config.get('state_dim', 512)
    action_dim = env_config.get('action_space_size', 100)
    hidden_dims = policy_net_config.get('hidden_dims', [256, 256])
    
    logger.info(f"配置参数: state_dim={state_dim}, action_dim={action_dim}, hidden_dims={hidden_dims}")
    
    # 使用第一个隐藏层维度
    if isinstance(hidden_dims, (list, tuple)) and len(hidden_dims) > 0:
        hidden_dim = hidden_dims[0]
    else:
        hidden_dim = 256
    
    logger.info(f"使用的hidden_dim: {hidden_dim}")
    
    # 创建PolicyNetwork
    policy_net = PolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )
    
    logger.info(f"PolicyNetwork创建成功")
    logger.info(f"PolicyNetwork.state_dim: {policy_net.state_dim}")
    logger.info(f"PolicyNetwork.action_dim: {policy_net.action_dim}")
    
    # 检查网络结构
    logger.info("\n=== 网络结构 ===")
    for name, module in policy_net.named_modules():
        if isinstance(module, torch.nn.Linear):
            logger.info(f"{name}: {module.in_features} -> {module.out_features}")
    
    # 测试前向传播
    logger.info("\n=== 测试前向传播 ===")
    
    # 单个样本
    test_state = torch.randn(1, state_dim)
    logger.info(f"测试状态形状: {test_state.shape}")
    
    try:
        mean, std = policy_net.forward(test_state)
        logger.info(f"前向传播成功 - 均值形状: {mean.shape}, 标准差形状: {std.shape}")
    except Exception as e:
        logger.error(f"前向传播失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 批次样本
    batch_size = 32
    test_batch = torch.randn(batch_size, state_dim)
    logger.info(f"测试批次形状: {test_batch.shape}")
    
    try:
        mean, std = policy_net.forward(test_batch)
        logger.info(f"批次前向传播成功 - 均值形状: {mean.shape}, 标准差形状: {std.shape}")
    except Exception as e:
        logger.error(f"批次前向传播失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试通过RLAgent创建
    logger.info("\n=== 通过RLAgent创建 ===")
    
    try:
        rl_agent = RLAgent(rl_config)
        logger.info(f"RLAgent创建成功")
        logger.info(f"RLAgent.state_dim: {rl_agent.state_dim}")
        logger.info(f"RLAgent.action_dim: {rl_agent.action_dim}")
        logger.info(f"RLAgent.policy_hidden_dim: {rl_agent.policy_hidden_dim}")
        
        # 检查PolicyNetwork参数
        logger.info(f"PolicyNetwork.state_dim: {rl_agent.policy_net.state_dim}")
        logger.info(f"PolicyNetwork.action_dim: {rl_agent.policy_net.action_dim}")
        
        # 测试select_action
        test_state = torch.randn(1, state_dim)
        action, log_prob = rl_agent.select_action(test_state)
        logger.info(f"select_action成功 - 动作形状: {action.shape}")
        
        # 测试批次select_action
        test_batch = torch.randn(batch_size, state_dim)
        batch_action, batch_log_prob = rl_agent.select_action(test_batch)
        logger.info(f"批次select_action成功 - 动作形状: {batch_action.shape}")
        
    except Exception as e:
        logger.error(f"RLAgent测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_policy_network_dimensions()