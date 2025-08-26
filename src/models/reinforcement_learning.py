"""强化学习模块

实现用于分子优化的强化学习智能体，包括策略网络和价值网络。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from torch.optim import Adam
from torch.distributions import Categorical, Normal
import random
from collections import deque
import psutil
import gc
import time
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

class RLAgent:
    """强化学习智能体"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 网络参数 - 修复配置读取
        # 从environment配置中读取state_dim和action_space_size
        env_config = config.get('environment', {})
        self.state_dim = env_config.get('state_dim', config.get('state_dim', 512))
        self.action_dim = env_config.get('action_space_size', config.get('action_dim', 100))
        self.hidden_dim = config.get('hidden_dim', 256)
        
        self.logger.info(f"RL智能体配置: state_dim={self.state_dim}, action_dim={self.action_dim}")
        self.logger.debug(f"完整配置: {config}")
        
        # 处理多层隐藏维度配置 - 修复agent配置读取
        agent_config = config.get('agent', {})
        policy_net_config = agent_config.get('policy_network', {})
        value_net_config = agent_config.get('value_network', {})
        
        self.policy_hidden_dims = policy_net_config.get('hidden_dims', config.get('policy_hidden_dims', (256, 256)))
        self.value_hidden_dims = value_net_config.get('hidden_dims', config.get('value_hidden_dims', (256, 256)))
        
        self.logger.info(f"策略网络隐藏层维度: {self.policy_hidden_dims}")
        self.logger.info(f"价值网络隐藏层维度: {self.value_hidden_dims}")
        
        # 为了兼容性，使用第一个隐藏层维度作为主要hidden_dim
        if isinstance(self.policy_hidden_dims, (list, tuple)) and len(self.policy_hidden_dims) > 0:
            self.policy_hidden_dim = self.policy_hidden_dims[0]
        else:
            self.policy_hidden_dim = self.hidden_dim
            
        if isinstance(self.value_hidden_dims, (list, tuple)) and len(self.value_hidden_dims) > 0:
            self.value_hidden_dim = self.value_hidden_dims[0]
        else:
            self.value_hidden_dim = self.hidden_dim
        
        # 训练参数
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.batch_size = config.get('batch_size', 64)
        self.buffer_size = config.get('buffer_size', 100000)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化网络
        self.policy_net = PolicyNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.policy_hidden_dim
        ).to(self.device)
        
        self.value_net = ValueNetwork(
            state_dim=self.state_dim,
            hidden_dim=self.value_hidden_dim
        ).to(self.device)
        
        self.target_value_net = ValueNetwork(
            state_dim=self.state_dim,
            hidden_dim=self.value_hidden_dim
        ).to(self.device)
        
        # 复制参数到目标网络
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        
        # 优化器
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=self.learning_rate)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        # 训练统计
        self.training_step = 0
        
        # 内存管理和并行处理配置
        self.memory_threshold = config.get('memory_threshold', 0.85)  # 内存使用阈值
        self.gc_frequency = config.get('gc_frequency', 100)  # 垃圾回收频率
        self.operation_count = 0
        self.max_workers = config.get('max_workers', min(4, psutil.cpu_count()))
        self.enable_parallel = config.get('enable_parallel', True)
        
        # 内存监控
        self.memory_monitor_enabled = config.get('memory_monitor_enabled', True)
        self.last_memory_check = time.time()
        self.memory_check_interval = config.get('memory_check_interval', 30)  # 秒
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="RL_Thread"
        ) if self.enable_parallel else None
        
        # 启动内存监控
        if self.memory_monitor_enabled:
            self.memory_monitor_thread = threading.Thread(
                target=self._memory_monitor_loop,
                daemon=True
            )
            self.memory_monitor_thread.start()
        
        self.logger.info(f"RL智能体初始化完成，设备: {self.device}")
        self.logger.info(f"内存管理: 阈值={self.memory_threshold}, 并行处理: {self.enable_parallel}")
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """选择动作
        
        Args:
            state: 状态
            deterministic: 是否确定性选择
            
        Returns:
            动作和对数概率
        """
        self.policy_net.eval()
        with torch.no_grad():
            state = state.to(self.device)
            # 添加调试信息
            self.logger.debug(f"select_action - 输入状态形状: {state.shape}")
            self.logger.debug(f"select_action - 期望状态维度: {self.state_dim}")
            self.logger.debug(f"select_action - 策略网络状态维度: {self.policy_net.state_dim}")
            
            # 检查维度匹配
            if state.shape[-1] != self.state_dim:
                self.logger.error(f"状态维度不匹配！输入: {state.shape[-1]}, 期望: {self.state_dim}")
                raise ValueError(f"状态维度不匹配！输入: {state.shape[-1]}, 期望: {self.state_dim}")
            
            action, log_prob = self.policy_net.sample(state, deterministic)
        return action, log_prob
    
    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估动作
        
        Args:
            state: 状态
            action: 动作
            
        Returns:
            对数概率和熵
        """
        log_prob, entropy = self.policy_net.evaluate(state, action)
        return log_prob, entropy
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """获取状态价值
        
        Args:
            state: 状态
            
        Returns:
            状态价值
        """
        return self.value_net(state)
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, 
                        next_state: np.ndarray, done: bool):
        """存储转移
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update_policy(self, states: torch.Tensor, actions: torch.Tensor, 
                     advantages: torch.Tensor, old_log_probs: torch.Tensor) -> float:
        """更新策略网络
        
        Args:
            states: 状态批次
            actions: 动作批次
            advantages: 优势函数
            old_log_probs: 旧的对数概率
            
        Returns:
            策略损失
        """
        with self.memory_efficient_context():
            # 计算新的对数概率
            new_log_probs, entropy = self.evaluate_action(states, actions)
            
            # 计算重要性采样比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO裁剪
            clip_ratio = self.config.get('clip_ratio', 0.2)
            clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            
            # 策略损失
            policy_loss1 = ratio * advantages
            policy_loss2 = clipped_ratio * advantages
            policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
            
            # 熵正则化
            entropy_coef = self.config.get('entropy_coef', 0.01)
            entropy_loss = -entropy_coef * entropy.mean()
            
            total_loss = policy_loss + entropy_loss
            
            # 更新策略网络
            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()
            
            return total_loss.item()
    
    def update_value(self, states: torch.Tensor, returns: torch.Tensor) -> float:
        """更新价值网络
        
        Args:
            states: 状态批次
            returns: 回报
            
        Returns:
            价值损失
        """
        with self.memory_efficient_context():
            # 预测价值
            predicted_values = self.value_net(states)
            
            # 价值损失
            value_loss = F.mse_loss(predicted_values.squeeze(), returns)
            
            # 更新价值网络
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
            
            return value_loss.item()
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   next_values: List[float], dones: List[bool]) -> Tuple[List[float], List[float]]:
        """计算广义优势估计
        
        Args:
            rewards: 奖励序列
            values: 价值序列
            next_values: 下一状态价值序列
            dones: 结束标志序列
            
        Returns:
            优势和回报
        """
        gae_lambda = self.config.get('gae_lambda', 0.95)
        
        advantages = []
        returns = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else next_values[i]
            else:
                next_value = values[i + 1]
                
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * gae_lambda * gae * (1 - dones[i])
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            
        return advantages, returns
    
    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练一步
        
        Args:
            batch_data: 批次数据
            
        Returns:
            训练统计
        """
        with self.memory_efficient_context():
            self.policy_net.train()
            self.value_net.train()
            
            states = batch_data['states'].to(self.device)
            actions = batch_data['actions'].to(self.device)
            rewards = batch_data['rewards'].to(self.device)
            next_states = batch_data['next_states'].to(self.device)
            dones = batch_data['dones'].to(self.device)
            old_log_probs = batch_data['log_probs'].to(self.device)
            
            # 计算价值
            with torch.no_grad():
                values = self.value_net(states).squeeze()
                next_values = self.value_net(next_states).squeeze()
                
            # 计算优势和回报
            advantages, returns = self.compute_gae(
                rewards.cpu().numpy().tolist(),
                values.cpu().numpy().tolist(),
                next_values.cpu().numpy().tolist(),
                dones.cpu().numpy().tolist()
            )
            
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 更新网络
            policy_loss = self.update_policy(states, actions, advantages, old_log_probs)
            value_loss = self.update_value(states, returns)
            
            # 软更新目标网络
            self.soft_update_target_network()
            
            self.training_step += 1
            
            # 定期内存清理和监控
            if self.training_step % 50 == 0:
                memory_info = self._get_memory_usage()
                if memory_info.get('gpu_percent', 0) > 80:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            # 返回训练统计，包含内存信息
            result = {
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'training_step': self.training_step
            }
            
            # 添加内存信息
            if self.training_step % 100 == 0:
                memory_info = self._get_memory_usage()
                result.update({
                    'memory_gpu_gb': memory_info.get('gpu_allocated_gb', 0),
                    'memory_gpu_percent': memory_info.get('gpu_percent', 0),
                    'memory_system_percent': memory_info.get('system_percent', 0)
                })
            
            return result
    
    def soft_update_target_network(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_model(self, path: str):
        """保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'target_value_net_state_dict': self.target_value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step
        }, path)
        self.logger.info(f"RL模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.target_value_net.load_state_dict(checkpoint['target_value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        self.logger.info(f"RL模型已从 {path} 加载")
    
    def _memory_monitor_loop(self):
        """后台内存监控循环"""
        while True:
            try:
                time.sleep(self.memory_check_interval)
                self._check_memory_usage()
            except Exception as e:
                self.logger.warning(f"内存监控异常: {e}")
    
    def _check_memory_usage(self):
        """检查内存使用情况"""
        try:
            # 系统内存
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            
            # GPU内存
            gpu_memory_percent = 0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
                gpu_memory_percent = gpu_memory
            
            # 检查是否超过阈值
            if memory_percent > self.memory_threshold or gpu_memory_percent > self.memory_threshold:
                self.logger.warning(f"内存使用率过高: 系统={memory_percent:.2%}, GPU={gpu_memory_percent:.2%}")
                self._force_memory_cleanup()
                
        except Exception as e:
            self.logger.warning(f"内存检查失败: {e}")
    
    def _force_memory_cleanup(self):
        """强制内存清理"""
        try:
            # Python垃圾回收
            collected = gc.collect()
            
            # CUDA缓存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.logger.info(f"内存清理完成，回收对象数: {collected}")
            
        except Exception as e:
            self.logger.warning(f"内存清理失败: {e}")
    
    @contextmanager
    def memory_efficient_context(self):
        """内存高效上下文管理器"""
        try:
            yield
        finally:
            self.operation_count += 1
            if self.operation_count % self.gc_frequency == 0:
                self._force_memory_cleanup()
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况"""
        memory_info = {}
        
        try:
            # 系统内存
            memory = psutil.virtual_memory()
            memory_info['system_used_gb'] = memory.used / (1024**3)
            memory_info['system_percent'] = memory.percent
            
            # GPU内存
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                memory_info['gpu_allocated_gb'] = gpu_memory_allocated
                memory_info['gpu_reserved_gb'] = gpu_memory_reserved
                
                # GPU使用百分比
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                memory_info['gpu_percent'] = (gpu_memory_allocated / total_gpu_memory) * 100
            else:
                memory_info['gpu_allocated_gb'] = 0
                memory_info['gpu_reserved_gb'] = 0
                memory_info['gpu_percent'] = 0
                
        except Exception as e:
            self.logger.warning(f"获取内存信息失败: {e}")
            
        return memory_info
    
    def parallel_train_batch(self, batch_list: List[Dict[str, torch.Tensor]]) -> List[Dict[str, float]]:
        """并行训练多个批次
        
        Args:
            batch_list: 批次数据列表
            
        Returns:
            训练统计列表
        """
        if not self.enable_parallel or len(batch_list) <= 1:
            # 串行处理
            return [self.train_step(batch) for batch in batch_list]
        
        # 并行处理
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {executor.submit(self.train_step, batch): batch 
                             for batch in batch_list}
            
            for future in as_completed(future_to_batch):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"并行训练批次失败: {e}")
                    # 降级到串行处理
                    batch = future_to_batch[future]
                    result = self.train_step(batch)
                    results.append(result)
        
        return results
    
    def batch_update_networks(self, states_list: List[torch.Tensor], 
                            actions_list: List[torch.Tensor],
                            advantages_list: List[torch.Tensor],
                            old_log_probs_list: List[torch.Tensor],
                            returns_list: List[torch.Tensor]) -> Dict[str, float]:
        """批量更新网络
        
        Args:
            states_list: 状态列表
            actions_list: 动作列表
            advantages_list: 优势列表
            old_log_probs_list: 旧对数概率列表
            returns_list: 回报列表
            
        Returns:
            平均训练统计
        """
        with self.memory_efficient_context():
            policy_losses = []
            value_losses = []
            
            if self.enable_parallel and len(states_list) > 1:
                # 并行处理策略更新
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    policy_futures = [executor.submit(self.update_policy, states, actions, advantages, old_log_probs)
                                    for states, actions, advantages, old_log_probs in 
                                    zip(states_list, actions_list, advantages_list, old_log_probs_list)]
                    
                    value_futures = [executor.submit(self.update_value, states, returns)
                                   for states, returns in zip(states_list, returns_list)]
                    
                    # 收集结果
                    for future in as_completed(policy_futures):
                        try:
                            policy_losses.append(future.result())
                        except Exception as e:
                            logging.error(f"并行策略更新失败: {e}")
                    
                    for future in as_completed(value_futures):
                        try:
                            value_losses.append(future.result())
                        except Exception as e:
                            logging.error(f"并行价值更新失败: {e}")
            else:
                # 串行处理
                for states, actions, advantages, old_log_probs, returns in zip(
                    states_list, actions_list, advantages_list, old_log_probs_list, returns_list):
                    policy_loss = self.update_policy(states, actions, advantages, old_log_probs)
                    value_loss = self.update_value(states, returns)
                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)
            
            return {
                'avg_policy_loss': sum(policy_losses) / len(policy_losses) if policy_losses else 0.0,
                'avg_value_loss': sum(value_losses) / len(value_losses) if value_losses else 0.0,
                'num_batches': len(states_list)
            }
    
    def cleanup_resources(self):
        """清理资源"""
        try:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
                
            self._force_memory_cleanup()
            self.logger.info("RL智能体资源清理完成")
            
        except Exception as e:
            self.logger.warning(f"资源清理失败: {e}")
    
    def __del__(self):
        """析构函数"""
        try:
            self.cleanup_resources()
        except:
            pass


class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 共享层 - 确保维度匹配
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值和标准差头 - 确保输出维度正确
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # 初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=0.01)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            state: 状态
            
        Returns:
            均值和标准差
        """
        # 添加调试信息
        print(f"PolicyNetwork.forward - 输入状态形状: {state.shape}")
        print(f"PolicyNetwork.forward - 网络期望状态维度: {self.state_dim}")
        print(f"PolicyNetwork.forward - 网络动作维度: {self.action_dim}")
        
        # 检查输入维度
        if state.shape[-1] != self.state_dim:
            print(f"错误：状态维度不匹配！输入: {state.shape[-1]}, 期望: {self.state_dim}")
            raise ValueError(f"状态维度不匹配！输入: {state.shape[-1]}, 期望: {self.state_dim}")
        
        features = self.shared_layers(state)
        print(f"PolicyNetwork.forward - 共享层输出形状: {features.shape}")
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        print(f"PolicyNetwork.forward - 均值形状: {mean.shape}, 标准差形状: {std.shape}")
        
        return mean, std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作
        
        Args:
            state: 状态
            deterministic: 是否确定性
            
        Returns:
            动作和对数概率
        """
        mean, std = self.forward(state)
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            normal = Normal(mean, std)
            action = normal.sample()
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
            
        return action, log_prob
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估动作
        
        Args:
            state: 状态
            action: 动作
            
        Returns:
            对数概率和熵
        """
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            state: 状态
            
        Returns:
            状态价值
        """
        return self.network(state)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        """添加经验
        
        Args:
            state: 状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            批次数据
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return {
            'states': torch.FloatTensor(state),
            'actions': torch.FloatTensor(action),
            'rewards': torch.FloatTensor(reward),
            'next_states': torch.FloatTensor(next_state),
            'dones': torch.BoolTensor(done)
        }
    
    def __len__(self) -> int:
        return len(self.buffer)