"""扩散模型模块

实现用于分子生成的扩散模型，包括U-Net架构和扩散过程。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union
import logging
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
import gc
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import psutil
import time
import threading
from contextlib import contextmanager

class DiffusionModel(nn.Module):
    """扩散模型主类"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 模型参数
        self.input_dim = config.get('input_dim', 512)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.output_dim = config.get('output_dim', 128)
        self.num_timesteps = config.get('num_timesteps', 1000)
        
        # 性能优化参数
        self.use_mixed_precision = config.get('use_mixed_precision', True)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.max_batch_size = config.get('max_batch_size', 32)
        self.enable_memory_efficient_attention = config.get('enable_memory_efficient_attention', True)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = UNetModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_timesteps=self.num_timesteps,
            enable_memory_efficient_attention=self.enable_memory_efficient_attention
        ).to(self.device)
        
        # 噪声调度
        self.beta_schedule = self._create_beta_schedule()
        self.alpha = 1.0 - self.beta_schedule
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        
        # 将调度参数移到GPU
        self.beta_schedule = self.beta_schedule.to(self.device)
        self.alpha = self.alpha.to(self.device)
        self.alpha_cumprod = self.alpha_cumprod.to(self.device)
        self.alpha_cumprod_prev = self.alpha_cumprod_prev.to(self.device)
        
        # 优化器（使用AdamW以获得更好的泛化性能）
        optimizer_class = AdamW if config.get('use_adamw', True) else Adam
        self.optimizer = optimizer_class(
            self.model.parameters(), 
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-2)
        )
        
        # 学习率调度器
        if config.get('use_onecycle', False):
            self.scheduler = OneCycleLR(
                self.optimizer, 
                max_lr=config.get('learning_rate', 1e-4),
                total_steps=config.get('total_steps', 10000)
            )
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.get('max_epochs', 100))
        
        # 混合精度训练
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # 线程池用于并行处理
        self.thread_pool = ThreadPoolExecutor(max_workers=min(4, mp.cpu_count()))
        
        # 内存监控和管理
        self.memory_threshold = config.get('memory_threshold', 0.85)  # 内存使用阈值
        self.gc_frequency = config.get('gc_frequency', 50)  # 垃圾回收频率
        self.operation_count = 0
        self.memory_monitor_enabled = config.get('memory_monitor_enabled', True)
        self.last_memory_check = time.time()
        self.memory_check_interval = config.get('memory_check_interval', 10)  # 秒
        
        # 内存监控线程
        if self.memory_monitor_enabled:
            self.memory_monitor_thread = threading.Thread(
                target=self._memory_monitor_loop, daemon=True
            )
            self.memory_monitor_thread.start()
        
        self.logger.info(f"扩散模型初始化完成，设备: {self.device}, 混合精度: {self.use_mixed_precision}")
        self.logger.info(f"内存监控: {self.memory_monitor_enabled}, 阈值: {self.memory_threshold}")
    
    def _create_beta_schedule(self, schedule_type: str = 'linear') -> torch.Tensor:
        """创建噪声调度
        
        Args:
            schedule_type: 调度类型
            
        Returns:
            beta值序列
        """
        if schedule_type == 'linear':
            beta_start = self.config.get('beta_start', 1e-4)
            beta_end = self.config.get('beta_end', 2e-2)
            return torch.linspace(beta_start, beta_end, self.num_timesteps)
        elif schedule_type == 'cosine':
            return self._cosine_beta_schedule()
        else:
            raise ValueError(f"不支持的调度类型: {schedule_type}")
    
    def _cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        """余弦噪声调度
        
        Args:
            s: 偏移参数
            
        Returns:
            beta值序列
        """
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向扩散过程
        
        Args:
            x_start: 原始数据
            t: 时间步
            noise: 噪声（可选）
            
        Returns:
            加噪后的数据
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self._extract(torch.sqrt(self.alpha_cumprod), t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            torch.sqrt(1.0 - self.alpha_cumprod), t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算损失
        
        Args:
            x_start: 原始数据
            t: 时间步
            condition: 条件信息
            
        Returns:
            损失值
        """
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        
        # 预测噪声
        predicted_noise = self.model(x_noisy, t, condition)
        
        # 计算MSE损失
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """反向采样一步
        
        Args:
            x: 当前状态
            t: 时间步
            condition: 条件信息
            
        Returns:
            去噪后的数据
        """
        betas_t = self._extract(self.beta_schedule, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            torch.sqrt(1.0 - self.alpha_cumprod), t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alpha), t, x.shape)
        
        # 预测噪声
        predicted_noise = self.model(x, t, condition)
        
        # 计算均值
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self._posterior_variance(), t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def p_sample_loop(self, shape: Tuple, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """完整的反向采样过程
        
        Args:
            shape: 输出形状
            condition: 条件信息
            
        Returns:
            生成的数据
        """
        device = next(self.model.parameters()).device
        b = shape[0]
        
        # 从纯噪声开始
        img = torch.randn(shape, device=device)
        
        # 内存优化：处理大批次时分块处理
        if b > self.max_batch_size:
            chunks = torch.split(img, self.max_batch_size, dim=0)
            if condition is not None:
                condition_chunks = torch.split(condition, self.max_batch_size, dim=0)
            else:
                condition_chunks = [None] * len(chunks)
            
            results = []
            for chunk, cond_chunk in zip(chunks, condition_chunks):
                chunk_result = self._p_sample_loop_chunk(chunk, cond_chunk)
                results.append(chunk_result)
                
                # 定期清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            return torch.cat(results, dim=0)
        
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, condition)
            
            # 定期内存清理
            if i % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return img
    
    def _p_sample_loop_chunk(self, img: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        对单个数据块进行采样循环
        
        Args:
            img: 输入噪声张量
            condition: 条件信息
            
        Returns:
            生成的数据块
        """
        device = img.device
        b = img.shape[0]
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, condition)
            
        return img
    
    def sample(self, batch_size: int, condition: Optional[torch.Tensor] = None, 
               use_ddim: bool = True, ddim_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        """
        生成样本（支持DDIM快速采样）
        
        Args:
            batch_size: 批次大小
            condition: 条件信息
            use_ddim: 是否使用DDIM采样
            ddim_steps: DDIM采样步数
            eta: DDIM随机性参数
            
        Returns:
            生成的样本
        """
        with self.memory_efficient_context():
            shape = (batch_size, self.output_dim)
            
            if use_ddim:
                return self.ddim_sample_loop(shape, condition, ddim_steps, eta)
            else:
                return self.p_sample_loop(shape, condition)
    
    def ddim_sample_loop(self, shape: Tuple, condition: Optional[torch.Tensor] = None,
                        ddim_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        """
        DDIM采样循环（更快的采样方法）
        """
        device = self.device
        batch_size = shape[0]
        
        # 创建DDIM时间步序列
        ddim_timesteps = self._make_ddim_timesteps(ddim_steps)
        ddim_timesteps_prev = torch.cat([torch.tensor([0]), ddim_timesteps[:-1]])
        
        # 初始化噪声
        img = torch.randn(shape, device=device)
        
        # 批处理优化
        if batch_size > self.max_batch_size:
            return self._ddim_sample_chunked(shape, condition, ddim_steps, eta)
        
        for i, (t, t_prev) in enumerate(zip(ddim_timesteps.flip(0), ddim_timesteps_prev.flip(0))):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            t_prev_batch = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
            
            with torch.no_grad():
                if self.use_mixed_precision:
                    with autocast():
                        img = self.ddim_step(img, t_batch, t_prev_batch, condition, eta)
                else:
                    img = self.ddim_step(img, t_batch, t_prev_batch, condition, eta)
            
            # 定期清理内存
            if i % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return img
    
    def ddim_step(self, x: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor,
                  condition: Optional[torch.Tensor] = None, eta: float = 0.0) -> torch.Tensor:
        """
        DDIM单步采样
        """
        # 预测噪声
        noise_pred = self.model(x, t, condition)
        
        # 获取alpha值
        alpha_t = self._extract(self.alpha_cumprod, t, x.shape)
        alpha_t_prev = self._extract(self.alpha_cumprod, t_prev, x.shape)
        
        # 计算预测的原始图像
        pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        
        # 计算方向向量
        dir_xt = torch.sqrt(1 - alpha_t_prev - eta**2 * (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) * noise_pred
        
        # 添加随机性
        if eta > 0:
            noise = torch.randn_like(x)
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
            dir_xt += sigma_t * noise
        
        # 计算下一步
        x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
        
        return x_prev
    
    def _make_ddim_timesteps(self, ddim_steps: int) -> torch.Tensor:
        """
        创建DDIM时间步序列
        """
        c = self.num_timesteps // ddim_steps
        ddim_timesteps = torch.arange(0, ddim_steps) * c + 1
        return ddim_timesteps.to(self.device)
    
    def _ddim_sample_chunked(self, shape: Tuple, condition: Optional[torch.Tensor] = None,
                            ddim_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        """
        分块DDIM采样，处理大批次
        """
        batch_size = shape[0]
        results = []
        
        for i in range(0, batch_size, self.max_batch_size):
            end_idx = min(i + self.max_batch_size, batch_size)
            chunk_size = end_idx - i
            chunk_shape = (chunk_size,) + shape[1:]
            
            chunk_condition = None
            if condition is not None:
                chunk_condition = condition[i:end_idx]
            
            chunk_result = self.ddim_sample_loop(chunk_shape, chunk_condition, ddim_steps, eta)
            results.append(chunk_result)
        
        return torch.cat(results, dim=0)
    
    def train_step(self, x: torch.Tensor, step: int = 0) -> Dict[str, float]:
        """
        执行一步训练（支持混合精度和梯度累积）
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            step: 当前训练步数
            
        Returns:
            包含损失信息的字典
        """
        with self.memory_efficient_context():
            self.model.train()
            
            # 动态批处理大小调整
            batch_size = x.shape[0]
            if batch_size > self.max_batch_size:
                # 分批处理大批次数据
                return self._train_step_chunked(x, step)
            
            # 随机采样时间步
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
            
            # 混合精度训练
            if self.use_mixed_precision and self.scaler is not None:
                with autocast():
                    loss = self.p_losses(x, t)
                    loss = loss / self.gradient_accumulation_steps
                
                # 梯度缩放和累积
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # 智能内存管理
                    if step % 50 == 0:  # 减少清理频率
                        memory_info = self._get_memory_usage()
                        if memory_info.get('gpu_memory_percent', 0) > 0.8:
                            torch.cuda.empty_cache()
                            gc.collect()
            else:
                # 标准训练
                loss = self.p_losses(x, t)
                loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # 获取内存使用信息
            memory_info = self._get_memory_usage()
            
            return {
                'loss': loss.item() * self.gradient_accumulation_steps,
                'lr': self.optimizer.param_groups[0]['lr'],
                'gpu_memory': memory_info.get('gpu_memory_used_gb', 0),
                'gpu_memory_percent': memory_info.get('gpu_memory_percent', 0),
                'system_memory_percent': memory_info.get('system_memory_percent', 0)
            }
    
    def _train_step_chunked(self, x: torch.Tensor, step: int) -> Dict[str, float]:
        """
        分块处理大批次数据
        """
        total_loss = 0.0
        num_chunks = (x.shape[0] + self.max_batch_size - 1) // self.max_batch_size
        
        for i in range(num_chunks):
            start_idx = i * self.max_batch_size
            end_idx = min((i + 1) * self.max_batch_size, x.shape[0])
            chunk = x[start_idx:end_idx]
            
            result = self.train_step(chunk, step * num_chunks + i)
            total_loss += result['loss'] * chunk.shape[0]
        
        return {
            'loss': total_loss / x.shape[0],
            'lr': self.optimizer.param_groups[0]['lr'],
            'gpu_memory': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        }
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """提取指定时间步的值
        
        Args:
            a: 源张量
            t: 时间步
            x_shape: 目标形状
            
        Returns:
            提取的值
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def _posterior_variance(self) -> torch.Tensor:
        """计算后验方差
        
        Returns:
            后验方差
        """
        posterior_variance = (
            self.beta_schedule * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        return posterior_variance
    
    def save_model(self, path: str):
        """保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        self.logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"模型已从 {path} 加载")
    
    def _memory_monitor_loop(self):
        """内存监控循环（在后台线程中运行）"""
        while True:
            try:
                time.sleep(self.memory_check_interval)
                self._check_memory_usage()
            except Exception as e:
                self.logger.warning(f"内存监控线程异常: {e}")
                time.sleep(5)  # 异常后等待5秒再继续
    
    def _check_memory_usage(self) -> Dict[str, float]:
        """检查内存使用情况
        
        Returns:
            内存使用信息字典
        """
        try:
            # 系统内存
            system_memory = psutil.virtual_memory()
            memory_percent = system_memory.percent / 100.0
            
            # GPU内存（如果可用）
            gpu_memory_percent = 0.0
            gpu_memory_used = 0.0
            gpu_memory_total = 0.0
            
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                gpu_memory_percent = gpu_memory_used / gpu_memory_total if gpu_memory_total > 0 else 0.0
            
            memory_info = {
                'system_memory_percent': memory_percent,
                'gpu_memory_percent': gpu_memory_percent,
                'gpu_memory_used_gb': gpu_memory_used,
                'gpu_memory_total_gb': gpu_memory_total
            }
            
            # 如果内存使用超过阈值，触发清理
            if memory_percent > self.memory_threshold or gpu_memory_percent > self.memory_threshold:
                self.logger.warning(
                    f"内存使用过高 - 系统: {memory_percent:.1%}, GPU: {gpu_memory_percent:.1%}"
                )
                self._force_memory_cleanup()
            
            return memory_info
            
        except Exception as e:
            self.logger.error(f"内存检查失败: {e}")
            return {}
    
    def _force_memory_cleanup(self):
        """强制内存清理"""
        try:
            # 清理Python垃圾
            collected = gc.collect()
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.logger.info(f"强制内存清理完成，回收对象数: {collected}")
            
        except Exception as e:
            self.logger.error(f"内存清理失败: {e}")
    
    @contextmanager
    def memory_efficient_context(self):
        """内存高效的上下文管理器"""
        initial_memory = self._get_memory_usage()
        try:
            yield
        finally:
            # 操作完成后清理内存
            self.operation_count += 1
            if self.operation_count % self.gc_frequency == 0:
                self._force_memory_cleanup()
                
            final_memory = self._get_memory_usage()
            if final_memory.get('gpu_memory_percent', 0) > 0.9:  # GPU内存使用超过90%
                self.logger.warning("GPU内存使用过高，执行紧急清理")
                self._force_memory_cleanup()
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况"""
        try:
            memory_info = {}
            
            # 系统内存
            system_memory = psutil.virtual_memory()
            memory_info['system_memory_percent'] = system_memory.percent / 100.0
            
            # GPU内存
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_info['gpu_memory_percent'] = gpu_memory_used / gpu_memory_total if gpu_memory_total > 0 else 0.0
                memory_info['gpu_memory_used_gb'] = gpu_memory_used
            
            return memory_info
        except Exception:
            return {}
    
    def cleanup_resources(self):
        """清理所有资源"""
        try:
            # 关闭线程池
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 垃圾回收
            gc.collect()
            
            self.logger.info("资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")
    
    def __del__(self):
        """析构函数"""
        try:
            self.cleanup_resources()
        except Exception:
            pass  # 析构函数中不应抛出异常


class UNetModel(nn.Module):
    """优化的U-Net模型用于扩散过程"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_timesteps: int,
                 enable_memory_efficient_attention: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_timesteps = num_timesteps
        self.enable_memory_efficient_attention = enable_memory_efficient_attention
        
        # 时间嵌入
        self.time_embed = TimeEmbedding(hidden_dim)
        
        # 条件嵌入
        self.condition_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ) if input_dim > 0 else None
        
        # 编码器（使用更深的网络）
        self.encoder = nn.ModuleList([
            ResidualBlock(input_dim, hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim * 2, hidden_dim),
            ResidualBlock(hidden_dim * 2, hidden_dim * 4, hidden_dim),
            ResidualBlock(hidden_dim * 4, hidden_dim * 8, hidden_dim),
        ])
        
        # 中间层（使用注意力机制）
        self.middle = nn.ModuleList([
            ResidualBlock(hidden_dim * 8, hidden_dim * 8, hidden_dim),
            ResidualBlock(hidden_dim * 8, hidden_dim * 8, hidden_dim)
        ])
        
        # 解码器
        self.decoder = nn.ModuleList([
            ResidualBlock(hidden_dim * 16, hidden_dim * 4, hidden_dim),  # 输入: middle(8) + skip(8) = 16
            ResidualBlock(hidden_dim * 8, hidden_dim * 2, hidden_dim),   # 输入: decoder(4) + skip(4) = 8
            ResidualBlock(hidden_dim * 4, hidden_dim, hidden_dim),       # 输入: decoder(2) + skip(2) = 4
            ResidualBlock(hidden_dim * 2, hidden_dim, hidden_dim),       # 输入: decoder(1) + skip(1) = 2
        ])
        
        # 输出层（使用组归一化）
        def get_num_groups(dim):
            if dim < 4:
                return 1
            # 找到最大的能整除dim的组数，不超过32
            for num_groups in range(min(32, dim), 0, -1):
                if dim % num_groups == 0:
                    return num_groups
            return 1
        
        self.output_norm = nn.GroupNorm(get_num_groups(hidden_dim), hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征
            t: 时间步
            condition: 条件信息
            
        Returns:
            预测的噪声
        """
        # 时间嵌入
        t_emb = self.time_embed(t)
        
        # 条件嵌入
        if condition is not None and self.condition_embed is not None:
            c_emb = self.condition_embed(condition)
            t_emb = t_emb + c_emb
        
        # 编码器
        skip_connections = []
        h = x
        
        for encoder_block in self.encoder:
            h = encoder_block(h, t_emb)
            skip_connections.append(h)
        
        # 中间层
        for middle_block in self.middle:
            h = middle_block(h, t_emb)
        
        # 解码器
        for decoder_block, skip in zip(self.decoder, reversed(skip_connections)):
            h = torch.cat([h, skip], dim=-1)
            h = decoder_block(h, t_emb)
        
        # 输出
        h = self.output_norm(h)
        h = F.silu(h)
        output = self.output_layer(h)
        return output


class TimeEmbedding(nn.Module):
    """时间嵌入层"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            t: 时间步
            
        Returns:
            时间嵌入
        """
        # 位置编码
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # 线性变换
        emb = self.linear1(emb)
        emb = F.silu(emb)
        emb = self.linear2(emb)
        
        return emb


class ResidualBlock(nn.Module):
    """优化的残差块，支持注意力机制"""
    
    def __init__(self, in_dim: int, out_dim: int, time_emb_dim: int = None, 
                 use_attention: bool = False, dropout: float = 0.1):
        super().__init__()
        
        # 如果没有指定时间嵌入维度，默认使用out_dim
        if time_emb_dim is None:
            time_emb_dim = out_dim
            
        self.use_attention = use_attention
        
        # 使用GroupNorm替代LayerNorm以提高性能，确保num_groups能整除num_channels
        def get_num_groups(dim):
            if dim < 4:
                return 1
            # 找到最大的能整除dim的组数，不超过32
            for num_groups in range(min(32, dim), 0, -1):
                if dim % num_groups == 0:
                    return num_groups
            return 1
        
        self.norm1 = nn.GroupNorm(get_num_groups(in_dim), in_dim) if in_dim >= 4 else nn.LayerNorm(in_dim)
        self.norm2 = nn.GroupNorm(get_num_groups(out_dim), out_dim) if out_dim >= 4 else nn.LayerNorm(out_dim)
        
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 时间嵌入投影
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_dim)
        )
        
        # 注意力机制（可选）
        if use_attention:
            self.attention = MultiHeadAttention(out_dim, num_heads=8, dropout=dropout)
            self.attention_norm = nn.GroupNorm(get_num_groups(out_dim), out_dim) if out_dim >= 4 else nn.LayerNorm(out_dim)
        
        # 残差连接
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = nn.Identity()
            
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征
            t_emb: 时间嵌入
            
        Returns:
            输出特征
        """
        residual = self.residual_proj(x)
        
        # 第一层
        h = self.norm1(x)
        h = F.silu(h)
        h = self.linear1(h)
        
        # 添加时间嵌入
        h = h + self.time_proj(t_emb).unsqueeze(1) if len(h.shape) == 3 else h + self.time_proj(t_emb)
        
        # 第二层
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        
        # 注意力机制（如果启用）
        if self.use_attention:
            h_norm = self.attention_norm(h)
            h = h + self.attention(h_norm)
        
        # 残差连接
        return h + residual


class MultiHeadAttention(nn.Module):
    """内存高效的多头注意力机制"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # 计算Q, K, V
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 使用Flash Attention风格的优化（如果可用）
        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 的优化注意力
            attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            # 标准注意力计算
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # 重塑输出
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        return self.out_proj(attn_output)