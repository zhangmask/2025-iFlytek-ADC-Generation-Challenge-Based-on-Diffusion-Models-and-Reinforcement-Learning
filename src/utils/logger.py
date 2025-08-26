#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的日志记录模块

提供统一的日志记录功能，支持多种日志级别、输出格式、性能监控和调试信息。

Author: AI Developer
Date: 2025
"""

import logging
import os
import sys
import json
import threading
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import functools
import time
from contextlib import contextmanager
from collections import defaultdict, deque
import torch


class PerformanceMetrics:
    """性能指标收集器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(int)
        self.lock = threading.Lock()
    
    def record_timing(self, operation: str, duration: float):
        """记录操作耗时"""
        with self.lock:
            self.metrics[f"{operation}_duration"].append(duration)
            self.counters[f"{operation}_count"] += 1
    
    def record_memory(self, operation: str, memory_mb: float):
        """记录内存使用"""
        with self.lock:
            self.metrics[f"{operation}_memory"].append(memory_mb)
    
    def get_stats(self, operation: str) -> Dict[str, Any]:
        """获取操作统计信息"""
        with self.lock:
            duration_key = f"{operation}_duration"
            memory_key = f"{operation}_memory"
            count_key = f"{operation}_count"
            
            stats = {"operation": operation, "count": self.counters[count_key]}
            
            if duration_key in self.metrics and self.metrics[duration_key]:
                durations = list(self.metrics[duration_key])
                stats.update({
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "total_duration": sum(durations)
                })
            
            if memory_key in self.metrics and self.metrics[memory_key]:
                memories = list(self.metrics[memory_key])
                stats.update({
                    "avg_memory_mb": sum(memories) / len(memories),
                    "max_memory_mb": max(memories)
                })
            
            return stats


def setup_logger(config: Dict[str, Any], name: Optional[str] = None, enable_json: bool = False) -> logging.Logger:
    """
    设置增强的日志记录器
    
    Args:
        config: 日志配置字典
        name: 日志记录器名称，如果为None则使用根记录器
        enable_json: 是否启用JSON格式日志
        
    Returns:
        配置好的日志记录器
    """
    # 获取配置参数
    level = config.get('level', 'INFO')
    format_str = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
    log_file = config.get('file', None)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 创建格式化器
    if enable_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(format_str)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 添加文件处理器（如果指定了日志文件）
    if log_file:
        # 确保日志目录存在
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 添加时间戳到文件名
        log_path = Path(log_file)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_with_timestamp = log_path.parent / f"{log_path.stem}_{timestamp}{log_path.suffix}"
        
        file_handler = logging.FileHandler(log_file_with_timestamp, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class JsonFormatter(logging.Formatter):
    """JSON格式化器"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 添加自定义字段
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


class ADCLogger:
    """
    ADC项目专用增强日志记录器类
    
    提供更高级的日志功能，包括模块化日志、性能监控、内存监控等。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化ADC日志记录器
        
        Args:
            config: 日志配置字典
        """
        self.config = config
        self.main_logger = setup_logger(config, 'adc_main')
        self.module_loggers = {}
        self.performance_metrics = PerformanceMetrics()
        self.debug_mode = config.get('debug', False)
    
    def get_module_logger(self, module_name: str) -> logging.Logger:
        """
        获取模块专用日志记录器
        
        Args:
            module_name: 模块名称
            
        Returns:
            模块日志记录器
        """
        if module_name not in self.module_loggers:
            logger_name = f"adc_{module_name}"
            self.module_loggers[module_name] = setup_logger(self.config, logger_name)
        
        return self.module_loggers[module_name]
    
    def log_data_info(self, data_info: Dict[str, Any]) -> None:
        """
        记录数据信息
        
        Args:
            data_info: 数据信息字典
        """
        logger = self.get_module_logger('data')
        logger.info("=== 数据信息 ===")
        for key, value in data_info.items():
            logger.info(f"{key}: {value}")
    
    def log_model_info(self, model_name: str, model_info: Dict[str, Any]) -> None:
        """
        记录模型信息
        
        Args:
            model_name: 模型名称
            model_info: 模型信息字典
        """
        logger = self.get_module_logger('models')
        logger.info(f"=== {model_name} 模型信息 ===")
        for key, value in model_info.items():
            logger.info(f"{key}: {value}")
    
    def log_training_progress(self, epoch: int, loss: float, metrics: Dict[str, float], 
                            memory_info: Dict[str, Any] = None) -> None:
        """
        记录训练进度
        
        Args:
            epoch: 当前轮次
            loss: 损失值
            metrics: 评估指标字典
            memory_info: 内存信息
        """
        logger = self.get_module_logger('training')
        log_data = {
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if memory_info:
            log_data['memory'] = memory_info
        
        logger.info(f"Training Progress: {json.dumps(log_data, ensure_ascii=False)}")
    
    def log_generation_results(self, num_generated: int, generation_time: float) -> None:
        """
        记录分子生成结果
        
        Args:
            num_generated: 生成的分子数量
            generation_time: 生成耗时
        """
        logger = self.get_module_logger('generation')
        logger.info(f"生成完成 - 分子数量: {num_generated}, 耗时: {generation_time:.2f}秒")
    
    def log_evaluation_results(self, results: Dict[str, Any], detailed: bool = False) -> None:
        """
        记录评估结果
        
        Args:
            results: 评估结果字典
            detailed: 是否记录详细信息
        """
        logger = self.get_module_logger('evaluation')
        
        if detailed and self.debug_mode:
            logger.info(f"Detailed Evaluation Results: {json.dumps(results, ensure_ascii=False, indent=2)}")
        else:
            # 简化输出
            summary = {k: v for k, v in results.items() if not isinstance(v, (list, dict)) or k in ['summary', 'metrics']}
            logger.info(f"Evaluation Results: {json.dumps(summary, ensure_ascii=False)}")
    
    def log_error(self, module_name: str, error_msg: str, exception: Optional[Exception] = None, 
                 severity: str = 'ERROR') -> None:
        """
        记录错误信息
        
        Args:
            module_name: 模块名称
            error_msg: 错误消息
            exception: 异常对象
            severity: 错误严重程度
        """
        logger = self.get_module_logger(module_name)
        
        error_data = {
            'error_type': type(exception).__name__ if exception else 'Unknown',
            'error_message': error_msg,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.error(f"Error Details: {json.dumps(error_data, ensure_ascii=False)}")
        
        if exception:
            logger.error(f"{error_msg}: {str(exception)}", exc_info=True)
        else:
            logger.error(error_msg)
    
    def log_memory_usage(self, operation: str, memory_info: Dict[str, Any]) -> None:
        """
        记录内存使用情况
        
        Args:
            operation: 操作名称
            memory_info: 内存信息字典
        """
        logger = self.get_module_logger('memory')
        
        memory_data = {
            'operation': operation,
            'timestamp': datetime.now().isoformat(),
            **memory_info
        }
        
        logger.info(f"Memory Usage: {json.dumps(memory_data, ensure_ascii=False)}")
        
        # 记录到性能指标
        if 'system_memory_mb' in memory_info:
            self.performance_metrics.record_memory(operation, memory_info['system_memory_mb'])
    
    def log_debug(self, message: str, data: Dict[str, Any] = None) -> None:
        """
        记录调试信息
        
        Args:
            message: 调试消息
            data: 调试数据
        """
        if self.debug_mode:
            logger = self.get_module_logger('debug')
            debug_data = {
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'data': data or {}
            }
            logger.debug(f"Debug: {json.dumps(debug_data, ensure_ascii=False)}")
    
    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        """
        记录性能信息
        
        Args:
            operation: 操作名称
            duration: 持续时间（秒）
            **kwargs: 其他性能指标
        """
        logger = self.get_module_logger('performance')
        
        log_data = {
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        logger.info(f"Performance: {json.dumps(log_data, ensure_ascii=False)}")
        
        # 记录到性能指标
        self.performance_metrics.record_timing(operation, duration)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息
        
        Returns:
            系统信息字典
        """
        info = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
            'timestamp': datetime.now().isoformat()
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,
                'gpu_count': torch.cuda.device_count()
            })
        
        return info


def create_adc_logger(config: Dict[str, Any]) -> ADCLogger:
    """
    创建ADC项目日志记录器
    
    Args:
        config: 日志配置字典
        
    Returns:
        ADC日志记录器实例
    """
    return ADCLogger(config)


# 增强的性能监控装饰器
def log_performance(operation_name: str = None, log_memory: bool = False, 
                   log_system_info: bool = False):
    """
    增强的性能监控装饰器
    
    Args:
        operation_name: 操作名称，默认使用函数名
        log_memory: 是否记录内存使用
        log_system_info: 是否记录系统信息
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            start_time = time.time()
            
            # 获取开始时的内存信息
            start_memory = None
            if log_memory:
                start_memory = psutil.virtual_memory().used / 1024**2  # MB
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 获取日志记录器
                logger = logging.getLogger('adc.performance')
                
                log_data = {
                    'operation': name,
                    'duration': duration,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                }
                
                # 记录内存使用
                if log_memory and start_memory is not None:
                    end_memory = psutil.virtual_memory().used / 1024**2
                    log_data['memory_delta_mb'] = end_memory - start_memory
                    log_data['peak_memory_mb'] = end_memory
                
                # 记录系统信息
                if log_system_info:
                    adc_logger = ADCLogger({})
                    log_data['system_info'] = adc_logger.get_system_info()
                
                logger.info(f"Performance: {json.dumps(log_data, ensure_ascii=False)}")
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger = logging.getLogger('adc.performance')
                
                error_data = {
                    'operation': name,
                    'duration': duration,
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.error(f"Performance Error: {json.dumps(error_data, ensure_ascii=False)}")
                raise
        
        return wrapper
    return decorator


@contextmanager
def log_context(logger: logging.Logger, operation: str, level: str = 'INFO'):
    """
    日志上下文管理器
    
    Args:
        logger: 日志记录器
        operation: 操作名称
        level: 日志级别
    """
    log_level = getattr(logging, level.upper())
    start_time = time.time()
    
    logger.log(log_level, f"开始操作: {operation}")
    
    try:
        yield
        duration = time.time() - start_time
        logger.log(log_level, f"操作完成: {operation} (耗时: {duration:.4f}s)")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"操作失败: {operation} (耗时: {duration:.4f}s) - 错误: {str(e)}")
        raise