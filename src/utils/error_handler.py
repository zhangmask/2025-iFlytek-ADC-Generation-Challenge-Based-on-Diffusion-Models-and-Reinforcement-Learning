#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误处理和异常管理模块

提供统一的错误处理、异常捕获和恢复机制。

Author: AI Developer
Date: 2025
"""

import logging
import traceback
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union
from contextlib import contextmanager
from enum import Enum
import torch
import gc


class ErrorSeverity(Enum):
    """错误严重程度枚举"""
    LOW = "low"          # 轻微错误，可以继续执行
    MEDIUM = "medium"    # 中等错误，需要处理但不致命
    HIGH = "high"        # 严重错误，可能影响结果
    CRITICAL = "critical" # 致命错误，必须停止执行


class ADCException(Exception):
    """ADC项目自定义异常基类"""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 module: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.module = module or "unknown"
        self.context = context or {}
        self.timestamp = time.time()
    
    def __str__(self):
        return f"[{self.severity.value.upper()}] {self.module}: {self.message}"


class DataProcessingError(ADCException):
    """数据处理错误"""
    pass


class ModelError(ADCException):
    """模型相关错误"""
    pass


class GenerationError(ADCException):
    """分子生成错误"""
    pass


class EvaluationError(ADCException):
    """评估错误"""
    pass


class MemoryError(ADCException):
    """内存相关错误"""
    pass


class ErrorHandler:
    """错误处理器类"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: List[ADCException] = []
        self.retry_counts: Dict[str, int] = {}
        self.max_retries = 3
        self.recovery_strategies = {
            DataProcessingError: self._recover_data_processing,
            ModelError: self._recover_model_error,
            GenerationError: self._recover_generation_error,
            MemoryError: self._recover_memory_error,
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> bool:
        """处理错误
        
        Args:
            error: 异常对象
            context: 错误上下文信息
            
        Returns:
            是否成功处理错误
        """
        # 转换为ADC异常
        if not isinstance(error, ADCException):
            adc_error = self._convert_to_adc_exception(error, context)
        else:
            adc_error = error
        
        # 记录错误
        self._log_error(adc_error)
        self.error_history.append(adc_error)
        
        # 尝试恢复
        if adc_error.severity != ErrorSeverity.CRITICAL:
            return self._attempt_recovery(adc_error)
        
        return False
    
    def _convert_to_adc_exception(self, error: Exception, context: Dict[str, Any] = None) -> ADCException:
        """将普通异常转换为ADC异常"""
        error_type = type(error).__name__
        message = str(error)
        
        # 根据错误类型确定严重程度
        if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)):
            return MemoryError(message, ErrorSeverity.HIGH, context=context)
        elif isinstance(error, (ValueError, TypeError)):
            return DataProcessingError(message, ErrorSeverity.MEDIUM, context=context)
        elif isinstance(error, (RuntimeError, OSError)):
            return ModelError(message, ErrorSeverity.HIGH, context=context)
        else:
            return ADCException(message, ErrorSeverity.MEDIUM, context=context)
    
    def _log_error(self, error: ADCException):
        """记录错误信息"""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[error.severity]
        
        self.logger.log(log_level, f"错误发生: {error}")
        if error.context:
            self.logger.log(log_level, f"错误上下文: {error.context}")
        
        # 记录堆栈跟踪
        if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
    
    def _attempt_recovery(self, error: ADCException) -> bool:
        """尝试错误恢复"""
        error_type = type(error)
        
        if error_type in self.recovery_strategies:
            try:
                self.logger.info(f"尝试恢复错误: {error}")
                success = self.recovery_strategies[error_type](error)
                if success:
                    self.logger.info(f"错误恢复成功: {error}")
                else:
                    self.logger.warning(f"错误恢复失败: {error}")
                return success
            except Exception as recovery_error:
                self.logger.error(f"错误恢复过程中发生异常: {recovery_error}")
        
        return False
    
    def _recover_data_processing(self, error: DataProcessingError) -> bool:
        """数据处理错误恢复"""
        # 清理数据缓存
        gc.collect()
        return True
    
    def _recover_model_error(self, error: ModelError) -> bool:
        """模型错误恢复"""
        # 重置模型状态
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return True
    
    def _recover_generation_error(self, error: GenerationError) -> bool:
        """生成错误恢复"""
        # 降低批次大小或其他参数
        return True
    
    def _recover_memory_error(self, error: MemoryError) -> bool:
        """内存错误恢复"""
        # 强制内存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return True
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        if not self.error_history:
            return {"total_errors": 0}
        
        severity_counts = {}
        module_counts = {}
        
        for error in self.error_history:
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
            module_counts[error.module] = module_counts.get(error.module, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "severity_distribution": severity_counts,
            "module_distribution": module_counts,
            "recent_errors": [str(error) for error in self.error_history[-5:]]
        }


def error_handler(severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 module: str = None,
                 retry_count: int = 3,
                 fallback_value: Any = None):
    """错误处理装饰器
    
    Args:
        severity: 错误严重程度
        module: 模块名称
        retry_count: 重试次数
        fallback_value: 失败时的回退值
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler()
            
            for attempt in range(retry_count + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retry_count:
                        # 最后一次尝试失败
                        adc_error = ADCException(
                            f"函数 {func.__name__} 执行失败: {str(e)}",
                            severity=severity,
                            module=module or func.__module__,
                            context={"args": str(args), "kwargs": str(kwargs)}
                        )
                        
                        if handler.handle_error(adc_error):
                            continue  # 恢复成功，再次尝试
                        else:
                            if fallback_value is not None:
                                handler.logger.warning(f"使用回退值: {fallback_value}")
                                return fallback_value
                            raise adc_error
                    else:
                        # 还有重试机会
                        handler.logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败，重试中...")
                        time.sleep(0.1 * (attempt + 1))  # 指数退避
        
        return wrapper
    return decorator


@contextmanager
def safe_execution(logger: logging.Logger = None, 
                  error_handler_instance: ErrorHandler = None):
    """安全执行上下文管理器
    
    Args:
        logger: 日志记录器
        error_handler_instance: 错误处理器实例
    """
    handler = error_handler_instance or ErrorHandler(logger)
    
    try:
        yield handler
    except Exception as e:
        if not handler.handle_error(e):
            raise


def validate_input(validation_func: Callable, error_message: str = None):
    """输入验证装饰器
    
    Args:
        validation_func: 验证函数
        error_message: 错误消息
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not validation_func(*args, **kwargs):
                raise DataProcessingError(
                    error_message or f"输入验证失败: {func.__name__}",
                    ErrorSeverity.MEDIUM,
                    module=func.__module__
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


class CircuitBreaker:
    """熔断器模式实现"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """通过熔断器调用函数"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise ADCException(
                    "熔断器开启，拒绝执行",
                    ErrorSeverity.HIGH,
                    context={"failure_count": self.failure_count}
                )
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise