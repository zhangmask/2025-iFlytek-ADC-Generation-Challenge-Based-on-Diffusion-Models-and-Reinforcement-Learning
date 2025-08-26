#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块

提供灵活的配置管理功能，支持多种配置源、环境变量、配置验证和热重载。

Author: AI Developer
Date: 2025
"""

import os
import json
import yaml
import logging
from typing import Any, Dict, List, Optional, Union, Type, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from copy import deepcopy


class ConfigFormat(Enum):
    """配置文件格式"""
    YAML = "yaml"
    JSON = "json"
    INI = "ini"
    TOML = "toml"


@dataclass
class ConfigSchema:
    """配置模式定义"""
    name: str
    type: Type
    default: Any = None
    required: bool = False
    description: str = ""
    validator: Optional[Callable[[Any], bool]] = None
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None


@dataclass
class ConfigSection:
    """配置节定义"""
    name: str
    description: str = ""
    schemas: Dict[str, ConfigSchema] = field(default_factory=dict)
    required: bool = False


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class ConfigFileWatcher(FileSystemEventHandler):
    """配置文件监控器"""
    
    def __init__(self, config_manager, file_path: str):
        self.config_manager = config_manager
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        """文件修改事件处理"""
        if not event.is_directory and event.src_path == self.file_path:
            self.logger.info(f"配置文件 {self.file_path} 已修改，重新加载配置")
            try:
                self.config_manager.reload_config()
            except Exception as e:
                self.logger.error(f"重新加载配置失败: {str(e)}")


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None, auto_reload: bool = False):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
            auto_reload: 是否启用自动重载
        """
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file
        self.auto_reload = auto_reload
        self.config_data: Dict[str, Any] = {}
        self.schemas: Dict[str, ConfigSection] = {}
        self.observers: List[Observer] = []
        self.change_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._lock = threading.RLock()
        
        # 加载配置
        if config_file:
            self.load_config(config_file)
        
        # 启用自动重载
        if auto_reload and config_file:
            self.enable_auto_reload()
    
    def register_schema(self, section: ConfigSection) -> None:
        """注册配置模式
        
        Args:
            section: 配置节定义
        """
        with self._lock:
            self.schemas[section.name] = section
            self.logger.debug(f"注册配置模式: {section.name}")
    
    def load_config(self, config_file: str) -> None:
        """加载配置文件
        
        Args:
            config_file: 配置文件路径
        """
        with self._lock:
            try:
                config_path = Path(config_file)
                if not config_path.exists():
                    self.logger.warning(f"配置文件不存在: {config_file}")
                    return
                
                # 根据文件扩展名确定格式
                file_format = self._detect_format(config_path)
                
                # 加载配置数据
                with open(config_path, 'r', encoding='utf-8') as f:
                    if file_format == ConfigFormat.YAML:
                        self.config_data = yaml.safe_load(f) or {}
                    elif file_format == ConfigFormat.JSON:
                        self.config_data = json.load(f)
                    else:
                        raise ValueError(f"不支持的配置文件格式: {file_format}")
                
                # 处理环境变量替换
                self._process_environment_variables()
                
                # 验证配置
                self.validate_config()
                
                self.config_file = config_file
                self.logger.info(f"成功加载配置文件: {config_file}")
                
                # 通知配置变更
                self._notify_config_change()
                
            except Exception as e:
                self.logger.error(f"加载配置文件失败: {str(e)}")
                raise
    
    def reload_config(self) -> None:
        """重新加载配置文件"""
        if self.config_file:
            self.load_config(self.config_file)
    
    def save_config(self, config_file: Optional[str] = None) -> None:
        """保存配置到文件
        
        Args:
            config_file: 配置文件路径，如果为None则使用当前配置文件
        """
        with self._lock:
            target_file = config_file or self.config_file
            if not target_file:
                raise ValueError("未指定配置文件路径")
            
            try:
                config_path = Path(target_file)
                config_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_format = self._detect_format(config_path)
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    if file_format == ConfigFormat.YAML:
                        yaml.dump(self.config_data, f, default_flow_style=False, 
                                allow_unicode=True, indent=2)
                    elif file_format == ConfigFormat.JSON:
                        json.dump(self.config_data, f, indent=2, ensure_ascii=False)
                    else:
                        raise ValueError(f"不支持的配置文件格式: {file_format}")
                
                self.logger.info(f"配置已保存到: {target_file}")
                
            except Exception as e:
                self.logger.error(f"保存配置文件失败: {str(e)}")
                raise
    
    def get(self, key: str, default: Any = None, section: Optional[str] = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            section: 配置节名称
            
        Returns:
            配置值
        """
        with self._lock:
            try:
                if section:
                    return self.config_data.get(section, {}).get(key, default)
                else:
                    # 支持点号分隔的嵌套键
                    keys = key.split('.')
                    value = self.config_data
                    for k in keys:
                        if isinstance(value, dict) and k in value:
                            value = value[k]
                        else:
                            return default
                    return value
            except Exception:
                return default
    
    def set(self, key: str, value: Any, section: Optional[str] = None) -> None:
        """设置配置值
        
        Args:
            key: 配置键
            value: 配置值
            section: 配置节名称
        """
        with self._lock:
            if section:
                if section not in self.config_data:
                    self.config_data[section] = {}
                self.config_data[section][key] = value
            else:
                # 支持点号分隔的嵌套键
                keys = key.split('.')
                current = self.config_data
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            
            # 验证新配置
            self.validate_config()
            
            # 通知配置变更
            self._notify_config_change()
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """批量更新配置
        
        Args:
            config_dict: 配置字典
        """
        with self._lock:
            self._deep_update(self.config_data, config_dict)
            
            # 验证新配置
            self.validate_config()
            
            # 通知配置变更
            self._notify_config_change()
    
    def validate_config(self) -> None:
        """验证配置
        
        Raises:
            ConfigValidationError: 配置验证失败
        """
        errors = []
        
        for section_name, section in self.schemas.items():
            section_data = self.config_data.get(section_name, {})
            
            # 检查必需的节
            if section.required and not section_data:
                errors.append(f"缺少必需的配置节: {section_name}")
                continue
            
            # 验证节中的配置项
            for schema_name, schema in section.schemas.items():
                value = section_data.get(schema_name)
                
                # 检查必需的配置项
                if schema.required and value is None:
                    errors.append(f"缺少必需的配置项: {section_name}.{schema_name}")
                    continue
                
                # 如果值为None且有默认值，使用默认值
                if value is None and schema.default is not None:
                    section_data[schema_name] = schema.default
                    value = schema.default
                
                if value is not None:
                    # 类型检查
                    if not isinstance(value, schema.type):
                        errors.append(
                            f"配置项 {section_name}.{schema_name} 类型错误: "
                            f"期望 {schema.type.__name__}, 实际 {type(value).__name__}"
                        )
                        continue
                    
                    # 选择值检查
                    if schema.choices and value not in schema.choices:
                        errors.append(
                            f"配置项 {section_name}.{schema_name} 值无效: "
                            f"必须是 {schema.choices} 中的一个"
                        )
                    
                    # 数值范围检查
                    if isinstance(value, (int, float)):
                        if schema.min_value is not None and value < schema.min_value:
                            errors.append(
                                f"配置项 {section_name}.{schema_name} 值过小: "
                                f"最小值为 {schema.min_value}"
                            )
                        if schema.max_value is not None and value > schema.max_value:
                            errors.append(
                                f"配置项 {section_name}.{schema_name} 值过大: "
                                f"最大值为 {schema.max_value}"
                            )
                    
                    # 自定义验证器
                    if schema.validator and not schema.validator(value):
                        errors.append(
                            f"配置项 {section_name}.{schema_name} 验证失败"
                        )
        
        if errors:
            error_message = "配置验证失败:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ConfigValidationError(error_message)
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """获取配置节
        
        Args:
            section_name: 配置节名称
            
        Returns:
            配置节数据
        """
        with self._lock:
            return deepcopy(self.config_data.get(section_name, {}))
    
    def has_section(self, section_name: str) -> bool:
        """检查是否存在配置节
        
        Args:
            section_name: 配置节名称
            
        Returns:
            是否存在
        """
        with self._lock:
            return section_name in self.config_data
    
    def list_sections(self) -> List[str]:
        """列出所有配置节
        
        Returns:
            配置节名称列表
        """
        with self._lock:
            return list(self.config_data.keys())
    
    def add_change_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """添加配置变更回调
        
        Args:
            callback: 回调函数
        """
        self.change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """移除配置变更回调
        
        Args:
            callback: 回调函数
        """
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
    
    def enable_auto_reload(self) -> None:
        """启用自动重载"""
        if not self.config_file:
            raise ValueError("未指定配置文件，无法启用自动重载")
        
        config_path = Path(self.config_file)
        if not config_path.exists():
            self.logger.warning(f"配置文件不存在，无法启用自动重载: {self.config_file}")
            return
        
        # 创建文件监控器
        event_handler = ConfigFileWatcher(self, str(config_path.absolute()))
        observer = Observer()
        observer.schedule(event_handler, str(config_path.parent), recursive=False)
        observer.start()
        
        self.observers.append(observer)
        self.logger.info(f"已启用配置文件自动重载: {self.config_file}")
    
    def disable_auto_reload(self) -> None:
        """禁用自动重载"""
        for observer in self.observers:
            observer.stop()
            observer.join()
        
        self.observers.clear()
        self.logger.info("已禁用配置文件自动重载")
    
    def export_config(self, format: ConfigFormat = ConfigFormat.YAML) -> str:
        """导出配置为字符串
        
        Args:
            format: 导出格式
            
        Returns:
            配置字符串
        """
        with self._lock:
            if format == ConfigFormat.YAML:
                return yaml.dump(self.config_data, default_flow_style=False, 
                               allow_unicode=True, indent=2)
            elif format == ConfigFormat.JSON:
                return json.dumps(self.config_data, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
    
    def import_config(self, config_str: str, format: ConfigFormat) -> None:
        """从字符串导入配置
        
        Args:
            config_str: 配置字符串
            format: 配置格式
        """
        with self._lock:
            try:
                if format == ConfigFormat.YAML:
                    new_config = yaml.safe_load(config_str) or {}
                elif format == ConfigFormat.JSON:
                    new_config = json.loads(config_str)
                else:
                    raise ValueError(f"不支持的导入格式: {format}")
                
                # 更新配置
                self.config_data = new_config
                
                # 处理环境变量替换
                self._process_environment_variables()
                
                # 验证配置
                self.validate_config()
                
                # 通知配置变更
                self._notify_config_change()
                
                self.logger.info("成功导入配置")
                
            except Exception as e:
                self.logger.error(f"导入配置失败: {str(e)}")
                raise
    
    def _detect_format(self, config_path: Path) -> ConfigFormat:
        """检测配置文件格式"""
        suffix = config_path.suffix.lower()
        if suffix in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        elif suffix == '.json':
            return ConfigFormat.JSON
        else:
            # 默认使用YAML格式
            return ConfigFormat.YAML
    
    def _process_environment_variables(self) -> None:
        """处理环境变量替换"""
        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            elif isinstance(obj, str):
                # 替换 ${VAR_NAME} 或 $VAR_NAME 格式的环境变量
                import re
                pattern = r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)'
                
                def replacer(match):
                    var_name = match.group(1) or match.group(2)
                    return os.environ.get(var_name, match.group(0))
                
                return re.sub(pattern, replacer, obj)
            else:
                return obj
        
        self.config_data = replace_env_vars(self.config_data)
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """深度更新字典"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _notify_config_change(self) -> None:
        """通知配置变更"""
        for callback in self.change_callbacks:
            try:
                callback(deepcopy(self.config_data))
            except Exception as e:
                self.logger.error(f"配置变更回调执行失败: {str(e)}")
    
    def __del__(self):
        """析构函数"""
        self.disable_auto_reload()


# 全局配置管理器实例
_global_config_manager: Optional[ConfigManager] = None


def get_global_config() -> ConfigManager:
    """获取全局配置管理器
    
    Returns:
        全局配置管理器实例
    """
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager


def init_global_config(config_file: str, auto_reload: bool = False) -> ConfigManager:
    """初始化全局配置管理器
    
    Args:
        config_file: 配置文件路径
        auto_reload: 是否启用自动重载
        
    Returns:
        全局配置管理器实例
    """
    global _global_config_manager
    _global_config_manager = ConfigManager(config_file, auto_reload)
    return _global_config_manager


def config_property(key: str, default: Any = None, section: Optional[str] = None):
    """配置属性装饰器
    
    Args:
        key: 配置键
        default: 默认值
        section: 配置节名称
    """
    def decorator(cls):
        def getter(self):
            config = get_global_config()
            return config.get(key, default, section)
        
        def setter(self, value):
            config = get_global_config()
            config.set(key, value, section)
        
        setattr(cls, f'_{key}', property(getter, setter))
        return cls
    
    return decorator