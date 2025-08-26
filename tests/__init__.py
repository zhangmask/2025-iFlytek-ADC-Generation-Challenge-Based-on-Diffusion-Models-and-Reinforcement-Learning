#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模块初始化文件

Author: AI Developer
Date: 2025
"""

# 测试配置
TEST_CONFIG = {
    'test_data_dir': 'tests/data',
    'test_output_dir': 'tests/output',
    'mock_models': True,
    'use_small_datasets': True,
    'timeout': 30  # 测试超时时间（秒）
}

# 测试工具函数
def setup_test_environment():
    """设置测试环境"""
    import os
    import tempfile
    from pathlib import Path
    
    # 创建临时测试目录
    test_dirs = [
        TEST_CONFIG['test_data_dir'],
        TEST_CONFIG['test_output_dir']
    ]
    
    for test_dir in test_dirs:
        Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置环境变量
    os.environ['TESTING'] = 'true'
    os.environ['TEST_MODE'] = 'unit'

def cleanup_test_environment():
    """清理测试环境"""
    import os
    import shutil
    from pathlib import Path
    
    # 清理测试目录
    test_dirs = [
        TEST_CONFIG['test_data_dir'],
        TEST_CONFIG['test_output_dir']
    ]
    
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir, ignore_errors=True)
    
    # 清理环境变量
    os.environ.pop('TESTING', None)
    os.environ.pop('TEST_MODE', None)