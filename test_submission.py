#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试提交文件生成脚本
专注于生成正确格式的submit1.csv和submit2.csv
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """加载训练和测试数据"""
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        logger.info(f"训练数据: {len(train_df)} 条")
        logger.info(f"测试数据: {len(test_df)} 条")
        return train_df, test_df
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        raise

def generate_simple_submit1(train_df, num_samples=500):
    """生成简化的submit1.csv"""
    logger.info("开始生成submit1.csv...")
    
    # 从训练数据中随机采样
    sampled_indices = np.random.choice(len(train_df), size=num_samples, replace=True)
    
    submit1_data = {
        'index': list(range(1, num_samples + 1)),
        'Antibody Light Chain Sequence': [],
        'Antibody Heavy Chain Sequence': [],
        'Antigen Sequence': [],
        'Payload Isosmiles': [],
        'Linker Isosmiles': [],
        'DAR': [],
        'C1': [],
        'C2': [],
        'C3': [],
        'C4': []
    }
    
    for i in sampled_indices:
        row = train_df.iloc[i]
        submit1_data['Antibody Light Chain Sequence'].append(row['Antibody Light Chain Sequence'])
        submit1_data['Antibody Heavy Chain Sequence'].append(row['Antibody Heavy Chain Sequence'])
        submit1_data['Antigen Sequence'].append(row['Antigen Sequence'])
        submit1_data['Payload Isosmiles'].append(row['Payload Isosmiles'])
        submit1_data['Linker Isosmiles'].append(row['Linker Isosmiles'])
        submit1_data['DAR'].append(row['DAR'])
        
        # 生成随机的C1-C4二分类标签
        submit1_data['C1'].append(int(np.random.choice([0, 1], p=[0.3, 0.7])))
        submit1_data['C2'].append(int(np.random.choice([0, 1], p=[0.4, 0.6])))
        submit1_data['C3'].append(int(np.random.choice([0, 1], p=[0.5, 0.5])))
        submit1_data['C4'].append(int(np.random.choice([0, 1], p=[0.6, 0.4])))
    
    submit1_df = pd.DataFrame(submit1_data)
    logger.info(f"submit1.csv生成完成，包含 {len(submit1_df)} 条记录")
    return submit1_df

def generate_simple_submit2(test_df):
    """生成简化的submit2.csv"""
    logger.info("开始生成submit2.csv...")
    
    submit2_data = {
        'index': test_df['index'].tolist(),
        'C1': [],
        'C2': [],
        'C3': [],
        'C4': []
    }
    
    for _ in range(len(test_df)):
        # 生成随机的C1-C4二分类预测
        submit2_data['C1'].append(int(np.random.choice([0, 1], p=[0.3, 0.7])))
        submit2_data['C2'].append(int(np.random.choice([0, 1], p=[0.4, 0.6])))
        submit2_data['C3'].append(int(np.random.choice([0, 1], p=[0.5, 0.5])))
        submit2_data['C4'].append(int(np.random.choice([0, 1], p=[0.6, 0.4])))
    
    submit2_df = pd.DataFrame(submit2_data)
    logger.info(f"submit2.csv生成完成，包含 {len(submit2_df)} 条记录")
    return submit2_df

def save_submission_files(submit1_df, submit2_df):
    """保存提交文件"""
    # 创建submit目录
    submit_dir = Path('submit')
    submit_dir.mkdir(exist_ok=True)
    
    # 保存submit1.csv（需要特殊处理序列数据的引号）
    submit1_path = submit_dir / 'submit1.csv'
    submit1_df.to_csv(submit1_path, index=False, encoding='utf-8', quoting=1)
    logger.info(f"submit1.csv已保存到: {submit1_path}")
    
    # 保存submit2.csv
    submit2_path = submit_dir / 'submit2.csv'
    submit2_df.to_csv(submit2_path, index=False, encoding='utf-8')
    logger.info(f"submit2.csv已保存到: {submit2_path}")
    
    return submit1_path, submit2_path

def validate_submission_files(submit1_path, submit2_path, test_df):
    """验证提交文件格式"""
    logger.info("验证提交文件格式...")
    
    # 验证submit1.csv
    submit1_df = pd.read_csv(submit1_path)
    expected_columns1 = ['index', 'Antibody Light Chain Sequence', 'Antibody Heavy Chain Sequence', 
                        'Antigen Sequence', 'Payload Isosmiles', 'Linker Isosmiles', 'DAR', 'C1', 'C2', 'C3', 'C4']
    
    assert list(submit1_df.columns) == expected_columns1, f"submit1.csv列名不正确: {list(submit1_df.columns)}"
    assert len(submit1_df) == 500, f"submit1.csv应包含500条记录，实际: {len(submit1_df)}"
    
    # 验证C1-C4为二分类
    for col in ['C1', 'C2', 'C3', 'C4']:
        unique_values = set(submit1_df[col].unique())
        assert unique_values.issubset({0, 1}), f"submit1.csv的{col}列应只包含0和1，实际: {unique_values}"
    
    logger.info("submit1.csv格式验证通过")
    
    # 验证submit2.csv
    submit2_df = pd.read_csv(submit2_path)
    expected_columns2 = ['index', 'C1', 'C2', 'C3', 'C4']
    
    assert list(submit2_df.columns) == expected_columns2, f"submit2.csv列名不正确: {list(submit2_df.columns)}"
    assert len(submit2_df) == len(test_df), f"submit2.csv应包含{len(test_df)}条记录，实际: {len(submit2_df)}"
    
    # 验证C1-C4为二分类
    for col in ['C1', 'C2', 'C3', 'C4']:
        unique_values = set(submit2_df[col].unique())
        assert unique_values.issubset({0, 1}), f"submit2.csv的{col}列应只包含0和1，实际: {unique_values}"
    
    logger.info("submit2.csv格式验证通过")
    logger.info("所有提交文件格式验证通过！")

def main():
    """主函数"""
    try:
        # 加载数据
        train_df, test_df = load_data()
        
        # 生成提交文件
        submit1_df = generate_simple_submit1(train_df)
        submit2_df = generate_simple_submit2(test_df)
        
        # 保存文件
        submit1_path, submit2_path = save_submission_files(submit1_df, submit2_df)
        
        # 验证文件格式
        validate_submission_files(submit1_path, submit2_path, test_df)
        
        logger.info("提交文件生成和验证完成！")
        
    except Exception as e:
        logger.error(f"生成提交文件失败: {e}")
        raise

if __name__ == "__main__":
    main()