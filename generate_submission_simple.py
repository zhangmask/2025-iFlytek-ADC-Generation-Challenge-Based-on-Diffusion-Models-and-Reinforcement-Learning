#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版提交文件生成脚本
直接使用随机生成方法创建submit1.csv和submit2.csv
"""

import os
import pandas as pd
import numpy as np
import random
from typing import List
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleSubmissionGenerator:
    """简化版提交文件生成器"""
    
    def __init__(self):
        self.logger = logger
        
        # 从训练数据中获取示例
        self.train_data = pd.read_csv('train.csv')
        self.test_data = pd.read_csv('test.csv')
        
        # 提取示例数据用于随机生成
        self.sample_antibody_light = self.train_data['Antibody Light Chain Sequence'].dropna().tolist()
        self.sample_antibody_heavy = self.train_data['Antibody Heavy Chain Sequence'].dropna().tolist()
        self.sample_antigen = self.train_data['Antigen Sequence'].dropna().tolist()
        self.sample_payload = self.train_data['Payload Isosmiles'].dropna().tolist()
        self.sample_linker = self.train_data['Linker Isosmiles'].dropna().tolist()
        self.sample_dar = self.train_data['DAR'].dropna().tolist()
        
    def generate_random_adc_molecules(self, n_molecules: int = 500) -> pd.DataFrame:
        """生成随机ADC分子"""
        self.logger.info(f"开始生成 {n_molecules} 个随机ADC分子")
        
        molecules = []
        for i in range(n_molecules):
            molecule = {
                'index': i,
                'Antibody Light Chain Sequence': random.choice(self.sample_antibody_light),
                'Antibody Heavy Chain Sequence': random.choice(self.sample_antibody_heavy),
                'Antigen Sequence': random.choice(self.sample_antigen),
                'Payload Isosmiles': random.choice(self.sample_payload),
                'Linker Isosmiles': random.choice(self.sample_linker),
                'DAR': random.choice(self.sample_dar),
                'C1': np.random.uniform(0.1, 10.0),  # 随机生成C1-C4值
                'C2': np.random.uniform(0.1, 10.0),
                'C3': np.random.uniform(0.1, 10.0),
                'C4': np.random.uniform(0.1, 10.0)
            }
            molecules.append(molecule)
            
        df = pd.DataFrame(molecules)
        self.logger.info(f"成功生成 {len(df)} 个ADC分子")
        return df
    
    def predict_test_set(self) -> pd.DataFrame:
        """为测试集生成C1-C4预测"""
        self.logger.info(f"开始为 {len(self.test_data)} 个测试样本生成预测")
        
        predictions = []
        for idx, row in self.test_data.iterrows():
            prediction = {
                'index': row['index'],
                'C1': np.random.uniform(0.1, 10.0),  # 随机预测C1-C4值
                'C2': np.random.uniform(0.1, 10.0),
                'C3': np.random.uniform(0.1, 10.0),
                'C4': np.random.uniform(0.1, 10.0)
            }
            predictions.append(prediction)
            
        df = pd.DataFrame(predictions)
        self.logger.info(f"成功生成 {len(df)} 个预测结果")
        return df
    
    def create_submission_files(self):
        """创建提交文件"""
        try:
            # 创建submit文件夹
            submit_dir = 'submit'
            os.makedirs(submit_dir, exist_ok=True)
            self.logger.info(f"创建提交文件夹: {submit_dir}")
            
            # 生成submit1.csv - 500个ADC分子
            self.logger.info("生成submit1.csv...")
            submit1_data = self.generate_random_adc_molecules(500)
            submit1_path = os.path.join(submit_dir, 'submit1.csv')
            submit1_data.to_csv(submit1_path, index=False, encoding='utf-8')
            self.logger.info(f"submit1.csv已保存到: {submit1_path}")
            
            # 生成submit2.csv - 测试集预测
            self.logger.info("生成submit2.csv...")
            submit2_data = self.predict_test_set()
            submit2_path = os.path.join(submit_dir, 'submit2.csv')
            submit2_data.to_csv(submit2_path, index=False, encoding='utf-8')
            self.logger.info(f"submit2.csv已保存到: {submit2_path}")
            
            # 验证文件
            self.validate_submission_files(submit1_path, submit2_path)
            
            self.logger.info("\n=== 提交文件生成成功！ ===")
            self.logger.info(f"submit1.csv: {submit1_data.shape[0]} 行, {submit1_data.shape[1]} 列")
            self.logger.info(f"submit2.csv: {submit2_data.shape[0]} 行, {submit2_data.shape[1]} 列")
            
        except Exception as e:
            self.logger.error(f"\n=== 提交文件生成失败！ ===")
            self.logger.error(f"错误信息: {e}")
            raise
    
    def validate_submission_files(self, submit1_path: str, submit2_path: str):
        """验证提交文件格式"""
        # 验证submit1.csv
        submit1 = pd.read_csv(submit1_path)
        expected_cols1 = ['index', 'Antibody Light Chain Sequence', 'Antibody Heavy Chain Sequence', 
                         'Antigen Sequence', 'Payload Isosmiles', 'Linker Isosmiles', 'DAR', 
                         'C1', 'C2', 'C3', 'C4']
        
        assert submit1.shape[0] == 500, f"submit1.csv应有500行，实际有{submit1.shape[0]}行"
        assert list(submit1.columns) == expected_cols1, f"submit1.csv列名不匹配"
        
        # 验证submit2.csv
        submit2 = pd.read_csv(submit2_path)
        expected_cols2 = ['index', 'C1', 'C2', 'C3', 'C4']
        
        assert submit2.shape[0] == len(self.test_data), f"submit2.csv行数应与测试集一致"
        assert list(submit2.columns) == expected_cols2, f"submit2.csv列名不匹配"
        
        self.logger.info("提交文件格式验证通过")

def main():
    """主函数"""
    logger.info("开始生成提交文件...")
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 创建生成器并生成提交文件
    generator = SimpleSubmissionGenerator()
    generator.create_submission_files()
    
    logger.info("提交文件生成完成！")

if __name__ == "__main__":
    main()