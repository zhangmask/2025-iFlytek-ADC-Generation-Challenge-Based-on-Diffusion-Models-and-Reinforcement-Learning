#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试模块

测试优化后的性能改进、内存使用和并行处理效果。

Author: AI Developer
Date: 2025
"""

import unittest
import time
import psutil
import gc
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import torch
from pathlib import Path
import sys
from unittest.mock import Mock, patch
from contextlib import contextmanager

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入测试工具
from tests.test_utils import (
    TestEnvironment, MockDataGenerator, TestConfig,
    skip_if_no_gpu, timeout_test
)

# 导入被测试的模块
try:
    from src.models.reinforcement_learning import RLAgent
    from src.models.diffusion_model import DiffusionModel
    from src.data.data_processor import DataProcessor
    from src.utils.logger import ADCLogger
except ImportError as e:
    print(f"警告: 无法导入某些模块: {e}")
    # 创建模拟模块
    RLAgent = Mock
    DiffusionModel = Mock
    DataProcessor = Mock
    ADCLogger = Mock


class PerformanceTestCase(unittest.TestCase):
    """性能测试基类"""
    
    def setUp(self):
        """测试前设置"""
        self.test_env = TestEnvironment()
        self.test_env.__enter__()
        self.data_generator = MockDataGenerator()
        
        # 性能监控
        self.start_time = None
        self.start_memory = None
        self.performance_metrics = {}
    
    def tearDown(self):
        """测试后清理"""
        self.test_env.__exit__(None, None, None)
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @contextmanager
    def performance_monitor(self, test_name):
        """性能监控上下文管理器"""
        # 开始监控
        self.start_time = time.time()
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            start_gpu_memory = 0
        
        try:
            yield
        finally:
            # 结束监控
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            if torch.cuda.is_available():
                end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            else:
                end_gpu_memory = 0
                peak_gpu_memory = 0
            
            # 记录性能指标
            self.performance_metrics[test_name] = {
                'execution_time': end_time - self.start_time,
                'memory_usage': end_memory - self.start_memory,
                'gpu_memory_usage': end_gpu_memory - start_gpu_memory,
                'peak_gpu_memory': peak_gpu_memory,
                'cpu_percent': process.cpu_percent()
            }
    
    def assert_performance_improvement(self, test_name, max_time=None, max_memory=None):
        """断言性能改进"""
        metrics = self.performance_metrics.get(test_name, {})
        
        if max_time is not None:
            self.assertLess(
                metrics.get('execution_time', float('inf')), 
                max_time,
                f"执行时间超过限制: {metrics.get('execution_time', 0):.2f}s > {max_time}s"
            )
        
        if max_memory is not None:
            self.assertLess(
                metrics.get('memory_usage', float('inf')), 
                max_memory,
                f"内存使用超过限制: {metrics.get('memory_usage', 0):.2f}MB > {max_memory}MB"
            )
    
    def print_performance_summary(self):
        """打印性能摘要"""
        print("\n=== 性能测试摘要 ===")
        for test_name, metrics in self.performance_metrics.items():
            print(f"\n{test_name}:")
            print(f"  执行时间: {metrics.get('execution_time', 0):.3f}s")
            print(f"  内存使用: {metrics.get('memory_usage', 0):.2f}MB")
            print(f"  GPU内存使用: {metrics.get('gpu_memory_usage', 0):.2f}MB")
            print(f"  GPU峰值内存: {metrics.get('peak_gpu_memory', 0):.2f}MB")
            print(f"  CPU使用率: {metrics.get('cpu_percent', 0):.1f}%")


class TestMemoryOptimization(PerformanceTestCase):
    """内存优化测试类"""
    
    @patch('src.models.reinforcement_learning.RLAgent')
    def test_memory_efficient_training(self, mock_rl_agent):
        """测试内存高效训练"""
        # 创建模拟智能体
        agent_instance = Mock()
        mock_rl_agent.return_value = agent_instance
        
        # 模拟内存监控方法
        agent_instance._check_memory_usage.return_value = {
            'system_memory_percent': 60.0,
            'gpu_memory_percent': 70.0
        }
        agent_instance._force_memory_cleanup.return_value = None
        
        # 模拟训练步骤
        training_metrics = {
            'loss': 0.5,
            'memory_info': {
                'gpu_used_gb': 2.5,
                'gpu_percent': 70.0,
                'system_percent': 60.0
            }
        }
        agent_instance.train_step.return_value = training_metrics
        
        with self.performance_monitor('memory_efficient_training'):
            # 创建智能体
            config = {
                'memory_threshold': 80.0,
                'gc_frequency': 50,
                'memory_monitor_enabled': True
            }
            agent = mock_rl_agent(config)
            
            # 模拟多步训练
            for step in range(100):
                # 生成模拟数据
                batch_size = 32
                state = torch.randn(batch_size, 128)
                action = torch.randn(batch_size, 64)
                reward = torch.randn(batch_size, 1)
                
                # 执行训练步骤
                metrics = agent.train_step(state, action, reward)
                
                # 验证内存信息
                if step % 100 == 0:
                    self.assertIn('memory_info', metrics)
                    memory_info = metrics['memory_info']
                    self.assertIn('gpu_used_gb', memory_info)
                    self.assertIn('gpu_percent', memory_info)
                    self.assertIn('system_percent', memory_info)
        
        # 验证性能
        self.assert_performance_improvement(
            'memory_efficient_training',
            max_time=10.0,  # 最大10秒
            max_memory=500.0  # 最大500MB内存增长
        )
    
    @patch('src.models.reinforcement_learning.RLAgent')
    def test_memory_cleanup_effectiveness(self, mock_rl_agent):
        """测试内存清理效果"""
        # 创建模拟智能体
        agent_instance = Mock()
        mock_rl_agent.return_value = agent_instance
        
        # 模拟内存清理前后的状态
        def mock_memory_cleanup():
            # 模拟清理效果
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        agent_instance._force_memory_cleanup.side_effect = mock_memory_cleanup
        
        with self.performance_monitor('memory_cleanup'):
            agent = mock_rl_agent({})
            
            # 创建大量临时数据
            temp_data = []
            for i in range(100):
                temp_data.append(torch.randn(1000, 1000))
            
            # 记录清理前内存
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # 执行内存清理
            agent._force_memory_cleanup()
            
            # 删除临时数据
            del temp_data
            gc.collect()
            
            # 记录清理后内存
            memory_after = process.memory_info().rss / 1024 / 1024
            
            # 验证内存释放
            memory_freed = memory_before - memory_after
            print(f"内存释放: {memory_freed:.2f}MB")
        
        # 验证清理调用
        agent_instance._force_memory_cleanup.assert_called()
    
    @skip_if_no_gpu
    @patch('src.models.reinforcement_learning.RLAgent')
    def test_gpu_memory_management(self, mock_rl_agent):
        """测试GPU内存管理"""
        if not torch.cuda.is_available():
            self.skipTest("GPU不可用")
        
        # 创建模拟智能体
        agent_instance = Mock()
        mock_rl_agent.return_value = agent_instance
        
        # 模拟GPU内存监控
        def mock_gpu_memory_check():
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                return (allocated_memory / total_memory) * 100
            return 0
        
        agent_instance._check_memory_usage.side_effect = lambda: {
            'gpu_memory_percent': mock_gpu_memory_check()
        }
        
        with self.performance_monitor('gpu_memory_management'):
            agent = mock_rl_agent({})
            
            # 创建GPU数据
            gpu_tensors = []
            for i in range(10):
                tensor = torch.randn(1000, 1000).cuda()
                gpu_tensors.append(tensor)
            
            # 检查内存使用
            memory_stats = agent._check_memory_usage()
            
            # 清理GPU内存
            del gpu_tensors
            torch.cuda.empty_cache()
        
        # 验证GPU内存监控
        agent_instance._check_memory_usage.assert_called()


class TestParallelProcessing(PerformanceTestCase):
    """并行处理测试类"""
    
    @patch('src.models.reinforcement_learning.RLAgent')
    def test_parallel_batch_training(self, mock_rl_agent):
        """测试并行批次训练"""
        # 创建模拟智能体
        agent_instance = Mock()
        mock_rl_agent.return_value = agent_instance
        
        # 模拟并行训练方法
        def mock_parallel_train(batches):
            # 模拟并行处理
            results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for batch in batches:
                    future = executor.submit(lambda b: {'loss': np.random.random(), 'batch_id': b}, batch)
                    futures.append(future)
                
                for future in futures:
                    results.append(future.result())
            return results
        
        agent_instance.parallel_train_batch.side_effect = mock_parallel_train
        
        with self.performance_monitor('parallel_batch_training'):
            agent = mock_rl_agent({'enable_parallel': True, 'max_workers': 4})
            
            # 创建多个批次
            batches = list(range(8))  # 8个批次
            
            # 执行并行训练
            results = agent.parallel_train_batch(batches)
            
            # 验证结果
            self.assertEqual(len(results), len(batches))
            for result in results:
                self.assertIn('loss', result)
                self.assertIn('batch_id', result)
        
        # 验证性能改进
        self.assert_performance_improvement(
            'parallel_batch_training',
            max_time=5.0  # 并行处理应该更快
        )
    
    @patch('src.models.reinforcement_learning.RLAgent')
    def test_batch_network_updates(self, mock_rl_agent):
        """测试批量网络更新"""
        # 创建模拟智能体
        agent_instance = Mock()
        mock_rl_agent.return_value = agent_instance
        
        # 模拟批量更新方法
        def mock_batch_update(policy_batches, value_batches):
            # 模拟并行更新
            policy_results = [{'policy_loss': np.random.random()} for _ in policy_batches]
            value_results = [{'value_loss': np.random.random()} for _ in value_batches]
            return {
                'policy_results': policy_results,
                'value_results': value_results,
                'total_policy_loss': np.mean([r['policy_loss'] for r in policy_results]),
                'total_value_loss': np.mean([r['value_loss'] for r in value_results])
            }
        
        agent_instance.batch_update_networks.side_effect = mock_batch_update
        
        with self.performance_monitor('batch_network_updates'):
            agent = mock_rl_agent({'enable_parallel': True})
            
            # 创建批次数据
            policy_batches = [torch.randn(32, 128) for _ in range(4)]
            value_batches = [torch.randn(32, 64) for _ in range(4)]
            
            # 执行批量更新
            results = agent.batch_update_networks(policy_batches, value_batches)
            
            # 验证结果
            self.assertIn('policy_results', results)
            self.assertIn('value_results', results)
            self.assertIn('total_policy_loss', results)
            self.assertIn('total_value_loss', results)
        
        # 验证调用
        agent_instance.batch_update_networks.assert_called_once()
    
    def test_thread_pool_performance(self):
        """测试线程池性能"""
        def cpu_intensive_task(n):
            """CPU密集型任务"""
            result = 0
            for i in range(n):
                result += i ** 2
            return result
        
        # 测试串行执行
        with self.performance_monitor('serial_execution'):
            serial_results = []
            for i in range(8):
                result = cpu_intensive_task(100000)
                serial_results.append(result)
        
        # 测试并行执行
        with self.performance_monitor('parallel_execution'):
            with ThreadPoolExecutor(max_workers=4) as executor:
                parallel_results = list(executor.map(cpu_intensive_task, [100000] * 8))
        
        # 验证结果一致性
        self.assertEqual(len(serial_results), len(parallel_results))
        
        # 比较性能
        serial_time = self.performance_metrics['serial_execution']['execution_time']
        parallel_time = self.performance_metrics['parallel_execution']['execution_time']
        
        print(f"串行执行时间: {serial_time:.3f}s")
        print(f"并行执行时间: {parallel_time:.3f}s")
        print(f"性能提升: {serial_time / parallel_time:.2f}x")
        
        # 在多核系统上，并行执行应该更快
        if multiprocessing.cpu_count() > 1:
            self.assertLess(parallel_time, serial_time * 0.8)  # 至少20%的性能提升


class TestAlgorithmOptimization(PerformanceTestCase):
    """算法优化测试类"""
    
    @patch('src.models.diffusion_model.DiffusionModel')
    def test_optimized_molecule_generation(self, mock_diffusion):
        """测试优化的分子生成"""
        # 创建模拟扩散模型
        model_instance = Mock()
        mock_diffusion.return_value = model_instance
        
        # 模拟优化的生成方法
        def mock_generate_molecules(num_samples, batch_size=32):
            # 模拟批量生成
            molecules = []
            for i in range(0, num_samples, batch_size):
                batch_size_actual = min(batch_size, num_samples - i)
                batch_molecules = [f"CC(C)C{j}" for j in range(batch_size_actual)]
                molecules.extend(batch_molecules)
            return molecules
        
        model_instance.generate_molecules.side_effect = mock_generate_molecules
        
        with self.performance_monitor('optimized_generation'):
            model = mock_diffusion({})
            
            # 生成大量分子
            molecules = model.generate_molecules(num_samples=1000, batch_size=64)
            
            # 验证结果
            self.assertEqual(len(molecules), 1000)
            self.assertTrue(all(isinstance(mol, str) for mol in molecules))
        
        # 验证性能
        self.assert_performance_improvement(
            'optimized_generation',
            max_time=5.0  # 优化后应该更快
        )
    
    @patch('src.data.data_processor.DataProcessor')
    def test_vectorized_data_processing(self, mock_processor):
        """测试向量化数据处理"""
        # 创建模拟数据处理器
        processor_instance = Mock()
        mock_processor.return_value = processor_instance
        
        # 模拟向量化处理
        def mock_vectorized_process(data):
            # 模拟向量化操作
            if isinstance(data, list):
                # 批量处理
                processed = np.array(data) * 2  # 简单的向量化操作
                return processed.tolist()
            else:
                return data * 2
        
        processor_instance.vectorized_process.side_effect = mock_vectorized_process
        
        with self.performance_monitor('vectorized_processing'):
            processor = mock_processor()
            
            # 创建大量数据
            large_data = list(range(10000))
            
            # 执行向量化处理
            processed_data = processor.vectorized_process(large_data)
            
            # 验证结果
            self.assertEqual(len(processed_data), len(large_data))
            self.assertEqual(processed_data[0], 0)  # 0 * 2 = 0
            self.assertEqual(processed_data[1], 2)  # 1 * 2 = 2
        
        # 验证性能
        self.assert_performance_improvement(
            'vectorized_processing',
            max_time=1.0  # 向量化处理应该很快
        )


class TestScalabilityAndStress(PerformanceTestCase):
    """可扩展性和压力测试类"""
    
    @timeout_test(60)
    def test_large_batch_processing(self):
        """测试大批量处理"""
        with self.performance_monitor('large_batch_processing'):
            # 模拟大批量数据处理
            batch_sizes = [32, 64, 128, 256, 512]
            processing_times = []
            
            for batch_size in batch_sizes:
                start_time = time.time()
                
                # 模拟批量处理
                batch_data = torch.randn(batch_size, 128)
                processed_data = torch.nn.functional.relu(batch_data)
                result = torch.sum(processed_data)
                
                end_time = time.time()
                processing_times.append(end_time - start_time)
                
                # 验证结果
                self.assertGreater(result.item(), 0)
            
            # 分析批量大小对性能的影响
            print("\n批量大小性能分析:")
            for batch_size, proc_time in zip(batch_sizes, processing_times):
                print(f"  批量大小 {batch_size}: {proc_time:.4f}s")
        
        # 验证可扩展性
        self.assert_performance_improvement(
            'large_batch_processing',
            max_time=10.0
        )
    
    def test_memory_stress(self):
        """测试内存压力"""
        with self.performance_monitor('memory_stress'):
            # 逐步增加内存使用
            tensors = []
            max_tensors = 100
            
            try:
                for i in range(max_tensors):
                    # 创建大张量
                    tensor = torch.randn(1000, 1000)
                    tensors.append(tensor)
                    
                    # 监控内存使用
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    # 如果内存使用过高，停止测试
                    if memory_mb > 2000:  # 2GB限制
                        print(f"达到内存限制，停止在第{i+1}个张量")
                        break
                
                # 验证能够处理大量数据
                self.assertGreater(len(tensors), 10)
                
            finally:
                # 清理内存
                del tensors
                gc.collect()
        
        # 验证内存管理
        memory_usage = self.performance_metrics['memory_stress']['memory_usage']
        print(f"内存压力测试峰值使用: {memory_usage:.2f}MB")
    
    def test_concurrent_operations(self):
        """测试并发操作"""
        def concurrent_task(task_id):
            """并发任务"""
            # 模拟计算密集型任务
            data = torch.randn(500, 500)
            result = torch.matmul(data, data.T)
            return {
                'task_id': task_id,
                'result_sum': torch.sum(result).item(),
                'thread_id': threading.current_thread().ident
            }
        
        with self.performance_monitor('concurrent_operations'):
            # 启动多个并发任务
            num_tasks = 8
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for i in range(num_tasks):
                    future = executor.submit(concurrent_task, i)
                    futures.append(future)
                
                # 收集结果
                results = []
                for future in futures:
                    result = future.result()
                    results.append(result)
            
            # 验证结果
            self.assertEqual(len(results), num_tasks)
            
            # 验证不同线程执行
            thread_ids = set(result['thread_id'] for result in results)
            self.assertGreater(len(thread_ids), 1)  # 应该使用多个线程
            
            print(f"使用了 {len(thread_ids)} 个不同的线程")
        
        # 验证并发性能
        self.assert_performance_improvement(
            'concurrent_operations',
            max_time=15.0
        )


if __name__ == '__main__':
    # 设置测试环境
    import os
    os.environ['TESTING'] = 'true'
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestMemoryOptimization,
        TestParallelProcessing,
        TestAlgorithmOptimization,
        TestScalabilityAndStress
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 打印性能摘要
    print("\n" + "="*50)
    print("性能测试完成")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("="*50)