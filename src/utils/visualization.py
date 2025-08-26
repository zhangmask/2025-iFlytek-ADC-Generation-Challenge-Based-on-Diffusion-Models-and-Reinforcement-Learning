#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果可视化模块

提供训练进度监控、结果可视化、性能分析和报告生成功能。

Author: AI Developer
Date: 2025
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import base64
from io import BytesIO


@dataclass
class TrainingMetrics:
    """训练指标数据类"""
    epoch: int
    step: int
    timestamp: datetime
    loss: float
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """生成结果数据类"""
    molecule_smiles: str
    dar_score: float
    diversity_score: float
    validity: bool
    uniqueness: bool
    properties: Dict[str, float] = field(default_factory=dict)
    generation_time: Optional[float] = None


class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, log_dir: str = "logs", save_interval: int = 100):
        """
        初始化训练监控器
        
        Args:
            log_dir: 日志目录
            save_interval: 保存间隔
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.logger = logging.getLogger(__name__)
        
        self.metrics_history: List[TrainingMetrics] = []
        self.current_epoch = 0
        self.current_step = 0
        self.start_time = datetime.now()
        
        # 创建子目录
        (self.log_dir / "plots").mkdir(exist_ok=True)
        (self.log_dir / "data").mkdir(exist_ok=True)
        (self.log_dir / "reports").mkdir(exist_ok=True)
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int = None, step: int = None) -> None:
        """记录训练指标
        
        Args:
            metrics: 指标字典
            epoch: 训练轮次
            step: 训练步数
        """
        if epoch is not None:
            self.current_epoch = epoch
        if step is not None:
            self.current_step = step
        
        # 提取标准指标
        loss = metrics.get('loss', 0.0)
        accuracy = metrics.get('accuracy')
        learning_rate = metrics.get('learning_rate')
        memory_usage = metrics.get('memory_usage')
        gpu_usage = metrics.get('gpu_usage')
        
        # 提取自定义指标
        custom_metrics = {k: v for k, v in metrics.items() 
                         if k not in ['loss', 'accuracy', 'learning_rate', 'memory_usage', 'gpu_usage']}
        
        # 创建指标记录
        metric_record = TrainingMetrics(
            epoch=self.current_epoch,
            step=self.current_step,
            timestamp=datetime.now(),
            loss=loss,
            accuracy=accuracy,
            learning_rate=learning_rate,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            custom_metrics=custom_metrics
        )
        
        self.metrics_history.append(metric_record)
        
        # 定期保存数据
        if len(self.metrics_history) % self.save_interval == 0:
            self.save_metrics()
        
        self.logger.debug(f"记录训练指标: Epoch {self.current_epoch}, Step {self.current_step}, Loss {loss:.4f}")
    
    def save_metrics(self) -> None:
        """保存指标数据"""
        try:
            # 转换为DataFrame
            data = []
            for metric in self.metrics_history:
                row = {
                    'epoch': metric.epoch,
                    'step': metric.step,
                    'timestamp': metric.timestamp.isoformat(),
                    'loss': metric.loss,
                    'accuracy': metric.accuracy,
                    'learning_rate': metric.learning_rate,
                    'memory_usage': metric.memory_usage,
                    'gpu_usage': metric.gpu_usage
                }
                row.update(metric.custom_metrics)
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # 保存为CSV
            csv_path = self.log_dir / "data" / "training_metrics.csv"
            df.to_csv(csv_path, index=False)
            
            # 保存为JSON
            json_path = self.log_dir / "data" / "training_metrics.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"训练指标已保存: {len(self.metrics_history)} 条记录")
            
        except Exception as e:
            self.logger.error(f"保存训练指标失败: {str(e)}")
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> str:
        """绘制训练曲线
        
        Args:
            save_path: 保存路径
            
        Returns:
            图片文件路径
        """
        if not self.metrics_history:
            self.logger.warning("没有训练指标数据")
            return ""
        
        # 准备数据
        epochs = [m.epoch for m in self.metrics_history]
        losses = [m.loss for m in self.metrics_history]
        accuracies = [m.accuracy for m in self.metrics_history if m.accuracy is not None]
        learning_rates = [m.learning_rate for m in self.metrics_history if m.learning_rate is not None]
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # 损失曲线
        axes[0, 0].plot(epochs, losses, 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率曲线
        if accuracies:
            acc_epochs = [m.epoch for m in self.metrics_history if m.accuracy is not None]
            axes[0, 1].plot(acc_epochs, accuracies, 'g-', linewidth=2)
            axes[0, 1].set_title('Training Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No Accuracy Data', ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # 学习率曲线
        if learning_rates:
            lr_epochs = [m.epoch for m in self.metrics_history if m.learning_rate is not None]
            axes[1, 0].plot(lr_epochs, learning_rates, 'r-', linewidth=2)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Learning Rate Data', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 内存使用情况
        memory_usage = [m.memory_usage for m in self.metrics_history if m.memory_usage is not None]
        if memory_usage:
            mem_epochs = [m.epoch for m in self.metrics_history if m.memory_usage is not None]
            axes[1, 1].plot(mem_epochs, memory_usage, 'm-', linewidth=2)
            axes[1, 1].set_title('Memory Usage')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Memory (GB)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Memory Data', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = self.log_dir / "plots" / f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"训练曲线已保存: {save_path}")
        return str(save_path)
    
    def create_interactive_dashboard(self, save_path: Optional[str] = None) -> str:
        """创建交互式仪表板
        
        Args:
            save_path: 保存路径
            
        Returns:
            HTML文件路径
        """
        if not self.metrics_history:
            self.logger.warning("没有训练指标数据")
            return ""
        
        # 准备数据
        df_data = []
        for metric in self.metrics_history:
            row = {
                'epoch': metric.epoch,
                'step': metric.step,
                'timestamp': metric.timestamp,
                'loss': metric.loss,
                'accuracy': metric.accuracy,
                'learning_rate': metric.learning_rate,
                'memory_usage': metric.memory_usage,
                'gpu_usage': metric.gpu_usage
            }
            row.update(metric.custom_metrics)
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # 创建子图
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Training Loss', 'Accuracy', 'Learning Rate', 'Memory Usage', 'GPU Usage', 'Custom Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 损失曲线
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['loss'], mode='lines', name='Loss', line=dict(color='blue')),
            row=1, col=1
        )
        
        # 准确率曲线
        if 'accuracy' in df.columns and df['accuracy'].notna().any():
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['accuracy'], mode='lines', name='Accuracy', line=dict(color='green')),
                row=1, col=2
            )
        
        # 学习率曲线
        if 'learning_rate' in df.columns and df['learning_rate'].notna().any():
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['learning_rate'], mode='lines', name='Learning Rate', line=dict(color='red')),
                row=2, col=1
            )
        
        # 内存使用情况
        if 'memory_usage' in df.columns and df['memory_usage'].notna().any():
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['memory_usage'], mode='lines', name='Memory Usage', line=dict(color='purple')),
                row=2, col=2
            )
        
        # GPU使用情况
        if 'gpu_usage' in df.columns and df['gpu_usage'].notna().any():
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['gpu_usage'], mode='lines', name='GPU Usage', line=dict(color='orange')),
                row=3, col=1
            )
        
        # 自定义指标
        custom_cols = [col for col in df.columns if col not in ['epoch', 'step', 'timestamp', 'loss', 'accuracy', 'learning_rate', 'memory_usage', 'gpu_usage']]
        if custom_cols:
            for i, col in enumerate(custom_cols[:5]):  # 最多显示5个自定义指标
                if df[col].notna().any():
                    fig.add_trace(
                        go.Scatter(x=df['epoch'], y=df[col], mode='lines', name=col),
                        row=3, col=2
                    )
        
        # 更新布局
        fig.update_layout(
            title='Training Dashboard',
            height=900,
            showlegend=True,
            template='plotly_white'
        )
        
        # 保存HTML文件
        if save_path is None:
            save_path = self.log_dir / "reports" / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        pyo.plot(fig, filename=str(save_path), auto_open=False)
        
        self.logger.info(f"交互式仪表板已保存: {save_path}")
        return str(save_path)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要
        
        Returns:
            训练摘要字典
        """
        if not self.metrics_history:
            return {}
        
        # 计算统计信息
        losses = [m.loss for m in self.metrics_history]
        accuracies = [m.accuracy for m in self.metrics_history if m.accuracy is not None]
        
        summary = {
            'total_epochs': self.current_epoch,
            'total_steps': len(self.metrics_history),
            'training_duration': str(datetime.now() - self.start_time),
            'final_loss': losses[-1] if losses else None,
            'best_loss': min(losses) if losses else None,
            'average_loss': np.mean(losses) if losses else None,
            'final_accuracy': accuracies[-1] if accuracies else None,
            'best_accuracy': max(accuracies) if accuracies else None,
            'average_accuracy': np.mean(accuracies) if accuracies else None
        }
        
        return summary


class MoleculeVisualizer:
    """分子可视化器"""
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        初始化分子可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def draw_molecule(self, smiles: str, size: Tuple[int, int] = (300, 300)) -> str:
        """绘制分子结构
        
        Args:
            smiles: SMILES字符串
            size: 图片大小
            
        Returns:
            Base64编码的图片字符串
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.logger.warning(f"无效的SMILES: {smiles}")
                return ""
            
            # 绘制分子
            drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            # 获取图片数据
            img_data = drawer.GetDrawingText()
            
            # 转换为Base64
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            self.logger.error(f"绘制分子失败: {str(e)}")
            return ""
    
    def create_molecule_grid(self, molecules: List[str], titles: Optional[List[str]] = None, 
                           cols: int = 4, mol_size: Tuple[int, int] = (200, 200)) -> str:
        """创建分子网格图
        
        Args:
            molecules: SMILES列表
            titles: 标题列表
            cols: 列数
            mol_size: 分子图片大小
            
        Returns:
            保存的图片路径
        """
        try:
            # 过滤有效分子
            valid_mols = []
            valid_titles = []
            
            for i, smiles in enumerate(molecules):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_mols.append(mol)
                    if titles and i < len(titles):
                        valid_titles.append(titles[i])
                    else:
                        valid_titles.append(f"Molecule {i+1}")
            
            if not valid_mols:
                self.logger.warning("没有有效的分子")
                return ""
            
            # 计算网格大小
            rows = (len(valid_mols) + cols - 1) // cols
            
            # 创建图片
            img = Draw.MolsToGridImage(
                valid_mols,
                molsPerRow=cols,
                subImgSize=mol_size,
                legends=valid_titles
            )
            
            # 保存图片
            save_path = self.output_dir / f"molecule_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            img.save(save_path)
            
            self.logger.info(f"分子网格图已保存: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"创建分子网格图失败: {str(e)}")
            return ""
    
    def plot_property_distribution(self, results: List[GenerationResult], 
                                 property_name: str = 'dar_score', 
                                 save_path: Optional[str] = None) -> str:
        """绘制属性分布图
        
        Args:
            results: 生成结果列表
            property_name: 属性名称
            save_path: 保存路径
            
        Returns:
            图片文件路径
        """
        try:
            # 提取属性值
            if property_name == 'dar_score':
                values = [r.dar_score for r in results]
            elif property_name == 'diversity_score':
                values = [r.diversity_score for r in results]
            elif property_name in results[0].properties:
                values = [r.properties.get(property_name, 0) for r in results]
            else:
                self.logger.warning(f"未知的属性: {property_name}")
                return ""
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            
            # 直方图
            plt.subplot(1, 2, 1)
            plt.hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'{property_name} Distribution')
            plt.xlabel(property_name)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # 箱线图
            plt.subplot(1, 2, 2)
            plt.boxplot(values)
            plt.title(f'{property_name} Box Plot')
            plt.ylabel(property_name)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            if save_path is None:
                save_path = self.output_dir / f"{property_name}_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"属性分布图已保存: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"绘制属性分布图失败: {str(e)}")
            return ""
    
    def create_correlation_heatmap(self, results: List[GenerationResult], 
                                 save_path: Optional[str] = None) -> str:
        """创建属性相关性热图
        
        Args:
            results: 生成结果列表
            save_path: 保存路径
            
        Returns:
            图片文件路径
        """
        try:
            # 准备数据
            data = []
            for result in results:
                row = {
                    'dar_score': result.dar_score,
                    'diversity_score': result.diversity_score,
                    'validity': int(result.validity),
                    'uniqueness': int(result.uniqueness)
                }
                row.update(result.properties)
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # 计算相关性矩阵
            correlation_matrix = df.corr()
            
            # 创建热图
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title('Property Correlation Heatmap')
            plt.tight_layout()
            
            # 保存图片
            if save_path is None:
                save_path = self.output_dir / f"correlation_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"相关性热图已保存: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"创建相关性热图失败: {str(e)}")
            return ""


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def generate_training_report(self, monitor: TrainingMonitor, 
                               save_path: Optional[str] = None) -> str:
        """生成训练报告
        
        Args:
            monitor: 训练监控器
            save_path: 保存路径
            
        Returns:
            报告文件路径
        """
        try:
            # 获取训练摘要
            summary = monitor.get_training_summary()
            
            # 生成图表
            curves_path = monitor.plot_training_curves()
            dashboard_path = monitor.create_interactive_dashboard()
            
            # 创建HTML报告
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Training Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .section {{ margin-bottom: 30px; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                    .metric-label {{ font-weight: bold; }}
                    .metric-value {{ color: #007bff; }}
                    img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Training Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Training Summary</h2>
                    <div class="metric">
                        <span class="metric-label">Total Epochs:</span>
                        <span class="metric-value">{summary.get('total_epochs', 'N/A')}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total Steps:</span>
                        <span class="metric-value">{summary.get('total_steps', 'N/A')}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Training Duration:</span>
                        <span class="metric-value">{summary.get('training_duration', 'N/A')}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Final Loss:</span>
                        <span class="metric-value">{summary.get('final_loss', 'N/A'):.4f if summary.get('final_loss') else 'N/A'}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Best Loss:</span>
                        <span class="metric-value">{summary.get('best_loss', 'N/A'):.4f if summary.get('best_loss') else 'N/A'}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Final Accuracy:</span>
                        <span class="metric-value">{summary.get('final_accuracy', 'N/A'):.4f if summary.get('final_accuracy') else 'N/A'}</span>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Training Curves</h2>
                    <img src="{curves_path}" alt="Training Curves">
                </div>
                
                <div class="section">
                    <h2>Interactive Dashboard</h2>
                    <p><a href="{dashboard_path}" target="_blank">Open Interactive Dashboard</a></p>
                </div>
            </body>
            </html>
            """
            
            # 保存报告
            if save_path is None:
                save_path = self.output_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"训练报告已生成: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"生成训练报告失败: {str(e)}")
            return ""
    
    def generate_generation_report(self, results: List[GenerationResult], 
                                 visualizer: MoleculeVisualizer,
                                 save_path: Optional[str] = None) -> str:
        """生成分子生成报告
        
        Args:
            results: 生成结果列表
            visualizer: 分子可视化器
            save_path: 保存路径
            
        Returns:
            报告文件路径
        """
        try:
            # 计算统计信息
            total_molecules = len(results)
            valid_molecules = sum(1 for r in results if r.validity)
            unique_molecules = sum(1 for r in results if r.uniqueness)
            
            avg_dar_score = np.mean([r.dar_score for r in results])
            avg_diversity_score = np.mean([r.diversity_score for r in results])
            
            # 生成可视化
            top_molecules = sorted(results, key=lambda x: x.dar_score, reverse=True)[:12]
            top_smiles = [r.molecule_smiles for r in top_molecules]
            top_titles = [f"DAR: {r.dar_score:.3f}" for r in top_molecules]
            
            grid_path = visualizer.create_molecule_grid(top_smiles, top_titles)
            dar_dist_path = visualizer.plot_property_distribution(results, 'dar_score')
            diversity_dist_path = visualizer.plot_property_distribution(results, 'diversity_score')
            correlation_path = visualizer.create_correlation_heatmap(results)
            
            # 创建HTML报告
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Molecule Generation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .section {{ margin-bottom: 30px; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                    .metric-label {{ font-weight: bold; }}
                    .metric-value {{ color: #007bff; }}
                    img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                    .molecule-grid {{ text-align: center; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Molecule Generation Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Generation Summary</h2>
                    <div class="metric">
                        <span class="metric-label">Total Molecules:</span>
                        <span class="metric-value">{total_molecules}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Valid Molecules:</span>
                        <span class="metric-value">{valid_molecules} ({valid_molecules/total_molecules*100:.1f}%)</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Unique Molecules:</span>
                        <span class="metric-value">{unique_molecules} ({unique_molecules/total_molecules*100:.1f}%)</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Average DAR Score:</span>
                        <span class="metric-value">{avg_dar_score:.4f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Average Diversity Score:</span>
                        <span class="metric-value">{avg_diversity_score:.4f}</span>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Top Molecules by DAR Score</h2>
                    <div class="molecule-grid">
                        <img src="{grid_path}" alt="Top Molecules">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Property Distributions</h2>
                    <img src="{dar_dist_path}" alt="DAR Score Distribution">
                    <img src="{diversity_dist_path}" alt="Diversity Score Distribution">
                </div>
                
                <div class="section">
                    <h2>Property Correlations</h2>
                    <img src="{correlation_path}" alt="Property Correlations">
                </div>
            </body>
            </html>
            """
            
            # 保存报告
            if save_path is None:
                save_path = self.output_dir / f"generation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"分子生成报告已生成: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"生成分子生成报告失败: {str(e)}")
            return ""