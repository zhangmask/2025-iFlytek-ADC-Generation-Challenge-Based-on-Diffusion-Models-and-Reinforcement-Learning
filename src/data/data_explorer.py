"""数据探索分析模块

提供数据统计分析、可视化等功能。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DataExplorer:
    """数据探索器"""
    
    def __init__(self, output_dir: str = "exploration_results"):
        """
        初始化数据探索器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # 可视化参数
        self.figure_size = (12, 8)
        self.dpi = 300
        
        # 设置样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.logger.info(f"数据探索器初始化完成，输出目录: {self.output_dir}")
    
    def basic_statistics(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        基本统计分析
        
        Args:
            data: 数据DataFrame
            
        Returns:
            统计结果字典
        """
        self.logger.info("开始基本统计分析")
        
        stats = {
            'basic_info': {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
                'duplicate_rows': data.duplicated().sum(),
                'duplicate_percentage': data.duplicated().sum() / len(data) * 100
            },
            'missing_values': {
                'counts': data.isnull().sum().to_dict(),
                'percentages': (data.isnull().sum() / len(data) * 100).to_dict()
            }
        }
        
        # 数值列统计
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            stats['numeric_statistics'] = data[numeric_columns].describe().to_dict()
            
            # 相关性分析
            if len(numeric_columns) > 1:
                correlation_matrix = data[numeric_columns].corr()
                stats['correlation_matrix'] = correlation_matrix.to_dict()
        
        # 文本列统计
        text_columns = data.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            text_stats = {}
            for col in text_columns:
                if not data[col].empty:
                    text_stats[col] = {
                        'unique_count': data[col].nunique(),
                        'unique_percentage': data[col].nunique() / len(data) * 100,
                        'most_common': data[col].value_counts().head(10).to_dict(),
                        'avg_length': data[col].astype(str).str.len().mean(),
                        'min_length': data[col].astype(str).str.len().min(),
                        'max_length': data[col].astype(str).str.len().max()
                    }
            stats['text_statistics'] = text_stats
        
        self.logger.info("基本统计分析完成")
        return stats
    
    def protein_sequence_analysis(self, data: pd.DataFrame, 
                                sequence_column: str = 'protein_sequence') -> Dict[str, any]:
        """
        蛋白质序列分析
        
        Args:
            data: 数据DataFrame
            sequence_column: 蛋白质序列列名
            
        Returns:
            蛋白质序列分析结果
        """
        if sequence_column not in data.columns:
            raise ValueError(f"列 '{sequence_column}' 不存在")
        
        self.logger.info("开始蛋白质序列分析")
        
        sequences = data[sequence_column].dropna()
        
        # 基本统计
        sequence_lengths = sequences.str.len()
        
        # 氨基酸组成分析
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_composition = {}
        
        for aa in amino_acids:
            aa_counts = sequences.str.count(aa)
            aa_composition[aa] = {
                'total_count': aa_counts.sum(),
                'avg_percentage': (aa_counts / sequence_lengths).mean() * 100,
                'std_percentage': (aa_counts / sequence_lengths).std() * 100
            }
        
        # 序列特征
        analysis_results = {
            'basic_stats': {
                'total_sequences': len(sequences),
                'unique_sequences': sequences.nunique(),
                'avg_length': sequence_lengths.mean(),
                'std_length': sequence_lengths.std(),
                'min_length': sequence_lengths.min(),
                'max_length': sequence_lengths.max(),
                'length_distribution': sequence_lengths.describe().to_dict()
            },
            'amino_acid_composition': aa_composition,
            'sequence_patterns': self._analyze_sequence_patterns(sequences)
        }
        
        self.logger.info("蛋白质序列分析完成")
        return analysis_results
    
    def _analyze_sequence_patterns(self, sequences: pd.Series) -> Dict[str, any]:
        """
        分析序列模式
        
        Args:
            sequences: 蛋白质序列Series
            
        Returns:
            序列模式分析结果
        """
        # 常见二肽分析
        dipeptides = []
        for seq in sequences:
            if len(seq) >= 2:
                dipeptides.extend([seq[i:i+2] for i in range(len(seq)-1)])
        
        dipeptide_counts = Counter(dipeptides)
        
        # 常见三肽分析
        tripeptides = []
        for seq in sequences:
            if len(seq) >= 3:
                tripeptides.extend([seq[i:i+3] for i in range(len(seq)-2)])
        
        tripeptide_counts = Counter(tripeptides)
        
        return {
            'most_common_dipeptides': dict(dipeptide_counts.most_common(20)),
            'most_common_tripeptides': dict(tripeptide_counts.most_common(20)),
            'total_dipeptides': len(dipeptide_counts),
            'total_tripeptides': len(tripeptide_counts)
        }
    
    def smiles_analysis(self, data: pd.DataFrame, 
                       smiles_column: str = 'smiles') -> Dict[str, any]:
        """
        SMILES分子分析
        
        Args:
            data: 数据DataFrame
            smiles_column: SMILES列名
            
        Returns:
            SMILES分析结果
        """
        if smiles_column not in data.columns:
            raise ValueError(f"列 '{smiles_column}' 不存在")
        
        self.logger.info("开始SMILES分子分析")
        
        smiles = data[smiles_column].dropna()
        
        # 基本统计
        smiles_lengths = smiles.str.len()
        
        # 字符组成分析
        all_chars = ''.join(smiles)
        char_counts = Counter(all_chars)
        
        # 常见原子分析
        atoms = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
        atom_composition = {}
        
        for atom in atoms:
            atom_counts = smiles.str.count(atom)
            atom_composition[atom] = {
                'total_count': atom_counts.sum(),
                'avg_count_per_molecule': atom_counts.mean(),
                'std_count_per_molecule': atom_counts.std(),
                'presence_percentage': (atom_counts > 0).mean() * 100
            }
        
        # 环结构分析
        ring_patterns = {
            'benzene_rings': smiles.str.count('c1ccccc1'),
            'six_membered_rings': smiles.str.count(r'[0-9]'),
            'aromatic_carbons': smiles.str.count('c'),
            'aliphatic_carbons': smiles.str.count('C')
        }
        
        analysis_results = {
            'basic_stats': {
                'total_smiles': len(smiles),
                'unique_smiles': smiles.nunique(),
                'avg_length': smiles_lengths.mean(),
                'std_length': smiles_lengths.std(),
                'min_length': smiles_lengths.min(),
                'max_length': smiles_lengths.max(),
                'length_distribution': smiles_lengths.describe().to_dict()
            },
            'character_composition': dict(char_counts.most_common(20)),
            'atom_composition': atom_composition,
            'structural_features': {
                'ring_statistics': {k: {'mean': v.mean(), 'std': v.std(), 'max': v.max()} 
                                  for k, v in ring_patterns.items()}
            }
        }
        
        self.logger.info("SMILES分子分析完成")
        return analysis_results
    
    def create_visualizations(self, data: pd.DataFrame, 
                            analysis_results: Dict = None,
                            save_plots: bool = True) -> Dict[str, str]:
        """
        创建可视化图表
        
        Args:
            data: 数据DataFrame
            analysis_results: 分析结果
            save_plots: 是否保存图表
            
        Returns:
            图表文件路径字典
        """
        self.logger.info("开始创建可视化图表")
        
        plots = {}
        
        # 1. 数据概览图
        plots.update(self._create_overview_plots(data, save_plots))
        
        # 2. 蛋白质序列可视化
        if 'protein_sequence' in data.columns:
            plots.update(self._create_protein_plots(data, save_plots))
        
        # 3. SMILES分子可视化
        if 'smiles' in data.columns:
            plots.update(self._create_smiles_plots(data, save_plots))
        
        # 4. 相关性分析图
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            plots.update(self._create_correlation_plots(data, numeric_columns, save_plots))
        
        self.logger.info(f"可视化图表创建完成，共生成 {len(plots)} 个图表")
        return plots
    
    def _create_overview_plots(self, data: pd.DataFrame, save_plots: bool) -> Dict[str, str]:
        """创建数据概览图表"""
        plots = {}
        
        # 1. 缺失值热力图
        if data.isnull().sum().sum() > 0:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            missing_data = data.isnull()
            sns.heatmap(missing_data, cbar=True, ax=ax, cmap='viridis')
            ax.set_title('Missing Values Heatmap')
            ax.set_xlabel('Columns')
            ax.set_ylabel('Rows')
            
            if save_plots:
                plot_path = self.output_dir / "missing_values_heatmap.png"
                plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
                plots['missing_values_heatmap'] = str(plot_path)
            
            plt.close()
        
        # 2. 数据类型分布
        fig, ax = plt.subplots(figsize=(10, 6))
        
        dtype_counts = data.dtypes.value_counts()
        ax.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        ax.set_title('Data Types Distribution')
        
        if save_plots:
            plot_path = self.output_dir / "data_types_distribution.png"
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plots['data_types_distribution'] = str(plot_path)
        
        plt.close()
        
        return plots
    
    def _create_protein_plots(self, data: pd.DataFrame, save_plots: bool) -> Dict[str, str]:
        """创建蛋白质序列相关图表"""
        plots = {}
        
        sequences = data['protein_sequence'].dropna()
        sequence_lengths = sequences.str.len()
        
        # 1. 序列长度分布
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 直方图
        axes[0].hist(sequence_lengths, bins=30, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Sequence Length')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Protein Sequence Length Distribution')
        
        # 箱线图
        axes[1].boxplot(sequence_lengths)
        axes[1].set_ylabel('Sequence Length')
        axes[1].set_title('Protein Sequence Length Box Plot')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / "protein_sequence_length.png"
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plots['protein_sequence_length'] = str(plot_path)
        
        plt.close()
        
        # 2. 氨基酸组成分析
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_composition = {}
        
        for aa in amino_acids:
            aa_counts = sequences.str.count(aa)
            aa_composition[aa] = (aa_counts / sequence_lengths).mean() * 100
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        aa_names = list(aa_composition.keys())
        aa_percentages = list(aa_composition.values())
        
        bars = ax.bar(aa_names, aa_percentages)
        ax.set_xlabel('Amino Acid')
        ax.set_ylabel('Average Percentage (%)')
        ax.set_title('Amino Acid Composition')
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, percentage in zip(bars, aa_percentages):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                   f'{percentage:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / "amino_acid_composition.png"
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plots['amino_acid_composition'] = str(plot_path)
        
        plt.close()
        
        return plots
    
    def _create_smiles_plots(self, data: pd.DataFrame, save_plots: bool) -> Dict[str, str]:
        """创建SMILES分子相关图表"""
        plots = {}
        
        smiles = data['smiles'].dropna()
        smiles_lengths = smiles.str.len()
        
        # 1. SMILES长度分布
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 直方图
        axes[0].hist(smiles_lengths, bins=30, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('SMILES Length')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('SMILES Length Distribution')
        
        # 箱线图
        axes[1].boxplot(smiles_lengths)
        axes[1].set_ylabel('SMILES Length')
        axes[1].set_title('SMILES Length Box Plot')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / "smiles_length_distribution.png"
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plots['smiles_length_distribution'] = str(plot_path)
        
        plt.close()
        
        # 2. 原子组成分析
        atoms = ['C', 'N', 'O', 'S', 'P', 'F']
        atom_composition = {}
        
        for atom in atoms:
            atom_counts = smiles.str.count(atom)
            atom_composition[atom] = atom_counts.mean()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        atom_names = list(atom_composition.keys())
        atom_counts = list(atom_composition.values())
        
        bars = ax.bar(atom_names, atom_counts)
        ax.set_xlabel('Atom Type')
        ax.set_ylabel('Average Count per Molecule')
        ax.set_title('Atom Composition in SMILES')
        
        # 添加数值标签
        for bar, count in zip(bars, atom_counts):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                   f'{count:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / "atom_composition.png"
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plots['atom_composition'] = str(plot_path)
        
        plt.close()
        
        return plots
    
    def _create_correlation_plots(self, data: pd.DataFrame, 
                                numeric_columns: List[str], 
                                save_plots: bool) -> Dict[str, str]:
        """创建相关性分析图表"""
        plots = {}
        
        # 相关性热力图
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        correlation_matrix = data[numeric_columns].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, ax=ax)
        ax.set_title('Correlation Matrix')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / "correlation_matrix.png"
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plots['correlation_matrix'] = str(plot_path)
        
        plt.close()
        
        return plots
    
    def create_interactive_plots(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        创建交互式图表
        
        Args:
            data: 数据DataFrame
            
        Returns:
            交互式图表文件路径字典
        """
        self.logger.info("开始创建交互式图表")
        
        plots = {}
        
        # 1. 数据概览仪表板
        if 'protein_sequence' in data.columns and 'smiles' in data.columns:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Protein Sequence Length', 'SMILES Length', 
                              'Data Completeness', 'Column Types'),
                specs=[[{"type": "histogram"}, {"type": "histogram"}],
                       [{"type": "bar"}, {"type": "pie"}]]
            )
            
            # 蛋白质序列长度分布
            protein_lengths = data['protein_sequence'].dropna().str.len()
            fig.add_trace(
                go.Histogram(x=protein_lengths, name="Protein Length"),
                row=1, col=1
            )
            
            # SMILES长度分布
            smiles_lengths = data['smiles'].dropna().str.len()
            fig.add_trace(
                go.Histogram(x=smiles_lengths, name="SMILES Length"),
                row=1, col=2
            )
            
            # 数据完整性
            completeness = (1 - data.isnull().sum() / len(data)) * 100
            fig.add_trace(
                go.Bar(x=completeness.index, y=completeness.values, name="Completeness"),
                row=2, col=1
            )
            
            # 列类型分布
            dtype_counts = data.dtypes.value_counts()
            fig.add_trace(
                go.Pie(labels=dtype_counts.index, values=dtype_counts.values, name="Data Types"),
                row=2, col=2
            )
            
            fig.update_layout(height=800, title_text="Data Overview Dashboard")
            
            # 保存交互式图表
            plot_path = self.output_dir / "interactive_overview.html"
            fig.write_html(str(plot_path))
            plots['interactive_overview'] = str(plot_path)
        
        self.logger.info(f"交互式图表创建完成，共生成 {len(plots)} 个图表")
        return plots
    
    def generate_report(self, data: pd.DataFrame, 
                       include_interactive: bool = True) -> str:
        """
        生成完整的数据探索报告
        
        Args:
            data: 数据DataFrame
            include_interactive: 是否包含交互式图表
            
        Returns:
            报告文件路径
        """
        self.logger.info("开始生成数据探索报告")
        
        # 进行各种分析
        basic_stats = self.basic_statistics(data)
        
        protein_analysis = None
        if 'protein_sequence' in data.columns:
            protein_analysis = self.protein_sequence_analysis(data)
        
        smiles_analysis_result = None
        if 'smiles' in data.columns:
            smiles_analysis_result = self.smiles_analysis(data)
        
        # 创建可视化
        static_plots = self.create_visualizations(data, save_plots=True)
        
        interactive_plots = {}
        if include_interactive:
            interactive_plots = self.create_interactive_plots(data)
        
        # 生成HTML报告
        report_content = self._generate_html_report(
            basic_stats, protein_analysis, smiles_analysis_result, 
            static_plots, interactive_plots
        )
        
        # 保存报告
        report_path = self.output_dir / "data_exploration_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"数据探索报告已生成: {report_path}")
        return str(report_path)
    
    def _generate_html_report(self, basic_stats: Dict, 
                            protein_analysis: Dict, 
                            smiles_analysis_result: Dict,
                            static_plots: Dict, 
                            interactive_plots: Dict) -> str:
        """生成HTML报告内容"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Exploration Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ margin: 10px 0; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .plot img {{ max-width: 100%; height: auto; }}
                .summary-box {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Data Exploration Report</h1>
            <p><strong>Generated on:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        # 基本统计信息
        html += f"""
            <h2>Basic Statistics</h2>
            <div class="summary-box">
                <h3>Dataset Overview</h3>
                <p><strong>Shape:</strong> {basic_stats['basic_info']['shape']}</p>
                <p><strong>Memory Usage:</strong> {basic_stats['basic_info']['memory_usage_mb']:.2f} MB</p>
                <p><strong>Duplicate Rows:</strong> {basic_stats['basic_info']['duplicate_rows']} ({basic_stats['basic_info']['duplicate_percentage']:.2f}%)</p>
            </div>
        """
        
        # 缺失值信息
        if any(basic_stats['missing_values']['counts'].values()):
            html += "<h3>Missing Values</h3><table><tr><th>Column</th><th>Missing Count</th><th>Missing Percentage</th></tr>"
            for col, count in basic_stats['missing_values']['counts'].items():
                if count > 0:
                    percentage = basic_stats['missing_values']['percentages'][col]
                    html += f"<tr><td>{col}</td><td>{count}</td><td>{percentage:.2f}%</td></tr>"
            html += "</table>"
        
        # 蛋白质序列分析
        if protein_analysis:
            html += f"""
                <h2>Protein Sequence Analysis</h2>
                <div class="summary-box">
                    <h3>Sequence Statistics</h3>
                    <p><strong>Total Sequences:</strong> {protein_analysis['basic_stats']['total_sequences']}</p>
                    <p><strong>Unique Sequences:</strong> {protein_analysis['basic_stats']['unique_sequences']}</p>
                    <p><strong>Average Length:</strong> {protein_analysis['basic_stats']['avg_length']:.1f}</p>
                    <p><strong>Length Range:</strong> {protein_analysis['basic_stats']['min_length']} - {protein_analysis['basic_stats']['max_length']}</p>
                </div>
            """
        
        # SMILES分析
        if smiles_analysis_result:
            html += f"""
                <h2>SMILES Analysis</h2>
                <div class="summary-box">
                    <h3>Molecule Statistics</h3>
                    <p><strong>Total SMILES:</strong> {smiles_analysis_result['basic_stats']['total_smiles']}</p>
                    <p><strong>Unique SMILES:</strong> {smiles_analysis_result['basic_stats']['unique_smiles']}</p>
                    <p><strong>Average Length:</strong> {smiles_analysis_result['basic_stats']['avg_length']:.1f}</p>
                    <p><strong>Length Range:</strong> {smiles_analysis_result['basic_stats']['min_length']} - {smiles_analysis_result['basic_stats']['max_length']}</p>
                </div>
            """
        
        # 静态图表
        if static_plots:
            html += "<h2>Visualizations</h2>"
            for plot_name, plot_path in static_plots.items():
                plot_filename = Path(plot_path).name
                html += f"""
                <div class="plot">
                    <h3>{plot_name.replace('_', ' ').title()}</h3>
                    <img src="{plot_filename}" alt="{plot_name}">
                </div>
                """
        
        # 交互式图表
        if interactive_plots:
            html += "<h2>Interactive Visualizations</h2>"
            for plot_name, plot_path in interactive_plots.items():
                plot_filename = Path(plot_path).name
                html += f"""
                <div class="plot">
                    <h3>{plot_name.replace('_', ' ').title()}</h3>
                    <p><a href="{plot_filename}" target="_blank">Open Interactive Plot</a></p>
                </div>
                """
        
        html += """
        </body>
        </html>
        """
        
        return html