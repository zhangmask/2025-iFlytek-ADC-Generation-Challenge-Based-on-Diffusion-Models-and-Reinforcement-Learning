"""评估流水线模块

整合多样性和有效性评估功能，提供完整的分子生成评估流程。
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

from .diversity_metrics import DiversityMetrics
from .validity_metrics import ValidityMetrics

class EvaluationPipeline:
    """评估流水线"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 初始化评估器
        self.diversity_metrics = DiversityMetrics(self.config.get('diversity', {}))
        self.validity_metrics = ValidityMetrics(self.config.get('validity', {}))
        
        # 输出目录
        self.output_dir = self.config.get('output_dir', 'evaluation_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 可视化参数
        self.figure_size = self.config.get('figure_size', (12, 8))
        self.dpi = self.config.get('dpi', 300)
        
        self.logger.info("评估流水线初始化完成")
    
    def evaluate_single_set(self, smiles_list: List[str], 
                           set_name: str = "Generated") -> Dict[str, any]:
        """评估单个分子集合
        
        Args:
            smiles_list: SMILES字符串列表
            set_name: 集合名称
            
        Returns:
            评估结果
        """
        self.logger.info(f"评估分子集合: {set_name} ({len(smiles_list)} 个分子)")
        
        # 计算多样性指标
        diversity_results = self.diversity_metrics.calculate_comprehensive_diversity(smiles_list)
        
        # 计算有效性指标
        validity_results = self.validity_metrics.calculate_comprehensive_validity(smiles_list)
        
        # 整合结果
        evaluation_results = {
            'set_name': set_name,
            'timestamp': datetime.now().isoformat(),
            'total_molecules': len(smiles_list),
            'diversity': diversity_results,
            'validity': validity_results,
            'summary': self._generate_summary(diversity_results, validity_results)
        }
        
        return evaluation_results
    
    def evaluate_multiple_sets(self, smiles_sets: Dict[str, List[str]]) -> Dict[str, any]:
        """评估多个分子集合
        
        Args:
            smiles_sets: 分子集合字典 {名称: SMILES列表}
            
        Returns:
            比较评估结果
        """
        self.logger.info(f"评估多个分子集合: {list(smiles_sets.keys())}")
        
        # 评估每个集合
        individual_results = {}
        for set_name, smiles_list in smiles_sets.items():
            individual_results[set_name] = self.evaluate_single_set(smiles_list, set_name)
        
        # 进行比较分析
        comparison_results = self._compare_multiple_sets(smiles_sets)
        
        # 整合结果
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'individual_results': individual_results,
            'comparison_results': comparison_results,
            'ranking': self._rank_sets(individual_results)
        }
        
        return evaluation_results
    
    def _compare_multiple_sets(self, smiles_sets: Dict[str, List[str]]) -> Dict[str, any]:
        """比较多个分子集合
        
        Args:
            smiles_sets: 分子集合字典
            
        Returns:
            比较结果
        """
        set_names = list(smiles_sets.keys())
        comparison_results = {}
        
        # 两两比较
        for i, name1 in enumerate(set_names):
            for j, name2 in enumerate(set_names[i+1:], i+1):
                comparison_key = f"{name1}_vs_{name2}"
                
                # 多样性比较
                diversity_comparison = self.diversity_metrics.compare_diversity(
                    smiles_sets[name1], smiles_sets[name2], [name1, name2]
                )
                
                # 有效性比较
                validity_comparison = self.validity_metrics.compare_validity(
                    smiles_sets[name1], smiles_sets[name2], [name1, name2]
                )
                
                comparison_results[comparison_key] = {
                    'diversity_comparison': diversity_comparison,
                    'validity_comparison': validity_comparison
                }
        
        return comparison_results
    
    def _rank_sets(self, individual_results: Dict[str, Dict]) -> Dict[str, any]:
        """对分子集合进行排名
        
        Args:
            individual_results: 个体评估结果
            
        Returns:
            排名结果
        """
        rankings = {
            'diversity_ranking': [],
            'validity_ranking': [],
            'overall_ranking': []
        }
        
        # 提取分数
        set_scores = {}
        for set_name, results in individual_results.items():
            diversity_score = results['diversity']['comprehensive_diversity_score']
            validity_score = results['validity']['comprehensive_validity_score']
            overall_score = (diversity_score + validity_score) / 2
            
            set_scores[set_name] = {
                'diversity_score': diversity_score,
                'validity_score': validity_score,
                'overall_score': overall_score
            }
        
        # 排名
        rankings['diversity_ranking'] = sorted(
            set_scores.items(), 
            key=lambda x: x[1]['diversity_score'], 
            reverse=True
        )
        
        rankings['validity_ranking'] = sorted(
            set_scores.items(), 
            key=lambda x: x[1]['validity_score'], 
            reverse=True
        )
        
        rankings['overall_ranking'] = sorted(
            set_scores.items(), 
            key=lambda x: x[1]['overall_score'], 
            reverse=True
        )
        
        return rankings
    
    def _generate_summary(self, diversity_results: Dict, validity_results: Dict) -> Dict[str, any]:
        """生成评估摘要
        
        Args:
            diversity_results: 多样性结果
            validity_results: 有效性结果
            
        Returns:
            评估摘要
        """
        summary = {
            'key_metrics': {
                'validity_rate': validity_results['basic_validity']['validity_rate'],
                'uniqueness_rate': validity_results['basic_validity']['uniqueness_rate'],
                'lipinski_compliance': validity_results['lipinski_compliance']['lipinski_compliance_rate'],
                'qed_score': validity_results['qed_scores']['mean_qed'],
                'tanimoto_diversity': diversity_results['tanimoto_diversity']['tanimoto_diversity_index'],
                'scaffold_diversity': diversity_results['scaffold_diversity']['scaffold_diversity_ratio']
            },
            'overall_scores': {
                'diversity_score': diversity_results['comprehensive_diversity_score'],
                'validity_score': validity_results['comprehensive_validity_score'],
                'combined_score': (diversity_results['comprehensive_diversity_score'] + 
                                 validity_results['comprehensive_validity_score']) / 2
            },
            'quality_assessment': self._assess_quality(
                diversity_results['comprehensive_diversity_score'],
                validity_results['comprehensive_validity_score']
            )
        }
        
        return summary
    
    def _assess_quality(self, diversity_score: float, validity_score: float) -> str:
        """评估质量等级
        
        Args:
            diversity_score: 多样性分数
            validity_score: 有效性分数
            
        Returns:
            质量等级
        """
        combined_score = (diversity_score + validity_score) / 2
        
        if combined_score >= 0.8:
            return "Excellent"
        elif combined_score >= 0.6:
            return "Good"
        elif combined_score >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def generate_visualizations(self, evaluation_results: Dict, 
                              save_plots: bool = True) -> Dict[str, any]:
        """生成可视化图表
        
        Args:
            evaluation_results: 评估结果
            save_plots: 是否保存图表
            
        Returns:
            可视化结果
        """
        self.logger.info("生成评估可视化图表")
        
        plots = {}
        
        # 如果是单个集合的结果
        if 'individual_results' not in evaluation_results:
            plots.update(self._plot_single_set_results(evaluation_results, save_plots))
        else:
            # 多个集合的比较结果
            plots.update(self._plot_multiple_sets_results(evaluation_results, save_plots))
        
        return plots
    
    def _plot_single_set_results(self, results: Dict, save_plots: bool) -> Dict[str, str]:
        """绘制单个集合的结果
        
        Args:
            results: 评估结果
            save_plots: 是否保存图表
            
        Returns:
            图表文件路径
        """
        plots = {}
        
        # 1. 关键指标雷达图
        fig, ax = plt.subplots(figsize=self.figure_size, subplot_kw=dict(projection='polar'))
        
        metrics = list(results['summary']['key_metrics'].keys())
        values = list(results['summary']['key_metrics'].values())
        
        # 添加第一个点到末尾以闭合雷达图
        metrics += [metrics[0]]
        values += [values[0]]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=True)
        
        ax.plot(angles, values, 'o-', linewidth=2, label=results['set_name'])
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics[:-1])
        ax.set_ylim(0, 1)
        ax.set_title(f"Key Metrics - {results['set_name']}", size=16, weight='bold')
        ax.legend()
        
        if save_plots:
            plot_path = os.path.join(self.output_dir, f"{results['set_name']}_radar.png")
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plots['radar_chart'] = plot_path
        
        plt.close()
        
        # 2. 分子属性分布图
        if 'detailed_results' in results['validity']['lipinski_compliance']:
            detailed_results = results['validity']['lipinski_compliance']['detailed_results']
            if detailed_results:
                fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
                
                # 分子量分布
                mw_values = [r['molecular_weight'] for r in detailed_results]
                axes[0, 0].hist(mw_values, bins=20, alpha=0.7, edgecolor='black')
                axes[0, 0].set_xlabel('Molecular Weight')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('Molecular Weight Distribution')
                
                # LogP分布
                logp_values = [r['logp'] for r in detailed_results]
                axes[0, 1].hist(logp_values, bins=20, alpha=0.7, edgecolor='black')
                axes[0, 1].set_xlabel('LogP')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('LogP Distribution')
                
                # 氢键供体分布
                hbd_values = [r['hbd'] for r in detailed_results]
                axes[1, 0].hist(hbd_values, bins=10, alpha=0.7, edgecolor='black')
                axes[1, 0].set_xlabel('H-bond Donors')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('H-bond Donors Distribution')
                
                # 氢键受体分布
                hba_values = [r['hba'] for r in detailed_results]
                axes[1, 1].hist(hba_values, bins=10, alpha=0.7, edgecolor='black')
                axes[1, 1].set_xlabel('H-bond Acceptors')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('H-bond Acceptors Distribution')
                
                plt.tight_layout()
                
                if save_plots:
                    plot_path = os.path.join(self.output_dir, f"{results['set_name']}_properties.png")
                    plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
                    plots['properties_distribution'] = plot_path
                
                plt.close()
        
        return plots
    
    def _plot_multiple_sets_results(self, results: Dict, save_plots: bool) -> Dict[str, str]:
        """绘制多个集合的比较结果
        
        Args:
            results: 评估结果
            save_plots: 是否保存图表
            
        Returns:
            图表文件路径
        """
        plots = {}
        
        # 1. 综合分数比较
        set_names = list(results['individual_results'].keys())
        diversity_scores = [results['individual_results'][name]['diversity']['comprehensive_diversity_score'] 
                          for name in set_names]
        validity_scores = [results['individual_results'][name]['validity']['comprehensive_validity_score'] 
                         for name in set_names]
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        x = np.arange(len(set_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, diversity_scores, width, label='Diversity Score', alpha=0.8)
        bars2 = ax.bar(x + width/2, validity_scores, width, label='Validity Score', alpha=0.8)
        
        ax.set_xlabel('Molecule Sets')
        ax.set_ylabel('Scores')
        ax.set_title('Comprehensive Scores Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(set_names, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.output_dir, "scores_comparison.png")
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plots['scores_comparison'] = plot_path
        
        plt.close()
        
        # 2. 关键指标热力图
        metrics_data = []
        metric_names = ['validity_rate', 'uniqueness_rate', 'lipinski_compliance', 
                       'qed_score', 'tanimoto_diversity', 'scaffold_diversity']
        
        for set_name in set_names:
            row = []
            summary = results['individual_results'][set_name]['summary']['key_metrics']
            for metric in metric_names:
                row.append(summary.get(metric, 0))
            metrics_data.append(row)
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        im = ax.imshow(metrics_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # 设置标签
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(set_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_yticklabels(set_names)
        
        # 添加数值标签
        for i in range(len(set_names)):
            for j in range(len(metric_names)):
                text = ax.text(j, i, f'{metrics_data[i][j]:.3f}',
                             ha="center", va="center", color="black")
        
        ax.set_title("Key Metrics Heatmap")
        plt.colorbar(im)
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.output_dir, "metrics_heatmap.png")
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plots['metrics_heatmap'] = plot_path
        
        plt.close()
        
        return plots
    
    def save_results(self, evaluation_results: Dict, filename: str = None) -> str:
        """保存评估结果
        
        Args:
            evaluation_results: 评估结果
            filename: 文件名
            
        Returns:
            保存路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 处理numpy类型以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 递归转换
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        converted_results = recursive_convert(evaluation_results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"评估结果已保存到: {filepath}")
        return filepath
    
    def generate_report(self, evaluation_results: Dict, 
                       include_plots: bool = True) -> str:
        """生成评估报告
        
        Args:
            evaluation_results: 评估结果
            include_plots: 是否包含图表
            
        Returns:
            报告文件路径
        """
        self.logger.info("生成评估报告")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"evaluation_report_{timestamp}.html"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # 生成可视化图表
        plots = {}
        if include_plots:
            plots = self.generate_visualizations(evaluation_results, save_plots=True)
        
        # 生成HTML报告
        html_content = self._generate_html_report(evaluation_results, plots)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"评估报告已生成: {report_path}")
        return report_path
    
    def _generate_html_report(self, results: Dict, plots: Dict) -> str:
        """生成HTML报告内容
        
        Args:
            results: 评估结果
            plots: 图表路径
            
        Returns:
            HTML内容
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Molecule Generation Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ margin: 10px 0; }}
                .score {{ font-weight: bold; color: #2e7d32; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .plot img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Molecule Generation Evaluation Report</h1>
            <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        # 如果是单个集合的结果
        if 'individual_results' not in results:
            html += self._generate_single_set_html(results, plots)
        else:
            # 多个集合的比较结果
            html += self._generate_multiple_sets_html(results, plots)
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_single_set_html(self, results: Dict, plots: Dict) -> str:
        """生成单个集合的HTML内容"""
        summary = results['summary']
        
        html = f"""
            <h2>Dataset: {results['set_name']}</h2>
            <p><strong>Total Molecules:</strong> {results['total_molecules']}</p>
            
            <h3>Summary</h3>
            <div class="metric">Overall Quality: <span class="score">{summary['quality_assessment']}</span></div>
            <div class="metric">Combined Score: <span class="score">{summary['overall_scores']['combined_score']:.3f}</span></div>
            
            <h3>Key Metrics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        
        for metric, value in summary['key_metrics'].items():
            html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value:.3f}</td></tr>"
        
        html += "</table>"
        
        # 添加图表
        if plots:
            html += "<h3>Visualizations</h3>"
            for plot_name, plot_path in plots.items():
                plot_filename = os.path.basename(plot_path)
                html += f"""
                <div class="plot">
                    <h4>{plot_name.replace('_', ' ').title()}</h4>
                    <img src="{plot_filename}" alt="{plot_name}">
                </div>
                """
        
        return html
    
    def _generate_multiple_sets_html(self, results: Dict, plots: Dict) -> str:
        """生成多个集合的HTML内容"""
        html = "<h2>Multiple Sets Comparison</h2>"
        
        # 排名表
        if 'ranking' in results:
            html += "<h3>Overall Ranking</h3><table><tr><th>Rank</th><th>Dataset</th><th>Score</th></tr>"
            for i, (set_name, scores) in enumerate(results['ranking']['overall_ranking'], 1):
                html += f"<tr><td>{i}</td><td>{set_name}</td><td>{scores['overall_score']:.3f}</td></tr>"
            html += "</table>"
        
        # 个体结果摘要
        html += "<h3>Individual Results Summary</h3><table>"
        html += "<tr><th>Dataset</th><th>Validity Rate</th><th>Diversity Score</th><th>Overall Score</th></tr>"
        
        for set_name, set_results in results['individual_results'].items():
            validity_rate = set_results['summary']['key_metrics']['validity_rate']
            diversity_score = set_results['diversity']['comprehensive_diversity_score']
            overall_score = set_results['summary']['overall_scores']['combined_score']
            
            html += f"""
            <tr>
                <td>{set_name}</td>
                <td>{validity_rate:.3f}</td>
                <td>{diversity_score:.3f}</td>
                <td>{overall_score:.3f}</td>
            </tr>
            """
        
        html += "</table>"
        
        # 添加图表
        if plots:
            html += "<h3>Visualizations</h3>"
            for plot_name, plot_path in plots.items():
                plot_filename = os.path.basename(plot_path)
                html += f"""
                <div class="plot">
                    <h4>{plot_name.replace('_', ' ').title()}</h4>
                    <img src="{plot_filename}" alt="{plot_name}">
                </div>
                """
        
        return html
    
    def run_full_evaluation(self, smiles_sets: Dict[str, List[str]], 
                          generate_report: bool = True) -> Dict[str, any]:
        """运行完整评估流程
        
        Args:
            smiles_sets: 分子集合字典
            generate_report: 是否生成报告
            
        Returns:
            完整评估结果
        """
        self.logger.info("开始完整评估流程")
        
        # 评估分子集合
        if len(smiles_sets) == 1:
            set_name, smiles_list = next(iter(smiles_sets.items()))
            evaluation_results = self.evaluate_single_set(smiles_list, set_name)
        else:
            evaluation_results = self.evaluate_multiple_sets(smiles_sets)
        
        # 保存结果
        results_path = self.save_results(evaluation_results)
        
        # 生成报告
        if generate_report:
            report_path = self.generate_report(evaluation_results)
            evaluation_results['report_path'] = report_path
        
        evaluation_results['results_path'] = results_path
        
        self.logger.info("完整评估流程完成")
        return evaluation_results