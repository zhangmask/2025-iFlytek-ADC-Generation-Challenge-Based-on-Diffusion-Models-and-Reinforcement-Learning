#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码质量和结构优化模块

提供代码质量检查、类型注解验证、文档字符串生成和代码规范检查功能。

Author: AI Developer
Date: 2025
"""

import ast
import inspect
import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import functools
import warnings


class CodeQualityLevel(Enum):
    """代码质量等级"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class QualityMetric:
    """质量指标数据类"""
    name: str
    score: float
    max_score: float
    description: str
    suggestions: List[str]


@dataclass
class CodeAnalysisResult:
    """代码分析结果数据类"""
    file_path: str
    overall_score: float
    quality_level: CodeQualityLevel
    metrics: List[QualityMetric]
    issues: List[str]
    suggestions: List[str]


class TypeAnnotationChecker:
    """类型注解检查器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def check_function_annotations(self, func: Callable) -> Dict[str, Any]:
        """检查函数的类型注解
        
        Args:
            func: 要检查的函数
            
        Returns:
            检查结果字典
        """
        sig = inspect.signature(func)
        result = {
            'function_name': func.__name__,
            'has_return_annotation': sig.return_annotation != inspect.Signature.empty,
            'parameter_annotations': {},
            'missing_annotations': [],
            'annotation_score': 0.0
        }
        
        total_params = len(sig.parameters)
        annotated_params = 0
        
        for param_name, param in sig.parameters.items():
            has_annotation = param.annotation != inspect.Parameter.empty
            result['parameter_annotations'][param_name] = has_annotation
            
            if has_annotation:
                annotated_params += 1
            else:
                result['missing_annotations'].append(param_name)
        
        # 计算注解完整性分数
        if total_params > 0:
            param_score = annotated_params / total_params
        else:
            param_score = 1.0
        
        return_score = 1.0 if result['has_return_annotation'] else 0.0
        result['annotation_score'] = (param_score + return_score) / 2
        
        return result
    
    def check_class_annotations(self, cls: Type) -> Dict[str, Any]:
        """检查类的类型注解
        
        Args:
            cls: 要检查的类
            
        Returns:
            检查结果字典
        """
        result = {
            'class_name': cls.__name__,
            'method_annotations': {},
            'attribute_annotations': {},
            'overall_score': 0.0
        }
        
        # 检查方法注解
        methods = inspect.getmembers(cls, predicate=inspect.isfunction)
        method_scores = []
        
        for method_name, method in methods:
            if not method_name.startswith('_') or method_name in ['__init__', '__call__']:
                method_result = self.check_function_annotations(method)
                result['method_annotations'][method_name] = method_result
                method_scores.append(method_result['annotation_score'])
        
        # 检查属性注解
        if hasattr(cls, '__annotations__'):
            result['attribute_annotations'] = cls.__annotations__
        
        # 计算总体分数
        if method_scores:
            result['overall_score'] = sum(method_scores) / len(method_scores)
        
        return result


class DocstringChecker:
    """文档字符串检查器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def check_function_docstring(self, func: Callable) -> Dict[str, Any]:
        """检查函数的文档字符串
        
        Args:
            func: 要检查的函数
            
        Returns:
            检查结果字典
        """
        docstring = inspect.getdoc(func)
        result = {
            'function_name': func.__name__,
            'has_docstring': docstring is not None,
            'docstring_length': len(docstring) if docstring else 0,
            'has_args_section': False,
            'has_returns_section': False,
            'has_raises_section': False,
            'docstring_score': 0.0
        }
        
        if docstring:
            # 检查是否包含Args、Returns、Raises部分
            result['has_args_section'] = 'Args:' in docstring or 'Arguments:' in docstring
            result['has_returns_section'] = 'Returns:' in docstring or 'Return:' in docstring
            result['has_raises_section'] = 'Raises:' in docstring or 'Raise:' in docstring
            
            # 计算文档字符串质量分数
            score = 0.0
            if result['docstring_length'] > 10:  # 基础分数
                score += 0.4
            if result['has_args_section']:
                score += 0.2
            if result['has_returns_section']:
                score += 0.2
            if result['has_raises_section']:
                score += 0.1
            if result['docstring_length'] > 100:  # 详细文档
                score += 0.1
            
            result['docstring_score'] = min(score, 1.0)
        
        return result
    
    def generate_docstring_template(self, func: Callable) -> str:
        """为函数生成文档字符串模板
        
        Args:
            func: 要生成文档字符串的函数
            
        Returns:
            文档字符串模板
        """
        sig = inspect.signature(func)
        
        template = f'"""TODO: 添加函数描述\n\n'
        
        # 添加参数部分
        if sig.parameters:
            template += '    Args:\n'
            for param_name, param in sig.parameters.items():
                if param_name != 'self':
                    param_type = 'TODO: 类型'
                    if param.annotation != inspect.Parameter.empty:
                        param_type = str(param.annotation)
                    template += f'        {param_name} ({param_type}): TODO: 参数描述\n'
        
        # 添加返回值部分
        if sig.return_annotation != inspect.Signature.empty:
            template += '\n    Returns:\n'
            template += f'        {sig.return_annotation}: TODO: 返回值描述\n'
        
        # 添加异常部分
        template += '\n    Raises:\n'
        template += '        TODO: 可能抛出的异常\n'
        
        template += '    """'
        
        return template


class CodeComplexityAnalyzer:
    """代码复杂度分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_cyclomatic_complexity(self, source_code: str) -> int:
        """计算圈复杂度
        
        Args:
            source_code: 源代码字符串
            
        Returns:
            圈复杂度值
        """
        try:
            tree = ast.parse(source_code)
            complexity = 1  # 基础复杂度
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                    complexity += 1
                elif isinstance(node, ast.ExceptHandler):
                    complexity += 1
                elif isinstance(node, (ast.And, ast.Or)):
                    complexity += 1
                elif isinstance(node, ast.comprehension):
                    complexity += 1
            
            return complexity
        except SyntaxError:
            self.logger.warning("代码语法错误，无法计算复杂度")
            return 0
    
    def analyze_function_complexity(self, func: Callable) -> Dict[str, Any]:
        """分析函数复杂度
        
        Args:
            func: 要分析的函数
            
        Returns:
            复杂度分析结果
        """
        try:
            source = inspect.getsource(func)
            complexity = self.calculate_cyclomatic_complexity(source)
            
            # 计算代码行数
            lines = source.strip().split('\n')
            total_lines = len(lines)
            code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            
            result = {
                'function_name': func.__name__,
                'cyclomatic_complexity': complexity,
                'total_lines': total_lines,
                'code_lines': code_lines,
                'complexity_level': self._get_complexity_level(complexity),
                'suggestions': self._get_complexity_suggestions(complexity, code_lines)
            }
            
            return result
        except (OSError, TypeError):
            return {
                'function_name': func.__name__,
                'error': '无法获取源代码'
            }
    
    def _get_complexity_level(self, complexity: int) -> str:
        """根据复杂度值获取复杂度等级"""
        if complexity <= 5:
            return 'low'
        elif complexity <= 10:
            return 'medium'
        elif complexity <= 20:
            return 'high'
        else:
            return 'very_high'
    
    def _get_complexity_suggestions(self, complexity: int, code_lines: int) -> List[str]:
        """根据复杂度提供优化建议"""
        suggestions = []
        
        if complexity > 10:
            suggestions.append('考虑将函数拆分为更小的函数')
            suggestions.append('使用策略模式或状态模式简化条件逻辑')
        
        if code_lines > 50:
            suggestions.append('函数过长，建议拆分为多个小函数')
        
        if complexity > 20:
            suggestions.append('复杂度过高，需要重构')
            suggestions.append('考虑使用设计模式简化代码结构')
        
        return suggestions


class CodeQualityAnalyzer:
    """代码质量分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.type_checker = TypeAnnotationChecker()
        self.doc_checker = DocstringChecker()
        self.complexity_analyzer = CodeComplexityAnalyzer()
    
    def analyze_file(self, file_path: str) -> CodeAnalysisResult:
        """分析文件的代码质量
        
        Args:
            file_path: 文件路径
            
        Returns:
            代码分析结果
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # 解析AST
            tree = ast.parse(source_code)
            
            metrics = []
            issues = []
            suggestions = []
            
            # 分析函数
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_metrics = self._analyze_function_node(node, source_code)
                    metrics.extend(func_metrics)
            
            # 计算总体分数
            if metrics:
                overall_score = sum(metric.score for metric in metrics) / len(metrics)
            else:
                overall_score = 0.0
            
            # 确定质量等级
            quality_level = self._get_quality_level(overall_score)
            
            return CodeAnalysisResult(
                file_path=file_path,
                overall_score=overall_score,
                quality_level=quality_level,
                metrics=metrics,
                issues=issues,
                suggestions=suggestions
            )
        
        except Exception as e:
            self.logger.error(f"分析文件 {file_path} 时出错: {str(e)}")
            return CodeAnalysisResult(
                file_path=file_path,
                overall_score=0.0,
                quality_level=CodeQualityLevel.POOR,
                metrics=[],
                issues=[f"分析失败: {str(e)}"],
                suggestions=["检查文件语法和编码"]
            )
    
    def _analyze_function_node(self, node: ast.FunctionDef, source_code: str) -> List[QualityMetric]:
        """分析函数节点"""
        metrics = []
        
        # 类型注解检查
        type_score = self._check_function_type_annotations(node)
        metrics.append(QualityMetric(
            name="type_annotations",
            score=type_score,
            max_score=1.0,
            description="类型注解完整性",
            suggestions=["添加缺失的类型注解"] if type_score < 1.0 else []
        ))
        
        # 文档字符串检查
        doc_score = self._check_function_docstring(node)
        metrics.append(QualityMetric(
            name="docstring",
            score=doc_score,
            max_score=1.0,
            description="文档字符串质量",
            suggestions=["添加或改进文档字符串"] if doc_score < 0.8 else []
        ))
        
        # 复杂度检查
        complexity_score = self._check_function_complexity(node, source_code)
        metrics.append(QualityMetric(
            name="complexity",
            score=complexity_score,
            max_score=1.0,
            description="代码复杂度",
            suggestions=["降低函数复杂度"] if complexity_score < 0.7 else []
        ))
        
        return metrics
    
    def _check_function_type_annotations(self, node: ast.FunctionDef) -> float:
        """检查函数类型注解"""
        total_params = len(node.args.args)
        annotated_params = sum(1 for arg in node.args.args if arg.annotation is not None)
        
        if total_params > 0:
            param_score = annotated_params / total_params
        else:
            param_score = 1.0
        
        return_score = 1.0 if node.returns is not None else 0.0
        
        return (param_score + return_score) / 2
    
    def _check_function_docstring(self, node: ast.FunctionDef) -> float:
        """检查函数文档字符串"""
        if not node.body or not isinstance(node.body[0], ast.Expr):
            return 0.0
        
        if not isinstance(node.body[0].value, ast.Constant):
            return 0.0
        
        docstring = node.body[0].value.value
        if not isinstance(docstring, str):
            return 0.0
        
        score = 0.0
        if len(docstring) > 10:
            score += 0.4
        if 'Args:' in docstring or 'Arguments:' in docstring:
            score += 0.2
        if 'Returns:' in docstring or 'Return:' in docstring:
            score += 0.2
        if 'Raises:' in docstring:
            score += 0.1
        if len(docstring) > 100:
            score += 0.1
        
        return min(score, 1.0)
    
    def _check_function_complexity(self, node: ast.FunctionDef, source_code: str) -> float:
        """检查函数复杂度"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        # 复杂度评分（反向）
        if complexity <= 5:
            return 1.0
        elif complexity <= 10:
            return 0.8
        elif complexity <= 15:
            return 0.6
        elif complexity <= 20:
            return 0.4
        else:
            return 0.2
    
    def _get_quality_level(self, score: float) -> CodeQualityLevel:
        """根据分数确定质量等级"""
        if score >= 0.9:
            return CodeQualityLevel.EXCELLENT
        elif score >= 0.7:
            return CodeQualityLevel.GOOD
        elif score >= 0.5:
            return CodeQualityLevel.FAIR
        else:
            return CodeQualityLevel.POOR


def quality_check(min_score: float = 0.7):
    """代码质量检查装饰器
    
    Args:
        min_score: 最低质量分数要求
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            analyzer = CodeQualityAnalyzer()
            
            # 检查类型注解
            type_result = analyzer.type_checker.check_function_annotations(func)
            if type_result['annotation_score'] < min_score:
                warnings.warn(
                    f"函数 {func.__name__} 类型注解不完整 (分数: {type_result['annotation_score']:.2f})",
                    UserWarning
                )
            
            # 检查文档字符串
            doc_result = analyzer.doc_checker.check_function_docstring(func)
            if doc_result['docstring_score'] < min_score:
                warnings.warn(
                    f"函数 {func.__name__} 文档字符串质量较低 (分数: {doc_result['docstring_score']:.2f})",
                    UserWarning
                )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def deprecated(reason: str = None):
    """标记函数为已弃用的装饰器
    
    Args:
        reason: 弃用原因
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"函数 {func.__name__} 已弃用"
            if reason:
                message += f": {reason}"
            
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_types(*type_args, **type_kwargs):
    """运行时类型验证装饰器
    
    Args:
        *type_args: 位置参数的类型
        **type_kwargs: 关键字参数的类型
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 验证位置参数类型
            for i, (arg, expected_type) in enumerate(zip(args, type_args)):
                if not isinstance(arg, expected_type):
                    raise TypeError(
                        f"参数 {i} 类型错误: 期望 {expected_type.__name__}, 实际 {type(arg).__name__}"
                    )
            
            # 验证关键字参数类型
            for key, value in kwargs.items():
                if key in type_kwargs:
                    expected_type = type_kwargs[key]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"参数 {key} 类型错误: 期望 {expected_type.__name__}, 实际 {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator