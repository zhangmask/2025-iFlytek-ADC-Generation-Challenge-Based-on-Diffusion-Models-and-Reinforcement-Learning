"""配置管理模块

定义项目的各种配置参数。
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json
from dataclasses import dataclass, asdict

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

@dataclass
class DataConfig:
    """数据相关配置"""
    # 数据路径
    data_dir: str = "data"
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    
    # 数据处理参数
    max_sequence_length: int = 1000
    min_sequence_length: int = 10
    max_smiles_length: int = 500
    min_smiles_length: int = 5
    
    # 数据划分
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    
    # 数据清洗
    remove_duplicates: bool = True
    missing_threshold: float = 0.5
    handle_missing: str = "drop"  # 'drop', 'fill', 'interpolate'
    
    # 异常值处理
    outlier_method: str = "iqr"  # 'iqr', 'zscore'
    outlier_threshold: float = 1.5

@dataclass
class FeatureConfig:
    """特征工程配置"""
    # 蛋白质序列编码
    sequence_encoding_methods: list = None
    sequence_normalize: bool = True
    
    # 分子特征提取
    fingerprint_types: list = None
    fingerprint_size: int = 2048
    molecular_descriptors: list = None
    
    # 特征融合
    fusion_method: str = "concatenation"  # 'concatenation', 'weighted', 'pca', 'attention', 'multimodal'
    fusion_dim: int = 512
    
    def __post_init__(self):
        if self.sequence_encoding_methods is None:
            self.sequence_encoding_methods = ["one_hot", "amino_acid_properties", "kmer", "statistical"]
        
        if self.fingerprint_types is None:
            self.fingerprint_types = ["morgan", "rdkit", "maccs"]
        
        if self.molecular_descriptors is None:
            self.molecular_descriptors = [
                "MolWt", "LogP", "NumHDonors", "NumHAcceptors", 
                "TPSA", "NumRotatableBonds", "NumAromaticRings"
            ]

@dataclass
class ModelConfig:
    """模型配置"""
    # 基础模型配置
    embedding_dim: int = 512
    hidden_dim: int = 256
    
    # 扩散模型配置
    diffusion_timesteps: int = 1000
    noise_schedule: str = "linear"  # 'linear', 'cosine'
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # U-Net配置
    unet_dim: int = 128
    unet_dim_mults: tuple = (1, 2, 4, 8)
    unet_channels: int = 3
    unet_resnet_block_groups: int = 8
    
    # 强化学习配置
    action_dim: int = 100  # 匹配config.yaml中的action_space_size
    state_dim: int = 512   # 匹配强化学习环境的state_dim
    rl_lr: float = 3e-4
    rl_gamma: float = 0.99
    rl_tau: float = 0.005
    rl_buffer_size: int = 100000
    rl_batch_size: int = 256
    rl_update_frequency: int = 4
    
    # 策略网络配置
    policy_hidden_dims: tuple = (256, 256)
    value_hidden_dims: tuple = (256, 256)
    
    # DAR预测配置
    dar_model_type: str = "neural_network"  # 'neural_network', 'random_forest'
    dar_hidden_dims: tuple = (512, 256, 128)
    dar_dropout: float = 0.2

@dataclass
class TrainingConfig:
    """训练配置"""
    # 通用训练参数
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    device: str = "cuda"  # 'cuda', 'cpu', 'auto'
    
    # 优化器配置
    optimizer: str = "adam"  # 'adam', 'adamw', 'sgd'
    weight_decay: float = 1e-5
    scheduler: str = "cosine"  # 'cosine', 'step', 'plateau'
    
    # 早停配置
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-6
    
    # 检查点配置
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10
    
    # 验证配置
    validation_frequency: int = 5
    
    # 扩散模型特定参数
    diffusion_lr: float = 1e-4
    diffusion_epochs: int = 200
    
    # 强化学习特定参数
    rl_episodes: int = 1000
    rl_max_steps: int = 100
    
    # DAR预测特定参数
    dar_epochs: int = 50
    dar_lr: float = 1e-3

@dataclass
class GenerationConfig:
    """生成配置"""
    # 分子生成参数
    num_samples: int = 1000
    guidance_scale: float = 7.5
    
    # Linker生成参数
    linker_types: list = None
    max_linker_length: int = 20
    
    # 优化参数
    optimization_steps: int = 100
    target_properties: dict = None
    
    # 多样性控制
    diversity_threshold: float = 0.7
    max_attempts: int = 10000
    
    def __post_init__(self):
        if self.linker_types is None:
            self.linker_types = ["peptide", "peg", "alkyl", "aromatic"]
        
        if self.target_properties is None:
            self.target_properties = {
                "molecular_weight": (200, 800),
                "logp": (-2, 5),
                "qed": (0.5, 1.0),
                "sa_score": (1, 4)
            }

@dataclass
class EvaluationConfig:
    """评估配置"""
    # 多样性评估
    diversity_metrics: list = None
    
    # 有效性评估
    validity_metrics: list = None
    
    # 可视化配置
    plot_format: str = "png"  # 'png', 'pdf', 'svg'
    plot_dpi: int = 300
    figure_size: tuple = (12, 8)
    
    # 报告配置
    generate_html_report: bool = True
    include_interactive_plots: bool = True
    
    def __post_init__(self):
        if self.diversity_metrics is None:
            self.diversity_metrics = [
                "tanimoto_diversity", "scaffold_diversity", 
                "descriptor_diversity", "cluster_diversity"
            ]
        
        if self.validity_metrics is None:
            self.validity_metrics = [
                "validity_rate", "lipinski_compliance", 
                "qed_score", "sa_score", "pains_alerts"
            ]

@dataclass
class Config:
    """主配置类"""
    data: DataConfig = None
    features: FeatureConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    generation: GenerationConfig = None
    evaluation: EvaluationConfig = None
    
    # 项目配置
    project_name: str = "ADC生成挑战赛"
    version: str = "1.0.0"
    
    # 路径配置
    output_dir: str = "outputs"
    model_dir: str = "models"
    log_dir: str = "logs"
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 随机种子
    seed: int = 42
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.features is None:
            self.features = FeatureConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.generation is None:
            self.generation = GenerationConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save(self, filepath: str) -> None:
        """保存配置到文件"""
        config_dict = self.to_dict()
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix.lower() == '.yaml' or filepath.suffix.lower() == '.yml':
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """从文件加载配置"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"配置文件不存在: {filepath}")
        
        if filepath.suffix.lower() == '.yaml' or filepath.suffix.lower() == '.yml':
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """从字典创建配置"""
        # 创建子配置对象
        data_config = DataConfig(**config_dict.get('data', {}))
        features_config = FeatureConfig(**config_dict.get('features', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        generation_config = GenerationConfig(**config_dict.get('generation', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        # 提取主配置参数
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['data', 'features', 'model', 'training', 'generation', 'evaluation']}
        
        return cls(
            data=data_config,
            features=features_config,
            model=model_config,
            training=training_config,
            generation=generation_config,
            evaluation=evaluation_config,
            **main_config
        )
    
    def update(self, **kwargs) -> None:
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # 尝试更新子配置
                for config_name in ['data', 'features', 'model', 'training', 'generation', 'evaluation']:
                    config_obj = getattr(self, config_name)
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
                        break
                else:
                    raise ValueError(f"未知的配置参数: {key}")
    
    def get_paths(self) -> Dict[str, Path]:
        """获取所有路径配置"""
        base_paths = {
            'project_root': PROJECT_ROOT,
            'data_dir': PROJECT_ROOT / self.data.data_dir,
            'output_dir': PROJECT_ROOT / self.output_dir,
            'model_dir': PROJECT_ROOT / self.model_dir,
            'log_dir': PROJECT_ROOT / self.log_dir,
        }
        
        # 创建目录
        for path in base_paths.values():
            if path != PROJECT_ROOT:
                path.mkdir(parents=True, exist_ok=True)
        
        return base_paths

# 默认配置实例
default_config = Config()

# 配置管理器
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config = None
    
    @property
    def config(self) -> Config:
        """获取配置"""
        if self._config is None:
            if self.config_path and Path(self.config_path).exists():
                self._config = Config.load(self.config_path)
            else:
                self._config = Config()
        return self._config
    
    def update_config(self, **kwargs) -> None:
        """更新配置"""
        self.config.update(**kwargs)
    
    def save_config(self, filepath: Optional[str] = None) -> None:
        """保存配置"""
        if filepath is None:
            filepath = self.config_path or "config.yaml"
        self.config.save(filepath)
    
    def reset_config(self) -> None:
        """重置配置"""
        self._config = Config()

# 全局配置管理器
config_manager = ConfigManager()

# 便捷函数
def get_config() -> Config:
    """获取全局配置"""
    return config_manager.config

def update_config(**kwargs) -> None:
    """更新全局配置"""
    config_manager.update_config(**kwargs)

def save_config(filepath: Optional[str] = None) -> None:
    """保存全局配置"""
    config_manager.save_config(filepath)

def load_config(filepath: str) -> Config:
    """加载配置文件"""
    global config_manager
    config_manager = ConfigManager(filepath)
    return config_manager.config

# 环境变量配置
def setup_environment():
    """设置环境变量"""
    config = get_config()
    
    # 设置随机种子
    import random
    import numpy as np
    import torch
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    
    # 设置设备
    if config.training.device == "auto":
        config.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建必要的目录
    paths = config.get_paths()
    
    return config

if __name__ == "__main__":
    # 示例用法
    config = Config()
    
    # 保存默认配置
    config.save("config.yaml")
    
    # 加载配置
    loaded_config = Config.load("config.yaml")
    
    print("配置加载成功！")
    print(f"项目名称: {loaded_config.project_name}")
    print(f"数据目录: {loaded_config.data.data_dir}")
    print(f"批次大小: {loaded_config.training.batch_size}")