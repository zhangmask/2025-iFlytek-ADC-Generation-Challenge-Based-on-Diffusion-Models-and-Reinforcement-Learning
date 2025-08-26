# ADC生成挑战赛 🧬

基于扩散模型和强化学习的抗体药物偶联物(ADC)生成项目

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 目录

- [项目概述](#项目概述)
- [特色功能](#特色功能)
- [系统架构](#系统架构)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [配置说明](#配置说明)
- [模块详解](#模块详解)
- [性能优化](#性能优化)
- [故障排除](#故障排除)
- [开发指南](#开发指南)
- [更新日志](#更新日志)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 🎯 项目概述

本项目是一个基于深度学习的抗体药物偶联物(ADC)生成系统，结合了最先进的扩散模型和强化学习技术。项目旨在生成高质量、多样化的ADC分子，并预测其药物抗体比值(DAR)，为药物发现和开发提供强有力的计算工具。

### 🎨 特色功能

- **🔬 智能分子生成**: 基于扩散模型的高质量分子生成
- **🤖 强化学习优化**: 使用RL技术优化分子属性
- **🔗 Linker设计**: 智能连接子生成和优化
- **📊 DAR预测**: 准确的药物抗体比值预测
- **📈 多样性控制**: 先进的分子多样性评估和控制
- **⚡ 高性能计算**: GPU加速和并行处理支持
- **📋 完整评估**: 全面的分子有效性和多样性评估
- **🎯 可视化分析**: 丰富的数据可视化和结果展示
- **🛠️ 模块化设计**: 高度可扩展的模块化架构
- **📝 详细日志**: 完整的训练和生成过程记录

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    ADC生成系统架构                          │
├─────────────────────────────────────────────────────────────┤
│  数据层     │  特征层     │  模型层     │  生成层     │  评估层  │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────┤
│ DataLoader  │ Protein     │ Diffusion   │ Molecule    │ Diversity│
│ Explorer    │ Encoder     │ Model       │ Generator   │ Metrics │
│ Preprocessor│ Molecular   │ RL Agent    │ Linker      │ Validity │
│             │ Features    │ DAR Model   │ Generator   │ Metrics │
│             │ Fusion      │             │             │ Pipeline │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────┘
```

## 📦 安装指南

### 🔧 环境要求

- **Python**: 3.8 或更高版本
- **操作系统**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **内存**: 建议 16GB 或更多
- **GPU**: NVIDIA GPU (可选，用于加速训练)
- **CUDA**: 11.0+ (如果使用GPU)

### 📥 安装步骤

#### 1. 克隆项目

```bash
git clone <repository-url>
cd ADC生成挑战赛
```

#### 2. 创建虚拟环境

```bash
# 使用 conda (推荐)
conda create -n adc-gen python=3.8
conda activate adc-gen

# 或使用 venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

#### 3. 安装依赖

```bash
# 安装核心依赖
pip install -r requirements_core.txt

# 安装完整依赖（包括可视化和开发工具）
pip install -r requirements.txt
```

#### 4. GPU支持 (可选)

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 5. 验证安装

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import rdkit; print('RDKit安装成功')"
```

## 🚀 快速开始

### 📊 准备数据

1. 将训练数据 `train.csv` 和测试数据 `test.csv` 放入 `data/` 目录
2. 确保数据格式正确（包含蛋白质序列和SMILES列）

### ⚡ 一键运行

```bash
# 运行完整流程（数据处理 + 训练 + 生成 + 评估）
python main.py --mode full

# 查看帮助
python main.py --help
```

### 📁 查看结果

运行完成后，结果将保存在以下目录：

- `outputs/`: 所有输出结果
- `models/`: 训练好的模型
- `submit/`: 提交文件
- `logs/`: 运行日志

## 📖 详细使用说明

### 🎛️ 命令行参数

```bash
python main.py [OPTIONS]

选项:
  --mode {full,train,generate,evaluate}  运行模式 [默认: full]
  --config PATH                          配置文件路径 [默认: config.yaml]
  --data-dir PATH                        数据目录 [默认: data]
  --output-dir PATH                      输出目录 [默认: outputs]
  --model-dir PATH                       模型目录 [默认: models]
  --device {auto,cuda,cpu}               计算设备 [默认: auto]
  --batch-size INT                       批次大小 [默认: 32]
  --num-epochs INT                       训练轮数 [默认: 100]
  --learning-rate FLOAT                  学习率 [默认: 1e-4]
  --num-samples INT                      生成样本数 [默认: 1000]
  --verbose                              详细输出
  --debug                                调试模式
  --help                                 显示帮助信息
```

### 🔄 分步执行

#### 1. 仅训练模型

```bash
python main.py --mode train --num-epochs 200 --batch-size 64
```

#### 2. 仅生成分子

```bash
python main.py --mode generate --num-samples 5000
```

#### 3. 仅评估结果

```bash
python main.py --mode evaluate
```

### 📋 生成提交文件

```bash
# 使用默认配置生成提交文件
python generate_submission.py

# 使用自定义参数
python generate_submission.py --num-samples 10000 --batch-size 128
```

## ⚙️ 配置说明

项目使用YAML配置文件管理所有参数。主要配置文件：

- `config.yaml`: 主配置文件
- `config/config.yaml`: 备用配置文件

### 📝 配置文件结构

```yaml
# 数据配置
data:
  data_dir: "data"
  train_file: "train.csv"
  test_file: "test.csv"
  max_sequence_length: 1000
  max_smiles_length: 500

# 模型配置
model:
  embedding_dim: 512
  hidden_dim: 256
  diffusion_timesteps: 1000
  noise_schedule: "linear"
  unet_dim: 128
  action_space_size: 100
  state_dim: 512

# 训练配置
training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  device: "cuda"
  early_stopping: true
  patience: 10

# 生成配置
generation:
  num_samples: 1000
  guidance_scale: 7.5
  diversity_threshold: 0.7
  max_attempts: 10000

# 评估配置
evaluation:
  diversity_metrics: ["tanimoto", "scaffold", "functional_group"]
  validity_metrics: ["lipinski", "qed", "sa_score"]
  generate_html_report: true
```

### 🎯 自定义配置

```bash
# 创建自定义配置文件
cp config.yaml my_config.yaml

# 编辑配置文件
vim my_config.yaml

# 使用自定义配置运行
python main.py --config my_config.yaml
```

## 🧩 模块详解

### 📊 数据处理模块 (`src/data/`)

- **`data_loader.py`**: 数据加载和验证
- **`data_explorer.py`**: 数据探索和统计分析
- **`data_preprocessor.py`**: 数据清洗和预处理

```python
from src.data import DataLoader, DataExplorer, DataPreprocessor

# 加载数据
loader = DataLoader(config.data)
data = loader.load_data()

# 数据探索
explorer = DataExplorer()
explorer.generate_report(data)

# 数据预处理
preprocessor = DataPreprocessor(config.data)
processed_data = preprocessor.preprocess(data)
```

### 🔧 特征工程模块 (`src/features/`)

- **`sequence_encoder.py`**: 蛋白质序列编码
- **`molecule_features.py`**: 分子特征提取
- **`feature_fusion.py`**: 多模态特征融合

```python
from src.features import SequenceEncoder, MoleculeFeatures, FeatureFusion

# 序列编码
encoder = SequenceEncoder(config.features)
sequence_features = encoder.encode(sequences)

# 分子特征
mol_extractor = MoleculeFeatures(config.features)
mol_features = mol_extractor.extract(smiles)

# 特征融合
fusion = FeatureFusion(config.features)
fused_features = fusion.fuse(sequence_features, mol_features)
```

### 🤖 模型模块 (`src/models/`)

- **`diffusion_model.py`**: 扩散模型实现
- **`reinforcement_learning.py`**: 强化学习智能体

```python
from src.models import DiffusionModel, RLAgent

# 扩散模型
diffusion = DiffusionModel(config.model)
diffusion.train(train_data)

# 强化学习
rl_agent = RLAgent(config.model)
rl_agent.train(environment)
```

### 🎯 生成模块 (`src/generation/`)

- **`molecule_generator.py`**: 分子生成器
- **`linker_generator.py`**: 连接子生成器
- **`dar_predictor.py`**: DAR预测器

```python
from src.generation import MoleculeGenerator, LinkerGenerator, DARPredictor

# 分子生成
generator = MoleculeGenerator(diffusion, rl_agent)
molecules = generator.generate(num_samples=1000)

# 连接子生成
linker_gen = LinkerGenerator(config.generation)
linkers = linker_gen.generate(molecules)

# DAR预测
dar_predictor = DARPredictor(config.model)
dar_values = dar_predictor.predict(molecules)
```

### 📈 评估模块 (`src/evaluation/`)

- **`diversity_metrics.py`**: 多样性评估指标
- **`validity_metrics.py`**: 有效性评估指标
- **`evaluation_pipeline.py`**: 综合评估流程

```python
from src.evaluation import DiversityMetrics, ValidityMetrics, EvaluationPipeline

# 评估流程
evaluator = EvaluationPipeline(config.evaluation)
results = evaluator.evaluate(generated_molecules)
```

### 🛠️ 工具模块 (`src/utils/`)

- **`error_handler.py`**: 错误处理和异常管理
- **`visualization.py`**: 训练监控和可视化
- **`code_quality.py`**: 代码质量检查
- **`config_manager.py`**: 配置管理
- **`logger.py`**: 日志系统

## ⚡ 性能优化

### 🚀 GPU加速

```python
# 自动检测GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 多GPU训练
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

### 💾 内存优化

```yaml
# 配置文件中的内存优化设置
training:
  batch_size: 16  # 减小批次大小
  gradient_accumulation_steps: 4  # 梯度累积
  mixed_precision: true  # 混合精度训练
  dataloader_num_workers: 4  # 数据加载并行
```

### 🔄 并行处理

```python
# 特征提取并行化
from multiprocessing import Pool

def parallel_feature_extraction(smiles_list, num_processes=4):
    with Pool(num_processes) as pool:
        features = pool.map(extract_features, smiles_list)
    return features
```

### 📊 性能监控

```python
# 使用训练监控器
from src.utils.visualization import TrainingMonitor

monitor = TrainingMonitor()
monitor.log_metrics({
    'loss': loss.item(),
    'accuracy': accuracy,
    'memory_usage': torch.cuda.memory_allocated()
})
```

## 🔧 故障排除

### ❗ 常见问题

#### 1. CUDA内存不足

```bash
# 错误信息
RuntimeError: CUDA out of memory

# 解决方案
# 1. 减小批次大小
python main.py --batch-size 16

# 2. 启用梯度检查点
# 在配置文件中设置 gradient_checkpointing: true

# 3. 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"
```

#### 2. RDKit安装问题

```bash
# 错误信息
ModuleNotFoundError: No module named 'rdkit'

# 解决方案
# 使用conda安装（推荐）
conda install -c conda-forge rdkit

# 或使用pip
pip install rdkit-pypi
```

#### 3. 数据格式错误

```bash
# 错误信息
KeyError: 'protein_sequence' or 'smiles'

# 解决方案
# 检查CSV文件列名，确保包含必要的列
python -c "import pandas as pd; print(pd.read_csv('data/train.csv').columns.tolist())"
```

#### 4. 模型加载失败

```bash
# 错误信息
RuntimeError: size mismatch

# 解决方案
# 删除旧的模型文件，重新训练
rm -rf models/*.pth
python main.py --mode train
```

### 🐛 调试模式

```bash
# 启用调试模式
python main.py --debug --verbose

# 查看详细日志
tail -f logs/adc_pipeline_*.log
```

### 📋 系统检查

```bash
# 运行系统检查脚本
python -c "
print('=== 系统信息 ===')
import sys, torch, platform
print(f'Python版本: {sys.version}')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'操作系统: {platform.system()} {platform.release()}')
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU型号: {torch.cuda.get_device_name(0)}')
    print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

## 👨‍💻 开发指南

### 🏗️ 项目结构

```
ADC生成挑战赛/
├── src/                          # 源代码
│   ├── data/                     # 数据处理
│   ├── features/                 # 特征工程
│   ├── models/                   # 模型定义
│   ├── generation/               # 生成模块
│   ├── evaluation/               # 评估模块
│   └── utils/                    # 工具模块
├── tests/                        # 测试文件
├── config/                       # 配置文件
├── data/                         # 数据目录
├── models/                       # 模型保存
├── outputs/                      # 输出结果
├── logs/                         # 日志文件
└── submit/                       # 提交文件
```

### 🧪 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_core_modules.py -v

# 运行性能测试
pytest tests/test_performance.py -v

# 生成测试覆盖率报告
pytest --cov=src tests/
```

### 📝 代码规范

```bash
# 代码格式化
black src/ tests/

# 代码检查
flake8 src/ tests/

# 类型检查
mypy src/

# 导入排序
isort src/ tests/
```

### 🔍 代码质量检查

```bash
# 运行代码质量检查
python -m src.utils.code_quality

# 或使用内置工具
python -c "from src.utils.code_quality import CodeQualityChecker; CodeQualityChecker().run_all_checks()"
```

### 📊 添加新功能

1. **创建新模块**

```python
# src/new_module/new_feature.py
class NewFeature:
    def __init__(self, config):
        self.config = config
    
    def process(self, data):
        # 实现新功能
        pass
```

2. **添加配置**

```yaml
# config.yaml
new_feature:
  parameter1: value1
  parameter2: value2
```

3. **编写测试**

```python
# tests/test_new_feature.py
import pytest
from src.new_module.new_feature import NewFeature

def test_new_feature():
    feature = NewFeature(config)
    result = feature.process(test_data)
    assert result is not None
```

4. **更新文档**

更新README.md和相关文档，说明新功能的使用方法。

## 📅 更新日志

### v2.0.0 (2025-01-26)

#### 🎉 新功能
- ✨ 添加了完整的错误处理和异常管理系统
- ✨ 实现了训练监控和可视化功能
- ✨ 添加了代码质量检查工具
- ✨ 支持多种特征融合策略
- ✨ 实现了并行处理和内存优化

#### 🚀 性能改进
- ⚡ 优化了分子生成算法，提升了30%的生成速度
- ⚡ 改进了GPU内存使用效率
- ⚡ 优化了数据加载和预处理流程
- ⚡ 实现了批处理优化

#### 🐛 Bug修复
- 🔧 修复了模型加载时的尺寸不匹配问题
- 🔧 解决了训练过程中的内存泄漏
- 🔧 修复了配置文件解析错误
- 🔧 改进了错误信息的可读性

#### 📚 文档更新
- 📖 完善了安装指南和使用说明
- 📖 添加了详细的API文档
- 📖 增加了故障排除指南
- 📖 更新了配置文件说明

### v1.0.0 (2024-12-01)

#### 🎉 初始版本
- 🎯 实现了基础的扩散模型框架
- 🤖 集成了强化学习优化
- 🔗 添加了Linker生成功能
- 📊 实现了DAR预测
- 📈 完成了基础评估系统

## 🤝 贡献指南

我们欢迎所有形式的贡献！请遵循以下步骤：

### 📋 贡献流程

1. **Fork项目**

```bash
# 在GitHub上Fork项目
# 然后克隆你的Fork
git clone https://github.com/your-username/ADC生成挑战赛.git
cd ADC生成挑战赛
```

2. **创建分支**

```bash
# 创建新的功能分支
git checkout -b feature/amazing-feature

# 或创建修复分支
git checkout -b fix/bug-fix
```

3. **开发和测试**

```bash
# 进行开发
# 运行测试确保代码质量
pytest tests/ -v

# 代码格式化
black src/ tests/
flake8 src/ tests/
```

4. **提交更改**

```bash
# 添加更改
git add .

# 提交更改（使用清晰的提交信息）
git commit -m "feat: 添加新的分子生成算法"

# 推送到你的Fork
git push origin feature/amazing-feature
```

5. **创建Pull Request**

在GitHub上创建Pull Request，详细描述你的更改。

### 📝 提交信息规范

使用[Conventional Commits](https://www.conventionalcommits.org/)规范：

- `feat:` 新功能
- `fix:` Bug修复
- `docs:` 文档更新
- `style:` 代码格式化
- `refactor:` 代码重构
- `test:` 测试相关
- `chore:` 构建过程或辅助工具的变动

### 🎯 贡献领域

- 🐛 Bug报告和修复
- ✨ 新功能开发
- 📚 文档改进
- 🧪 测试用例添加
- ⚡ 性能优化
- 🎨 UI/UX改进
- 🌐 国际化支持

### 📞 联系方式

- 📧 邮箱: [your-email@example.com](mailto:your-email@example.com)
- 💬 讨论: [GitHub Discussions](https://github.com/your-repo/discussions)
- 🐛 问题报告: [GitHub Issues](https://github.com/your-repo/issues)

## 📄 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

```
MIT License

Copyright (c) 2024 ADC生成挑战赛项目

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMplied, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 致谢

感谢以下开源项目和社区的支持：

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [RDKit](https://www.rdkit.org/) - 化学信息学工具包
- [scikit-learn](https://scikit-learn.org/) - 机器学习库
- [Plotly](https://plotly.com/) - 数据可视化
- [Transformers](https://huggingface.co/transformers/) - 预训练模型
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - 强化学习

特别感谢所有贡献者和用户的支持！

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给我们一个星标！ ⭐**

[🏠 主页](https://github.com/your-repo) |
[📖 文档](https://your-docs-site.com) |
[🐛 问题反馈](https://github.com/your-repo/issues) |
[💬 讨论](https://github.com/your-repo/discussions)

</div>

---

> **免责声明**: 本项目仅用于研究和教育目的。生成的分子和预测结果需要进一步的实验验证才能用于实际的药物开发应用。使用本项目进行任何商业或临床应用时，请确保遵循相关的法律法规和伦理准则。