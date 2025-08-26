"""DAR预测器模块

用于预测药物抗体比值（Drug-to-Antibody Ratio）。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from ..features.sequence_encoder import SequenceEncoder
from ..features.molecule_features import MoleculeFeatureExtractor

class DARPredictor:
    """DAR预测器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 模型参数
        self.model_type = config.get('model_type', 'neural_network')  # 'neural_network' or 'random_forest'
        self.hidden_dims = config.get('hidden_dims', [512, 256, 128])
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 特征提取器
        self.sequence_encoder = SequenceEncoder(config.get('sequence_encoder', {}))
        self.molecule_extractor = MoleculeFeatureExtractor(config.get('molecule_features', {}))
        
        # 模型
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # DAR范围
        self.min_dar = config.get('min_dar', 0.0)
        self.max_dar = config.get('max_dar', 8.0)
        
        self.logger.info(f"DAR预测器初始化完成，模型类型: {self.model_type}")
    
    def build_model(self, input_dim: int):
        """构建预测模型
        
        Args:
            input_dim: 输入特征维度
        """
        if self.model_type == 'neural_network':
            self.model = DARNeuralNetwork(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                dropout_rate=self.dropout_rate
            ).to(self.device)
            
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
            
        self.logger.info(f"模型构建完成，输入维度: {input_dim}")
    
    def prepare_features(self, protein_sequences: List[str],
                        molecule_smiles: List[str],
                        linker_smiles: Optional[List[str]] = None) -> np.ndarray:
        """准备特征
        
        Args:
            protein_sequences: 蛋白质序列列表
            molecule_smiles: 分子SMILES列表
            linker_smiles: Linker SMILES列表（可选）
            
        Returns:
            特征矩阵
        """
        self.logger.info(f"准备 {len(protein_sequences)} 个样本的特征")
        
        # 编码蛋白质序列
        protein_features = []
        for seq in protein_sequences:
            features = self.sequence_encoder.encode_sequence(seq, method='combined')
            protein_features.append(features)
        protein_features = np.array(protein_features)
        
        # 提取分子特征
        molecule_features = self.molecule_extractor.extract_features(molecule_smiles)
        
        # 合并特征
        combined_features = np.concatenate([protein_features, molecule_features], axis=1)
        
        # 如果有Linker特征，也加入
        if linker_smiles:
            linker_features = self.molecule_extractor.extract_features(linker_smiles)
            combined_features = np.concatenate([combined_features, linker_features], axis=1)
            
        self.logger.info(f"特征准备完成，特征维度: {combined_features.shape[1]}")
        return combined_features
    
    def fit(self, protein_sequences: List[str],
           molecule_smiles: List[str],
           dar_values: List[float],
           linker_smiles: Optional[List[str]] = None,
           validation_split: float = 0.2) -> Dict[str, float]:
        """训练DAR预测模型
        
        Args:
            protein_sequences: 蛋白质序列列表
            molecule_smiles: 分子SMILES列表
            dar_values: DAR值列表
            linker_smiles: Linker SMILES列表（可选）
            validation_split: 验证集比例
            
        Returns:
            训练结果
        """
        self.logger.info("开始训练DAR预测模型")
        
        # 准备特征
        features = self.prepare_features(protein_sequences, molecule_smiles, linker_smiles)
        targets = np.array(dar_values)
        
        # 数据标准化
        features_scaled = self.scaler.fit_transform(features)
        
        # 构建模型
        if self.model is None:
            self.build_model(features_scaled.shape[1])
            
        # 划分训练和验证集
        n_samples = len(features_scaled)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        X_train, X_val = features_scaled[train_indices], features_scaled[val_indices]
        y_train, y_val = targets[train_indices], targets[val_indices]
        
        # 训练模型
        if self.model_type == 'neural_network':
            results = self._train_neural_network(X_train, y_train, X_val, y_val)
        else:
            results = self._train_random_forest(X_train, y_train, X_val, y_val)
            
        self.is_fitted = True
        self.logger.info("DAR预测模型训练完成")
        
        return results
    
    def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """训练神经网络
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            训练结果
        """
        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # 训练参数
        num_epochs = self.config.get('num_epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        
        best_val_loss = float('inf')
        patience = self.config.get('patience', 10)
        patience_counter = 0
        
        # 训练循环
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            
            # 批次训练
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                predictions = self.model(batch_X).squeeze()
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            # 验证
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val_tensor).squeeze()
                val_loss = criterion(val_predictions, y_val_tensor).item()
                
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                self.logger.info(f"早停于第 {epoch+1} 轮")
                break
                
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(X_train_tensor)*batch_size:.4f}, Val Loss: {val_loss:.4f}")
        
        # 计算最终指标
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(X_train_tensor).squeeze().cpu().numpy()
            val_pred = self.model(X_val_tensor).squeeze().cpu().numpy()
            
        train_mse = mean_squared_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        return {
            'train_mse': train_mse,
            'train_r2': train_r2,
            'val_mse': val_mse,
            'val_r2': val_r2
        }
    
    def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """训练随机森林
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            训练结果
        """
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 预测
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # 计算指标
        train_mse = mean_squared_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        return {
            'train_mse': train_mse,
            'train_r2': train_r2,
            'val_mse': val_mse,
            'val_r2': val_r2
        }
    
    def predict(self, protein_sequences: List[str],
               molecule_smiles: List[str],
               linker_smiles: Optional[List[str]] = None) -> np.ndarray:
        """预测DAR值
        
        Args:
            protein_sequences: 蛋白质序列列表
            molecule_smiles: 分子SMILES列表
            linker_smiles: Linker SMILES列表（可选）
            
        Returns:
            预测的DAR值
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
            
        # 准备特征
        features = self.prepare_features(protein_sequences, molecule_smiles, linker_smiles)
        features_scaled = self.scaler.transform(features)
        
        # 预测
        if self.model_type == 'neural_network':
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled).to(self.device)
                predictions = self.model(features_tensor).squeeze().cpu().numpy()
        else:
            predictions = self.model.predict(features_scaled)
            
        # 限制DAR范围
        predictions = np.clip(predictions, self.min_dar, self.max_dar)
        
        return predictions
    
    def predict_with_uncertainty(self, protein_sequences: List[str],
                               molecule_smiles: List[str],
                               linker_smiles: Optional[List[str]] = None,
                               n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """预测DAR值及不确定性
        
        Args:
            protein_sequences: 蛋白质序列列表
            molecule_smiles: 分子SMILES列表
            linker_smiles: Linker SMILES列表（可选）
            n_samples: 采样次数（用于估计不确定性）
            
        Returns:
            (预测均值, 预测标准差)
        """
        if self.model_type == 'random_forest':
            # 随机森林可以通过树的预测方差估计不确定性
            features = self.prepare_features(protein_sequences, molecule_smiles, linker_smiles)
            features_scaled = self.scaler.transform(features)
            
            # 获取所有树的预测
            tree_predictions = np.array([tree.predict(features_scaled) for tree in self.model.estimators_])
            
            mean_pred = np.mean(tree_predictions, axis=0)
            std_pred = np.std(tree_predictions, axis=0)
            
            return mean_pred, std_pred
            
        else:
            # 神经网络使用Dropout采样估计不确定性
            self.model.train()  # 启用Dropout
            
            features = self.prepare_features(protein_sequences, molecule_smiles, linker_smiles)
            features_scaled = self.scaler.transform(features)
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            
            predictions = []
            
            with torch.no_grad():
                for _ in range(n_samples):
                    pred = self.model(features_tensor).squeeze().cpu().numpy()
                    predictions.append(pred)
                    
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            return mean_pred, std_pred
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """获取特征重要性
        
        Returns:
            特征重要性数组
        """
        if self.model_type == 'random_forest' and self.is_fitted:
            return self.model.feature_importances_
        else:
            self.logger.warning("特征重要性仅适用于已训练的随机森林模型")
            return None
    
    def save_model(self, filepath: str):
        """保存模型
        
        Args:
            filepath: 保存路径
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
            
        model_data = {
            'model_type': self.model_type,
            'scaler': self.scaler,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        if self.model_type == 'neural_network':
            model_data['model_state_dict'] = self.model.state_dict()
        else:
            model_data['model'] = self.model
            
        joblib.dump(model_data, filepath)
        self.logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型
        
        Args:
            filepath: 模型路径
        """
        model_data = joblib.load(filepath)
        
        self.model_type = model_data['model_type']
        self.scaler = model_data['scaler']
        self.config.update(model_data['config'])
        self.is_fitted = model_data['is_fitted']
        
        if self.model_type == 'neural_network':
            # 重建神经网络结构
            input_dim = len(self.scaler.mean_)  # 从scaler获取输入维度
            self.build_model(input_dim)
            self.model.load_state_dict(model_data['model_state_dict'])
        else:
            self.model = model_data['model']
            
        self.logger.info(f"模型已从 {filepath} 加载")

class DARNeuralNetwork(nn.Module):
    """DAR预测神经网络"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入特征
            
        Returns:
            预测输出
        """
        return self.network(x)