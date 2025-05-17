"""
集成模型 - 组合多个基础模型进行预测
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, Dropout
from tensorflow.keras.models import Model

class EnsembleModel:
    """
    集成模型类，整合多个基础模型以提高预测准确性
    """
    def __init__(self, base_model=None, num_models=3):
        """
        初始化集成模型
        
        参数:
            base_model: 基础模型，作为集成的起点
            num_models: 集成模型的数量
        """
        self.base_model = base_model
        self.num_models = num_models
        self.models = []
        
        # 如果提供了基础模型，将其添加到模型列表
        if base_model is not None:
            self.models.append(base_model)
        
    def build_ensemble(self, input_shape, model_types=None):
        """
        构建多样化的模型集合
        
        参数:
            input_shape: 输入数据形状
            model_types: 模型类型列表，None则使用默认混合
            
        返回:
            self.models: 构建好的模型列表
        """
        # 默认使用混合模型类型
        if model_types is None:
            model_types = ['lstm', 'gru', 'rnn'] * (self.num_models // 3 + 1)
        
        # 确保有足够的模型类型
        model_types = model_types[:self.num_models]
        
        # 为每种类型创建一个模型
        for i, model_type in enumerate(model_types):
            # 跳过第一个，如果已经有基础模型的话
            if i == 0 and self.base_model is not None:
                continue
                
            if model_type.lower() == 'lstm':
                # LSTM模型
                inputs = Input(shape=input_shape)
                x = LSTM(units=64, return_sequences=True)(inputs)
                x = Dropout(0.3)(x)
                x = LSTM(units=32)(x)
                x = Dropout(0.3)(x)
                outputs = Dense(1)(x)
                model = Model(inputs=inputs, outputs=outputs)
            elif model_type.lower() == 'gru':
                # GRU模型
                inputs = Input(shape=input_shape)
                x = GRU(units=48, return_sequences=True)(inputs)
                x = Dropout(0.3)(x)
                x = GRU(units=24)(x)
                x = Dropout(0.3)(x)
                outputs = Dense(1)(x)
                model = Model(inputs=inputs, outputs=outputs)
            elif model_type.lower() == 'rnn':
                # SimpleRNN模型
                inputs = Input(shape=input_shape)
                x = SimpleRNN(units=32, return_sequences=True)(inputs)
                x = Dropout(0.3)(x)
                x = SimpleRNN(units=16)(x)
                x = Dropout(0.3)(x)
                outputs = Dense(1)(x)
                model = Model(inputs=inputs, outputs=outputs)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 编译模型
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # 添加到模型列表
            self.models.append(model)
        
        return self.models
    
    def create_variants(self, base_model, num_variants=2):
        """
        从基础模型创建变体
        
        参数:
            base_model: 基础模型
            num_variants: 要创建的变体数量
            
        返回:
            variants: 变体模型列表
        """
        variants = []
        
        for i in range(num_variants):
            # 克隆基础模型
            variant = clone_model(base_model.model if hasattr(base_model, 'model') else base_model)
            variant.compile(optimizer='adam', loss='mean_squared_error')
            
            # 随机调整权重
            weights = [w.numpy() for w in variant.weights]
            for j in range(len(weights)):
                # 添加随机噪声，幅度根据层次调整
                noise_scale = 0.1 / (i + 1)  # 逐渐减小噪声
                noise = np.random.normal(0, noise_scale, weights[j].shape)
                weights[j] = weights[j] + noise
            
            # 设置调整后的权重
            variant.set_weights(weights)
            
            variants.append(variant)
        
        return variants
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, verbose=1):
        """
        训练集成中的各个模型
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批量大小
            verbose: 显示详细程度
            
        返回:
            histories: 各模型的训练历史
        """
        histories = []
        
        for i, model in enumerate(self.models):
            print(f"训练模型 {i+1}/{len(self.models)}...")
            
            # 创建早停回调
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
            
            # 训练模型
            if X_val is not None and y_val is not None:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=verbose
                )
            else:
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=verbose
                )
            
            histories.append(history)
        
        return histories
    
    def predict(self, X, verbose=0, return_individual=False):
        """
        使用集成模型进行预测
        
        参数:
            X: 输入特征
            verbose: 显示详细程度
            return_individual: 是否返回各模型的预测结果
            
        返回:
            ensemble_mean: 集成预测均值
            ensemble_std: 集成预测标准差
            lower_bound: 预测下界
            upper_bound: 预测上界
            individual_predictions: (可选) 各模型的预测结果
        """
        if not self.models:
            raise ValueError("没有可用的模型进行预测，请先构建模型")
        
        # 收集各模型的预测结果
        individual_predictions = []
        
        for model in self.models:
            pred = model.predict(X, verbose=verbose)
            individual_predictions.append(pred.flatten())
        
        # 转换为numpy数组，形状为(num_models, num_samples)
        individual_predictions = np.array(individual_predictions)
        
        # 计算集成预测的均值和标准差
        ensemble_mean = np.mean(individual_predictions, axis=0)
        ensemble_std = np.std(individual_predictions, axis=0)
        
        # 计算置信区间
        confidence_multiplier = 1.96  # 95%置信区间
        lower_bound = ensemble_mean - confidence_multiplier * ensemble_std
        upper_bound = ensemble_mean + confidence_multiplier * ensemble_std
        
        if return_individual:
            return ensemble_mean, ensemble_std, lower_bound, upper_bound, individual_predictions
        else:
            return ensemble_mean, ensemble_std, lower_bound, upper_bound
    
    def save(self, directory):
        """
        保存集成中的所有模型
        
        参数:
            directory: 保存目录
        """
        os.makedirs(directory, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_path = os.path.join(directory, f"model_{i}.keras")
            model.save(model_path)
            
        # 保存模型数量信息
        with open(os.path.join(directory, "ensemble_info.json"), 'w') as f:
            json.dump({"num_models": len(self.models)}, f)
    
    @classmethod
    def load(cls, directory):
        """
        加载保存的集成模型
        
        参数:
            directory: 保存目录
            
        返回:
            ensemble: 加载的集成模型实例
        """
        # 读取模型数量信息
        with open(os.path.join(directory, "ensemble_info.json"), 'r') as f:
            info = json.load(f)
            num_models = info["num_models"]
        
        # 创建集成模型实例
        ensemble = cls(num_models=num_models)
        ensemble.models = []
        
        # 加载各个模型
        for i in range(num_models):
            model_path = os.path.join(directory, f"model_{i}.keras")
            model = tf.keras.models.load_model(model_path)
            ensemble.models.append(model)
        
        return ensemble