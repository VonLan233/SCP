"""
LSTM模型 - 学生成绩预测系统的核心模型
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

class LSTMModel:
    """
    LSTM模型类，用于学生成绩预测
    """
    def __init__(self, input_shape=(5, 1), units=64, dropout_rate=0.2, learning_rate=0.001):
        """
        初始化LSTM模型
        
        参数:
            input_shape: 输入数据形状，默认是(5, 1)，表示5个时间步，每步1个特征
            units: LSTM单元数量
            dropout_rate: Dropout比率，防止过拟合
            learning_rate: 学习率
        """
        self.input_shape = input_shape
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self):
        """
        构建标准LSTM模型
        
        返回:
            model: 构建好的Keras模型
        """
        # 使用Sequential API创建模型
        model = Sequential()
        
        # 添加LSTM层
        model.add(LSTM(units=self.units, 
                       return_sequences=True, 
                       input_shape=self.input_shape))
        model.add(Dropout(self.dropout_rate))
        
        # 添加第二个LSTM层
        model.add(LSTM(units=self.units//2))
        model.add(Dropout(self.dropout_rate))
        
        # 添加输出层
        model.add(Dense(1))
        
        # 编译模型
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        
        return model
    
    def build_advanced_model(self):
        """
        构建高级LSTM模型，包含双向LSTM和Batch Normalization
        
        返回:
            model: 构建好的高级Keras模型
        """
        # 使用函数式API构建更复杂的模型
        inputs = Input(shape=self.input_shape)
        
        # 双向LSTM增强特征提取能力
        x = Bidirectional(LSTM(units=self.units, 
                              return_sequences=True,
                              kernel_regularizer=l2(0.001)))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # 第二层LSTM
        x = Bidirectional(LSTM(units=self.units//2, 
                              return_sequences=False))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # 全连接层
        x = Dense(units=32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate/2)(x)
        
        # 输出层
        outputs = Dense(1)(x)
        
        # 创建模型
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        
        # 替换当前模型
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, model_path=None, verbose=1):
        """
        训练模型
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批量大小
            model_path: 模型保存路径
            verbose: 显示详细程度
            
        返回:
            history: 训练历史
        """
        # 创建回调函数
        callbacks = []
        
        # 早停，避免过拟合
        early_stopping = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=15,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # 学习率调度
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )
        callbacks.append(reduce_lr)
        
        # 如果指定了模型保存路径
        if model_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # 模型检查点
            checkpoint = ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True
            )
            callbacks.append(checkpoint)
        
        # 训练模型
        if X_val is not None and y_val is not None:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
        
        return history
    
    def predict(self, X, verbose=0):
        """
        使用模型进行预测
        
        参数:
            X: 输入特征
            verbose: 显示详细程度
            
        返回:
            predictions: 预测结果
        """
        return self.model.predict(X, verbose=verbose)
    
    def save(self, filepath):
        """
        保存模型
        
        参数:
            filepath: 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath):
        """
        加载模型
        
        参数:
            filepath: 模型文件路径
            
        返回:
            model_instance: 加载的模型实例
        """
        model = tf.keras.models.load_model(filepath)
        
        # 创建实例并设置模型
        instance = cls()
        instance.model = model
        
        # 推断输入形状
        input_shape = model.input_shape[1:]
        instance.input_shape = input_shape
        
        return instance
    
    def summary(self):
        """
        打印模型摘要
        """
        self.model.summary()
    
    def monte_carlo_dropout_predict(self, X, n_samples=100):
        """
        使用Monte Carlo Dropout进行不确定性估计
        
        参数:
            X: 输入特征
            n_samples: 采样次数
            
        返回:
            mean_predictions: 平均预测值
            std_predictions: 预测标准差
            lower_bound: 预测下界
            upper_bound: 预测上界
        """
        # 创建一个新模型，基于原有模型的结构
        mc_model = tf.keras.models.clone_model(self.model)
        mc_model.set_weights(self.model.get_weights())
        
        # 多次预测
        predictions = []
        for _ in range(n_samples):
            # 预测一次
            pred = mc_model.predict(X, verbose=0)
            predictions.append(pred.flatten())
        
        # 转换为numpy数组
        predictions = np.array(predictions)  # shape: (n_samples, n_test_samples)
        
        # 计算每个测试样本的预测均值和标准差
        mean_predictions = np.mean(predictions, axis=0)
        std_predictions = np.std(predictions, axis=0)
        
        # 计算95%置信区间
        confidence_multiplier = 1.96
        lower_bound = mean_predictions - confidence_multiplier * std_predictions
        upper_bound = mean_predictions + confidence_multiplier * std_predictions
        
        return mean_predictions, std_predictions, lower_bound, upper_bound