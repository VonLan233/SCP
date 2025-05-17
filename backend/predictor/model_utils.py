"""
模型工具函数 - 提供模型操作的通用工具
"""

import os
import json
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split

def save_model(model, model_dir, model_name="model.keras", scalers=None, metadata=None):
    """
    保存模型及相关数据
    
    参数:
        model: 模型对象
        model_dir: 保存目录
        model_name: 模型文件名
        scalers: 标准化器字典
        metadata: 元数据字典
    """
    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    if hasattr(model, 'save'):
        # 如果模型有save方法
        model.save(os.path.join(model_dir, model_name))
    elif isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
        # 如果是Keras模型
        model.save(os.path.join(model_dir, model_name))
    else:
        # 其他情况，尝试使用joblib保存
        joblib.dump(model, os.path.join(model_dir, model_name.replace('.keras', '.pkl')))
    
    # 保存标准化器
    if scalers is not None:
        joblib.dump(scalers, os.path.join(model_dir, 'scalers.pkl'))
    
    # 保存元数据
    if metadata is not None:
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"模型已保存到 {model_dir}")

def load_model(model_dir, model_name="model.keras", load_scalers=True, custom_objects=None):
    """
    加载模型及相关数据
    
    参数:
        model_dir: 保存目录
        model_name: 模型文件名
        load_scalers: 是否加载标准化器
        custom_objects: 自定义对象字典，用于加载具有自定义层的模型
        
    返回:
        model: 加载的模型
        scalers: 加载的标准化器
        metadata: 加载的元数据
    """
    model_path = os.path.join(model_dir, model_name)
    scalers = None
    metadata = None
    
    # 加载模型
    if os.path.exists(model_path):
        # 尝试加载Keras模型
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        except:
            # 如果失败，尝试加载joblib模型
            model = joblib.load(model_path.replace('.keras', '.pkl'))
    else:
        # 尝试加载joblib模型
        joblib_path = model_path.replace('.keras', '.pkl')
        if os.path.exists(joblib_path):
            model = joblib.load(joblib_path)
        else:
            raise FileNotFoundError(f"找不到模型文件: {model_path} 或 {joblib_path}")
    
    # 加载标准化器
    if load_scalers and os.path.exists(os.path.join(model_dir, 'scalers.pkl')):
        scalers = joblib.load(os.path.join(model_dir, 'scalers.pkl'))
    
    # 加载元数据
    if os.path.exists(os.path.join(model_dir, 'metadata.json')):
        with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
    
    print(f"已加载模型: {model_dir}")
    
    return model, scalers, metadata

def train_test_split_time_series(X, y, test_size=0.2, val_size=0.2):
    """
    为时间序列数据执行训练-测试分割
    
    参数:
        X: 特征数据
        y: 标签数据
        test_size: 测试集比例
        val_size: 验证集比例
        
    返回:
        X_train, X_val, X_test: 训练、验证和测试特征
        y_train, y_val, y_test: 训练、验证和测试标签
    """
    # 分割出测试集（保持时间顺序）
    if isinstance(test_size, int):
        # 如果test_size是整数，表示样本数量
        test_idx = test_size
    else:
        # 如果test_size是浮点数，表示比例
        test_idx = int(len(X) * test_size)
    
    X_temp, X_test = X[:-test_idx], X[-test_idx:]
    y_temp, y_test = y[:-test_idx], y[-test_idx:]
    
    # 从剩余数据中分割出验证集
    if isinstance(val_size, int):
        # 如果val_size是整数，表示样本数量
        val_idx = val_size
    else:
        # 如果val_size是浮点数，表示比例
        val_idx = int(len(X_temp) * val_size)
    
    X_train, X_val = X_temp[:-val_idx], X_temp[-val_idx:]
    y_train, y_val = y_temp[:-val_idx], y_temp[-val_idx:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_sequences(data, seq_length):
    """
    将时间序列数据转换为监督学习格式
    
    参数:
        data: 输入序列
        seq_length: 序列长度
        
    返回:
        X: 输入序列
        y: 目标值
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    return np.array(X), np.array(y)

def apply_monte_carlo_dropout(model, X, n_samples=100):
    """
    使用Monte Carlo Dropout进行不确定性估计
    
    参数:
        model: Keras模型
        X: 输入特征
        n_samples: 采样次数
        
    返回:
        mean_predictions: 平均预测值
        std_predictions: 预测标准差
        lower_bound: 预测下界
        upper_bound: 预测上界
    """
    # 收集预测结果
    predictions = []
    
    # 创建一个新模型，让Dropout层在预测时也工作
    mc_model = tf.keras.models.clone_model(model)
    mc_model.set_weights(model.get_weights())
    
    for _ in range(n_samples):
        # 进行一次预测
        pred = mc_model.predict(X, verbose=0)
        predictions.append(pred.flatten())
    
    # 转换为numpy数组
    predictions = np.array(predictions)
    
    # 计算均值和标准差
    mean_predictions = np.mean(predictions, axis=0)
    std_predictions = np.std(predictions, axis=0)
    
    # 计算95%置信区间
    confidence_multiplier = 1.96
    lower_bound = mean_predictions - confidence_multiplier * std_predictions
    upper_bound = mean_predictions + confidence_multiplier * std_predictions
    
    return mean_predictions, std_predictions, lower_bound, upper_bound

def normalize_predictions(predictions, min_val=0, max_val=100):
    """
    将预测结果限制在合理范围内
    
    参数:
        predictions: 预测结果
        min_val: 最小值
        max_val: 最大值
        
    返回:
        normalized_predictions: 归一化后的预测结果
    """
    return np.clip(predictions, min_val, max_val)

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    
    参数:
        model: 模型对象
        X_test: 测试特征
        y_test: 测试标签
        
    返回:
        metrics: 评估指标字典
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # 获取预测结果
    y_pred = model.predict(X_test).flatten()
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2)
    }
    
    return metrics