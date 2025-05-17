"""
模型训练模块 - 提供通用的模型训练和管理功能
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import joblib

from .lstm_model import LSTMModel
from .gru_model import GRUModel
from .ensemble_model import EnsembleModel

def train_model(X, y, model_type='lstm', input_shape=None, test_size=0.2, val_size=0.2, 
                epochs=100, batch_size=32, model_path=None, return_history=False, verbose=1):
    """
    训练模型的通用函数
    
    参数:
        X: 特征数据
        y: 标签数据
        model_type: 模型类型('lstm', 'gru', 'ensemble')
        input_shape: 输入形状，如果为None则自动推断
        test_size: 测试集比例
        val_size: 验证集比例
        epochs: 训练轮数
        batch_size: 批量大小
        model_path: 模型保存路径
        return_history: 是否返回训练历史
        verbose: 显示详细程度
        
    返回:
        model: 训练好的模型
        history: (可选) 训练历史
        test_results: 测试集评估结果
    """
    # 检查输入数据
    if len(X) == 0:
        raise ValueError("输入数据X不能为空")
    
    # 推断输入形状
    if input_shape is None:
        if len(X.shape) == 3:
            input_shape = (X.shape[1], X.shape[2])
        else:
            raise ValueError("无法推断输入形状，请手动指定input_shape")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # 从训练集中划分验证集
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size_adjusted, shuffle=False)
    
    # 创建模型
    if model_type.lower() == 'lstm':
        model = LSTMModel(input_shape=input_shape)
        model.build_advanced_model()  # 使用高级模型
    elif model_type.lower() == 'gru':
        model = GRUModel(input_shape=input_shape)
        model.build_advanced_model()  # 使用高级模型
    elif model_type.lower() == 'ensemble':
        # 创建基础LSTM模型
        base_model = LSTMModel(input_shape=input_shape)
        base_model.build_advanced_model()
        
        # 创建集成模型
        model = EnsembleModel(base_model=base_model.model, num_models=3)
        model.build_ensemble(input_shape)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 训练模型
    if model_type.lower() in ['lstm', 'gru']:
        # 单模型训练
        if model_path:
            model_save_path = model_path
        else:
            model_save_path = None
        
        history = model.train(X_train, y_train, X_val, y_val, 
                            epochs=epochs, batch_size=batch_size, 
                            model_path=model_save_path, verbose=verbose)
    else:
        # 集成模型训练
        history = model.train(X_train, y_train, X_val, y_val, 
                            epochs=epochs, batch_size=batch_size, verbose=verbose)
        
        # 保存集成模型
        if model_path:
            model.save(model_path)
    
    # 评估模型
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    if model_type.lower() in ['lstm', 'gru']:
        y_pred = model.predict(X_test).flatten()
    else:
        y_pred, _, _, _ = model.predict(X_test)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    test_results = {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2)
    }
    
    # 如果指定了模型路径，保存评估结果
    if model_path:
        # 确保目录存在
        if model_type.lower() in ['lstm', 'gru']:
            result_dir = os.path.dirname(model_path)
        else:
            result_dir = model_path
            
        os.makedirs(result_dir, exist_ok=True)
        
        with open(os.path.join(result_dir, 'evaluation.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
    
    if return_history:
        return model, history, test_results
    else:
        return model, test_results

def load_model(model_path, model_type='lstm'):
    """
    加载训练好的模型
    
    参数:
        model_path: 模型保存路径
        model_type: 模型类型('lstm', 'gru', 'ensemble')
        
    返回:
        model: 加载的模型
    """
    if model_type.lower() == 'lstm':
        return LSTMModel.load(model_path)
    elif model_type.lower() == 'gru':
        return GRUModel.load(model_path)
    elif model_type.lower() == 'ensemble':
        return EnsembleModel.load(model_path)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def save_model_with_scalers(model, scalers, model_dir):
    """
    保存模型及其对应的标准化器
    
    参数:
        model: 模型对象
        scalers: 标准化器字典
        model_dir: 保存目录
    """
    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    if hasattr(model, 'save'):
        # 如果模型有save方法
        model.save(os.path.join(model_dir, 'model.keras'))
    elif isinstance(model, (LSTMModel, GRUModel)):
        # 如果是自定义模型类的实例
        model.save(os.path.join(model_dir, 'model.keras'))
    elif isinstance(model, EnsembleModel):
        # 如果是集成模型
        model.save(model_dir)
    else:
        # 其他情况，尝试使用joblib保存
        joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    
    # 保存标准化器
    joblib.dump(scalers, os.path.join(model_dir, 'scalers.pkl'))
    
    # 保存模型类型信息
    if isinstance(model, LSTMModel):
        model_type = 'lstm'
    elif isinstance(model, GRUModel):
        model_type = 'gru'
    elif isinstance(model, EnsembleModel):
        model_type = 'ensemble'
    else:
        model_type = 'unknown'
    
    with open(os.path.join(model_dir, 'model_info.json'), 'w') as f:
        json.dump({'model_type': model_type}, f)

def load_model_with_scalers(model_dir):
    """
    加载模型及其对应的标准化器
    
    参数:
        model_dir: 保存目录
        
    返回:
        model: 加载的模型
        scalers: 加载的标准化器
    """
    # 读取模型类型信息
    try:
        with open(os.path.join(model_dir, 'model_info.json'), 'r') as f:
            info = json.load(f)
            model_type = info.get('model_type', 'unknown')
    except FileNotFoundError:
        # 如果信息文件不存在，尝试推断模型类型
        if os.path.exists(os.path.join(model_dir, 'model.keras')):
            model_type = 'lstm'  # 默认为LSTM
        elif os.path.exists(os.path.join(model_dir, 'ensemble_info.json')):
            model_type = 'ensemble'
        else:
            model_type = 'unknown'
    
    # 加载模型
    if model_type == 'lstm':
        model = LSTMModel.load(os.path.join(model_dir, 'model.keras'))
    elif model_type == 'gru':
        model = GRUModel.load(os.path.join(model_dir, 'model.keras'))
    elif model_type == 'ensemble':
        model = EnsembleModel.load(model_dir)
    else:
        # 尝试使用joblib加载
        try:
            model = joblib.load(os.path.join(model_dir, 'model.pkl'))
        except:
            raise ValueError(f"无法加载模型，未知的模型类型: {model_type}")
    
    # 加载标准化器
    try:
        scalers = joblib.load(os.path.join(model_dir, 'scalers.pkl'))
    except:
        scalers = None
    
    return model, scalers

def train_student_model(student_df, seq_length=5, model_type='lstm', output_dir=None):
    """
    为单个学生训练模型
    
    参数:
        student_df: 学生的成绩数据框
        seq_length: 序列长度
        model_type: 模型类型
        output_dir: 输出目录
        
    返回:
        model: 训练好的模型
        scalers: 标准化器
        metrics: 评估指标
    """
    from ..data_processor import DataProcessor
    
    # 创建数据处理器
    processor = DataProcessor(seq_length=seq_length)
    
    # 处理数据
    features_df = processor.create_features(student_df)
    scaled_features = processor.scale_features(features_df)
    X, y = processor.create_sequences(scaled_features)
    
    # 如果数据不足
    if len(X) == 0:
        print(f"警告：该学生的数据不足以进行有效训练(仅有{len(features_df)}条记录)")
        return None, processor.scalers, None
    
    # 训练模型
    model_path = os.path.join(output_dir, 'model.keras') if output_dir else None
    model, metrics = train_model(
        X, y, 
        model_type=model_type,
        model_path=model_path,
        epochs=100,
        batch_size=32
    )
    
    # 如果指定了输出目录，保存标准化器
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(processor.scalers, os.path.join(output_dir, 'scalers.pkl'))
    
    return model, processor.scalers, metrics