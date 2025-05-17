"""
学生成绩预测系统 - 机器学习模型包
"""

from predictor.improved_lstm import StudentScorePredictor
from predictor.lstm_model import LSTMModel
from predictor.gru_model import GRUModel
from predictor.ensemble_model import EnsembleModel
from predictor.model_utils import (
    save_model, 
    load_model, 
    train_test_split_time_series,
    create_sequences,
    apply_monte_carlo_dropout,
    normalize_predictions,
    evaluate_model
)
from predictor.model_training import (
    train_model,
    load_model_with_scalers,
    save_model_with_scalers,
    train_student_model
)

# 定义可用模型类型
AVAILABLE_MODELS = {
    'lstm': LSTMModel,
    'gru': GRUModel,
    'ensemble': EnsembleModel,
    'student_predictor': StudentScorePredictor
}

# 定义模型默认参数
DEFAULT_MODEL_PARAMS = {
    'lstm': {
        'timeSteps': 5,
        'units': 64,
        'dropout_rate': 0.2,
        'learning_rate': 0.001
    },
    'gru': {
        'timeSteps': 5,
        'units': 64,
        'dropout_rate': 0.2,
        'learning_rate': 0.001
    },
    'ensemble': {
        'timeSteps': 5,
        'num_models': 3
    },
    'student_predictor': {
        'seq_length': 5
    }
}

# 模型训练默认配置
DEFAULT_TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'test_size': 0.2,
    'val_size': 0.2,
    'early_stopping_patience': 15,
    'use_monte_carlo_dropout': True
}

def create_model(model_type='lstm', params=None):
    """
    创建指定类型的预测模型
    
    参数:
        model_type: 模型类型 ('lstm', 'gru', 'ensemble', 'student_predictor')
        params: 模型参数字典
        
    返回:
        model: 创建的模型实例
    """
    if model_type not in AVAILABLE_MODELS:
        raise ValueError(f"不支持的模型类型: {model_type}. 可用类型: {list(AVAILABLE_MODELS.keys())}")
    
    # 如果没有提供参数，使用默认参数
    if params is None:
        params = DEFAULT_MODEL_PARAMS.get(model_type, {})
    
    # 创建模型实例
    model_class = AVAILABLE_MODELS[model_type]
    
    if model_type == 'lstm' or model_type == 'gru':
        input_shape = (params.get('timeSteps', 5), 1)  # 默认单特征
        units = params.get('units', 64)
        dropout_rate = params.get('dropout_rate', 0.2)
        learning_rate = params.get('learning_rate', 0.001)
        
        model = model_class(
            input_shape=input_shape,
            units=units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
    elif model_type == 'ensemble':
        # 创建基础模型
        base_model_type = params.get('base_model_type', 'lstm')
        base_params = params.get('base_params', DEFAULT_MODEL_PARAMS.get(base_model_type, {}))
        base_model = create_model(base_model_type, base_params)
        
        if hasattr(base_model, 'model'):
            base_keras_model = base_model.model
        else:
            base_keras_model = base_model
            
        num_models = params.get('num_models', 3)
        model = model_class(base_model=base_keras_model, num_models=num_models)
        
    elif model_type == 'student_predictor':
        seq_length = params.get('seq_length', 5)
        model_path = params.get('model_path', 'models')
        model = model_class(seq_length=seq_length, model_path=model_path)
    
    return model

def get_model_info():
    """
    获取所有可用模型的信息
    
    返回:
        info: 包含模型信息的字典
    """
    info = {
        'available_models': list(AVAILABLE_MODELS.keys()),
        'default_params': DEFAULT_MODEL_PARAMS,
        'training_config': DEFAULT_TRAINING_CONFIG
    }
    return info

# 版本信息
__version__ = '1.0.0'