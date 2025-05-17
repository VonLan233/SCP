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

# 修改模型类型常量，使线性回归成为默认模型
DEFAULT_MODEL_TYPE = "linear_regression"  # 改为线性回归

# 更新模型参数默认值
DEFAULT_MODEL_PARAMS = {
    "linear_regression": {
        "model_path": "predictor/models",
        "confidence_level": 0.95
    },
    "student_predictor": {  # 保留LSTM模型配置但不作为默认
        "seq_length": 5,
        "model_path": "predictor/models"
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

# 创建模型工厂函数
def create_model(model_type=None, model_params=None):
    """
    创建预测模型实例
    
    参数:
        model_type: 模型类型，默认为线性回归
        model_params: 模型参数
        
    返回:
        model: 模型实例
    """
    if model_type is None:
        model_type = DEFAULT_MODEL_TYPE
        
    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS.get(model_type, {})
        
    if model_type == "linear_regression":
        from .linear_regression import LinearRegressionPredictor
        return LinearRegressionPredictor(**model_params)
    elif model_type == "student_predictor":
        from .improved_lstm import StudentScorePredictor
        return StudentScorePredictor(**model_params)
    else:
        # 默认返回线性回归模型
        from .linear_regression import LinearRegressionPredictor
        return LinearRegressionPredictor(**model_params)

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