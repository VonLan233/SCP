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

# 模型训练默认配置
DEFAULT_TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'test_size': 0.2,
    'val_size': 0.2,
    'early_stopping_patience': 15,
    'use_monte_carlo_dropout': True
}

# 在 predictor/__init__.py 文件中

# 默认模型类型
DEFAULT_MODEL_TYPE = "linear_regression"

# 更新 DEFAULT_MODEL_PARAMS 的定义
DEFAULT_MODEL_PARAMS = {
    "linear_regression": {
        "model_path": "predictor/models",
        "confidence_level": 0.95
    },
    "student_predictor": {
        "seq_length": 5,
        "model_path": "predictor/models"
    }
}

# 确保 create_model 函数正确引用线性回归模型
def create_model(model_type=None, model_params=None):
    """创建预测模型实例"""
    if model_type is None:
        model_type = DEFAULT_MODEL_TYPE
        
    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS.get(model_type, {})
    
    # 注意：通过绝对导入方式确保模块能被正确找到
    if model_type == "linear_regression":
        from predictor.linear_regression import LinearRegressionPredictor
        return LinearRegressionPredictor(**model_params)
    elif model_type == "student_predictor":
        from predictor.improved_lstm import StudentScorePredictor
        return StudentScorePredictor(**model_params)
    else:
        # 默认使用线性回归
        from predictor.linear_regression import LinearRegressionPredictor
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