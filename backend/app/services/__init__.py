# app/services/__init__.py
"""
服务模块包 - 提供预测、文件处理和模型管理服务
"""

from app.services.predictor import PredictorService
from app.services.file_handler import FileHandler
from app.services.model_manager import ModelManager
from app.services.score_predictor_service import ScorePredictorService

# 创建服务实例
predictor_service = None
file_handler_service = None
model_manager_service = None
score_predictor_service = None

def init_services(app):
    """初始化所有服务"""
    global predictor_service, file_handler_service, model_manager_service, score_predictor_service
    
    # 配置服务路径
    model_path = app.config.get('MODEL_PATH', 'predictor/models')
    upload_folder = app.config.get('UPLOAD_FOLDER', 'data/uploads')
    
    # 初始化服务实例
    predictor_service = PredictorService(model_path=model_path)
    file_handler_service = FileHandler(upload_folder=upload_folder)
    model_manager_service = ModelManager(model_path=model_path)
    score_predictor_service = ScorePredictorService(model_path=model_path)
    
    app.logger.info("服务层初始化完成")
    
    return {
        'predictor': predictor_service,
        'file_handler': file_handler_service,
        'model_manager': model_manager_service,
        'score_predictor': score_predictor_service
    }

def get_predictor_service():
    """获取预测服务实例"""
    global predictor_service
    if predictor_service is None:
        predictor_service = PredictorService()
    return predictor_service

def get_file_handler_service():
    """获取文件处理服务实例"""
    global file_handler_service
    if file_handler_service is None:
        file_handler_service = FileHandler()
    return file_handler_service

def get_score_predictor_service():
    """获取成绩预测服务实例"""
    global score_predictor_service
    if score_predictor_service is None:
        score_predictor_service = ScorePredictorService()
    return score_predictor_service