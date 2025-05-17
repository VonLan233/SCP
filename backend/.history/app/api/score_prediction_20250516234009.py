# app/api/score_prediction.py
from flask import Blueprint, jsonify, request, current_app
from app.services import get_score_predictor_service
import logging
import traceback
import traceback
import numpy as np
import json
from json import JSONEncoder

class NumpyJSONEncoder(JSONEncoder):
    """处理 NumPy 和 TensorFlow 数据类型的 JSON 编码器"""
    def default(self, obj):
        import numpy as np
        import tensorflow as tf
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64, tf.Tensor)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'numpy'):  # 处理 TensorFlow tensor
            return obj.numpy().tolist() if hasattr(obj.numpy(), 'tolist') else float(obj.numpy())
        return super().default(obj)
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建蓝图
score_prediction_bp = Blueprint('score_prediction', __name__, url_prefix='/api/prediction')

@score_prediction_bp.record
def record_custom_encoder(state):
    state.app.json_encoder = NumpyJSONEncoder

@score_prediction_bp.route('/student/<string:student_id>', methods=['POST'])
def predict_student_scores(student_id):
    try:
        # 获取请求参数
        data = request.get_json() or {}
        steps = data.get('steps', 3)
        model_params = data.get('modelParams')
        model_type = data.get('modelType', 'linear_regression')  # 默认使用线性回归
        subject = data.get('subject')
        
        # 获取预测服务
        predictor = get_score_predictor_service()
        
        # 调用预测方法
        result = predictor.predict_student_scores(
            student_id=student_id,
            steps=steps,
            model_params=model_params,
            model_type=model_type,
            subject=subject
        )
        
        # 预处理数据，确保所有数值类型正确
        def convert_to_native_types(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_to_native_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_to_native_types(obj.tolist())
            elif hasattr(obj, 'numpy'):  # TensorFlow tensor
                return convert_to_native_types(obj.numpy())
            else:
                return obj
                
        processed_result = convert_to_native_types(result)
        
        return jsonify({
            'success': True,
            'data': processed_result
        })
        
    except Exception as e:
        logger.error(f"预测学生 {student_id} 成绩失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': f"预测失败: {str(e)}"
        }), 500

@score_prediction_bp.route('/class/<int:class_id>', methods=['POST'])
def predict_class_average(class_id):
    """
    预测班级平均分
    
    路径参数:
        class_id: 班级ID
        
    请求体参数:
        steps: 预测步数
        model_params: 模型参数
        
    返回:
        predictions: 预测结果
    """
    try:
        # 获取请求参数
        data = request.get_json() or {}
        steps = data.get('steps', 3)
        model_params = data.get('modelParams')
        
        # 获取预测服务
        predictor = get_score_predictor_service()
        
        # 调用预测方法
        result = predictor.predict_class_average(
            class_id=class_id,
            steps=steps,
            model_params=model_params
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"预测班级 {class_id} 平均分失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': f"预测失败: {str(e)}"
        }), 500

@score_prediction_bp.route('/batch', methods=['POST'])
def batch_predict_students():
    """
    批量预测多个学生的成绩
    
    请求体参数:
        student_ids: 学生ID列表
        steps: 预测步数
        model_params: 模型参数
        model_type: 模型类型
        
    返回:
        predictions: 按学生ID组织的预测结果
    """
    try:
        # 获取请求参数
        data = request.get_json() or {}
        student_ids = data.get('studentIds', [])
        steps = data.get('steps', 3)
        model_params = data.get('modelParams')
        model_type = data.get('modelType')
        
        if not student_ids:
            return jsonify({
                'success': False,
                'error': "未提供学生ID列表"
            }), 400
        
        # 获取预测服务
        predictor = get_score_predictor_service()
        
        # 调用批量预测方法
        results = predictor.batch_predict_students(
            student_ids=student_ids,
            steps=steps,
            model_params=model_params,
            model_type=model_type
        )
        
        return jsonify({
            'success': True,
            'data': results
        })
        
    except Exception as e:
        logger.error(f"批量预测失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': f"批量预测失败: {str(e)}"
        }), 500

@score_prediction_bp.route('/model/status/<string:student_id>', methods=['GET'])
def get_model_status(student_id):
    """
    获取学生模型状态
    
    路径参数:
        student_id: 学生ID
        
    请求参数:
        model_type: 模型类型
        
    返回:
        status: 状态信息
    """
    try:
        # 获取请求参数
        model_type = request.args.get('modelType')
        
        # 获取预测服务
        predictor = get_score_predictor_service()
        
        # 调用状态检查方法
        status = predictor.check_model_status(
            student_id=student_id,
            model_type=model_type
        )
        
        return jsonify({
            'success': True,
            'data': status
        })
        
    except Exception as e:
        logger.error(f"获取模型状态失败: {str(e)}")
        
        return jsonify({
            'success': False,
            'error': f"获取状态失败: {str(e)}"
        }), 500

@score_prediction_bp.route('/model/train/<string:student_id>', methods=['POST'])
def train_model(student_id):
    """
    训练学生模型
    
    路径参数:
        student_id: 学生ID
        
    请求体参数:
        model_type: 模型类型
        params: 训练参数
        
    返回:
        result: 训练结果
    """
    try:
        # 获取请求参数
        data = request.get_json() or {}
        model_type = data.get('modelType')
        params = data.get('params')
        subject=data.get('subject')
        
        # 获取预测服务
        predictor = get_score_predictor_service()
        
        # 调用训练方法
        result = predictor.train_model(
            student_id=student_id,
            model_type=model_type,
            model_params=params,  # 修改这里，使参数名一致
            subject=subject # 添加科目参数
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"训练模型失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': f"训练失败: {str(e)}"
        }), 500
        
@score_prediction_bp.route('/analysis/student/<string:student_id>', methods=['GET'])
def analyze_student_performance(student_id):
    """
    分析学生表现
    
    路径参数:
        student_id: 学生ID
        
    返回:
        analysis: 分析结果
    """
    try:
        # 获取预测服务
        predictor = get_score_predictor_service()
        
        # 调用分析方法
        analysis = predictor.analyze_student_performance(student_id)
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"分析学生表现失败: {str(e)}")
        
        return jsonify({
            'success': False,
            'error': f"分析失败: {str(e)}"
        }), 500

@score_prediction_bp.route('/analysis/class/<int:class_id>/anomalies', methods=['GET'])
def detect_class_anomalies(class_id):
    """
    检测班级异常表现
    
    路径参数:
        class_id: 班级ID
        
    返回:
        anomalies: 异常信息
    """
    try:
        # 获取预测服务
        predictor = get_score_predictor_service()
        
        # 调用异常检测方法
        anomalies = predictor.detect_performance_anomalies(class_id)
        
        return jsonify(anomalies)
        
    except Exception as e:
        logger.error(f"检测班级异常失败: {str(e)}")
        
        return jsonify({
            'success': False,
            'error': f"检测失败: {str(e)}"
        }), 500

@score_prediction_bp.route('/insights/class/<int:class_id>', methods=['GET'])
def generate_class_insights(class_id):
    """
    生成班级预测性分析洞察
    
    路径参数:
        class_id: 班级ID
        
    返回:
        insights: 洞察信息
    """
    try:
        # 获取预测服务
        predictor = get_score_predictor_service()
        
        # 调用洞察生成方法
        insights = predictor.generate_predictive_insights(class_id)
        
        return jsonify(insights)
        
    except Exception as e:
        logger.error(f"生成班级洞察失败: {str(e)}")
        
        return jsonify({
            'success': False,
            'error': f"生成洞察失败: {str(e)}"
        }), 500

@score_prediction_bp.route('/service/status', methods=['GET'])
def get_service_status():
    """
    获取预测服务状态
    
    返回:
        status: 状态信息
    """
    try:
        # 获取预测服务
        predictor = get_score_predictor_service()
        
        # 调用状态获取方法
        status = predictor.get_service_status()
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"获取服务状态失败: {str(e)}")
        
        return jsonify({
            'success': False,
            'error': f"获取状态失败: {str(e)}"
        }), 500

@score_prediction_bp.route('/service/cache/clear', methods=['POST'])
def clear_model_cache():
    """
    清除模型缓存
    
    返回:
        result: 操作结果
    """
    try:
        # 获取预测服务
        predictor = get_score_predictor_service()
        
        # 调用缓存清除方法
        result = predictor.clear_model_cache()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"清除模型缓存失败: {str(e)}")
        
        return jsonify({
            'success': False,
            'error': f"清除缓存失败: {str(e)}"
        }), 500