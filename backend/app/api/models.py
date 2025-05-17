# app/api/models.py
from flask import Blueprint, request, jsonify
from app.services.model_manager import ModelManager

model_bp = Blueprint('model', __name__, url_prefix='/api/model')

@model_bp.route('/config', methods=['GET'])
def get_model_config():
    # 获取模型配置
    model_manager = ModelManager()
    config = model_manager.get_config()
    
    return jsonify({
        'success': True,
        'data': config
    })

@model_bp.route('/config', methods=['PUT'])
def update_model_config():
    # 更新模型配置
    data = request.get_json()
    if not data:
        return jsonify({
            'success': False,
            'error': '请求体不能为空'
        }), 400
    
    model_manager = ModelManager()
    updated_config = model_manager.update_config(data)
    
    return jsonify({
        'success': True,
        'data': updated_config
    })

@model_bp.route('/train', methods=['POST'])
def train_model():
    # 训练模型
    data = request.get_json() or {}
    model_type = data.get('modelType')
    parameters = data.get('parameters')
    
    model_manager = ModelManager()
    result = model_manager.train_model(data=None, model_type=model_type, params=parameters)
    
    return jsonify(result)