# app/__init__.py
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from app.services import init_services
import traceback
import os
from app.database import db

# 创建扩展对象，但暂不初始化
# db = SQLAlchemy()
migrate = Migrate()

def create_app(config_class=None):
    app = Flask(__name__)
    
    # 如果没有传入配置类，使用默认配置
    if config_class is None:
        from app.config import Config
        config_class = Config
    
    app.config.from_object(config_class)
    # 初始化服务
    services = init_services(app)
    
    # 你可以将服务添加到应用上下文中，便于在视图函数中访问
    app.predictor = services['predictor']
    app.file_handler = services['file_handler']
    app.model_manager = services['model_manager']
    
    # 初始化扩展
    db.init_app(app)
    migrate.init_app(app, db)
    CORS(app)  # 允许跨域请求
    
    # 注册蓝图
    from app.api.students import student_bp
    from app.api.classes import class_bp
    from app.api.grades import grade_bp
    from app.api.exams import exam_bp
    from app.api.uploads import upload_bp
    from app.api.models import model_bp
    from app.api.search import search_bp
    from app.api.score_prediction import score_prediction_bp
    
    app.register_blueprint(student_bp)
    app.register_blueprint(class_bp)
    app.register_blueprint(grade_bp)
    app.register_blueprint(exam_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(score_prediction_bp)
    
    # 全局错误处理
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({
            'success': False,
            'error': '请求的资源不存在'
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'success': False,
            'error': '服务器内部错误'
        }), 500
    
    # 捕获所有异常
    @app.errorhandler(Exception)
    def handle_exception(e):
        # 打印详细错误堆栈
        traceback.print_exc()
        
        # 如果是HTTP异常，使用其状态码
        if isinstance(e, HTTPException):
            return jsonify({
                'success': False,
                'error': str(e)
            }), e.code
        
        # 其他异常视为500错误
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
    # 确保上传目录和模型目录存在
    upload_folder = app.config['UPLOAD_FOLDER']
    model_folder = os.path.join('predictor', 'models')
    for folder in [upload_folder, model_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # 添加一个简单的健康检查路由
    @app.route('/health')
    def health_check():
        return jsonify({'status': 'ok'})
    
    # # 如果开发环境，添加测试数据
    # if app.config['ENV'] == 'development':
    #     with app.app_context():
    #         try:
    #             from app.models.student import Student
    #             # 如果数据库为空，添加测试数据
    #             if Student.query.count() == 0:
    #                 print("数据库为空，添加测试数据...")
    #                 from app.seed_data import seed_data
    #                 seed_data()
    #         except Exception as e:
    #             print(f"尝试添加测试数据时出错: {e}")
    
    return app