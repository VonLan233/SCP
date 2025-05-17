# app/api/uploads.py
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import pandas as pd
import traceback
from app.database import db
from app.models.student import Student
from app.models.exam import Exam
from app.models.score import Score
from app.models.class_model import Class
from app.models.grade import Grade
from app.services.file_handler import FileHandler
from datetime import datetime

upload_bp = Blueprint('upload', __name__, url_prefix='/api/upload')

@upload_bp.route('/scores', methods=['POST'])
def upload_scores():
    try:
        print("接收到成绩文件上传请求")
        
        # 检查请求中是否包含文件
        if 'file' not in request.files:
            print("请求中没有文件部分")
            return jsonify({
                'success': False,
                'message': '请求中没有文件部分'
            }), 400
        
        file = request.files['file']
        print(f"接收到文件: {file.filename}")
        
        # 检查文件名是否为空
        if file.filename == '':
            print("未选择文件")
            return jsonify({
                'success': False,
                'message': '未选择文件'
            }), 400
        
        # 验证文件扩展名
        allowed_extensions = {'csv', 'xlsx', 'xls'}
        if '.' not in file.filename or \
           file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            print(f"不支持的文件格式: {file.filename}")
            return jsonify({
                'success': False,
                'message': '不支持的文件格式，请上传CSV或Excel文件'
            }), 400
        
        # 使用FileHandler保存和处理文件
        file_handler = FileHandler(current_app.config['UPLOAD_FOLDER'])
        file_path = file_handler.save_uploaded_file(file)
        print(f"文件已保存到: {file_path}")
        
        # 处理文件
        try:
            result = file_handler.process_score_file(file_path)
            print(f"文件处理结果: {result}")
            
            return jsonify({
                'success': True,
                'message': '成绩文件上传并处理成功',
                'data': result
            })
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"处理文件时出错: {str(e)}\n{error_details}")
            return jsonify({
                'success': False,
                'message': f'处理文件时出错: {str(e)}'
            }), 500
            
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"文件上传过程中出错: {str(e)}\n{error_details}")
        
        return jsonify({
            'success': False,
            'message': f'服务器错误: {str(e)}'
        }), 500

@upload_bp.route('/students', methods=['POST'])
def upload_students():
    try:
        print("接收到学生数据上传请求")
        
        # 检查请求中是否包含文件
        if 'file' not in request.files:
            print("请求中没有文件部分")
            return jsonify({
                'success': False,
                'message': '请求中没有文件部分'
            }), 400
        
        file = request.files['file']
        print(f"接收到文件: {file.filename}")
        
        # 检查文件名是否为空
        if file.filename == '':
            print("未选择文件")
            return jsonify({
                'success': False,
                'message': '未选择文件'
            }), 400
        
        # 验证文件扩展名
        allowed_extensions = {'csv', 'xlsx', 'xls'}
        if '.' not in file.filename or \
           file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            print(f"不支持的文件格式: {file.filename}")
            return jsonify({
                'success': False,
                'message': '不支持的文件格式，请上传CSV或Excel文件'
            }), 400
        
        # 使用FileHandler保存和处理文件
        file_handler = FileHandler(current_app.config['UPLOAD_FOLDER'])
        file_path = file_handler.save_uploaded_file(file)
        print(f"学生数据文件已保存到: {file_path}")
        
        # 处理文件
        try:
            result = file_handler.process_student_file(file_path)
            print(f"学生数据处理结果: {result}")
            
            return jsonify({
                'success': True,
                'message': '学生数据上传并处理成功',
                'data': result
            })
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"处理学生数据文件时出错: {str(e)}\n{error_details}")
            return jsonify({
                'success': False,
                'message': f'处理学生数据文件时出错: {str(e)}'
            }), 500
            
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"学生数据上传过程中出错: {str(e)}\n{error_details}")
        
        return jsonify({
            'success': False,
            'message': f'服务器错误: {str(e)}'
        }), 500