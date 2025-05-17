# app/api/students.py
from flask import Blueprint, request, jsonify
from app.database import db
from app.models.student import Student
from app.models.score import Score
from app.services.predictor import PredictorService
import random
from datetime import datetime, timedelta

student_bp = Blueprint('student', __name__, url_prefix='/api/students')

@student_bp.route('/', methods=['GET'])
def get_students():
    # 获取查询参数
    # grade_id = request.args.get('gradeId', type=int)
    # class_id = request.args.get('classId', type=int)
    
    # 构建查询
    query = Student.query
    # if grade_id:
    #     query = query.filter_by(grade_id=grade_id)
    # if class_id:
    #     query = query.filter_by(class_id=class_id)
    
    # 执行查询
    students = query.all()
    
    # 格式化响应
    return jsonify({
        'success': True,
        'data': [student.to_dict() for student in students]
    })

@student_bp.route('/<int:student_id>', methods=['GET'])
def get_student_detail(student_id):
    # student = Student.query.get_or_404(student_id)
    student = Student.query.filter_by(student_id=student_id).first()
    if not student:
        return jsonify({
            'success': False,
            'error': '学生不存在'
        }), 404
    
    return jsonify({
        'success': True,
        'data': student.to_dict(include_details=True)
    })

@student_bp.route('/<int:student_id>/scores', methods=['GET'])
def get_student_scores(student_id):
    student = Student.query.filter_by(student_id=student_id).first()
    if not student:
        return jsonify({
            'success': False,
            'error': '学生不存在'
        }), 404
    
    # 获取学生所有成绩
    scores = Score.query.filter_by(student_id=student_id).order_by(Score.exam_id).all()
    
    # 格式化响应
    return jsonify({
        'success': True,
        'data': [score.to_dict() for score in scores]
    })

@student_bp.route('/<int:student_id>/predict', methods=['POST'])
def predict_student_scores(student_id):
    try:
        # 打印调试信息
        print(f"接收到学生ID={student_id}的预测请求")
        
        # 确保学生存在
        student = Student.query.get_or_404(student_id)
        print(f"找到学生: {student.name} (ID: {student.id})")
        
        # 获取参数
        data = request.get_json() or {}
        steps = data.get('steps', 3)  # 默认预测3步
        print(f"预测步数: {steps}")
        
        # 创建预测服务实例
        predictor = PredictorService()
        
        # 调用预测方法
        predictions = predictor.predict_student_scores(student_id, steps)
        print(f"生成了 {len(predictions)} 条预测记录")
        
        # 返回预测结果
        return jsonify({
            'success': True,
            'data': predictions
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"预测失败: {str(e)}\n{error_details}")
        
        return jsonify({
            'success': False,
            'error': f'预测失败: {str(e)}'
        }), 500