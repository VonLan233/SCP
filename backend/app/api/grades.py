# app/api/grades.py
from flask import Blueprint, request, jsonify
from app.database import db
from app.models.grade import Grade
from app.models.score import Score
from app.models.exam import Exam
from app.models.student import Student
import random
from datetime import datetime, timedelta

grade_bp = Blueprint('grade', __name__, url_prefix='/api/grades')

@grade_bp.route('/', methods=['GET'])
def get_grades():
    # 获取所有年级
    grades = Grade.query.all()
    
    # 格式化响应
    return jsonify({
        'success': True,
        'data': [grade.to_dict() for grade in grades]
    })

@grade_bp.route('/<int:grade_id>', methods=['GET'])
def get_grade_detail(grade_id):
    grade = Grade.query.get_or_404(grade_id)
    
    # 格式化响应
    return jsonify({
        'success': True,
        'data': grade.to_dict(include_details=True)
    })

@grade_bp.route('/<int:grade_id>/average', methods=['GET'])
def get_grade_average(grade_id):
    # 确保年级存在
    grade = Grade.query.get_or_404(grade_id)
    
    # 获取该年级的所有考试
    exams = Exam.query.filter_by(grade_id=grade_id).order_by(Exam.date).all()
    
    # 计算每次考试的年级平均分
    result = []
    for exam in exams:
        # 获取该年级该次考试的所有学生成绩
        scores = Score.query.join(Student).filter(
            Score.exam_id == exam.id,
            Student.grade_id == grade_id
        ).all()
        
        # 计算平均分
        if scores and any(s.score is not None for s in scores):
            avg_score = sum(s.score for s in scores if s.score is not None) / len([s for s in scores if s.score is not None])
            
            result.append({
                'examId': exam.id,
                'date': exam.date.strftime('%Y-%m-%d'),
                'average': round(avg_score, 1),
                'subject': scores[0].subject if scores and scores[0].subject else None
            })
    
    # 格式化响应
    return jsonify({
        'success': True,
        'data': result
    })

@grade_bp.route('/<int:grade_id>/predict', methods=['POST'])
def predict_grade_average(grade_id):
    try:
        # 确保年级存在
        grade = Grade.query.get_or_404(grade_id)
        
        # 获取参数
        data = request.get_json() or {}
        steps = data.get('steps', 3)  # 默认预测3步
        
        # 获取历史年级平均分
        # 这里为简单起见，使用模拟数据
        historical_averages = []
        # 如果历史数据太少，返回错误
        if len(historical_averages) < 3:
            return jsonify({
                'success': False,
                'error': '历史数据不足，无法预测'
            }), 400
            
        # 获取最近一次考试信息
        last_avg = historical_averages[-1]
        base_average = last_avg['average']
        last_exam_id = last_avg['examId']
        
        # 生成预测结果
        predictions = []
        for i in range(steps):
            exam_id = last_exam_id + i + 1
            predicted_avg = base_average + random.uniform(-2, 4)  # 随机波动
            predicted_avg = max(75, min(90, predicted_avg))  # 限制在75-90之间
            
            # 创建预测结果
            prediction = {
                'examId': exam_id,
                'date': (datetime.now() + timedelta(days=30 * (i + 1))).strftime('%Y-%m-%d'),
                'average': round(predicted_avg, 1),
                'subject': last_avg['subject']
            }
            
            predictions.append(prediction)
            
            # 更新基准
            base_average = predicted_avg
        
        # 合并历史数据和预测结果
        result_data = historical_averages + predictions
        
        return jsonify({
            'success': True,
            'data': result_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'预测失败: {str(e)}'
        }), 500