# app/api/classes.py
from flask import Blueprint, request, jsonify
from app.database import db
from app.models.class_model import Class
from app.models.score import Score
from app.models.exam import Exam
import random
from datetime import datetime, timedelta

class_bp = Blueprint('class', __name__, url_prefix='/api/classes')

@class_bp.route('/', methods=['GET'])
def get_classes():
    # 获取查询参数
    grade_id = request.args.get('gradeId', type=int)
    
    # 构建查询
    query = Class.query
    if grade_id:
        query = query.filter_by(grade_id=grade_id)
    
    # 执行查询
    classes = query.all()
    
    # 格式化响应
    return jsonify({
        'success': True,
        'data': [cls.to_dict() for cls in classes]
    })

@class_bp.route('/<int:class_id>', methods=['GET'])
def get_class_detail(class_id):
    cls = Class.query.get_or_404(class_id)
    
    # 格式化响应，包含详细信息
    return jsonify({
        'success': True,
        'data': cls.to_dict(include_details=True)
    })

@class_bp.route('/<int:class_id>/average', methods=['GET'])
def get_class_average(class_id):
    # 确保班级存在
    cls = Class.query.get_or_404(class_id)
    
    # 获取该班级的所有考试
    exams = Exam.query.filter_by(grade_id=cls.grade_id).order_by(Exam.date).all()
    
    # 计算每次考试的班级平均分
    result = []
    for exam in exams:
        # 获取该班级该次考试的所有学生成绩
        scores = Score.query.filter(
            Score.exam_id == exam.id,
            Score.student_id.in_([s.id for s in cls.students])
        ).all()
        
        # 计算平均分
        if scores:
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

@class_bp.route('/<int:class_id>/predict', methods=['POST'])
def predict_class_average(class_id):
    try:
        # 确保班级存在
        cls = Class.query.get_or_404(class_id)
        
        # 获取参数
        data = request.get_json() or {}
        steps = data.get('steps', 3)  # 默认预测3步
        
        # 获取历史班级平均分
        exams = Exam.query.filter_by(grade_id=cls.grade_id).order_by(Exam.date).all()
        
        # 计算每次考试的班级平均分
        historical_averages = []
        for exam in exams:
            # 获取该班级该次考试的所有学生成绩
            scores = Score.query.filter(
                Score.exam_id == exam.id,
                Score.student_id.in_([s.id for s in cls.students])
            ).all()
            
            # 计算平均分
            if scores and any(s.score is not None for s in scores):
                avg_score = sum(s.score for s in scores if s.score is not None) / len([s for s in scores if s.score is not None])
                
                historical_averages.append({
                    'examId': exam.id,
                    'date': exam.date.strftime('%Y-%m-%d'),
                    'average': round(avg_score, 1),
                    'subject': scores[0].subject if scores and scores[0].subject else None
                })
        
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
            predicted_avg = base_average + random.uniform(-3, 5)  # 随机波动
            predicted_avg = max(70, min(95, predicted_avg))  # 限制在70-95之间
            
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