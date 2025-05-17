# app/api/exams.py
from flask import Blueprint, request, jsonify
from app.database import db
from app.models.exam import Exam
from app.models.score import Score

exam_bp = Blueprint('exam', __name__, url_prefix='/api/exams')

@exam_bp.route('/', methods=['GET'])
def get_exams():
    # 获取查询参数
    grade_id = request.args.get('gradeId', type=int)
    
    # 构建查询
    query = Exam.query
    if grade_id:
        query = query.filter_by(grade_id=grade_id)
    
    # 执行查询
    exams = query.order_by(Exam.date.desc()).all()
    
    # 格式化响应
    return jsonify({
        'success': True,
        'data': [exam.to_dict() for exam in exams]
    })

@exam_bp.route('/<int:exam_id>', methods=['GET'])
def get_exam_detail(exam_id):
    exam = Exam.query.get_or_404(exam_id)
    
    # 获取该考试的所有成绩
    scores = Score.query.filter_by(exam_id=exam_id).all()
    
    # 计算基本统计信息
    actual_scores = [s.score for s in scores if s.score is not None]
    
    stats = {
        'averageScore': round(sum(actual_scores) / len(actual_scores), 2) if actual_scores else 0,
        'highestScore': max(actual_scores) if actual_scores else 0,
        'lowestScore': min(actual_scores) if actual_scores else 0,
        'passRate': len([s for s in actual_scores if s >= 60]) / len(actual_scores) * 100 if actual_scores else 0,
        'excellentRate': len([s for s in actual_scores if s >= 90]) / len(actual_scores) * 100 if actual_scores else 0
    }
    
    # 按科目分组统计
    subjects = {}
    for score in scores:
        if score.subject not in subjects:
            subjects[score.subject] = []
        if score.score is not None:
            subjects[score.subject].append(score.score)
    
    subject_stats = {}
    for subject, subject_scores in subjects.items():
        if subject_scores:
            subject_stats[subject] = {
                'average': round(sum(subject_scores) / len(subject_scores), 2),
                'highest': max(subject_scores),
                'lowest': min(subject_scores),
                'passRate': len([s for s in subject_scores if s >= 60]) / len(subject_scores) * 100,
                'excellentRate': len([s for s in subject_scores if s >= 90]) / len(subject_scores) * 100
            }
    
    # 构建响应
    response = exam.to_dict()
    response.update(stats)
    response['subjectDetails'] = subject_stats
    
    return jsonify({
        'success': True,
        'data': response
    })