# app/api/search.py

from venv import logger
from flask import Blueprint, request, jsonify
from app.models.student import Student
from app.models.class_model import Class
from app.models.score import Score

search_bp = Blueprint('search', __name__, url_prefix='/api/search')

@search_bp.route('/students', methods=['GET'])
def search_students():
    """按学号或姓名搜索学生"""
    try:
        # 获取查询参数
        student_id = request.args.get('student_id')
        name = request.args.get('name')
        
        # logger.info(f"Searching for student with ID: {student_id}, name: {name}")
        logger.info(f"搜索请求: 学号={student_id}, 姓名={name}")
        logger.info(f"请求参数: {request.args}")
        
        # 创建初始查询
        query = Student.query
        
        # 如果提供了学号，按学号查询
        if student_id:
            query = query.filter_by(student_id=student_id)
            
        # 如果提供了姓名，按姓名模糊查询
        if name:
            query = query.filter(Student.name.like(f'%{name}%'))
            
        # 执行查询
        students = query.all()
        
        if not students:
            return jsonify({
                'success': True,
                'data': []
            })
        
        # 格式化结果
        result = []
        for student in students:
            # 获取学生的平均分
            scores = Score.query.filter_by(student_id=student.student_id).all()
            avg_score = 0
            if scores:
                valid_scores = [s.score for s in scores if s.score is not None]
                avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
            
            # 获取学生的班级信息
            class_info = Class.query.get(student.class_id)
            class_name = class_info.name if class_info else "未分配班级"
            
            # 确定成绩趋势
            trend = "stable"
            if len(scores) >= 2:
                sorted_scores = sorted(scores, key=lambda s: s.exam.date if s.exam else None)
                recent_scores = [s.score for s in sorted_scores[-3:] if s.score is not None]
                if len(recent_scores) >= 2:
                    if recent_scores[-1] > recent_scores[-2]:
                        trend = "up"
                    elif recent_scores[-1] < recent_scores[-2]:
                        trend = "down"
            
            result.append({
                'id': student.student_id,
                'name': student.name,
                'class_id': student.class_id,
                'class_name': class_name,
                'grade_id': student.grade_id,
                'average': round(avg_score, 1),
                'trend': trend,
                'alerts': 0  # 默认值，可根据实际需求设置警报
            })
        logger.info(f"搜索结果: 找到 {len(result)} 个学生")
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'搜索学生失败: {str(e)}'
        }), 500

# 在app/__init__.py中注册蓝图
# 