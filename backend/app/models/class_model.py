from app.database import db
from datetime import datetime

class Class(db.Model):
    __tablename__ = 'classes'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)  # 班级名称，例如"高一1班"
    grade_id = db.Column(db.Integer, db.ForeignKey('grades.id'))
    teacher_id = db.Column(db.Integer, db.ForeignKey('teachers.id'))
    student_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # 关系
    students = db.relationship('Student', backref='class', lazy=True)
    grade = db.relationship('Grade', backref='classes', lazy=True)
    
    def calculate_average(self, exam_id=None, subject=None):
        """计算班级平均分"""
        from app.models.score import Score
        from app.models.student import Student
        
        # 获取班级所有学生ID
        student_ids = [s.student_id for s in self.students]
        
        if not student_ids:
            return 0
        
        # 构建查询
        query = Score.query.filter(Score.student_id.in_(student_ids))
        
        if exam_id:
            query = query.filter_by(exam_id=exam_id)
        
        if subject:
            query = query.filter_by(subject=subject)
        
        scores = query.all()
        
        if not scores:
            return 0
        
        # 计算平均分
        valid_scores = [s.score for s in scores if s.score is not None]
        if not valid_scores:
            return 0
        
        return sum(valid_scores) / len(valid_scores)
    
    def get_score_distribution(self, exam_id=None, subject=None):
        """获取班级成绩分布"""
        from app.models.score import Score
        from app.models.student import Student
        
        # 获取班级所有学生ID
        student_ids = [s.student_id for s in self.students]
        
        if not student_ids:
            return {"distribution": []}
        
        # 构建查询
        query = Score.query.filter(Score.student_id.in_(student_ids))
        
        if exam_id:
            query = query.filter_by(exam_id=exam_id)
        
        if subject:
            query = query.filter_by(subject=subject)
        
        scores = query.all()
        
        if not scores:
            return {"distribution": []}
        
        # 计算分数分布
        valid_scores = [s.score for s in scores if s.score is not None]
        
        if not valid_scores:
            return {"distribution": []}
        
        # 定义分数段
        ranges = [
            (0, 60),      # 不及格
            (60, 70),     # 及格
            (70, 80),     # 中等
            (80, 90),     # 良好
            (90, 100.1)   # 优秀
        ]
        
        distribution = []
        for low, high in ranges:
            count = sum(1 for s in valid_scores if low <= s < high)
            distribution.append({
                "range": f"{low}-{high-0.1:.1f}",
                "count": count,
                "percentage": count / len(valid_scores) * 100
            })
        
        return {"distribution": distribution}
    
    def to_dict(self, include_stats=False):
        result = {
            'id': self.id,
            'name': self.name,
            'gradeId': self.grade_id,
            'teacherId': self.teacher_id,
            'studentCount': self.student_count
        }
        
        if include_stats:
            result.update({
                'average': self.calculate_average(),
                'distribution': self.get_score_distribution()
            })
        
        return result