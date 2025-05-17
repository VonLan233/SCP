from app.database import db
from datetime import datetime

class Grade(db.Model):
    __tablename__ = 'grades'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)  # 年级名称，例如"高一"
    class_count = db.Column(db.Integer, default=0)
    director_id = db.Column(db.Integer, db.ForeignKey('teachers.id'))
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # 关系
    exams = db.relationship('Exam', backref='grade', lazy=True)
    
    def calculate_average(self, exam_id=None, subject=None):
        """计算年级平均分"""
        from app.models.score import Score
        from app.models.student import Student
        
        # 获取年级所有学生ID
        students = Student.query.filter_by(grade_id=self.id).all()
        student_ids = [s.student_id for s in students]
        
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
    
    def get_class_comparison(self, exam_id=None, subject=None):
        """获取班级间对比数据"""
        from app.models.class_model import Class
        
        classes = Class.query.filter_by(grade_id=self.id).all()
        
        comparison = []
        for cls in classes:
            avg = cls.calculate_average(exam_id, subject)
            distribution = cls.get_score_distribution(exam_id, subject)
            
            comparison.append({
                'id': cls.id,
                'name': cls.name,
                'average': avg,
                'distribution': distribution
            })
        
        # 排序
        comparison.sort(key=lambda x: x['average'], reverse=True)
        
        return comparison
    
    def to_dict(self, include_stats=False):
        result = {
            'id': self.id,
            'name': self.name,
            'classCount': self.class_count,
            'directorId': self.director_id
        }
        
        if include_stats:
            result.update({
                'average': self.calculate_average(),
                'classComparison': self.get_class_comparison()
            })
        
        return result