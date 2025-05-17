from app.database import db
from datetime import datetime

class Score(db.Model):
    __tablename__ = 'scores'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(12), db.ForeignKey('students.student_id'), nullable=False)  # 关联学生的学号
    exam_id = db.Column(db.Integer, db.ForeignKey('exams.id'), nullable=False)
    subject = db.Column(db.String(50), nullable=False)  # 科目名称
    score = db.Column(db.Float)  # 实际得分
    total_score = db.Column(db.Float)  # 满分值 (语数英为150, 其他默认100)
    predicted_score = db.Column(db.Float)  # 预测分数
    lower_bound = db.Column(db.Float)  # 预测下限
    upper_bound = db.Column(db.Float)  # 预测上限
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    def __init__(self, **kwargs):
        super(Score, self).__init__(**kwargs)
        # 根据科目自动设置满分值
        if self.subject in ['语文', '数学', '英语'] and not self.total_score:
            self.total_score = 150
        elif not self.total_score:
            self.total_score = 100
    
    def get_percentage(self):
        """计算得分百分比"""
        if self.score is None or self.total_score is None or self.total_score == 0:
            return None
        return (self.score / self.total_score) * 100
    
    def to_dict(self):
        return {
            'examId': self.exam_id,
            'date': self.exam.date.strftime('%Y-%m-%d') if hasattr(self, 'exam') and self.exam and self.exam.date else None,
            'subject': self.subject,
            'actual': self.score,
            'total': self.total_score,
            'percentage': self.get_percentage(),
            'predicted': self.predicted_score,
            'lower': self.lower_bound,
            'upper': self.upper_bound
        }