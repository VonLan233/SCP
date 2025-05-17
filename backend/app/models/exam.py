from app.database import db
from datetime import datetime

class Exam(db.Model):
    __tablename__ = 'exams'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)  # 考试名称
    date = db.Column(db.Date, nullable=False)  # 考试日期
    type = db.Column(db.String(50))  # 考试类型 (月考、期中、期末等)
    grade_id = db.Column(db.Integer, db.ForeignKey('grades.id'))  # 适用年级
    # subject = db.Column(db.String(50))  # 科目名称
    
    # 关系
    scores = db.relationship('Score', backref='exam', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'date': self.date.strftime('%Y-%m-%d') if self.date else None,
            'type': self.type,
            'gradeId': self.grade_id,
            # 'subject': self.subject,
        }