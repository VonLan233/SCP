# app/models/teacher.py

from app.database import db
from datetime import datetime

class Teacher(db.Model):
    __tablename__ = 'teachers'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    subject = db.Column(db.String(50))
    contact = db.Column(db.String(100))
    other_info = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 修改关系定义，确保关联到正确的外键
    classes = db.relationship('Class', backref='teacher_ref', lazy=True, 
                              foreign_keys='Class.teacher_id')
    
    # 删除或注释掉错误的scores关系定义
    # scores = db.relationship('Score', backref='teacher', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'subject': self.subject,
            'contact': self.contact,
            'other_info': self.other_info
        }