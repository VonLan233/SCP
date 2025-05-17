from app.database import db
from datetime import datetime

class Student(db.Model):
    __tablename__ = 'students'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(12), unique=True, nullable=False)  # 学号，如202440134
    name = db.Column(db.String(50), nullable=False)
    gender = db.Column(db.String(10))
    join_date = db.Column(db.Date, default=datetime.now)
    class_id = db.Column(db.Integer, db.ForeignKey('classes.id'))
    grade_id = db.Column(db.Integer, db.ForeignKey('grades.id'))
    
    # 关系
    scores = db.relationship('Score', backref='student', lazy=True)
    
    @property
    def enrollment_year(self):
        """获取入学年份"""
        if len(self.student_id) >= 4:
            return self.student_id[:4]
        return None
    
    @property
    def grade_number(self):
        """获取年级编号"""
        if len(self.student_id) >= 5:
            return self.student_id[4:5]
        return None
    
    @property
    def class_number(self):
        """获取班级编号"""
        if len(self.student_id) >= 7:
            return self.student_id[5:7]
        return None
    
    @property
    def student_number(self):
        """获取班级内学号"""
        if len(self.student_id) >= 9:
            return self.student_id[7:]
        return None
    
    def calculate_average(self):
        if not self.scores:
            return 0
        return sum(score.score for score in self.scores if score.score is not None) / len([s for s in self.scores if s.score is not None])
    
    def calculate_trend(self):
        if len(self.scores) < 3:
            return 'stable'
        
        # 按考试ID排序
        sorted_scores = sorted(self.scores, key=lambda s: s.exam_id, reverse=True)
        recent_scores = [s.score for s in sorted_scores[:3] if s.score is not None]
        
        if len(recent_scores) < 2:
            return 'stable'
        
        if recent_scores[0] > recent_scores[-1]:
            return 'up'
        elif recent_scores[0] < recent_scores[-1]:
            return 'down'
        else:
            return 'stable'
    
    def count_alerts(self):
        # 检测异常成绩数量
        if len(self.scores) < 2:
            return 0
        
        sorted_scores = sorted(self.scores, key=lambda s: s.exam_id)
        alerts = 0
        
        for i in range(1, len(sorted_scores)):
            if sorted_scores[i].score is None or sorted_scores[i-1].score is None:
                continue
                
            # 成绩下降超过20%视为异常
            if sorted_scores[i].score < sorted_scores[i-1].score * 0.8:
                alerts += 1
        
        return alerts
    
    def get_contact_info(self):
        # 示例实现，实际中可能从其他表获取
        return {
            'parent': '家长姓名',  # 示例数据
            'phone': '13800138000',  # 示例数据
            'email': 'parent@example.com'  # 示例数据
        }
    
    def calculate_subject_average(self, subject):
        """计算某一科目的平均分"""
        subject_scores = [s.score for s in self.scores if s.subject == subject and s.score is not None]
        if not subject_scores:
            return 0
        return sum(subject_scores) / len(subject_scores)
    
    def calculate_strengths(self):
        """计算学生的强势科目"""
        subjects = ['语文', '数学', '英语', '物理', '化学', '生物', '历史', '地理', '政治','技术']
        averages = [(subject, self.calculate_subject_average(subject)) for subject in subjects]
        # 筛选平均分较高的科目(前三名)
        strengths = sorted(averages, key=lambda x: x[1], reverse=True)[:3]
        return [s[0] for s in strengths if s[1] > 0]
        
    def calculate_weaknesses(self):
        """计算学生的弱势科目"""
        subjects = ['语文', '数学', '英语', '物理', '化学', '生物', '历史', '地理', '政治','技术']
        averages = [(subject, self.calculate_subject_average(subject)) for subject in subjects]
        # 筛选平均分较低的科目(后三名)
        weaknesses = sorted(averages, key=lambda x: x[1])[:3]
        return [w[0] for w in weaknesses if w[1] > 0]
    
    def to_dict(self, include_details=False):
        result = {
            'id': self.id,
            'student_id': self.student_id,
            'name': self.name,
            'average': self.calculate_average(),
            'trend': self.calculate_trend(),
            'alerts': self.count_alerts(),
            'classId': self.class_id,
            'gradeId': self.grade_id
        }
        
        if include_details:
            result.update({
                'gender': self.gender,
                'enrollment_year': self.enrollment_year,
                'grade_number': self.grade_number,
                'class_number': self.class_number,
                'student_number': self.student_number,
                'joinDate': self.join_date.strftime('%Y-%m-%d') if self.join_date else None,
                'contactInfo': self.get_contact_info(),
                'strengths': self.calculate_strengths(),
                'weaknesses': self.calculate_weaknesses()
            })
        
        return result