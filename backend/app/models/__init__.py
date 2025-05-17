# app/models/__init__.py

"""
数据库模型包
这个文件导入所有的模型类，确保它们在应用程序启动时被正确加载，
以便SQLAlchemy能够创建相应的数据库表。
"""

# 导入各个模型
from app.models.student import Student
from app.models.class_model import Class  # 注意: 因为class是Python关键字，通常模型文件命名为class_model.py
from app.models.grade import Grade
from app.models.exam import Exam
from app.models.score import Score
from app.models.teacher import Teacher

# 如果有其他模型，也在这里导入
# 例如:
# from app.models.attendance import Attendance
# from app.models.behavior import Behavior

# 这个列表可以在应用程序的其他部分使用，以获取所有模型类
__all__ = [
    'Student',
    'Class',
    'Grade',
    'Exam',
    'Score',
    'Teacher',
    # 添加其他模型类
]