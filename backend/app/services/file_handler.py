# app/services/file_handler.py
import os
import pandas as pd
from werkzeug.utils import secure_filename
from flask import logging
from app.database import db
from app.models.score import Score
from app.models.student import Student
from app.models.exam import Exam
from app.models.class_model import Class
from app.models.grade import Grade
from datetime import datetime

class FileHandler:
    def __init__(self, upload_folder='data/uploads'):
        self.upload_folder = upload_folder
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
    
    def save_uploaded_file(self, file):
        """保存上传的文件"""
        filename = secure_filename(file.filename)
        file_path = os.path.join(self.upload_folder, filename)
        file.save(file_path)
        return file_path
    
    def process_score_file(self, file_path):
        """处理成绩文件"""
        # 获取文件扩展名
        ext = file_path.split('.')[-1].lower()
        
        # 读取文件
        if ext == 'csv':
            df = pd.read_csv(file_path)
        else:  # xlsx or xls
            df = pd.read_excel(file_path)
        
        # 标准化列名
        column_mapping = {
            'student_id': ['student_id', 'studentid', 'id', '学生id', '学号'],
            'student_name': ['student_name', 'studentname', 'name', '学生姓名', '姓名'],
            'exam_date': ['exam_date', 'examdate', 'date', '考试日期', '日期'],
            'exam_name': ['exam_name', 'examname', '考试名称', '考试'],
            'subject': ['subject', 'subjects', '科目'],
            'score': ['score', 'scores', '分数', '成绩']
        }
        
        # 尝试标准化列名
        renamed_columns = {}
        for standard_name, possible_names in column_mapping.items():
            for col in df.columns:
                if col.lower() in possible_names:
                    renamed_columns[col] = standard_name
        
        if renamed_columns:
            df = df.rename(columns=renamed_columns)
        
       # 检查文件结构
        expected_columns = ["student_id", 'student_name']
        subject_columns = ['语文', '数学', '英语', '物理', '化学', '生物', '政治', '历史', '地理', '技术', '总分']
        
        # 验证必要的列
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"文件缺少必要的列: {', '.join(missing_columns)}")
        
       # 获取考试信息（从文件名或额外参数中提取）
        file_basename = os.path.basename(file_path).split('.')[0]
        if '_' in file_basename:
            date_part = file_basename.split('_')[0]
            exam_name_part = file_basename.split('_')[1]
            
            # 尝试将日期字符串转换为日期对象
            try:
                # 假设日期格式是 "yyyy-mm-dd"
                year, month, day = date_part.split('-')
                exam_date = datetime(int(year), int(month), int(day)).date()
                
                # 设置考试名称
                exam_name = f"{year}年{month}月考试" if not exam_name_part else exam_name_part
            except Exception as e:
                # 如果日期解析失败，使用当前日期
                exam_date = datetime.now().date()
                exam_name = exam_name_part if '_' in file_basename else file_basename
        else:
            # 如果文件名没有下划线分隔，使用默认值
            exam_date = datetime.now().date()
            exam_name = file_basename
        
        # 处理数据
        processed_records = 0
        errors = []
        created_students = 0
        created_exams = 0
        updated_scores = 0
        
        for idx, row in df.iterrows():
            try:
                # 获取学生信息
                student_id = str(row['student_id']).strip()
                student_name = str(row['student_name']).strip()
                
                # 确保学号符合格式要求：YYYYGGCCSSS (年份+年级+班级+学号)
                if not student_id.isdigit() or len(student_id) != 9:
                    errors.append(f"行 {idx+1}: 学号 '{student_id}' 格式不正确，应为9位数字")
                    continue
                
                # 解析学号信息
                enrollment_year = "20" + student_id[:2]  # 前两位作为入学年份
                grade_number = enrollment_year #3位作为年级编号
                class_number = student_id[3:5]  # 第4-5位作为班级编号
                student_number = student_id[5:]  # 后4位作为班内学号
                
                # 构建完整学号
                full_student_id = student_id
                
                # 考试信息
                # try:
                #     exam_date = pd.to_datetime(row['exam_date']).date()
                # except:
                #     errors.append(f"行 {idx+1}: 无效的考试日期 '{row['exam_date']}'")
                #     exam_date = datetime.now().date()  # 默认使用当前日期
                #     # continue
                
                exam_name = f"{exam_date.year}年{exam_date.month}月考试"
                if 'exam_name' in df.columns and not pd.isna(row['exam_name']):
                    exam_name = str(row['exam_name'])
                 # # 查找或创建年级
                grade_name = str(grade_number)
                
                grade = Grade.query.filter_by(name=grade_name).first()
                if not grade:
                    grade = Grade(name=grade_name, class_count=0)
                    db.session.add(grade)
                    db.session.flush()
                
                # 查找或创建班级
                class_name = f"{grade_name}{class_number}班"
                target_class = Class.query.filter_by(name=class_name, grade_id=grade.id).first()
                if not target_class:
                    target_class = Class(name=class_name, grade_id=grade.id, student_count=0)
                    db.session.add(target_class)
                    db.session.flush()
                
                # 查找或创建学生
                # target_class =  Student.query.filter_by(student_id=full_student_id).first()
                student = Student.query.filter_by(student_id=full_student_id).first()
                if not student:
                    student = Student(
                        student_id=full_student_id,
                        name=student_name,
                        class_id=target_class.id,
                        grade_id=grade.id,
                        join_date=datetime(int(enrollment_year), 9, 1).date()  # 假设9月1日入学
                    )
                    db.session.add(student)
                    db.session.flush()
                    created_students += 1
                
                # 查找或创建考试
                exam = Exam.query.filter_by(date=exam_date, name=exam_name, grade_id=grade.id).first()
                if not exam:
                    exam = Exam(name=exam_name, date=exam_date, type='regular', grade_id=grade.id)
                    db.session.add(exam)
                    db.session.flush()
                    created_exams += 1
                    
                # 处理每个科目的成绩
                for subject in subject_columns:
                    if subject in df.columns and not pd.isna(row[subject]):
                        try:
                            score_value = float(row[subject])
                            
                            # 确定满分值
                            total_score = 100
                            if subject in ['语文', '数学', '英语']:
                                total_score = 150
                            elif subject == '总分':
                                total_score = 750  # 假设总分是1000
                            
                            # 查找或创建成绩记录
                            existing_score = Score.query.filter_by(
                                student_id=student_id,
                                exam_id=exam.id,
                                subject=subject
                            ).first()
                            
                            if existing_score:
                                # 更新成绩
                                existing_score.score = score_value
                                existing_score.total_score = total_score
                                existing_score.updated_at = datetime.now()
                                updated_scores += 1
                            else:
                                # 创建新成绩记录
                                new_score = Score(
                                    student_id=student_id,
                                    exam_id=exam.id,
                                    subject=subject,
                                    score=score_value,
                                    total_score=total_score
                                )
                                db.session.add(new_score)
                                processed_records += 1
                        except Exception as e:
                            errors.append(f"处理学生 {student_id} 的 {subject} 成绩时出错: {str(e)}")
                
                # # 查找或创建成绩记录
                # existing_score = Score.query.filter_by(
                #     student_id=full_student_id,
                #     exam_id=exam.id,
                #     subject=subject
                # ).first()
                
                # if existing_score:
                #     # 更新成绩
                #     existing_score.score = score_value
                #     existing_score.total_score = total_score
                #     existing_score.updated_at = datetime.now()
                #     updated_scores += 1
                # else:
                #     # 创建新成绩记录
                #     new_score = Score(
                #         student_id=full_student_id,
                #         exam_id=exam.id,
                #         subject=subject,
                #         score=score_value,
                #         total_score=total_score
                #     )
                #     db.session.add(new_score)
                
                # processed_records += 1
                
            except Exception as e:
                errors.append(f"处理行 {idx+1} 时出错: {str(e)}")
        
        # 提交更改
        if processed_records > 0:
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                errors.append(f"提交数据库时出错: {str(e)}")
        
        return {
            'processed_records': processed_records,
            'created_students': created_students,
            'created_exams': created_exams,
            'updated_scores': updated_scores,
            'errors': errors
        }
    
    def process_student_file(self, file_path):
        """处理学生文件"""
        # 获取文件扩展名
        ext = file_path.split('.')[-1].lower()
        
        # 读取文件
        if ext == 'csv':
            df = pd.read_csv(file_path)
        else:  # xlsx or xls
            df = pd.read_excel(file_path)
        
        # 标准化列名
        column_mapping = {
            'student_id': ['student_id', 'studentid', 'id', '学生id', '学号'],
            'student_name': ['student_name', 'studentname', 'name', '学生姓名', '姓名'],
            'gender': ['gender', 'sex', '性别'],
            'class_name': ['class_name', 'classname', '班级名称', '班级'],
            'contact': ['contact', 'phone', 'email', '联系方式', '电话', '邮箱'],
            'join_date': ['join_date', 'joindate', 'enroll_date', '入学日期', '加入日期']
        }
        
        # 尝试标准化列名
        renamed_columns = {}
        for standard_name, possible_names in column_mapping.items():
            for col in df.columns:
                if col.lower() in possible_names:
                    renamed_columns[col] = standard_name
        
        if renamed_columns:
            df = df.rename(columns=renamed_columns)
        
        # 确保必要的列存在
        required_columns = ['student_id', 'student_name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"文件缺少必要的列: {', '.join(missing_columns)}")
        
        # 处理数据
        processed_records = 0
        errors = []
        created_students = 0
        updated_students = 0
        
        for idx, row in df.iterrows():
            try:
                # 获取学生基本信息
                student_id = str(row['student_id']).strip()
                student_name = str(row['student_name']).strip()
                
                # 确保学号符合格式要求
                if not student_id.isdigit() or len(student_id) != 9:
                    errors.append(f"行 {idx+1}: 学号 '{student_id}' 格式不正确，应为9位数字")
                    continue
                
                # 解析学号信息
                enrollment_year = "20" + student_id[:2]  # 前两位作为入学年份
                # grade_number = student_id[2:3]  # 第3位作为年级编号
                grade_number=enrollment_year
                class_number = student_id[3:5]  # 第4-5位作为班级编号
                
                # 构建完整学号
                full_student_id = student_id
                
                # 获取其他可选信息
                gender = None
                if 'gender' in df.columns and not pd.isna(row['gender']):
                    gender = str(row['gender']).strip()
                
                join_date = None
                if 'join_date' in df.columns and not pd.isna(row['join_date']):
                    try:
                        join_date = pd.to_datetime(row['join_date']).date()
                    except:
                        # 使用默认入学日期
                        join_date = datetime(int(enrollment_year), 9, 1).date()
                else:
                    # 使用默认入学日期
                    join_date = datetime(int(enrollment_year), 9, 1).date()
                
                contact = None
                if 'contact' in df.columns and not pd.isna(row['contact']):
                    contact = str(row['contact']).strip()
                
                # 确定年级和班级
                grade_name = str(grade_number)
                
                class_name = f"{grade_name}{class_number}班"
                if 'class_name' in df.columns and not pd.isna(row['class_name']):
                    class_name = str(row['class_name']).strip()
                
                # 查找或创建年级
                grade = Grade.query.filter_by(name=grade_name).first()
                if not grade:
                    grade = Grade(name=grade_name, class_count=0)
                    db.session.add(grade)
                    db.session.flush()
                
                # 查找或创建班级
                target_class = Class.query.filter_by(name=class_name, grade_id=grade.id).first()
                if not target_class:
                    target_class = Class(name=class_name, grade_id=grade.id, student_count=0)
                    db.session.add(target_class)
                    db.session.flush()
                
                # 查找或创建学生
                student = Student.query.filter_by(student_id=full_student_id).first()
                if student:
                    # 更新学生信息
                    student.name = student_name
                    if gender:
                        student.gender = gender
                    student.class_id = target_class.id
                    student.grade_id = grade.id
                    student.join_date = join_date
                    updated_students += 1
                else:
                    # 创建新学生
                    student = Student(
                        student_id=full_student_id,
                        name=student_name,
                        gender=gender,
                        class_id=target_class.id,
                        grade_id=grade.id,
                        join_date=join_date
                    )
                    db.session.add(student)
                    created_students += 1
                
                processed_records += 1
                
            except Exception as e:
                errors.append(f"处理行 {idx+1} 时出错: {str(e)}")
        
        # 提交更改
        if processed_records > 0:
            try:
                db.session.commit()
                
                # 更新班级学生数量
                classes = Class.query.all()
                for cls in classes:
                    cls.student_count = Student.query.filter_by(class_id=cls.id).count()
                
                # 更新年级班级数量
                grades = Grade.query.all()
                for grade in grades:
                    grade.class_count = Class.query.filter_by(grade_id=grade.id).count()
                
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                errors.append(f"提交数据库时出错: {str(e)}")
        
        return {
            'processed_records': processed_records,
            'created_students': created_students,
            'updated_students': updated_students,
            'errors': errors
        }