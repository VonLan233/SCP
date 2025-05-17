# app/services/predictor.py
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from predictor.improved_lstm import StudentScorePredictor
from scipy import stats

class PredictorService:
    """学生成绩预测服务 - 使用LSTM模型"""
    
    def __init__(self, model_path='predictor/models'):
        self.model_path = model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # 模型缓存
        self.model_cache = {}
    
    def predict_student_scores(self, student_id, steps=3, model_params=None):
        """使用LSTM模型预测学生未来成绩"""
        from app.models.score import Score
        
        print(f"开始为学生 {student_id} 预测成绩...")
        
        # 获取学生历史成绩
        scores = Score.query.filter_by(student_id=student_id).order_by(Score.exam_id).all()
        
        if not scores or len(scores) < 5:  # LSTM模型需要足够的历史数据
            print(f"学生 {student_id} 的历史数据不足，无法使用LSTM模型预测")
            return self._fallback_prediction(scores, steps)
        
        try:
            # 转换为DataFrame
            scores_data = []
            for score in scores:
                scores_data.append({
                    'student_id': score.student_id,
                    'exam_id': score.exam_id,
                    'score': score.score,
                    'date': score.exam.date.strftime('%Y-%m-%d') if score.exam and score.exam.date else None,
                    'subject': score.subject
                })
            
            df = pd.DataFrame(scores_data)
            
            # 按学科分组处理
            subjects = df['subject'].unique()
            
            # 如果有多个学科，我们需要为每个学科创建单独的模型
            if len(subjects) > 1:
                all_results = []
                for subject in subjects:
                    subject_df = df[df['subject'] == subject]
                    if len(subject_df) >= 5:  # 确保每个学科有足够的数据
                        subject_scores = [s for s in scores if s.subject == subject]
                        subject_result = self._predict_for_subject(
                            student_id, subject, subject_df, subject_scores, steps, model_params
                        )
                        all_results.extend(subject_result)
                
                # 按照考试ID排序
                all_results.sort(key=lambda x: x['examId'])
                return all_results
            else:
                # 单一学科情况，正常处理
                subject = subjects[0] if len(subjects) > 0 else "未知"
                return self._predict_for_subject(student_id, subject, df, scores, steps, model_params)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"LSTM预测发生错误: {str(e)}，使用备用预测方法")
            return self._fallback_prediction(scores, steps)
    
    def _predict_for_subject(self, student_id, subject, df, scores, steps, model_params):
        """为特定学科预测成绩"""
        print(f"预测学生 {student_id} 的 {subject} 科目成绩...")
        
        # 创建LSTM预测器
        student_model_path = os.path.join(self.model_path, f'student_{student_id}_{subject}')
        if not os.path.exists(student_model_path):
            os.makedirs(student_model_path)
            
        # 设置模型参数
        seq_length = 5
        if model_params and 'timeSteps' in model_params:
            seq_length = model_params['timeSteps']
            
        # 创建预测器实例
        print(f"创建LSTM预测器，序列长度: {seq_length}")
        predictor = StudentScorePredictor(seq_length=seq_length, model_path=student_model_path)
        
        # 检查是否已有训练好的模型
        model_files = [f for f in os.listdir(student_model_path) if f.endswith('.keras')]
        if not model_files:
            print("没有找到预训练模型，开始训练...")
            # 训练模型
            model, metrics = predictor.train_and_evaluate_for_student(
                df, test_size=0.2, val_size=0.2, epochs=50
            )
            if not model:
                print("模型训练失败，使用备用预测方法")
                return self._fallback_prediction(scores, steps)
        else:
            # 加载模型，如果加载失败，重新训练
            if not predictor.load_model():
                print("模型加载失败，尝试重新训练...")
                model, metrics = predictor.train_and_evaluate_for_student(
                    df, test_size=0.2, val_size=0.2, epochs=50
                )
                if not model:
                    print("模型重新训练失败，使用备用预测方法")
                    return self._fallback_prediction(scores, steps)
        
        # 预测未来成绩
        print("使用LSTM模型进行预测...")
        result = []
        
        # 先添加历史成绩
        for score in scores:
            result.append({
                'examId': score.exam_id,
                'date': score.exam.date.strftime('%Y-%m-%d') if score.exam and score.exam.date else None,
                'actual': score.score,
                'predicted': None,
                'lower': None,
                'upper': None,
                'subject': score.subject
            })
        
        # 使用模型预测未来成绩
        last_exam = scores[-1]
        for i in range(steps):
            next_score, confidence_interval = predictor.predict_next_exam(df)
            
            # 创建预测结果
            prediction = {
                'examId': last_exam.exam_id + i + 1,
                'date': (datetime.now() + timedelta(days=30 * (i + 1))).strftime('%Y-%m-%d'),
                'actual': None,
                'predicted': round(next_score, 1) if next_score is not None else None,
                'lower': round(confidence_interval[0], 1) if confidence_interval is not None else None,
                'upper': round(confidence_interval[1], 1) if confidence_interval is not None else None,
                'subject': subject
            }
            
            result.append(prediction)
            
            # 添加预测结果到DataFrame以用于下一次预测
            new_row = {
                'student_id': student_id,
                'exam_id': last_exam.exam_id + i + 1,
                'score': next_score,
                'date': (datetime.now() + timedelta(days=30 * (i + 1))).strftime('%Y-%m-%d'),
                'subject': subject
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        print(f"LSTM预测完成，生成了 {steps} 步预测")
        return result
    
    def _fallback_prediction(self, scores, steps=3):
        """备用的简单预测方法，当LSTM模型不可用时使用"""
        print("使用备用预测方法...")
        
        if not scores:
            return []
            
        # 转换为列表形式
        result = []
        for score in scores:
            result.append({
                'examId': score.exam_id,
                'date': score.exam.date.strftime('%Y-%m-%d') if score.exam and score.exam.date else None,
                'actual': score.score,
                'predicted': None,
                'lower': None,
                'upper': None,
                'subject': score.subject
            })
        
        # 按学科分组处理
        subjects = set(score.subject for score in scores)
        
        if len(subjects) > 1:
            # 多个学科分别预测
            all_results = list(result)  # 复制历史记录
            
            for subject in subjects:
                subject_scores = [s for s in scores if s.subject == subject]
                if len(subject_scores) > 0:
                    # 跳过历史记录，只要预测结果
                    subject_predictions = self._fallback_predict_for_subject(subject_scores, steps)[len(subject_scores):]
                    all_results.extend(subject_predictions)
            
            # 按照考试ID排序
            all_results.sort(key=lambda x: x['examId'])
            return all_results
        else:
            # 单一学科
            return self._fallback_predict_for_subject(scores, steps)
    
    def _fallback_predict_for_subject(self, scores, steps=3):
        """为单一学科进行备用预测"""
        if not scores:
            return []
            
        # 转换为列表形式
        result = []
        for score in scores:
            result.append({
                'examId': score.exam_id,
                'date': score.exam.date.strftime('%Y-%m-%d') if score.exam and score.exam.date else None,
                'actual': score.score,
                'predicted': None,
                'lower': None,
                'upper': None,
                'subject': score.subject
            })
        
        # 获取最后一次考试信息
        last_exam = scores[-1]
        
        # 使用改进的备用预测方法
        if len(scores) >= 3:
            # 获取最近的成绩记录
            recent_scores = [s.score for s in scores[-5:] if s.score is not None]
            
            # 计算移动平均和标准差
            avg_score = np.mean(recent_scores)
            std_score = np.std(recent_scores) if len(recent_scores) > 1 else 5.0
            
            # 计算趋势（线性回归）
            if len(recent_scores) >= 3:
                x = np.arange(len(recent_scores))
                y = np.array(recent_scores)
                try:
                    slope, _, _, _, _ = stats.linregress(x, y)
                except:
                    slope = 0
            else:
                slope = 0
                
            for i in range(steps):
                # 基于趋势的预测
                predicted_score = avg_score + slope * (i + 1)
                
                # 增加随机波动以模拟真实情况
                noise = np.random.normal(0, std_score * 0.3)
                predicted_score = predicted_score + noise
                
                # 限制分数在合理范围内
                predicted_score = max(60, min(100, predicted_score))
                
                # 设置合理的置信区间
                ci_width = std_score * 1.96 + i * 1.0  # 随着预测步长增加，不确定性增加
                lower_bound = max(60, predicted_score - ci_width)
                upper_bound = min(100, predicted_score + ci_width)
                
                # 创建预测结果
                prediction = {
                    'examId': last_exam.exam_id + i + 1,
                    'date': (datetime.now() + timedelta(days=30 * (i + 1))).strftime('%Y-%m-%d'),
                    'actual': None,
                    'predicted': round(predicted_score, 1),
                    'lower': round(lower_bound, 1),
                    'upper': round(upper_bound, 1),
                    'subject': last_exam.subject
                }
                result.append(prediction)
        else:
            # 数据太少，使用极简预测
            base_score = last_exam.score or 75
            for i in range(steps):
                # 简单预测，加上小的随机波动
                predicted_score = base_score + np.random.uniform(-3, 3)
                predicted_score = max(60, min(100, predicted_score))
                
                # 宽松的置信区间
                lower_bound = max(60, predicted_score - 10)
                upper_bound = min(100, predicted_score + 10)
                
                # 创建预测结果
                prediction = {
                    'examId': last_exam.exam_id + i + 1,
                    'date': (datetime.now() + timedelta(days=30 * (i + 1))).strftime('%Y-%m-%d'),
                    'actual': None,
                    'predicted': round(predicted_score, 1),
                    'lower': round(lower_bound, 1),
                    'upper': round(upper_bound, 1),
                    'subject': last_exam.subject
                }
                result.append(prediction)
                base_score = predicted_score
        
        return result