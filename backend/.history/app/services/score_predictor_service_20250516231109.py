# app/services/score_predictor_service.py
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
import traceback

from predictor.improved_lstm import StudentScorePredictor
from predictor import create_model, DEFAULT_MODEL_PARAMS, DEFAULT_TRAINING_CONFIG
from app.models.score import Score

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScorePredictorService:
    """
    学生成绩预测服务 - 高级版
    
    此服务集成了多种预测模型，提供更全面的成绩预测和分析功能：
    1. 支持LSTM、GRU和集成模型
    2. 提供成绩预测、趋势分析和异常检测
    3. 包含模型性能评估和可视化
    4. 支持批量预测和实时反馈
    """
    
    def __init__(self, model_path='predictor/models', default_model_type='improved_lstm'):
        """
        初始化预测服务
        
        参数:
            model_path: 模型存储路径
            default_model_type: 默认模型类型
        """
        self.model_path = model_path
        self.default_model_type = default_model_type
        self.model_cache = {}  # 模型缓存
        self.scalers_cache = {}  # 数据缩放器缓存
        
        # 确保目录存在
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        logger.info(f"成绩预测服务初始化完成，使用默认模型: {default_model_type}")
    
    def get_predictor_instance(self, student_id, subject='总分', model_type=None, model_params=None):
        """
        获取或创建预测器实例，按科目区分
        
        参数:
            student_id: 学生ID
            subject: 学科名称，默认为'总分'
            model_type: 模型类型，默认使用self.default_model_type
            model_params: 模型参数
            
        返回:
            predictor: 预测器实例
        """
        # 如果没有指定模型类型，使用默认模型类型
        if model_type is None:
            model_type = self.default_model_type
            
        # 如果没有指定参数，初始化一个空字典
        if model_params is None:
            model_params = {}
        
        # 创建缓存键，加入科目信息
        cache_key = f"student_{student_id}_{subject}_{model_type}"
        
        # 检查缓存中是否有该模型
        if cache_key in self.model_cache:
            logger.info(f"从缓存获取学生 {student_id} 的 {subject} 科目 {model_type} 模型")
            return self.model_cache[cache_key]
        
        # 设置模型路径，加入科目信息
        student_model_path = os.path.join(self.model_path, f'student_{student_id}_{subject}')
        if not os.path.exists(student_model_path):
            os.makedirs(student_model_path)
        
        # 如果没有指定参数，使用默认参数
        if model_params is None:
            model_params = DEFAULT_MODEL_PARAMS.get(model_type, {})
        
        # 设置模型路径
        model_params['model_path'] = student_model_path
        
        # 创建预测器实例
        if model_type == 'student_predictor':
            seq_length = model_params.get('seq_length', 5)
            predictor = StudentScorePredictor(seq_length=seq_length, model_path=student_model_path)
        else:
            # 使用工厂函数创建其他类型的模型
            predictor = create_model(model_type, model_params)
        
        # 将模型添加到缓存
        self.model_cache[cache_key] = predictor
        
        logger.info(f"创建学生 {student_id} 的 {subject} 科目 {model_type} 模型")
        return predictor
    
    def batch_predict_students(self, student_ids, steps=1, model_params=None, model_type=None):
        """
        批量预测多个学生的未来成绩
        
        参数:
            student_ids: 学生ID列表
            steps: 预测步数
            model_params: 模型参数
            model_type: 模型类型
            
        返回:
            results: 按学生ID组织的预测结果字典
        """
        results = {}
        
        for student_id in student_ids:
            try:
                result = self.predict_student_scores(
                    student_id=student_id,
                    steps=steps,
                    model_params=model_params,
                    model_type=model_type
                )
                results[student_id] = {
                    'success': True,
                    'data': result
                }
            except Exception as e:
                logger.error(f"预测学生 {student_id} 成绩失败: {str(e)}")
                results[student_id] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def predict_student_scores(self, student_id, steps=1, subject='总分', model_params=None, model_type=None):
        """
        预测学生未来成绩，按科目区分
        
        参数:
            student_id: 学生ID
            steps: 预测步数
            subject: 学科名称，默认为'总分'
            model_params: 模型参数
            model_type: 模型类型
            
        返回:
            result: 预测结果列表
        """
        try:
            logger.info(f"开始为学生 {student_id} 预测 {subject} 科目未来 {steps} 步成绩...")
                    
            # 添加诊断代码，检查学生是否存在
            from app.models.student import Student
            student = Student.query.filter_by(student_id=student_id).first()
            if not student:
                logger.error(f"学生ID {student_id} 不存在!")
                return []
            else:
                logger.info(f"找到学生: {student.name}, 学号: {student.student_id}")
                
            # 获取学生指定科目的历史成绩
            scores = Score.query.filter_by(student_id=student_id, subject=subject).order_by(Score.exam_id).all()
            
            # 记录更详细的信息
            logger.info(f"学生 {student_id} 的 {subject} 科目历史成绩记录数: {len(scores) if scores else 0}")
            
            if not scores or len(scores) < 5:  # 模型需要足够的历史数据
                logger.warning(f"学生 {student_id} 的 {subject} 科目历史数据不足（{len(scores) if scores else 0}条），无法使用高级模型")
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
                
                # 获取预测器实例
                predictor = self.get_predictor_instance(
                    student_id=student_id, 
                    model_type=model_type,
                    model_params=model_params,
                    subject=subject
                )
                
                # 检查是否需要训练模型
                model_files = []
                student_model_path = os.path.join(self.model_path, f'student_{student_id}')
                
                if os.path.exists(student_model_path):
                    model_files = [f for f in os.listdir(student_model_path) if f.endswith('.keras')]
                
                if not model_files or model_params.get('force_retrain', False):
                    # 训练模型
                    logger.info(f"训练学生 {student_id} 的模型...")
                    
                    # 使用高级接口进行训练
                    if isinstance(predictor, StudentScorePredictor):
                        model, metrics = predictor.train_and_evaluate_for_student(
                            df, 
                            test_size=0.2, 
                            val_size=0.2, 
                            epochs=model_params.get('epochs', 50)
                        )
                        
                        if not model:
                            logger.warning(f"学生 {student_id} 的模型训练失败，使用备用预测方法")
                            return self._fallback_prediction(scores, steps)
                    else:
                        # 对于其他类型的模型，可能需要更具体的训练代码
                        # 这里使用备用预测方法
                        logger.warning(f"不支持训练 {type(predictor).__name__} 类型的模型，使用备用预测方法")
                        return self._fallback_prediction(scores, steps)
                
                # 准备预测结果
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
                
                # 预测未来成绩
                # 预测未来成绩
                # 预测未来成绩
                logger.info(f"使用线性回归预测学生 {student_id} 的未来 {steps} 步成绩...")
                last_exam = scores[-1]

                # 使用线性回归预测
                
                # 在获取学生成绩的代码后添加
                if subject == '总分':
                    # 获取该学生的所有科目
                    all_subjects = set(Score.query.filter_by(student_id=student_id).with_entities(Score.subject).distinct())
                    all_subjects.discard('总分')  # 排除总分本身
                    
                    # 如果有多个科目，分别预测每个科目然后求和
                    if len(all_subjects) > 1:
                        logger.info(f"总分预测: 将对 {len(all_subjects)} 个科目分别预测并求和")
                        
                        # 存储所有科目的预测结果
                        subject_predictions = {}
                        
                        # 对每个科目进行预测
                        for sub in all_subjects:
                            sub_scores = Score.query.filter_by(student_id=student_id, subject=sub).order_by(Score.exam_id).all()
                            
                            if len(sub_scores) >= 3:  # 确保有足够的历史数据
                                sub_data = []
                                for score in sub_scores:
                                    sub_data.append({
                                        'student_id': score.student_id,
                                        'exam_id': score.exam_id,
                                        'score': score.score,
                                        'date': score.exam.date.strftime('%Y-%m-%d') if score.exam and score.exam.date else None,
                                        'subject': score.subject
                                    })
                                
                                sub_df = pd.DataFrame(sub_data)
                                
                                # 预测该科目的未来成绩
                                sub_predictions = self._linear_regression_predict(sub_df, steps=steps, subject=sub)
                                subject_predictions[sub] = sub_predictions
                            else :
                                logger.warning(f"科目 {sub} 的历史数据不足（{len(sub_scores) if sub_scores else 0}条），无法进行预测")
                        
                        # 如果成功预测了至少一个科目
                        if subject_predictions:
                            # 为每一步的预测计算总分
                            for i in range(steps):
                                total_score = 0
                                lower_sum = 0
                                upper_sum = 0
                                
                                # 累加各科目的预测分数
                                for sub, preds in subject_predictions.items():
                                    if i < len(preds):
                                        total_score += preds[i][0]
                                        lower_sum += preds[i][1][0]
                                        upper_sum += preds[i][1][1]
                                
                                # 创建总分预测结果
                                total_prediction = {
                                    'examId': last_exam.exam_id + i + 1,
                                    'date': (datetime.now() + timedelta(days=30 * (i + 1))).strftime('%Y-%m-%d'),
                                    'actual': None,
                                    'predicted': round(total_score, 1),
                                    'lower': round(lower_sum, 1),
                                    'upper': round(upper_sum, 1),
                                    'subject': '总分'
                                }
                                
                                result.append(total_prediction)
                            
                            logger.info(f"总分预测完成，基于 {len(subject_predictions)} 个科目")
                            return result

                predictions = self._linear_regression_predict(df, steps=steps, subject=subject)

                for i, (prediction, confidence_interval) in enumerate(predictions):
                    # 创建预测结果
                    prediction_result = {
                        'examId': last_exam.exam_id + i + 1,
                        'date': (datetime.now() + timedelta(days=30 * (i + 1))).strftime('%Y-%m-%d'),
                        'actual': None,
                        'predicted': round(prediction, 1),
                        'lower': round(confidence_interval[0], 1),
                        'upper': round(confidence_interval[1], 1),
                        'subject': subject
                    }
                    
                    result.append(prediction_result)
                    
                    # 添加预测结果到DataFrame以用于下一次预测
                    new_row = {
                        'student_id': student_id,
                        'exam_id': last_exam.exam_id + i + 1,
                        'score': prediction,
                        'date': (datetime.now() + timedelta(days=30 * (i + 1))).strftime('%Y-%m-%d'),
                        'subject': subject
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                
                logger.info(f"学生 {student_id} 的预测完成，生成了 {steps} 步预测")
                for i, item in enumerate(result):
                    if 'predicted' in item and hasattr(item['predicted'], 'numpy'):
                        result[i]['predicted'] = float(item['predicted'].numpy())
                    if 'lower' in item and hasattr(item['lower'], 'numpy'):
                        result[i]['lower'] = float(item['lower'].numpy())
                    if 'upper' in item and hasattr(item['upper'], 'numpy'):
                        result[i]['upper'] = float(item['upper'].numpy())
                
                return result
                
            except Exception as e:
                logger.error(f"预测过程中发生错误: {str(e)}")
                logger.error(traceback.format_exc())
                logger.info(f"使用备用预测方法")
                return self._fallback_prediction(scores, steps)
                
        except Exception as e:
            logger.error(f"预测服务发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def _fallback_prediction(self, scores, steps=1):
        """
        备用预测方法，当高级模型不可用时使用
        
        参数:
            scores: 历史成绩记录列表
            steps: 预测步数
            
        返回:
            result: 预测结果列表
        """
        logger.info("使用备用预测方法...")
        
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
        base_score = last_exam.score or 75
        
        # 简单预测方法：使用均值和线性趋势
        if len(scores) >= 3:
            # 计算平均增长趋势
            recent_scores = [s.score for s in scores[-5:] if s.score is not None]
            if len(recent_scores) >= 3:
                # 计算最近几次的平均增长
                diffs = [recent_scores[i] - recent_scores[i-1] for i in range(1, len(recent_scores))]
                avg_growth = sum(diffs) / len(diffs)
            else:
                avg_growth = 0
        else:
            avg_growth = 0
        
        # 生成预测
        import random
        for i in range(steps):
            # 基于过去趋势的预测，加上随机波动
            predicted_score = base_score + avg_growth * (i+1) + random.uniform(-5, 5)
            if (last_exam.subject == "语文" or last_exam.subject == "数学" or last_exam.subject == "英语"):
                # 根据科目难度调整分数
                predicted_score *= self._adjust_by_difficulty(predicted_score, subject=last_exam.subject)
                # 生成置信区间
                confidence_interval = random.randint(5,8) + i * 2  # 随着预测步长增加，不确定性增加
                confidence_interval = confidence_interval / 100
                lower_bound = max(0, predicted_score *(1- confidence_interval))
                upper_bound =min(150, predicted_score*(1 + confidence_interval))
            elif (last_exam.subject != "总分"):
            # 根据科目难度调整分数
                predicted_score *= self._adjust_by_difficulty(predicted_score, subject=last_exam.subject)
                # 生成置信区间
                confidence_interval = random.randint(5,8) + i * 2  # 随着预测步长增加，不确定性增加
                confidence_interval = confidence_interval / 100
                lower_bound = max(0, predicted_score *(1- confidence_interval))
                upper_bound =min(100, predicted_score*(1 + confidence_interval))
            else:
                # 生成置信区间
                predicted_score *= self._adjust_by_difficulty(predicted_score, subject=last_exam.subject)
                # 生成置信区间
                confidence_interval = random.randint(5,8) + i * 2  # 随着预测步长增加，不确定性增加
                confidence_interval = confidence_interval / 100
                lower_bound = max(0, predicted_score *(1- confidence_interval))
                upper_bound =min(750, predicted_score*(1 + confidence_interval))
            # scaled_prediction, (lower_bound, upper_bound) = self._apply_subject_scaling(predicted_score, (lower_bound, upper_bound), subject=last_exam.subject)
            
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
            
            # 更新基准
            base_score = predicted_score
        
        logger.info("备用预测方法完成")
        return result
    
    def _adjust_by_difficulty(self, score, subject=None):
            """根据科目难度调整分数"""
            
            # 定义科目难度系数 (< 1.0意味着更难获得高分)
            subject_difficulty = {
                "语文": 0.6,
                "数学": 0.6,
                "英语": 0.6,
                "物理": 0.6,
                "化学": 0.6,
                "生物": 0.6,
                "政治": 0.6,
                "历史": 0.6,
                "地理": 0.6,
                "技术": 0.6,
                "总分": 0.6,
                "未知": 0.6
            }
            
            # 获取难度系数
            difficulty = subject_difficulty.get(subject, subject_difficulty["未知"])
            
            # 高分区域更难获得
            if score > 90:
                # 分数越高，难度系数效果越明显
                adjustment_factor = difficulty * (1 - (score - 90) / 100)
                adjusted_score = 90 + (score - 90) * adjustment_factor
            elif score > 80:
                # 80-90分区间有轻微调整
                adjustment_factor = difficulty * 0.95
                adjusted_score = 80 + (score - 80) * adjustment_factor
            else:
                # 低于80分的区间基本保持不变
                adjusted_score = score
            
            return adjusted_score/100
        
    def _apply_subject_scaling(self, prediction, confidence_interval, subject=None):
        """
        根据科目总分对预测结果进行转换和限制
        
        参数:
        prediction: 原始预测分数
        confidence_interval: 原始置信区间 (lower, upper)
        subject: 科目名称
        
        返回:
        scaled_prediction: 转换后的预测分数
        scaled_interval: 转换后的置信区间 (lower, upper)
        """
        lower_bound, upper_bound = confidence_interval
        
        # 定义科目总分映射
        subject_total_scores = {
            "语文": 150,
            "数学": 150, 
            "英语": 150,
            "物理": 100,
            "化学": 100,
            "生物": 100,
            "政治": 100,
            "历史": 100,
            "地理": 100,
            "技术": 100,
            "总分": 750  # 假设总分是750
        }
        
        subject_difficulty = {
            "语文": 0.6,
            "数学": 0.6,
            "英语": 0.6,
            "物理": 0.6,
            "化学": 0.6,
            "生物": 0.6,
            "政治": 0.6,
            "历史": 0.6,
            "地理": 0.6,
            "总分": 0.6
        }
        
            # 如果没有指定科目或科目不在列表中，默认使用百分制(0-100)
        if subject is None or subject not in subject_total_scores:
            # 应用默认难度系数
            difficulty = 0.85
            # 限制分数在0-100范围内，并应用难度系数
            max_score = 100 * difficulty
            prediction = max(0, min(max_score, prediction))
            # 缩小置信区间宽度
            interval_width = upper_bound - lower_bound
            reduced_width = interval_width * 0.9  # 减少30%的区间宽度
            mid_point = (upper_bound + lower_bound) / 2
            lower_bound = max(0, mid_point - reduced_width / 2)
            upper_bound = min(max_score, mid_point + reduced_width / 2)
            return prediction, (lower_bound, upper_bound)
        
        # 获取科目总分和难度系数
        total_score = subject_total_scores.get(subject, 100)
        difficulty = subject_difficulty.get(subject, 0.85)
        
        # 应用难度系数，满分变得更加困难
        max_score = total_score * difficulty
        
        # 转换预测分数：先转为百分比，再乘以调整后的最大分数
        prediction_percentage = prediction / 100.0
        scaled_prediction = prediction_percentage * max_score
        
        # 同样转换置信区间，并缩小区间宽度
        lower_percentage = lower_bound / 100.0
        upper_percentage = upper_bound / 100.0
        
        # 缩小置信区间宽度
        interval_width = upper_percentage - lower_percentage
        reduced_width = interval_width * 0.7  # 减少30%的区间宽度
        mid_point = (upper_percentage + lower_percentage) / 2
        lower_percentage = max(0, mid_point - reduced_width / 2)
        upper_percentage = min(difficulty, mid_point + reduced_width / 2)
        
        scaled_lower = lower_percentage * total_score
        scaled_upper = upper_percentage * total_score
        
        # 限制在合理范围内
        scaled_prediction = max(0, min(max_score, scaled_prediction))
        scaled_lower = max(0, min(max_score, scaled_lower))
        scaled_upper = max(0, min(max_score, scaled_upper))
        
        return scaled_prediction, (scaled_lower, scaled_upper)
    
    # 在 app/services/score_predictor_service.py 中添加新的预测方法

    def _linear_regression_predict(self, df, steps=1, subject=None):
        """
        使用线性回归预测未来成绩，考虑分数区间的难度差异
        
        参数:
            df: 包含历史成绩的DataFrame
            steps: 预测步数
            subject: 科目名称
            
        返回:
            predictions: 预测结果列表, 包含预测分数和置信区间
        """
        import numpy as np
        from sklearn.linear_model import LinearRegression
        import random
        
        # 提取历史成绩
        X = np.array(range(len(df))).reshape(-1, 1)  # 使用索引作为特征
        y = df['score'].values
        
        # 训练线性回归模型
        model = LinearRegression()
        model.fit(X, y)
        
        # 计算标准误差用于置信区间
        y_pred = model.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        std_error = np.sqrt(mse)
        
        # 预测未来分数
        predictions = []
        last_score = y[-1]
        
        for i in range(steps):
            # 预测下一个点
            next_index = len(df) + i
            next_x = np.array([[next_index]])
            
            # 基础预测值
            base_prediction = float(model.predict(next_x)[0])
            
            # 应用难度调整：高分更难提高，低分更容易提高
            difficulty_adjusted = self._apply_difficulty_adjustment(base_prediction, subject)
            
            # 添加适当的随机波动
            noise = random.uniform(-std_error * 0.5, std_error * 0.5)
            final_prediction = difficulty_adjusted + noise
            
            # 根据科目限制分数范围
            min_score, max_score = self._get_subject_score_range(subject)
            final_prediction = max(min_score, min(max_score, final_prediction))
            
            # 生成置信区间（根据标准误差和预测步长扩大）
            ci_width = std_error * 1.96 * (1 + i * 0.2)  # 置信区间随预测步长增加而扩大
            lower_bound = max(min_score, final_prediction - ci_width)
            upper_bound = min(max_score, final_prediction + ci_width)
            
            predictions.append((final_prediction, (lower_bound, upper_bound)))
        
        return predictions

    def _apply_difficulty_adjustment(self, score, subject=None):
        """
        根据当前分数应用难度调整：高分更难提高，低分更容易提高
        
        参数:
            score: 预测的原始分数
            subject: 科目名称
            
        返回:
            adjusted_score: 调整后的分数
        """
        min_score, max_score = self._get_subject_score_range(subject)
        mid_score = (min_score + max_score) / 2
        range_score = max_score - min_score
        
        # 确定难度系数（高分区系数小，低分区系数大）
        if score > mid_score + range_score * 0.2:  # 高分区
            # 分数越高，难度系数越小（提分越难）
            difficulty_factor = 0.6 - 0.3 * ((score - mid_score) / (range_score / 2))
            # 应用难度系数：减缓增长速度
            adjustment = model.coef_[0] * difficulty_factor if model.coef_[0] > 0 else model.coef_[0]
            adjusted_score = score + adjustment
        elif score < mid_score - range_score * 0.2:  # 低分区
            # 分数越低，提升系数越大（提分越容易）
            boost_factor = 1.2 + 0.4 * ((mid_score - score) / (range_score / 2))
            # 应用提升系数：加速增长
            adjustment = model.coef_[0] * boost_factor if model.coef_[0] > 0 else model.coef_[0] * 0.8
            adjusted_score = score + adjustment
        else:  # 中间区域
            # 中等分数区域，保持正常增长
            adjusted_score = score + model.coef_[0]
        
        return adjusted_score

    def _get_subject_score_range(self, subject=None):
        """
        获取不同科目的分数范围
        
        参数:
            subject: 科目名称
            
        返回:
            (min_score, max_score): 最低分和最高分的元组
        """
        # 定义不同科目的分数范围
        subject_ranges = {
            "语文": (0, 150),
            "数学": (0, 150),
            "英语": (0, 150),
            "物理": (0, 100),
            "化学": (0, 100),
            "生物": (0, 100),
            "政治": (0, 100),
            "历史": (0, 100),
            "地理": (0, 100),
            "总分": (0, 750)  # 假设总分是750
        }
        
        # 返回科目的分数范围，如果科目不在列表中，默认使用(0, 100)
        return subject_ranges.get(subject, (0, 100))
    
    def predict_class_average(self, class_id, steps=1, model_params=None):
        """
        预测班级平均分
        
        参数:
            class_id: 班级ID
            steps: 预测步数
            model_params: 模型参数
            
        返回:
            result: 预测结果列表
        """
        from app.models.class_model import Class
        from app.models.score import Score
        from app.models.student import Student
        
        logger.info(f"预测班级 {class_id} 的平均分...")
        
        # 获取该班级的所有学生
        students = Student.query.filter_by(class_id=class_id).all()
        
        if not students:
            logger.warning(f"班级 {class_id} 没有学生数据")
            return []
        
        # 获取历史考试数据
        all_exams = set()
        for student in students:
            scores = Score.query.filter_by(student_id=student.id).all()
            for score in scores:
                all_exams.add(score.exam_id)
        
        if not all_exams:
            logger.warning(f"班级 {class_id} 没有考试数据")
            return []
        
        # 计算每次考试的班级平均分
        class_averages = []
        for exam_id in sorted(all_exams):
            scores = Score.query.filter(
                Score.exam_id == exam_id,
                Score.student_id.in_([s.id for s in students])
            ).all()
            
            if scores and any(s.score is not None for s in scores):
                valid_scores = [s.score for s in scores if s.score is not None]
                avg_score = sum(valid_scores) / len(valid_scores)
                exam_date = scores[0].exam.date.strftime('%Y-%m-%d') if scores[0].exam and scores[0].exam.date else None
                
                class_averages.append({
                    'examId': exam_id,
                    'date': exam_date,
                    'average': round(avg_score, 1),
                    'subject': scores[0].subject if scores[0].subject else None
                })
        
        # 预测未来平均分
        # 这里可以选择不同的预测方法：
        # 1. 使用所有学生的单独预测结果平均
        # 2. 直接对班级历史平均分进行预测
        # 3. 构建班级专属模型
        
        # 这里使用简单的预测方法
        if len(class_averages) < 3:
            logger.warning(f"班级 {class_id} 的历史数据不足，使用简单预测")
            return self._predict_class_average_simple(class_averages, steps)
        
        try:
            return self._predict_class_average_simple(class_averages, steps)
        except Exception as e:
            logger.error(f"班级预测发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            return self._predict_class_average_simple(class_averages, steps)
    
    def _predict_class_average_simple(self, class_averages, steps=1):
        """
        简单的班级平均分预测方法
        
        参数:
            class_averages: 班级历史平均分列表
            steps: 预测步数
            
        返回:
            result: 预测结果列表
        """
        if not class_averages:
            return []
        
        result = list(class_averages)
        
        # 获取最近的考试数据
        last_average = class_averages[-1]
        base_average = last_average['average']
        last_exam_id = last_average['examId']
        
        # 计算趋势
        if len(class_averages) >= 3:
            recent_avgs = [avg['average'] for avg in class_averages[-5:]]
            diffs = [recent_avgs[i] - recent_avgs[i-1] for i in range(1, len(recent_avgs))]
            avg_growth = sum(diffs) / len(diffs)
        else:
            avg_growth = 0
        
        # 生成预测
        import random
        for i in range(steps):
            predicted_avg = base_average + avg_growth * (i+1) + random.uniform(-2, 3)
            predicted_avg = max(70, min(95, predicted_avg))  # 限制在合理范围
            
            prediction = {
                'examId': last_exam_id + i + 1,
                'date': (datetime.now() + timedelta(days=30 * (i + 1))).strftime('%Y-%m-%d'),
                'average': round(predicted_avg, 1),
                'subject': last_average['subject']
            }
            
            result.append(prediction)
            base_average = predicted_avg
        
        return result
    
    def check_model_status(self, student_id, model_type=None):
        """
        检查学生模型的训练状态
        
        参数:
            student_id: 学生ID
            model_type: 模型类型
            
        返回:
            status: 状态信息字典
        """
        # 如果没有指定模型类型，使用默认模型类型
        if model_type is None:
            model_type = self.default_model_type
            
        student_model_path = os.path.join(self.model_path, f'student_{student_id}')
        
        status = {
            'trained': False,
            'metrics': None,
            'last_trained': None,
            'model_type': model_type,
            'available_data': 0
        }
        
        # 检查学生数据
        try:
            scores = Score.query.filter_by(student_id=student_id).all()
            status['available_data'] = len(scores)
        except:
            pass
        
        # 检查模型是否已训练
        if os.path.exists(student_model_path):
            model_files = [f for f in os.listdir(student_model_path) if f.endswith('.keras')]
            if model_files:
                status['trained'] = True
                
                # 获取模型最后修改时间
                model_file_path = os.path.join(student_model_path, model_files[0])
                status['last_trained'] = datetime.fromtimestamp(
                    os.path.getmtime(model_file_path)
                ).isoformat()
                
                # 尝试获取模型评估指标
                metrics_file = os.path.join(student_model_path, 'evaluation.json')
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, 'r') as f:
                            status['metrics'] = json.load(f)
                    except:
                        pass
        
        return status
    
    # 修改 train_model 方法

def train_model(self, student_id, subject, model_type=None, model_params=None):
    """
    使用线性回归训练学生模型
    
    参数:
        student_id: 学生ID
        subject: 科目
        model_type: 模型类型 (现在默认为线性回归)
        model_params: 训练参数
        
    返回:
        result: 训练结果
    """
    try:
        # 获取学生成绩数据
        scores = Score.query.filter_by(student_id=student_id, subject=subject).order_by(Score.exam_id).all()
        
        if not scores or len(scores) < 3:  # 线性回归至少需要3个点
            return {
                'success': False,
                'message': f"学生 {student_id} 的 {subject} 科目数据不足，至少需要3条记录，当前仅有 {len(scores) if scores else 0} 条"
            }
        
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
        
        # 训练线性回归模型
        X = np.array(range(len(df))).reshape(-1, 1)
        y = df['score'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # 评估模型
        y_pred = model.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        r2 = model.score(X, y)
        
        # 保存模型和评估指标
        model_dir = os.path.join(self.model_path, f'student_{student_id}_{subject}')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # 保存评估指标
        metrics = {
            'mse': float(mse),
            'r2': float(r2),
            'coefficient': float(model.coef_[0]),
            'intercept': float(model.intercept_)
        }
        
        with open(os.path.join(model_dir, 'linear_regression_metrics.json'), 'w') as f:
            json.dump(metrics, f)
            
        # 可以选择保存模型本身，但对于简单的线性回归，保存系数和截距即可
        # 从缓存中删除旧模型
        cache_key = f"student_{student_id}_{subject}_linear_regression"
        if cache_key in self.model_cache:
            del self.model_cache[cache_key]
            
        return {
            'success': True,
            'message': f"学生 {student_id} 的 {subject} 科目线性回归模型训练成功",
            'metrics': metrics
        }
                
    except Exception as e:
        logger.error(f"模型训练失败: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'message': f"训练过程中发生错误: {str(e)}"
        }
    
    def _simple_performance_analysis(self, scores_data):
        """
        简单的学生表现分析
        
        参数:
            scores_data: 学生成绩数据列表
            
        返回:
            analysis: 分析结果字典
        """
        import numpy as np
        
        # 提取成绩
        scores = [item['score'] for item in scores_data if item['score'] is not None]
        
        if len(scores) < 3:
            return {
                'success': False,
                'message': "有效数据不足，无法进行分析"
            }
            
        # 计算基本统计量
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        
        # 计算趋势
        recent_scores = scores[-3:]
        historical_scores = scores[:-3] if len(scores) > 3 else scores
        
        recent_avg = np.mean(recent_scores)
        historical_avg = np.mean(historical_scores)
        
        trend = recent_avg - historical_avg
        
        # 判断趋势类型
        if trend > 5:
            trend_type = "显著提升"
        elif trend > 2:
            trend_type = "稳步提升"
        elif trend < -5:
            trend_type = "显著下降"
        elif trend < -2:
            trend_type = "轻微下降"
        else:
            trend_type = "基本稳定"
        
        # 计算波动性
        volatility = std_score / avg_score * 100  # 变异系数
        
        if volatility > 15:
            volatility_level = "高波动"
        elif volatility > 8:
            volatility_level = "中等波动"
        else:
            volatility_level = "低波动"
        
        # 异常检测
        z_scores = [(score - avg_score) / std_score for score in scores]
        outliers = [
            (i, scores[i]) 
            for i in range(len(scores)) 
            if abs(z_scores[i]) > 2
        ]
        
        # 生成结果
        analysis = {
            'success': True,
            'analysis': {
                'student_id': scores_data[0]['student_id'] if scores_data else None,
                'statistics': {
                    'average': round(avg_score, 1),
                    'std_dev': round(std_score, 1),
                    'max': max_score,
                    'min': min_score,
                    'range': max_score - min_score
                },
                'trend': {
                    'value': round(trend, 1),
                    'type': trend_type,
                    'recent_average': round(recent_avg, 1),
                    'historical_average': round(historical_avg, 1)
                },
                'volatility': {
                    'value': round(volatility, 1),
                    'level': volatility_level
                },
                'outliers': [
                    {
                        'index': idx,
                        'score': score,
                        'z_score': round(z_scores[idx], 2)
                    }
                    for idx, score in outliers
                ]
            },
            'recommendations': []
        }
        
        # 生成建议
        if trend_type == "显著提升":
            analysis['recommendations'].append("学生进步显著，建议肯定其努力并保持当前学习策略。")
        elif trend_type == "稳步提升":
            analysis['recommendations'].append("学生正在稳步进步，建议继续现有学习方法。")
        elif trend_type == "显著下降":
            analysis['recommendations'].append("学生成绩明显下滑，建议及时干预，找出原因并提供针对性辅导。")
        elif trend_type == "轻微下降":
            analysis['recommendations'].append("学生成绩有轻微下降趋势，建议关注并提供适当指导。")
        
        if volatility_level == "高波动":
            analysis['recommendations'].append("成绩波动较大，建议关注学生学习的稳定性，可能需要更系统的学习方法。")
        
        if outliers:
            if any(score > avg_score + std_score for _, score in outliers):
                analysis['recommendations'].append("学生有异常高分表现，可能在某些知识点上有特别的优势，建议继续发掘潜力。")
            if any(score < avg_score - std_score for _, score in outliers):
                analysis['recommendations'].append("学生有异常低分表现，可能在某些知识点上存在困难，建议针对性补强。")
        
        return analysis
        
    def detect_performance_anomalies(self, class_id):
        """
        检测班级中存在异常表现的学生
        
        参数:
            class_id: 班级ID
            
        返回:
            anomalies: 异常信息字典
        """
        from app.models.student import Student
        from app.models.score import Score
        
        try:
            # 获取班级所有学生
            students = Student.query.filter_by(class_id=class_id).all()
            
            if not students:
                return {
                    'success': False,
                    'message': f"班级 {class_id} 没有学生数据"
                }
            
            # 获取最近的考试ID
            all_scores = Score.query.filter(
                Score.student_id.in_([s.id for s in students])
            ).all()
            
            if not all_scores:
                return {
                    'success': False,
                    'message': f"班级 {class_id} 没有考试数据"
                }
            
            # 按考试ID分组获取最近几次考试
            exam_groups = {}
            for score in all_scores:
                if score.exam_id not in exam_groups:
                    exam_groups[score.exam_id] = []
                exam_groups[score.exam_id].append(score)
            
            # 按时间排序考试ID
            sorted_exam_ids = sorted(exam_groups.keys())
            
            # 获取最近的三次考试
            recent_exam_ids = sorted_exam_ids[-3:] if len(sorted_exam_ids) >= 3 else sorted_exam_ids
            
            # 按学生ID分组成绩
            student_scores = {}
            for exam_id in recent_exam_ids:
                for score in exam_groups[exam_id]:
                    if score.student_id not in student_scores:
                        student_scores[score.student_id] = []
                    student_scores[score.student_id].append({
                        'exam_id': score.exam_id,
                        'score': score.score,
                        'subject': score.subject
                    })
            
            # 计算每个学生的平均分和标准差
            stats = {}
            for student_id, scores in student_scores.items():
                if len(scores) >= 2:  # 至少需要两次考试数据
                    score_values = [s['score'] for s in scores if s['score'] is not None]
                    if score_values:
                        avg = np.mean(score_values)
                        std = np.std(score_values)
                        stats[student_id] = {
                            'average': avg,
                            'std_dev': std,
                            'scores': score_values,
                            'student': next((s for s in students if s.id == student_id), None)
                        }
            
            # 计算班级整体的平均分和标准差
            all_scores_values = []
            for student_stats in stats.values():
                all_scores_values.extend(student_stats['scores'])
            
            class_avg = np.mean(all_scores_values) if all_scores_values else 0
            class_std = np.std(all_scores_values) if all_scores_values else 0
            
            # 检测异常
            anomalies = []
            
            for student_id, student_stats in stats.items():
                student = student_stats['student']
                
                # 检查是否有显著波动
                if student_stats['std_dev'] > class_std * 1.5:
                    anomalies.append({
                        'student_id': student_id,
                        'student_name': student.name if student else 'Unknown',
                        'type': 'high_volatility',
                        'description': f"成绩波动较大 (标准差: {student_stats['std_dev']:.2f})",
                        'scores': student_stats['scores'],
                        'avg_score': student_stats['average']
                    })
                
                # 检查是否显著优于班级
                if student_stats['average'] > class_avg + class_std:
                    anomalies.append({
                        'student_id': student_id,
                        'student_name': student.name if student else 'Unknown',
                        'type': 'outperformer',
                        'description': f"成绩显著高于班级平均 ({student_stats['average']:.1f} vs {class_avg:.1f})",
                        'scores': student_stats['scores'],
                        'avg_score': student_stats['average']
                    })
                
                # 检查是否显著低于班级
                if student_stats['average'] < class_avg - class_std:
                    anomalies.append({
                        'student_id': student_id,
                        'student_name': student.name if student else 'Unknown',
                        'type': 'underperformer',
                        'description': f"成绩显著低于班级平均 ({student_stats['average']:.1f} vs {class_avg:.1f})",
                        'scores': student_stats['scores'],
                        'avg_score': student_stats['average']
                    })
                
                # 检查是否有急剧变化
                if len(student_stats['scores']) >= 3:
                    latest = student_stats['scores'][-1]
                    previous_avg = np.mean(student_stats['scores'][:-1])
                    change = latest - previous_avg
                    
                    if abs(change) > class_std * 1.5:
                        change_type = "提升" if change > 0 else "下降"
                        anomalies.append({
                            'student_id': student_id,
                            'student_name': student.name if student else 'Unknown',
                            'type': 'sudden_change',
                            'description': f"成绩突然{change_type} ({change:.1f}分)",
                            'scores': student_stats['scores'],
                            'avg_score': student_stats['average']
                        })
            
            return {
                'success': True,
                'class_id': class_id,
                'class_stats': {
                    'average': class_avg,
                    'std_dev': class_std,
                    'student_count': len(stats)
                },
                'anomalies': anomalies
            }
                
        except Exception as e:
            logger.error(f"异常检测失败: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'message': f"检测过程中发生错误: {str(e)}"
            }
    
    def generate_predictive_insights(self, class_id):
        """
        生成班级预测性分析洞察
        
        参数:
            class_id: 班级ID
            
        返回:
            insights: 洞察信息字典
        """
        try:
            # 获取班级成绩预测
            class_predictions = self.predict_class_average(class_id, steps=1)
            
            # 检测异常学生
            anomalies = self.detect_performance_anomalies(class_id)
            
            # 生成整体洞察
            insights = {
                'success': True,
                'class_id': class_id,
                'predictions': class_predictions,
                'anomalies': anomalies.get('anomalies', []) if anomalies.get('success', False) else [],
                'improvement_opportunities': [],
                'strengths': [],
                'recommendations': []
            }
            
            # 分析班级趋势
            if len(class_predictions) >= 2:
                historical = [p.get('average') for p in class_predictions if 'average' in p and p.get('actual') is not None]
                predicted = [p.get('average') for p in class_predictions if 'average' in p and p.get('actual') is None]
                
                if historical and predicted:
                    hist_avg = np.mean(historical)
                    pred_avg = np.mean(predicted)
                    trend = pred_avg - hist_avg
                    
                    if trend > 2:
                        insights['predictions_trend'] = {
                            'direction': 'upward',
                            'magnitude': trend,
                            'description': f"班级整体成绩预计将提升约 {trend:.1f} 分"
                        }
                        insights['strengths'].append("班级整体学习态势良好，预计将持续进步")
                    elif trend < -2:
                        insights['predictions_trend'] = {
                            'direction': 'downward',
                            'magnitude': abs(trend),
                            'description': f"班级整体成绩预计将下降约 {abs(trend):.1f} 分"
                        }
                        insights['improvement_opportunities'].append("班级整体成绩可能面临下滑风险，建议加强干预")
                    else:
                        insights['predictions_trend'] = {
                            'direction': 'stable',
                            'magnitude': abs(trend),
                            'description': "班级整体成绩预计将保持稳定"
                        }
            
            # 分析异常表现学生
            outperformers = [a for a in insights['anomalies'] if a['type'] == 'outperformer']
            underperformers = [a for a in insights['anomalies'] if a['type'] == 'underperformer']
            volatile_students = [a for a in insights['anomalies'] if a['type'] == 'high_volatility']
            
            if outperformers:
                insights['strengths'].append(f"班级有 {len(outperformers)} 名学生表现优异，显著高于平均水平")
                if len(outperformers) <= 3:
                    names = ", ".join([a['student_name'] for a in outperformers])
                    insights['strengths'].append(f"优秀学生包括: {names}，可以发挥他们的带动作用")
            
            if underperformers:
                insights['improvement_opportunities'].append(
                    f"班级有 {len(underperformers)} 名学生需要额外关注，成绩显著低于平均水平"
                )
                insights['recommendations'].append("建议对学习困难学生进行分组辅导或一对一辅导")
            
            if volatile_students:
                insights['improvement_opportunities'].append(
                    f"班级有 {len(volatile_students)} 名学生成绩波动较大，学习可能不够稳定"
                )
                insights['recommendations'].append("关注学习不稳定学生，帮助他们建立更系统的学习方法")
            
            # 生成教学建议
            if not insights['recommendations']:
                if outperformers and underperformers:
                    insights['recommendations'].append("可以考虑实施'对子互助'学习方法，让优秀学生帮助学习有困难的同学")
                
                if len(class_predictions) >= 2 and insights.get('predictions_trend', {}).get('direction') == 'downward':
                    insights['recommendations'].append("预测显示班级成绩可能下滑，建议及时调整教学策略，加强薄弱环节")
                
                insights['recommendations'].append("定期进行针对性测验，及时发现和解决学习中的问题")
            
            return insights
            
        except Exception as e:
            logger.error(f"生成洞察失败: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'message': f"生成洞察过程中发生错误: {str(e)}"
            }
    
    def get_service_status(self):
        """
        获取预测服务状态信息
        
        返回:
            status: 状态信息字典
        """
        try:
            # 统计已训练的模型数量
            trained_models = 0
            model_types = {}
            
            if os.path.exists(self.model_path):
                # 遍历所有学生模型目录
                student_dirs = [d for d in os.listdir(self.model_path) 
                                if os.path.isdir(os.path.join(self.model_path, d)) 
                                and d.startswith('student_')]
                
                for student_dir in student_dirs:
                    model_dir = os.path.join(self.model_path, student_dir)
                    model_files = [f for f in os.listdir(model_dir) 
                                  if f.endswith('.keras') or f.endswith('.pkl')]
                    
                    if model_files:
                        trained_models += 1
                        
                        # 检查模型类型
                        model_info_file = os.path.join(model_dir, 'model_info.json')
                        if os.path.exists(model_info_file):
                            try:
                                with open(model_info_file, 'r') as f:
                                    info = json.load(f)
                                    model_type = info.get('model_type', 'unknown')
                                    model_types[model_type] = model_types.get(model_type, 0) + 1
                            except:
                                pass
            
            # 获取缓存的模型数量
            cached_models = len(self.model_cache)
            
            # 统计成绩数据
            from app.models.score import Score
            from app.models.student import Student
            from app.models.class_model import Class
            
            total_students = Student.query.count()
            total_scores = Score.query.count()
            total_classes = Class.query.count()
            
            # 获取默认模型配置
            from predictor import DEFAULT_MODEL_PARAMS, DEFAULT_TRAINING_CONFIG
            
            # 构建状态信息
            status = {
                'success': True,
                'service': {
                    'name': 'ScorePredictorService',
                    'version': '1.0.0',
                    'default_model_type': self.default_model_type,
                    'model_path': self.model_path,
                    'cached_models': cached_models
                },
                'models': {
                    'trained_count': trained_models,
                    'model_types': model_types,
                    'default_params': DEFAULT_MODEL_PARAMS,
                    'training_config': DEFAULT_TRAINING_CONFIG
                },
                'data': {
                    'total_students': total_students,
                    'total_scores': total_scores,
                    'total_classes': total_classes,
                    'avg_scores_per_student': round(total_scores / total_students, 1) if total_students > 0 else 0
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"获取服务状态失败: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'message': f"获取服务状态过程中发生错误: {str(e)}"
            }
    
    def clear_model_cache(self):
        """
        清除模型缓存
        
        返回:
            result: 操作结果
        """
        try:
            cache_size = len(self.model_cache)
            self.model_cache = {}
            
            return {
                'success': True,
                'message': f"已清除 {cache_size} 个缓存模型"
            }
        except Exception as e:
            logger.error(f"清除模型缓存失败: {str(e)}")
            return {
                'success': False,
                'message': f"清除缓存过程中发生错误: {str(e)}"
            }
        
    def analyze_student_performance(self, student_id):
        """
        分析学生表现，包括趋势、波动性和异常检测
        
        参数:
            student_id: 学生ID
            
        返回:
            analysis: 分析结果字典
        """
        try:
            # 获取学生数据
            scores = Score.query.filter_by(student_id=student_id).order_by(Score.exam_id).all()
            
            if not scores or len(scores) < 3:
                return {
                    'success': False,
                    'message': f"学生 {student_id} 的数据不足，至少需要3条记录，当前仅有 {len(scores) if scores else 0} 条"
                }
            
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
            
            # 获取预测器实例
            predictor = self.get_predictor_instance(
                student_id=student_id, 
                model_type='student_predictor',
                subject=scores[0].subject if scores else None
            )
            
            # 使用预测器的分析功能
            if isinstance(predictor, StudentScorePredictor):
                analysis_results = predictor.analyze_student_performance(df)
                return {
                    'success': True,
                    'analysis': analysis_results
                }
            else:
                # 自行进行简单分析
                return self._simple_performance_analysis(scores_data)
                
        except Exception as e:
            logger.error(f"学生表现分析失败: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'message': f"分析过程中发生错误: {str(e)}"
            }