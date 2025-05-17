# predictor/linear_regression.py

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LinearRegressionPredictor:
    """线性回归预测器，用于预测学生成绩"""
    
    def __init__(self, model_path="predictor/models", confidence_level=0.95,**kwargs):
        """
        初始化线性回归预测器
        
        参数:
            model_path: 模型存储路径
            confidence_level: 置信区间水平，默认95%
        """
        self.model_path = model_path
        self.confidence_level = confidence_level
        self.model = LinearRegression()
        self.is_trained = False
        self.scaler = StandardScaler()
        
        # 确保目录存在
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        logger.info(f"初始化线性回归预测器: confidence_level={confidence_level}, 额外参数={kwargs}")
    
    def train_and_evaluate_for_student(self, df, test_size=0.2, val_size=0.0, epochs=None, **kwargs):
        """
        训练学生模型并评估性能
        
        参数:
            df: 包含学生成绩数据的DataFrame
            test_size: 测试集比例
            val_size: 验证集比例
            epochs: 不使用，保留参数兼容性
            
        返回:
            model: 训练好的模型
            metrics: 评估指标
        """
        try:
            logger.info(f"开始线性回归模型训练，数据大小: {len(df)}")
            
            # 提取特征和目标变量
            X = np.array(range(len(df))).reshape(-1, 1)  # 时间序列索引作为特征
            y = df['score'].values
            
            # 特征标准化
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # 训练模型
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # 预测并计算指标
            y_pred = self.model.predict(X_scaled)
            mse = np.mean((y - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y - y_pred))
            r2 = self.model.score(X_scaled, y)
            
            # 保存模型和指标
            self._save_model()
            
            metrics = {
                "MSE": float(mse),
                "RMSE": float(rmse),
                "MAE": float(mae),
                "R2": float(r2),
                "Coefficient": float(self.model.coef_[0]),
                "Intercept": float(self.model.intercept_)
            }
            
            logger.info(f"线性回归模型训练完成，指标: {metrics}")
            return self.model, metrics
            
        except Exception as e:
            logger.error(f"线性回归模型训练失败: {str(e)}")
            return None, None
    
    def predict_next_exam(self, df, subject=None):
        """
        预测下一次考试成绩
        
        参数:
            df: 包含历史成绩的DataFrame
            subject: 科目名称，用于非线性调整
            
        返回:
            prediction: 预测分数
            confidence_interval: 置信区间 (lower, upper)
        """
        try:
            if not self.is_trained:
                self._load_model()
                
            if not self.is_trained:
                logger.warning("模型未训练，无法预测")
                return None, (None, None)
            
            # 准备特征
            next_index = len(df)
            X_next = np.array([[next_index]])
            X_next_scaled = self.scaler.transform(X_next)
            
            # 基础预测
            predicted_score = float(self.model.predict(X_next_scaled)[0])
            
            # 应用非线性调整: 高分区更难提升，低分区更容易提升
            original_pred = predicted_score
            adjusted_pred = self._apply_non_linear_adjustment(predicted_score, subject)
            
            # 基于历史误差计算置信区间
            X = np.array(range(len(df))).reshape(-1, 1)
            X_scaled = self.scaler.transform(X)
            y = df['score'].values
            y_pred = self.model.predict(X_scaled)
            
            # 计算预测误差
            errors = y - y_pred
            std_error = np.std(errors)
            
            # 使用t分布计算置信区间
            from scipy import stats
            alpha = 1 - self.confidence_level
            t_value = stats.t.ppf(1 - alpha/2, len(df) - 2)
            
            # 置信区间宽度，随着预测步长增加而扩大
            margin = t_value * std_error
            
            # 获取科目的分数范围
            min_score, max_score = self._get_subject_score_range(subject)
            
            # 确保预测分数在合理范围内
            adjusted_pred = max(min_score, min(max_score, adjusted_pred))
            lower_bound = max(min_score, adjusted_pred - margin)
            upper_bound = min(max_score, adjusted_pred + margin)
            
            logger.info(f"预测下一次考试成绩: 原始={original_pred:.2f}, 调整后={adjusted_pred:.2f}, 区间=[{lower_bound:.2f}, {upper_bound:.2f}]")
            return adjusted_pred, (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            return None, (None, None)
    
    def _apply_non_linear_adjustment(self, score, subject=None):
        """
        应用非线性调整：高分区更难提升，低分区更容易提升
        
        参数:
            score: 预测分数
            subject: 科目名称
            
        返回:
            adjusted_score: 调整后的分数
        """
        min_score, max_score = self._get_subject_score_range(subject)
        mid_range = (min_score + max_score) / 2
        
        # 如果成绩高于中间值，提分难度增加
        if score > mid_range:
            # 分数越高，调整越大
            difficulty_factor = 0.8 - 0.4 * ((score - mid_range) / (max_score - mid_range))
            # 确保系数至少为0.4，避免极端情况
            difficulty_factor = max(0.4, difficulty_factor)
            
            # 如果系数为正，在原有基础上减缓增长
            if self.model.coef_[0] > 0:
                adjusted_score = score - (1 - difficulty_factor) * self.model.coef_[0]
            else:
                # 如果系数为负，在原有基础上加速下降
                adjusted_score = score + difficulty_factor * self.model.coef_[0]
        
        # 如果成绩低于中间值，提分容易
        elif score < mid_range:
            # 分数越低，提升越容易
            boost_factor = 1.0 + 0.5 * ((mid_range - score) / (mid_range - min_score))
            # 限制最大提升系数
            boost_factor = min(1.5, boost_factor)
            
            # 如果系数为正，在原有基础上加速增长
            if self.model.coef_[0] > 0:
                adjusted_score = score + (boost_factor - 1) * self.model.coef_[0]
            else:
                # 如果系数为负，在原有基础上减缓下降
                adjusted_score = score + boost_factor * self.model.coef_[0]
        
        # 中间区域，不做调整
        else:
            adjusted_score = score
            
        return adjusted_score
    
    def _get_subject_score_range(self, subject=None):
        """
        获取不同科目的分数范围
        
        参数:
            subject: 科目名称
            
        返回:
            (min_score, max_score): 最低分和最高分
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
        
        # 返回科目的分数范围，默认使用(0, 100)
        return subject_ranges.get(subject, (0, 100))
    
    def _save_model(self):
        """保存模型和标准化器"""
        try:
            # 保存线性回归模型
            joblib.dump(self.model, os.path.join(self.model_path, 'linear_model.pkl'))
            
            # 保存标准化器
            joblib.dump(self.scaler, os.path.join(self.model_path, 'scaler.pkl'))
            
            # 保存元数据
            metadata = {
                "model_type": "linear_regression",
                "coefficient": float(self.model.coef_[0]),
                "intercept": float(self.model.intercept_),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(os.path.join(self.model_path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
                
            self.is_trained = True
            logger.info(f"模型已保存到 {self.model_path}")
            
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
    
    def _load_model(self):
        """加载模型和标准化器"""
        try:
            model_file = os.path.join(self.model_path, 'linear_model.pkl')
            scaler_file = os.path.join(self.model_path, 'scaler.pkl')
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                self.model = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                self.is_trained = True
                logger.info(f"已加载模型和标准化器")
                return True
            else:
                logger.warning(f"未找到模型文件: {model_file}")
                return False
                
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return False
            
    def analyze_student_performance(self, df):
        """
        分析学生表现
        
        参数:
            df: 包含学生成绩的DataFrame
            
        返回:
            analysis: 分析结果
        """
        try:
            # 提取成绩数据
            scores = df['score'].values
            
            if len(scores) < 3:
                return {
                    "status": "insufficient_data",
                    "message": "数据不足，至少需要3个数据点进行分析"
                }
                
            # 基本统计量
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            # 计算趋势
            X = np.array(range(len(scores))).reshape(-1, 1)
            model = LinearRegression().fit(X, scores)
            slope = model.coef_[0]
            
            # 确定趋势类型
            if slope > 1.5:
                trend_type = "显著提升"
            elif slope > 0.5:
                trend_type = "稳步提升"
            elif slope < -1.5:
                trend_type = "显著下降"
            elif slope < -0.5:
                trend_type = "轻微下降"
            else:
                trend_type = "基本稳定"
                
            # 计算最近趋势与历史趋势
            if len(scores) >= 5:
                recent_scores = scores[-3:]
                historical_scores = scores[:-3]
                
                recent_avg = np.mean(recent_scores)
                historical_avg = np.mean(historical_scores)
                recent_trend = recent_avg - historical_avg
            else:
                recent_avg = mean_score
                historical_avg = mean_score
                recent_trend = 0
                
            # 计算波动性
            coefficient_of_variation = std_score / mean_score * 100
            
            if coefficient_of_variation > 15:
                volatility_level = "高波动"
            elif coefficient_of_variation > 8:
                volatility_level = "中等波动"
            else:
                volatility_level = "低波动"
                
            # 检测异常值
            z_scores = (scores - mean_score) / std_score
            outliers = []
            
            for i, z in enumerate(z_scores):
                if abs(z) > 2:
                    outliers.append({
                        "index": i,
                        "score": float(scores[i]),
                        "z_score": float(z)
                    })
                    
            # 预测下一次成绩
            next_value = len(scores)
            next_score_prediction = float(model.predict([[next_value]])[0])
            
            # 根据波动性计算置信区间
            margin = 1.96 * std_score
            lower_bound = next_score_prediction - margin
            upper_bound = next_score_prediction + margin
            
            # 生成分析结果
            analysis = {
                "statistics": {
                    "average": float(mean_score),
                    "std_dev": float(std_score),
                    "min": float(min_score),
                    "max": float(max_score),
                    "range": float(max_score - min_score)
                },
                "trend": {
                    "value": float(slope),
                    "type": trend_type,
                    "recent_average": float(recent_avg),
                    "historical_average": float(historical_avg),
                    "recent_vs_historical": float(recent_trend)
                },
                "volatility": {
                    "value": float(coefficient_of_variation),
                    "level": volatility_level
                },
                "outliers": outliers,
                "next_exam_prediction": {
                    "score": float(next_score_prediction),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound)
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"分析学生表现失败: {str(e)}")
            return {
                "status": "error",
                "message": f"分析失败: {str(e)}"
            }