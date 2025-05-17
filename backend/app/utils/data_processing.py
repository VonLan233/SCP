import numpy as np
import pandas as pd
from io import BytesIO
import os

def preprocess_time_series(data, time_steps):
    """
    将时间序列数据转换为LSTM可用的格式
    
    Args:
        data: 原始时间序列数据列表
        time_steps: 时间步长
        
    Returns:
        X: 特征数据，形状为(样本数, 时间步)
        y: 标签数据，形状为(样本数,)
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return X, y

def parse_excel_scores(file_data):
    """解析Excel格式的成绩数据
    
    Args:
        file_data: 上传的文件数据
        
    Returns:
        解析后的成绩数据列表
    """
    try:
        df = pd.read_excel(BytesIO(file_data))
        
        # 验证必要的列
        required_columns = ['student_id', 'exam_id', 'subject', 'score']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"缺少必要的列: {', '.join(missing)}")
        
        # 转换为列表字典
        records = df.to_dict('records')
        
        # 验证数据有效性
        for i, record in enumerate(records):
            # 验证学生ID
            if not record.get('student_id'):
                raise ValueError(f"第{i+2}行: 学生ID不能为空")
                
            # 验证考试ID
            if not record.get('exam_id'):
                raise ValueError(f"第{i+2}行: 考试ID不能为空")
                
            # 验证科目
            if not record.get('subject'):
                raise ValueError(f"第{i+2}行: 科目不能为空")
                
            # 验证分数
            try:
                score = float(record.get('score', 0))
                if not (0 <= score <= 100):
                    raise ValueError(f"第{i+2}行: 分数必须在0-100之间")
                record['score'] = score
            except (ValueError, TypeError):
                raise ValueError(f"第{i+2}行: 分数格式无效")
        
        return records
    except Exception as e:
        raise ValueError(f"解析Excel文件失败: {str(e)}")

def normalize_data(data, feature_range=(0, 1)):
    """归一化数据
    
    Args:
        data: 原始数据
        feature_range: 归一化范围
        
    Returns:
        归一化后的数据，最小值，最大值
    """
    min_val = np.min(data)
    max_val = np.max(data)
    
    # 防止除零错误
    if max_val == min_val:
        return np.zeros_like(data), min_val, max_val
    
    scaled_data = (data - min_val) / (max_val - min_val)
    scaled_data = scaled_data * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return scaled_data, min_val, max_val

def denormalize_data(data, min_val, max_val, feature_range=(0, 1)):
    """反归一化数据
    
    Args:
        data: 归一化后的数据
        min_val: 原始数据最小值
        max_val: 原始数据最大值
        feature_range: 归一化使用的范围
        
    Returns:
        反归一化后的数据
    """
    # 防止除零错误
    if max_val == min_val:
        return np.full_like(data, min_val)
    
    return ((data - feature_range[0]) / (feature_range[1] - feature_range[0])) * (max_val - min_val) + min_val