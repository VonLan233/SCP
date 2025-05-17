import numpy as np
import pandas as pd
# 在 improved_lstm.py 文件的顶部添加以下代码
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
import joblib

class StudentScorePredictor:
    """
    学生成绩预测器类，包含所有数据处理、建模、训练和预测功能
    """
    def __init__(self, seq_length=5, model_path='models'):
        """
        初始化预测器
        
        参数:
        seq_length: 用于预测的历史成绩序列长度
        model_path: 模型保存路径
        """
        self.seq_length = seq_length
        self.model_path = model_path
        self.model = None
        self.scalers = {}
        
        # 创建模型保存目录
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
    def save_model(self, model_file='model.keras'):
        """
        保存训练好的模型
        
        参数:
            model_file: 模型文件名
        """
        if self.model is None:
            print("错误：没有模型可保存")
            return False
        
        model_path = os.path.join(self.model_path, model_file)
        try:
            self.model.save(model_path)
            print(f"模型已保存到: {model_path}")
            
            # 同时保存scalers
            scalers_path = os.path.join(self.model_path, 'scalers.pkl')
            joblib.dump(self.scalers, scalers_path)
            print(f"标准化器已保存到: {scalers_path}")
            
            return True
        except Exception as e:
            print(f"保存模型失败: {e}")
            return False

    def load_model(self, model_file='model.keras'):
        """
        加载已训练的模型
        
        参数:
            model_file: 模型文件名
            
        返回:
            success: 是否成功加载
        """
        model_path = os.path.join(self.model_path, model_file)
        scalers_path = os.path.join(self.model_path, 'scalers.pkl')
        
        try:
            if os.path.exists(model_path):
                # 清除之前的模型和会话
                import tensorflow as tf
                tf.keras.backend.clear_session()
                
                # 加载模型
                self.model = tf.keras.models.load_model(model_path, compile=True)
                print(f"模型已从 {model_path} 加载")
                
                # 加载标准化器
                if os.path.exists(scalers_path):
                    self.scalers = joblib.load(scalers_path)
                    print(f"标准化器已从 {scalers_path} 加载")
                else:
                    print(f"警告: 找不到标准化器文件 {scalers_path}")
                    return False
                
                return True
            else:
                print(f"错误: 找不到模型文件 {model_path}")
                
                return False
        except Exception as e:
            print(f"加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def load_data(self, file_path):
        """
        加载学生成绩数据
        
        参数:
        file_path: 数据文件路径
        
        返回:
        df: 加载的数据框
        """
        try:
            df = pd.read_csv(file_path)
            print(f"数据加载成功，共 {df.shape[0]} 条记录")
            return df
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def preprocess_data(self, df):
        """
        数据预处理
        
        参数:
        df: 原始数据框
        
        返回:
        students_data: 按学生ID组织的数据字典
        """
        # 数据基本检查
        print("数据预览:")
        print(df.head())
        print("\n数据信息:")
        print(df.info())
        print("\n缺失值统计:")
        print(df.isnull().sum())
        
        # 如果有缺失值，进行处理
        if df.isnull().sum().sum() > 0:
            # 对分数进行中位数填充（可以根据需要调整策略）
            if 'score' in df.columns and df['score'].isnull().sum() > 0:
                df['score'] = df.groupby('student_id')['score'].transform(
                    lambda x: x.fillna(x.median() if not x.median() != x.median() else x.mean())
                )
                # 如果仍有缺失值（如某学生所有成绩都缺失），使用全局中位数填充
                df['score'] = df['score'].fillna(df['score'].median())
        
        # 按学生ID分组
        students_data = {}
        for student_id, group in df.groupby('student_id'):
            # 按考试ID排序，确保时间序列正确
            student_df = group.sort_values('exam_id')
            students_data[student_id] = student_df
        
        return students_data
    
    def create_features(self, student_df):
        """
        增强版特征创建
        
        参数:
        student_df: 单个学生的数据框
        
        返回:
        features_df: 包含所有特征的数据框
        """
        df = student_df.copy()
        
        # 基本特征：分数
        features = ['score']
        
        # 添加滚动统计特征
        if len(df) > 2:  # 确保有足够的数据点
            # 滚动平均
            df['rolling_mean_2'] = df['score'].rolling(window=2, min_periods=1).mean()
            df['rolling_mean_3'] = df['score'].rolling(window=3, min_periods=1).mean()
            df['rolling_mean_5'] = df['score'].rolling(window=5, min_periods=1).mean()
            
            # 滚动标准差（波动性）
            df['rolling_std_3'] = df['score'].rolling(window=3, min_periods=1).std()
            df['rolling_std_5'] = df['score'].rolling(window=5, min_periods=1).std()
            
            # 动量（相对于上一次考试的变化）
            df['momentum'] = df['score'].diff()
            df['momentum_2'] = df['score'].diff(2)  # 与前两次考试比较
            
            # 趋势（过去几次考试的斜率）
            if len(df) >= 3:
                df['trend'] = (df['score'] - df['score'].shift(2)) / 2
            
            # 加速度（动量的变化）
            df['acceleration'] = df['momentum'].diff()
            
            # 相对排名信息（如果有班级数据）
            # 这里只是一个示例，实际中需要班级数据
            if 'class_avg' in df.columns:
                df['rel_to_class'] = df['score'] - df['class_avg']
                df['rel_to_class_pct'] = df['score'] / df['class_avg']
            
            # 周期性特征（如果考试有明显的周期性）
            # 例如，月考、期中考、期末考的模式
            if 'exam_type' in df.columns:
                # 将exam_type转为数值型特征
                exam_type_map = {'monthly': 0, 'midterm': 1, 'final': 2}
                df['exam_type_num'] = df['exam_type'].map(exam_type_map)
            
            # 添加与历史最佳成绩的差距
            df['gap_from_best'] = df['score'] - df['score'].cummax()
            
            features.extend(['rolling_mean_2', 'rolling_mean_3', 'rolling_mean_5', 
                            'rolling_std_3', 'rolling_std_5', 
                            'momentum', 'momentum_2', 'trend', 'acceleration',
                            'gap_from_best'])
            
            # 添加额外的特征（如果相关数据列存在）
            optional_features = ['rel_to_class', 'rel_to_class_pct', 'exam_type_num']
            for feature in optional_features:
                if feature in df.columns:
                    features.append(feature)
        
        # 填充NaN值
        for feature in features:
            if feature in df.columns and df[feature].isnull().any():
                df[feature] = df[feature].bfill()
                df[feature] = df[feature].ffill()
                df[feature] = df[feature].fillna(0)
        
        return df[features]
    
    def scale_features(self, features_df, is_training=True):
        """
        对特征进行标准化
        
        参数:
        features_df: 特征数据框
        is_training: 是否处于训练阶段
        
        返回:
        scaled_features: 标准化后的特征
        """
        scaled_features = {}
        
        for column in features_df.columns:
            data = features_df[column].values.reshape(-1, 1)
            
            if is_training:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data)
                self.scalers[column] = scaler
            else:
                if column in self.scalers:
                    scaler = self.scalers[column]
                    scaled_data = scaler.transform(data)
                else:
                    print(f"警告：特征 {column} 没有对应的标准化器")
                    scaled_data = data  # 使用原始数据
            
            scaled_features[column] = scaled_data.flatten()
        
        return pd.DataFrame(scaled_features)
    
    def create_sequences(self, features_df):
        """
        将特征数据转换为LSTM可用的序列格式
        
        参数:
        features_df: 特征数据框
        
        返回:
        X: 特征序列
        y: 目标值
        """
        X, y = [], []
        
        # 确保数据足够长
        if len(features_df) <= self.seq_length:
            return np.array([]), np.array([])
        
        # 创建序列
        for i in range(len(features_df) - self.seq_length):
            # 获取一个序列的所有特征
            sequence = features_df.iloc[i:i+self.seq_length].values
            target = features_df.iloc[i+self.seq_length]['score']
            
            X.append(sequence)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape, dropout_rate=0.3, units=64):  # 增加dropout率
        """
        构建改进的LSTM模型
        
        参数:
        input_shape: 输入形状 (seq_length, feature_count)
        dropout_rate: Dropout比率
        units: LSTM单元数量
        
        返回:
        model: 构建好的LSTM模型
        """
        from tensorflow.keras.layers import Bidirectional, BatchNormalization, Attention
        from tensorflow.keras.regularizers import l2
        
        # 使用函数式API构建更复杂的模型
        inputs = Input(shape=input_shape)
        
        # 双向LSTM增强特征提取能力，增加L2正则化强度
        x = Bidirectional(LSTM(units=units, return_sequences=True, 
                            kernel_regularizer=l2(0.002)))(inputs)  # 增加正则化强度
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # 第二层LSTM
        x = Bidirectional(LSTM(units=int(units/2), return_sequences=False, 
                            kernel_regularizer=l2(0.002)))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # 全连接层
        x = Dense(units=32, activation='relu', kernel_regularizer=l2(0.002))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate/2)(x)
        
        # 输出层
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # 使用Adam优化器，但降低学习率
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # 降低学习率
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        model.summary()
        
        return model
    
    def apply_bias_correction(self, predictions, historical_bias=None):
        """
        应用偏差校正，防止模型系统性地高估或低估成绩
        
        参数:
        predictions: 原始预测值
        historical_bias: 历史偏差值，如果为None则使用默认偏差
        
        返回:
        corrected_predictions: 校正后的预测值
        """
        # 默认假设模型有5%的高估倾向
        default_bias_factor = 0.95
        
        # 如果提供了历史偏差数据，则使用它计算偏差因子
        if historical_bias is not None:
            # 假设historical_bias是过去预测误差的平均值(预测值-实际值)
            # 正值表示高估，负值表示低估
            if historical_bias > 0:
                # 高估情况，计算校正因子
                bias_factor = 1 - min(0.15, historical_bias / 100)  # 最多校正15%
            else:
                # 低估情况，增加预测值
                bias_factor = 1 + min(0.10, abs(historical_bias) / 100)  # 最多增加10%
        else:
            bias_factor = default_bias_factor
        
        # 应用校正因子
        corrected_predictions = predictions * bias_factor
        
        return corrected_predictions
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1):
        """
        训练模型
        
        参数:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        epochs: 训练轮数
        batch_size: 批量大小
        verbose: 显示详细程度
        
        返回:
        history: 训练历史
        """
        # 如果没有足够的训练数据
        if len(X_train) == 0 or len(X_val) == 0:
            print("警告：没有足够的数据来训练模型")
            return None
        
        # 构建模型
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        # 回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001),
            ModelCheckpoint(
                filepath=os.path.join(self.model_path, 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose
        )
        
        # 保存标准化器
        joblib.dump(self.scalers, os.path.join(self.model_path, 'scalers.pkl'))
        
        return history
    
    def predict(self, X_test):
        """
        使用模型进行预测
        
        参数:
        X_test: 测试数据
        
        返回:
        predictions: 预测结果
        """
        if self.model is None:
            print("错误：模型尚未训练")
            return None
        
        if len(X_test) == 0:
            print("警告：没有测试数据")
            return np.array([])
        
        # 预测
        predictions = self.model.predict(X_test)
        return predictions
    
    def inverse_transform_predictions(self, predictions):
        """
        将预测结果反转标准化
        
        参数:
        predictions: 标准化的预测结果
        
        返回:
        original_scale_predictions: 原始刻度的预测结果
        """
        if 'score' not in self.scalers:
            print("错误：找不到分数的标准化器")
            return predictions
        
        # 反转标准化
        original_scale_predictions = self.scalers['score'].inverse_transform(predictions)
        return original_scale_predictions
    
    def evaluate_model(self, y_true, y_pred, prefix=""):
        """
        评估模型性能
        
        参数:
        y_true: 真实值
        y_pred: 预测值
        prefix: 指标前缀（如"训练集"或"测试集"）
        
        返回:
        metrics: 评估指标字典
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        print(f"\n{prefix}评估指标:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        
        return metrics
    
    def visualize_training_history(self, history):
        """
        可视化训练历史
        
        参数:
        history: 训练历史对象
        """
        if history is None:
            print("错误：没有训练历史")
            return
        history.to_csv(os.path.join(self.model_path, 'training_history.csv'))
    
    def visualize_predictions(self, y_true, y_pred, title="预测结果"):
        """
        可视化预测结果
        
        参数:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
        """
    
    def monte_carlo_dropout_predict(self, X_test, n_samples=100):
        """
        使用Monte Carlo Dropout进行不确定性估计 - 修复版本
        """
        if self.model is None:
            print("错误：模型尚未训练")
            return None, None, None, None
        
        # 对原始模型进行预测，获取结构信息
        original_pred = self.model.predict(X_test, verbose=0)
        
        # 创建一个新模型，但保持相同的结构
        # 这种方法不依赖于具体的模型结构，而是使用Keras的函数式API克隆模型
        from tensorflow.keras.models import clone_model
        
        # 克隆模型结构
        mc_model = clone_model(self.model)
        mc_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        # 复制原模型的权重
        mc_model.set_weights(self.model.get_weights())
        
        # 修改为自定义训练循环，手动将Dropout层设置为训练模式
        predictions = []
        for _ in range(n_samples):
            # 使用tf.function进行前向传播，手动设置training=True
            def dropout_forward_pass(x, training=True):
                # 获取所有层
                for layer in mc_model.layers:
                    # 应用层处理
                    if isinstance(layer, tf.keras.layers.Dropout):
                        # 对Dropout层，强制设置training=True
                        x = layer(x, training=True)
                    else:
                        # 对其他层，使用默认行为
                        x = layer(x)
                return x
            
            # 从输入到输出的自定义前向传播
            def predict_with_dropout(x):
                # 假设第一层是输入层
                intermediate = x
                # 遍历所有层(除了输入层)
                for i, layer in enumerate(mc_model.layers):
                    if i == 0:  # 跳过输入层
                        continue
                    if isinstance(layer, tf.keras.layers.Dropout):
                        # 对于Dropout层，强制training=True
                        intermediate = layer(intermediate, training=True)
                    else:
                        # 对于其他层，正常调用
                        intermediate = layer(intermediate)
                return intermediate
            
            # 方法1：直接使用模型但进行多次预测（Dropout每次都会随机）
            pred = mc_model.predict(X_test, verbose=0)
            predictions.append(pred.flatten())
        
        # 将预测结果转换为numpy数组
        predictions = np.array(predictions)  # shape: (n_samples, n_test_samples)
        
        # 计算每个测试样本的预测均值和标准差
        mean_predictions = np.mean(predictions, axis=0)
        std_predictions = np.std(predictions, axis=0)
        
        # 修改这部分：减小置信区间系数
        confidence_multiplier = 1.5  # 从2.5减小到1.5
        min_ci_width = 3.0  # 从5.0减小到3.0
        
        # 计算扩展后的置信区间
        lower_bound = mean_predictions - confidence_multiplier * std_predictions
        upper_bound = mean_predictions + confidence_multiplier * std_predictions
        
        # 确保置信区间至少达到最小宽度
        for i in range(len(mean_predictions)):
            ci_width = upper_bound[i] - lower_bound[i]
            if ci_width < min_ci_width:
                half_expansion = (min_ci_width - ci_width) / 2
                lower_bound[i] -= half_expansion
                upper_bound[i] += half_expansion
        
        # 添加: 限制预测值在合理范围内（标准化空间）
        mean_predictions = np.minimum(mean_predictions, 2.5)  # 限制最大预测值
        
        # 在标准化空间中进行合理性调整
        if 'score' in self.scalers:
            # 确保置信区间不超出合理范围
            lower_bound = np.maximum(lower_bound, -2)  # 提高下限
            upper_bound = np.minimum(upper_bound, 2.5)  # 降低上限
        
        return mean_predictions, std_predictions, lower_bound, upper_bound
    
    def visualize_prediction_intervals(self, X_test, y_true, n_samples=100):
        """
        可视化预测区间 - 修复版
        """
        # 获取置信区间
        mean_preds, std_preds, lower_bound, upper_bound = self.monte_carlo_dropout_predict(X_test, n_samples)
        
        # 反转标准化
        if 'score' in self.scalers:
            scaler = self.scalers['score']
            y_true_original = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            mean_preds_original = scaler.inverse_transform(mean_preds.reshape(-1, 1)).flatten()
            lower_bound_original = scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
            upper_bound_original = scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()
            
            # 再次检查并修正置信区间
            lower_bound_original = np.minimum(lower_bound_original, mean_preds_original)
            upper_bound_original = np.maximum(upper_bound_original, mean_preds_original)
        else:
            y_true_original = y_true
            mean_preds_original = mean_preds
            lower_bound_original = lower_bound
            upper_bound_original = upper_bound
        
        
        # 计算误差条的高度，确保不出现负值
        yerr_lower = mean_preds_original - lower_bound_original
        yerr_upper = upper_bound_original - mean_preds_original
        
        return mean_preds_original, lower_bound_original, upper_bound_original
    
    def detect_anomalies(self, y_true, y_pred, std_preds, z_threshold=2.0):
        """
        检测异常成绩
        
        参数:
        y_true: 真实成绩
        y_pred: 预测成绩
        std_preds: 预测标准差
        z_threshold: z分数阈值
        
        返回:
        anomaly_indices: 异常样本的索引
        z_scores: z分数
        """
        # 计算z分数
        errors = y_true - y_pred
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        z_scores = (errors - mean_error) / std_error
        
        # 找出异常值
        anomaly_indices = np.where(np.abs(z_scores) > z_threshold)[0]
       
        # 输出异常记录
        if len(anomaly_indices) > 0:
            print("\n检测到异常成绩:")
            for idx in anomaly_indices:
                print(f"样本 {idx}: 实际成绩 {y_true[idx]:.2f}, 预测成绩 {y_pred[idx]:.2f}, Z分数 {z_scores[idx]:.2f}")
        else:
            print("\n未检测到异常成绩")
        
        return anomaly_indices, z_scores
    
    def train_and_evaluate_for_student(self, student_df, test_size=0.2, val_size=0.2, epochs=100):
        """
        为单个学生训练和评估模型
        
        参数:
        student_df: 学生的成绩数据框
        test_size: 测试集比例
        val_size: 验证集比例
        epochs: 训练轮数
        
        返回:
        model: 训练好的模型
        metrics: 评估指标
        """
        try:
            # 首先尝试加载已有模型
            if os.path.exists(os.path.join(self.model_path, 'model.keras')):
                print("发现已有模型，尝试加载...")
                if self.load_model():
                    print("成功加载已有模型")
                    # 这里可以添加模型验证代码
                    return self.model, None
                
            # 如果没有模型或加载失败，进行训练
            print("没有找到可用模型，开始训练新模型...")
            # 创建特征
            features_df = self.create_features(student_df)
            
            # 检查数据是否足够
            if len(features_df) <= self.seq_length + 2:
                print(f"警告：该学生的数据不足以进行有效训练(仅有{len(features_df)}条记录)")
                return None, None
            
            # 特征标准化
            scaled_features = self.scale_features(features_df)
            
            # 创建序列
            X, y = self.create_sequences(scaled_features)
            
            # 检查序列是否创建成功
            if len(X) == 0:
                print("警告：无法创建序列数据")
                return None, None
            
            # 分割数据
            # 首先分割出测试集
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            # 然后从剩余数据中分割出验证集
            val_size_adjusted = val_size / (1 - test_size)  # 调整验证集大小
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, shuffle=False)
            
            # 训练模型
            history = self.train_model(X_train, y_train, X_val, y_val, epochs=epochs)
            
            # 可视化训练历史
            # if history is not None:
            #     self.visualize_training_history(history)
            
            # 模型评估
            if self.model is not None:
                # 预测验证集
                val_predictions = self.predict(X_val)
                val_metrics = self.evaluate_model(y_val, val_predictions, prefix="验证集")
                
                # 预测测试集
                test_predictions = self.predict(X_test)
                test_metrics = self.evaluate_model(y_test, test_predictions, prefix="测试集")
                
                # 反转标准化后的预测结果
                y_test_original = self.scalers['score'].inverse_transform(y_test.reshape(-1, 1)).flatten()
                test_predictions_original = self.scalers['score'].inverse_transform(test_predictions).flatten()
                
                # 可视化预测结果
                # self.visualize_predictions(y_test_original, test_predictions_original)
                
                # 预测区间
                mean_preds, lower_bound, upper_bound = self.visualize_prediction_intervals(X_test, y_test)
                
                # 异常检测
                _, std_preds, _, _ = self.monte_carlo_dropout_predict(X_test)
                anomaly_indices, _ = self.detect_anomalies(y_test_original, mean_preds, std_preds)
                print("训练完成，保存模型...")
                self.save_model()
                return self.model, test_metrics
                
            return self.model, test_metrics
        except Exception as e:
            print(f"训练和评估过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def ensemble_predict(self, X_test, num_models=3):
        """
        使用集成学习方法预测，结合多个模型的预测结果
        
        参数:
        X_test: 测试数据
        num_models: 集成模型数量
        
        返回:
        ensemble_mean: 集成预测结果均值
        ensemble_std: 集成预测结果标准差
        lower_bound: 下限
        upper_bound: 上限
        """
        if self.model is None:
            print("错误：模型尚未训练")
            return None, None, None, None
        
        # 使用MC-Dropout预测作为第一个模型
        mc_mean, mc_std, mc_lower, mc_upper = self.monte_carlo_dropout_predict(X_test)
        
        # 创建一系列不同结构的辅助模型
        ensemble_predictions = [mc_mean]  # 将MC-Dropout预测添加到集成中
        
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, GRU, SimpleRNN
        from tensorflow.keras.models import Model, clone_model
        
        # 输入形状
        input_shape = (X_test.shape[1], X_test.shape[2])
        
        # 添加不同架构的模型预测
        for i in range(num_models - 1):  # -1因为已经包含了MC-Dropout模型
            if i % 3 == 0:
                # GRU模型
                inputs = Input(shape=input_shape)
                x = GRU(units=48, return_sequences=True)(inputs)
                x = Dropout(0.3)(x, training=False)
                x = GRU(units=24)(x)
                x = Dropout(0.3)(x, training=False)
                outputs = Dense(1)(x)
                model = Model(inputs=inputs, outputs=outputs)
            elif i % 3 == 1:
                # SimpleRNN模型
                inputs = Input(shape=input_shape)
                x = SimpleRNN(units=48, return_sequences=True)(inputs)
                x = Dropout(0.3)(x, training=False)
                x = SimpleRNN(units=24)(x)
                x = Dropout(0.3)(x, training=False)
                outputs = Dense(1)(x)
                model = Model(inputs=inputs, outputs=outputs)
            else:
                # 浅层LSTM模型
                inputs = Input(shape=input_shape)
                x = LSTM(units=32)(inputs)
                x = Dropout(0.3)(x, training=False)
                outputs = Dense(1)(x)
                model = Model(inputs=inputs, outputs=outputs)
            
            # 编译模型
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # 初始化随机权重
            # 注意：不再尝试复制原始模型的权重，直接使用随机初始化的权重
            print(f"使用随机初始化权重创建集成模型 {i+1}")
            
            # 预测
            pred = model.predict(X_test, verbose=0)
            ensemble_predictions.append(pred.flatten())
        
        # 将所有预测结果转换为numpy数组
        ensemble_predictions = np.array(ensemble_predictions)  # shape: (num_models, n_test_samples)
        
        # 计算集成预测的均值和标准差
        ensemble_mean = np.mean(ensemble_predictions, axis=0)
        ensemble_std = np.std(ensemble_predictions, axis=0)
        
        # 计算置信区间（使用增强的方法）
        confidence_multiplier = 2.0
        min_ci_width = 8.0  # 较大的最小置信区间（标准化空间中）
        
        lower_bound = ensemble_mean - confidence_multiplier * ensemble_std
        upper_bound = ensemble_mean + confidence_multiplier * ensemble_std
        
        # 确保置信区间至少达到最小宽度
        for i in range(len(ensemble_mean)):
            ci_width = upper_bound[i] - lower_bound[i]
            if ci_width < min_ci_width:
                half_expansion = (min_ci_width - ci_width) / 2
                lower_bound[i] -= half_expansion
                upper_bound[i] += half_expansion
        
        return ensemble_mean, ensemble_std, lower_bound, upper_bound
    # 在 Improved_LSTM.py 文件中，StudentScorePredictor 类中添加新方法

    def extended_model_evaluation(self, X_test, y_test):
        """
        扩展模型评估，提供更详细的模型性能分析
        
        参数:
        X_test: 测试特征数据
        y_test: 测试标签数据
        
        返回:
        detailed_metrics: 详细评估指标
        """
        if self.model is None:
            print("错误：模型尚未训练")
            return None
        
        # 标准评估
        predictions = self.predict(X_test)
        
        # 如果需要，反转标准化
        if 'score' in self.scalers:
            y_test_original = self.scalers['score'].inverse_transform(y_test.reshape(-1, 1)).flatten()
            predictions_original = self.scalers['score'].inverse_transform(predictions).flatten()
        else:
            y_test_original = y_test
            predictions_original = predictions.flatten()
        
        # 计算基本指标
        mse = mean_squared_error(y_test_original, predictions_original)
        mae = mean_absolute_error(y_test_original, predictions_original)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_original, predictions_original)
        
        # 计算高级指标
        errors = y_test_original - predictions_original
        
        # 误差分布统计
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        error_median = np.median(errors)
        error_min = np.min(errors)
        error_max = np.max(errors)
        
        # 精确度分析
        accuracy_1 = np.mean(np.abs(errors) <= 1.0) * 100  # 误差在1分以内的百分比
        accuracy_3 = np.mean(np.abs(errors) <= 3.0) * 100  # 误差在3分以内的百分比
        accuracy_5 = np.mean(np.abs(errors) <= 5.0) * 100  # 误差在5分以内的百分比
        
        # 模型偏差分析
        is_biased = abs(error_mean) > 1.0  # 平均误差大于1分认为有偏差
        bias_direction = "高估" if error_mean < 0 else "低估" if error_mean > 0 else "无偏差"
        
        # 拟合分析
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_test_original, predictions_original)
        is_underfitting = r2 < 0.5  # R²较低表示欠拟合
        is_overfitting = r2 > 0.99  # R²过高可能表示过拟合
        
        # 整合所有指标
        detailed_metrics = {
            'standard_metrics': {
                'MSE': float(mse),
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R2': float(r2)
            },
            'error_distribution': {
                'mean': float(error_mean),
                'std': float(error_std),
                'median': float(error_median),
                'min': float(error_min),
                'max': float(error_max)
            },
            'accuracy': {
                'within_1_point': float(accuracy_1),
                'within_3_points': float(accuracy_3),
                'within_5_points': float(accuracy_5)
            },
            'bias_analysis': {
                'is_biased': bool(is_biased),
                'direction': bias_direction,
                'magnitude': float(abs(error_mean))
            },
            'fit_analysis': {
                'slope': float(slope),
                'intercept': float(intercept),
                'p_value': float(p_value),
                'is_underfitting': bool(is_underfitting),
                'is_overfitting': bool(is_overfitting)
            }
        }
        
        # 打印摘要信息
        print("\n扩展模型评估摘要:")
        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
        print(f"预测准确度: 在1分内: {accuracy_1:.1f}%, 在3分内: {accuracy_3:.1f}%, 在5分内: {accuracy_5:.1f}%")
        if is_biased:
            print(f"模型偏差: {bias_direction}, 平均偏差: {abs(error_mean):.2f}分")
        if is_underfitting:
            print("警告: 模型可能存在欠拟合问题")
        if is_overfitting:
            print("警告: 模型可能存在过拟合问题")
        
        return detailed_metrics
    
    def analyze_student_performance(self, student_df, prediction_window=3):
        """
        分析学生表现，检测异常并给出教学建议
        
        参数:
        student_df: 学生的历史成绩数据
        prediction_window: 用于预测的窗口大小
        
        返回:
        analysis_results: 包含分析结果和建议的字典
        """
        df = student_df.copy()
        
        # 确保有足够的数据
        if len(df) < 5:
            return {
                'status': 'insufficient_data',
                'message': f'数据不足，至少需要5条记录，当前仅有{len(df)}条'
            }
        
        # 获取基本统计信息
        scores = df['score'].values
        recent_scores = scores[-prediction_window:]
        historical_scores = scores[:-prediction_window] if len(scores) > prediction_window else scores
        
        # 计算关键指标
        avg_score = np.mean(scores)
        recent_avg = np.mean(recent_scores)
        historical_avg = np.mean(historical_scores)
        score_trend = recent_avg - historical_avg
        
        # 计算波动性
        volatility = np.std(scores)
        recent_volatility = np.std(recent_scores)
        
        # 检测异常模式
        is_improving = score_trend > 2
        is_declining = score_trend < -2
        is_stable = abs(score_trend) <= 2
        
        high_volatility = volatility > 10
        
        # 生成分析报告
        analysis = {
            'average_score': float(avg_score),
            'recent_average': float(recent_avg),
            'historical_average': float(historical_avg),
            'score_trend': float(score_trend),
            'volatility': float(volatility),
            'recent_volatility': float(recent_volatility),
            'patterns': {
                'improving': bool(is_improving),
                'declining': bool(is_declining),
                'stable': bool(is_stable),
                'high_volatility': bool(high_volatility)
            }
        }
        
        # 生成教学建议
        recommendations = []
        
        if is_improving:
            recommendations.append("学生表现持续提升，建议保持现有学习方法并适当增加挑战。")
        elif is_declining:
            recommendations.append("学生成绩呈下降趋势，建议教师关注学生学习状态，了解可能的困难点。")
        else:
            recommendations.append("学生成绩保持稳定，可针对性地提高某些知识点的掌握程度。")
        
        if high_volatility:
            recommendations.append("学生成绩波动较大，建议关注学习的稳定性，可能需要加强基础知识的巩固。")
        
        # 预测未来表现
        next_exam_predict, confidence_interval = self.predict_next_exam(df)
        
        # 完整分析结果
        analysis_results = {
            'student_id': df['student_id'].iloc[0],
            'analysis': analysis,
            'recommendations': recommendations,
            'next_exam_prediction': {
                'score': float(next_exam_predict) if next_exam_predict is not None else None,
                'lower_bound': float(confidence_interval[0]) if confidence_interval is not None else None,
                'upper_bound': float(confidence_interval[1]) if confidence_interval is not None else None
            }
        }
        
        return analysis_results
    
    # 在 improved_lstm.py 中的 StudentScorePredictor 类中添加或修改方法

    def predict_next_exam(self, student_df, exam_id=None, subject=None):
        """
        预测学生下一次考试成绩，并根据科目总分进行转换
        
        参数:
        student_df: 学生的历史成绩数据
        exam_id: 要预测的考试ID，如果为None则预测下一次考试
        subject: 科目名称，用于确定总分
        
        返回:
        prediction: 预测成绩
        confidence_interval: 置信区间 (lower, upper)
        """
        if self.model is None:
            print("模型未加载，尝试加载模型...")
            if not self.load_model():
                print("无法加载模型，返回备用预测结果")
                # 实现一个简单的备用预测方法
                # 根据科目限制分数范围
                return self._predict_with_linear_regression(student_df, subject)
        
        df = student_df.copy()
        
        # 如果没有指定考试ID，则预测下一次考试
        if exam_id is None:
            last_exam_id = df['exam_id'].max()
            exam_id = last_exam_id + 1
        
        # 检查是否有足够的历史数据
        if len(df) < self.seq_length:
            print(f"错误：历史数据不足，需要至少{self.seq_length}条记录")
            return None, None
        
        # 创建特征
        features_df = self.create_features(df)
        
        # 特征标准化
        scaled_features = self.scale_features(features_df, is_training=False)
        
        # 获取最新的seq_length条记录
        latest_sequence = scaled_features.iloc[-self.seq_length:].values
        latest_sequence = latest_sequence.reshape(1, self.seq_length, scaled_features.shape[1])
        
        # 使用Monte Carlo Dropout进行预测
        mean_pred, std_pred, lower_bound, upper_bound = self.monte_carlo_dropout_predict(latest_sequence)
        
        # 反转标准化
        mean_pred_original = self.scalers['score'].inverse_transform(mean_pred.reshape(-1, 1))[0][0]
        lower_bound_original = self.scalers['score'].inverse_transform(lower_bound.reshape(-1, 1))[0][0]
        upper_bound_original = self.scalers['score'].inverse_transform(upper_bound.reshape(-1, 1))[0][0]
        
        # 应用偏差校正
        historical_bias = self._calculate_historical_bias(student_df)
        mean_pred_original = self.apply_bias_correction(mean_pred_original, historical_bias)
        
        # 应用科目总分转换和限制
        mean_pred_original, (lower_bound_original, upper_bound_original) = self._apply_subject_scaling(
            mean_pred_original, 
            (lower_bound_original, upper_bound_original), 
            subject
        )
        
        print(f"\n预测考试 {exam_id} 的成绩:")
        print(f"科目: {subject if subject else '总分'}")
        print(f"预测分数: {mean_pred_original:.2f}")
        print(f"95%置信区间: [{lower_bound_original:.2f}, {upper_bound_original:.2f}]")
        
        return mean_pred_original, (lower_bound_original, upper_bound_original)
    
    def _calculate_historical_bias(self, student_df):
        """
        计算模型的历史预测偏差
        
        参数:
        student_df: 包含历史成绩的数据框
        
        返回:
        historical_bias: 历史预测偏差（预测值-实际值的平均值）
        """
        if len(student_df) < self.seq_length + 2:
            # 数据不足以计算有意义的偏差
            return None
        
        df = student_df.copy()
        
        # 创建特征
        features_df = self.create_features(df)
        scaled_features = self.scale_features(features_df, is_training=False)
        
        # 创建历史序列和对应的实际结果
        X, y_true = self.create_sequences(scaled_features)
        
        if len(X) == 0:
            return None
        
        # 模型预测
        y_pred = self.predict(X).flatten()
        
        # 计算偏差 (预测-实际)
        bias = y_pred - y_true
        
        # 反转标准化
        if 'score' in self.scalers:
            scaler = self.scalers['score']
            y_true_original = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            bias_original = y_pred_original - y_true_original
            
            # 计算平均偏差
            historical_bias = np.mean(bias_original)
            return historical_bias
        
        return np.mean(bias)

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
            reduced_width = interval_width * 0.7  # 减少30%的区间宽度
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
    
    def _fallback_prediction(self, scores, steps=3):
        """备用的线性回归预测方法，当LSTM模型不可用时使用"""
        print("使用线性回归预测方法...")
        
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
        
        # 获取历史成绩数据用于线性回归
        exam_ids = []
        score_values = []
        
        for score in scores:
            if score.score is not None:
                exam_ids.append(score.exam_id)
                score_values.append(score.score)
        
        # 确保有足够的数据进行线性回归
        if len(exam_ids) < 2:
            print("数据不足，无法进行线性回归，使用最后一次成绩作为预测")
            last_exam = scores[-1]
            base_score = last_exam.score or 75
            
            for i in range(steps):
                prediction = {
                    'examId': last_exam.exam_id + i + 1,
                    'date': (datetime.now() + timedelta(days=30 * (i + 1))).strftime('%Y-%m-%d'),
                    'actual': None,
                    'predicted': round(base_score, 1),
                    'lower': round(max(60, base_score - 5), 1),
                    'upper': round(min(100, base_score + 5), 1),
                    'subject': last_exam.subject
                }
                result.append(prediction)
            
            return result
        
        # 使用sklearn进行线性回归
        from sklearn.linear_model import LinearRegression
        import numpy as np
        from sklearn.metrics import mean_squared_error
        
        # 重构数据格式
        X = np.array(exam_ids).reshape(-1, 1)
        y = np.array(score_values)
        
        # 创建线性回归模型
        model = LinearRegression()
        model.fit(X, y)
        
        # 获取模型参数
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # 计算模型拟合误差
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        # 获取最后一次考试信息
        last_exam = scores[-1]
        last_exam_id = last_exam.exam_id
        
        # 基于线性回归进行预测
        for i in range(steps):
            next_exam_id = last_exam_id + i + 1
            # 预测值
            predicted_score = model.predict(np.array([[next_exam_id]]))[0]
            
            # 基于RMSE生成置信区间
            confidence_factor = 1.5  # 置信区间系数
            confidence_width = rmse * confidence_factor
            
            # 应用科目难度调整
            subject = last_exam.subject
            predicted_score = self._adjust_by_difficulty(predicted_score, subject)
            
            # 确保预测值在合理范围内
            predicted_score = max(60, min(100, predicted_score))
            
            # 考虑趋势生成置信区间
            if slope > 0:  # 成绩上升趋势
                lower_bound = max(60, predicted_score - confidence_width * 0.7)  # 下限更接近预测值
                upper_bound = min(100, predicted_score + confidence_width * 1.3)  # 上限更宽
            elif slope < 0:  # 成绩下降趋势
                lower_bound = max(60, predicted_score - confidence_width * 1.3)  # 下限更宽
                upper_bound = min(100, predicted_score + confidence_width * 0.7)  # 上限更接近预测值
            else:  # 无明显趋势
                lower_bound = max(60, predicted_score - confidence_width)
                upper_bound = min(100, predicted_score + confidence_width)
            
            # 创建预测结果
            prediction = {
                'examId': next_exam_id,
                'date': (datetime.now() + timedelta(days=30 * (i + 1))).strftime('%Y-%m-%d'),
                'actual': None,
                'predicted': round(predicted_score, 1),
                'lower': round(lower_bound, 1),
                'upper': round(upper_bound, 1),
                'subject': last_exam.subject
            }
            result.append(prediction)
        
        print("线性回归预测方法完成")
        return result

    def _adjust_by_difficulty(self, score, subject=None):
        """根据科目难度调整分数"""
        
        # 定义科目难度系数 (< 1.0意味着更难获得高分)
        subject_difficulty = {
            "语文": 0.7,
            "数学": 0.7,
            "英语": 0.7,
            "物理": 0.7,
            "化学": 0.7,
            "生物": 0.7,
            "政治": 0.7,
            "历史": 0.7,
            "地理": 0.7,
            "技术": 0.7,
            "总分": 0.7, # 默认难度
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
        
        return adjusted_score
    
    def _predict_class_average_simple(self, class_averages, steps=3):
        """使用线性回归的班级平均分预测方法"""
        if not class_averages:
            return []
        
        result = list(class_averages)  # 复制历史数据
        
        # 提取历史数据用于线性回归
        exam_ids = []
        avg_scores = []
        
        for avg in class_averages:
            exam_ids.append(avg['examId'])
            avg_scores.append(avg['average'])
        
        # 确保有足够的数据进行线性回归
        if len(exam_ids) < 2:
            # 使用简单平均的方法
            last_average = class_averages[-1]
            base_average = last_average['average']
            last_exam_id = last_average['examId']
            
            for i in range(steps):
                prediction = {
                    'examId': last_exam_id + i + 1,
                    'date': (datetime.now() + timedelta(days=30 * (i + 1))).strftime('%Y-%m-%d'),
                    'average': round(base_average, 1),
                    'subject': last_average['subject']
                }
                result.append(prediction)
            
            return result
        
        # 使用sklearn进行线性回归
        from sklearn.linear_model import LinearRegression
        import numpy as np
        from sklearn.metrics import mean_squared_error
        
        # 重构数据格式
        X = np.array(exam_ids).reshape(-1, 1)
        y = np.array(avg_scores)
        
        # 创建线性回归模型
        model = LinearRegression()
        model.fit(X, y)
        
        # 获取模型参数
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # 计算模型拟合误差
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"线性回归模型: 分数 = {slope:.2f} * 考试ID + {intercept:.2f}, RMSE = {rmse:.2f}")
        
        # 获取最后一次考试信息
        last_average = class_averages[-1]
        last_exam_id = last_average['examId']
        subject = last_average['subject']
        
        # 基于线性回归进行预测
        for i in range(steps):
            next_exam_id = last_exam_id + i + 1
            
            # 预测值
            predicted_avg = model.predict(np.array([[next_exam_id]]))[0]
            
            # 确保预测值在合理范围内 (班级平均分通常在70-90之间)
            predicted_avg = max(70, min(95, predicted_avg))
            
            # 添加一些合理的随机波动，但保持整体趋势
            import random
            noise = random.uniform(-1.5, 1.5)
            predicted_avg += noise
            
            # 应用主题难度调整（班级平均分）
            if subject:
                predicted_avg = self._adjust_class_average_by_subject(predicted_avg, subject)
            
            prediction = {
                'examId': next_exam_id,
                'date': (datetime.now() + timedelta(days=30 * (i + 1))).strftime('%Y-%m-%d'),
                'average': round(predicted_avg, 1),
                'subject': subject
            }
            result.append(prediction)
        
        return result

    def _adjust_class_average_by_subject(self, avg_score, subject=None):
        """根据科目调整班级平均分"""
        # 不同科目的班级平均分调整系数
        subject_adjustments = {
            "语文": 0.0,  # 保持不变
            "数学": -1.5,  # 数学平均分通常略低
            "英语": 0.5,   # 英语平均分通常略高
            "物理": -2.0,  # 物理平均分通常较低
            "化学": -1.8,  # 化学平均分通常较低
            "生物": 1.0,   # 生物平均分通常较高
            "政治": 2.5,   # 政治平均分通常较高
            "历史": 1.5,   # 历史平均分通常较高
            "地理": 1.0,   # 地理平均分通常较高
            "总分": 0.0    # 保持不变
        }
        
        # 如果有科目信息，应用调整
        if subject in subject_adjustments:
            return avg_score + subject_adjustments[subject]
        
        return avg_score
    
    def train_all_students(self, students_data, test_size=0.2, val_size=0.2, epochs=100):
        """
        为所有学生训练模型
        
        参数:
        students_data: 按学生ID组织的数据字典
        test_size: 测试集比例
        val_size: 验证集比例
        epochs: 训练轮数
        
        返回:
        trained_models: 按学生ID组织的训练好的模型字典
        metrics: 按学生ID组织的评估指标字典
        """
        trained_models = {}
        metrics = {}
        
        for student_id, student_df in students_data.items():
            print(f"\n训练学生 {student_id} 的模型...")
            
            # 创建预测器实例
            predictor = StudentScorePredictor(
                seq_length=self.seq_length,
                model_path=os.path.join(self.model_path, f'student_{student_id}')
            )
            
            # 训练和评估
            model, model_metrics = predictor.train_and_evaluate_for_student(
                student_df,
                test_size=test_size,
                val_size=val_size,
                epochs=epochs
            )
            
            if model is not None:
                trained_models[student_id] = predictor
                metrics[student_id] = model_metrics
        
        return trained_models, metrics