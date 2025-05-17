# 已保存模型目录

此目录用于存储已训练好的模型文件。

## 目录结构

每个模型通常包含以下文件：

```
model_name/
├── model.keras           # 训练好的模型文件
├── scalers.pkl           # 标准化器，用于特征的标准化/反标准化
└── metadata.json         # 元数据，包含模型信息和训练参数
```

对于集成模型，目录结构如下：

```
ensemble_model/
├── model_0.keras         # 集成中的第一个模型
├── model_1.keras         # 集成中的第二个模型
├── model_2.keras         # 集成中的第三个模型
├── ensemble_info.json    # 集成模型信息
└── scalers.pkl           # 标准化器
```

## 使用说明

### 加载单个模型

```python
from ml_models.model_utils import load_model

# 加载模型
model, scalers, metadata = load_model('saved/student_model')

# 使用模型进行预测
predictions = model.predict(X_test)
```

### 加载集成模型

```python
from ml_models import EnsembleModel

# 加载集成模型
ensemble = EnsembleModel.load('saved/ensemble_model')

# 使用集成模型进行预测
mean_predictions, std_predictions, lower_bound, upper_bound = ensemble.predict(X_test)
```

## 模型列表

| 模型名称 | 描述 | 特点 | 准确度 |
|----------|------|------|--------|
| student_lstm | 学生成绩LSTM预测模型 | 双向LSTM, 序列长度=5 | RMSE: 3.24 |
| student_gru | 学生成绩GRU预测模型 | 双向GRU, 序列长度=5 | RMSE: 3.31 |
| student_ensemble | 学生成绩集成预测模型 | LSTM+GRU+RNN组合 | RMSE: 3.10 |

*注：具体准确度取决于实际训练数据和参数*