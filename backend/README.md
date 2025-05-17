student_prediction_backend/
├── app/
│   ├── __init__.py           # Flask应用初始化
│   ├── config.py             # 配置文件
│   ├── models/               # 数据库模型
│   │   ├── __init__.py
│   │   ├── student.py        # 学生模型
│   │   ├── class.py          # 班级模型
│   │   ├── grade.py          # 年级模型
│   │   ├── exam.py           # 考试模型
│   │   └── score.py          # 成绩模型
│   ├── api/                  # API路由
│   │   ├── __init__.py
│   │   ├── students.py       # 学生相关API
│   │   ├── classes.py        # 班级相关API
│   │   ├── grades.py         # 年级相关API
│   │   ├── exams.py          # 考试相关API
│   │   ├── scores.py         # 成绩相关API
│   │   └── model.py          # 预测模型API
│   ├── services/             # 业务逻辑
│   │   ├── __init__.py
│   │   └── prediction.py     # 预测服务
│   └── utils/                # 工具函数
│       ├── __init__.py
│       └── data_processing.py # 数据处理工具
├── ml_models/                # 机器学习模型
│   ├── __init__.py
│   ├── lstm_model.py         # LSTM模型定义
│   └── model_training.py     # 模型训练脚本
├── tests/                    # 测试代码
│   ├── __init__.py
│   ├── test_api.py           # API测试
│   └── test_models.py        # 模型测试
├── migrations/               # 数据库迁移
├── requirements.txt          # 依赖包
├── run.py                    # 启动脚本
└── README.md                 # 项目说明