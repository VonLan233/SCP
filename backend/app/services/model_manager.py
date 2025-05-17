# app/services/model_manager.py
import os
import json
from datetime import datetime

class ModelManager:
    """模型管理服务"""
    
    def __init__(self, model_path='predictor/models'):
        self.model_path = model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # 默认模型配置
        self.default_config = {
            'timeSteps': 5,
            'predictionSteps': 3,
            'epochs': 100,
            'batchSize': 32,
            'confidenceInterval': 95
        }
        
        # 加载配置
        self.config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'defaultParameters': self.default_config,
                'availableModels': ['LSTM', 'GRU', 'Ensemble'],
                'currentModel': 'LSTM',
                'lastTrainedAt': None,
                'accuracy': None
            }
            self._save_config()
    
    def get_config(self):
        """获取模型配置"""
        return self.config
    
    def update_config(self, new_config):
        """更新模型配置"""
        self.config.update(new_config)
        self._save_config()
        return self.config
    
    def _save_config(self):
        """保存配置到文件"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def train_model(self, data=None, model_type=None, params=None):
        """训练模型"""
        # 模拟训练过程
        model_type = model_type or self.config['currentModel']
        
        # 更新配置
        self.config['lastTrainedAt'] = datetime.now().isoformat()
        self.config['accuracy'] = 0.948  # 模拟准确度
        self._save_config()
        
        return {
            'success': True,
            'message': '模型训练成功',
            'trainingTime': 123,  # 模拟训练时间
            'accuracy': self.config['accuracy'],
            'lossHistory': [0.8, 0.5, 0.3, 0.2, 0.15, 0.12, 0.1]  # 模拟损失历史
        }