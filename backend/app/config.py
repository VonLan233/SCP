import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 限制上传文件大小为16MB
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'predictor/models')
    DEFAULT_SEQ_LENGTH = 5
    DEFAULT_PREDICTION_STEPS = 3
    DEFAULT_EPOCHS = 100
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_CONFIDENCE_INTERVAL = 95
