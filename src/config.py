import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data_processed')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'positive'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'negative'), exist_ok=True)

# Model Settings
svm_model_path = os.path.join(MODELS_DIR, 'ball_detector_svm.joblib')
