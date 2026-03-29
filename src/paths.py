import os

# Root project directory
PROJECT_ROOT = os.path.abspath(os.getcwd())

# Data paths
DATA_RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'titanic-dataset.csv')
DATA_INTERIM_PATH = os.path.join(PROJECT_ROOT, 'data', 'interim', 'cleaned-titanic-dataset.csv')
DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Model paths
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
