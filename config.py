"""
Proje konfigürasyon dosyası
Tüm hiperparametreler ve ayarlar burada
"""
import os

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "dataset")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")

# Data files
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
STORE_FILE = os.path.join(DATA_DIR, "store.csv")
SUBMISSION_FILE = os.path.join(ROOT_DIR, "submission.csv")

# Model artifacts
MODEL_FILE = os.path.join(MODELS_DIR, "model.pkl")
FEATURES_FILE = os.path.join(MODELS_DIR, "features.json")

# Cross-validation
CV_FOLDS = 3
CV_VAL_WEEKS = 6

# LightGBM hyperparameters
LIGHTGBM_PARAMS = {
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'max_depth': -1,
    'num_leaves': 63,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'max_depth': 8,
    'reg_alpha': 0.0,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
}

# sklearn HistGradientBoosting hyperparameters
HGBR_PARAMS = {
    'max_depth': None,
    'learning_rate': 0.05,
    'max_bins': 255,
    'l2_regularization': 0.0,
    'random_state': 42
}

# Early stopping
EARLY_STOPPING_ROUNDS = 100

# Feature engineering
LAG_WINDOWS = [7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]

# Target transformation
TARGET_TRANSFORM = 'log1p'  # log1p or None

# Evaluation metrics
PRIMARY_METRIC = 'rmspe'
SECONDARY_METRICS = ['rmse', 'mae', 'mape']

# Visualization
FIGURE_DPI = 150
PLOT_STYLE = 'whitegrid'
PLOT_FIGSIZE = (12, 8)

# Random seed
RANDOM_SEED = 42
