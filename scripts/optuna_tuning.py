"""
Hyperparameter Tuning with Optuna
LightGBM ve XGBoost için optimal parametreleri bulur
"""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import json
from joblib import dump

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("❌ Optuna yüklü değil! Yüklemek için: pip install optuna")
    exit(1)

import lightgbm as lgb
import xgboost as xgb

from src.data import load_datasets, merge_store
from src.features import build_features, select_feature_columns
from src.metrics import rmspe
from src.validation import time_series_cv_indices

DATA_DIR = os.path.join(ROOT, 'dataset')
MODELS_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 70)
print("HYPERPARAMETER TUNING - Optuna")
print("=" * 70)

# Load data
print("\n1. Veri hazırlanıyor...")
train, test, store = load_datasets(DATA_DIR)
train, test = merge_store(train, test, store)
train_fe, test_fe = build_features(train, test)

train_fit = train_fe[train_fe['Open'] == 1].dropna(subset=['Sales']).copy()
y = np.log1p(train_fit['Sales'].values)
feat_cols = select_feature_columns(train_fit)
X = train_fit[feat_cols].fillna(0)

print(f"✓ Eğitim boyutu: {len(train_fit):,}")
print(f"✓ Özellik sayısı: {len(feat_cols)}")

# CV folds
folds = time_series_cv_indices(train_fit, n_splits=3, val_weeks=6)

# ============================================================================
# LIGHTGBM TUNING
# ============================================================================

def objective_lgb(trial):
    """LightGBM için objective function"""
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    fold_scores = []
    for tr_idx, va_idx in folds:
        X_tr, y_tr = X.loc[tr_idx], y[train_fit.index.get_indexer(tr_idx)]
        X_va, y_va = X.loc[va_idx], y[train_fit.index.get_indexer(va_idx)]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        pred = np.expm1(model.predict(X_va))
        y_va_orig = np.expm1(y_va)
        
        # Handle Open=0
        val_df = train_fit.loc[va_idx]
        pred[val_df['Open'] == 0] = 0
        
        score = rmspe(y_va_orig, pred)
        fold_scores.append(score)
    
    return np.mean(fold_scores)


print("\n2. LightGBM tuning başlıyor (50 trial)...")
study_lgb = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    study_name='lgb_tuning'
)
study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True)

print(f"\n✓ LightGBM En İyi Skor: {study_lgb.best_value:.4f}")
print(f"✓ En İyi Parametreler:")
for key, value in study_lgb.best_params.items():
    print(f"   {key}: {value}")

# ============================================================================
# XGBOOST TUNING
# ============================================================================

def objective_xgb(trial):
    """XGBoost için objective function"""
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    fold_scores = []
    for tr_idx, va_idx in folds:
        X_tr, y_tr = X.loc[tr_idx], y[train_fit.index.get_indexer(tr_idx)]
        X_va, y_va = X.loc[va_idx], y[train_fit.index.get_indexer(va_idx)]
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
            early_stopping_rounds=100
        )
        
        pred = np.expm1(model.predict(X_va))
        y_va_orig = np.expm1(y_va)
        
        # Handle Open=0
        val_df = train_fit.loc[va_idx]
        pred[val_df['Open'] == 0] = 0
        
        score = rmspe(y_va_orig, pred)
        fold_scores.append(score)
    
    return np.mean(fold_scores)


print("\n3. XGBoost tuning başlıyor (50 trial)...")
study_xgb = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    study_name='xgb_tuning'
)
study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=True)

print(f"\n✓ XGBoost En İyi Skor: {study_xgb.best_value:.4f}")
print(f"✓ En İyi Parametreler:")
for key, value in study_xgb.best_params.items():
    print(f"   {key}: {value}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n4. Sonuçlar kaydediliyor...")

tuning_results = {
    'lightgbm': {
        'best_score': study_lgb.best_value,
        'best_params': study_lgb.best_params,
        'n_trials': len(study_lgb.trials)
    },
    'xgboost': {
        'best_score': study_xgb.best_value,
        'best_params': study_xgb.best_params,
        'n_trials': len(study_xgb.trials)
    },
    'improvement': {
        'lgb_baseline': 0.1366,  # Eski skor
        'xgb_baseline': 0.1347,  # Eski skor
        'lgb_tuned': study_lgb.best_value,
        'xgb_tuned': study_xgb.best_value,
        'lgb_improvement_pct': (0.1366 - study_lgb.best_value) / 0.1366 * 100,
        'xgb_improvement_pct': (0.1347 - study_xgb.best_value) / 0.1347 * 100
    }
}

with open(os.path.join(MODELS_DIR, 'tuning_results.json'), 'w') as f:
    json.dump(tuning_results, f, indent=2)

print(f"✓ Sonuçlar kaydedildi: models/tuning_results.json")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("TUNING SONUÇLARI")
print("=" * 70)

print(f"\nLightGBM:")
print(f"  Baseline:  0.1366 RMSPE")
print(f"  Tuned:     {study_lgb.best_value:.4f} RMSPE")
print(f"  İyileşme:  {tuning_results['improvement']['lgb_improvement_pct']:.2f}%")

print(f"\nXGBoost:")
print(f"  Baseline:  0.1347 RMSPE")
print(f"  Tuned:     {study_xgb.best_value:.4f} RMSPE")
print(f"  İyileşme:  {tuning_results['improvement']['xgb_improvement_pct']:.2f}%")

print(f"\n💡 Sonraki adım: Bu parametrelerle ensemble_train.py'yi güncelle!")
print("=" * 70)
