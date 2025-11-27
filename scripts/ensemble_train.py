"""
Ensemble Model Training - XGBoost + LightGBM
Performans iyileştirme için birden fazla modelin ortalaması
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
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.data import load_datasets, merge_store
from src.features import build_features, select_feature_columns
from src.metrics import rmspe
from src.validation import time_series_cv_indices

DATA_DIR = os.path.join(ROOT, 'dataset')
MODELS_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 70)
print("ENSEMBLE MODEL TRAINING - LightGBM + XGBoost")
print("=" * 70)

if not HAS_LIGHTGBM or not HAS_XGBOOST:
    print("\n⚠️  LightGBM ve XGBoost gerekli!")
    print("Yüklemek için: pip install lightgbm xgboost")
    exit(1)

# Load and prepare data
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

# Cross-validation
folds = time_series_cv_indices(train_fit, n_splits=3, val_weeks=6)

lgb_scores = []
xgb_scores = []
ensemble_scores = []

lgb_models = []
xgb_models = []

print("\n2. Ensemble eğitimi başlıyor...")
for i, (tr_idx, va_idx) in enumerate(folds, 1):
    print(f"\n--- Fold {i} ---")
    X_tr, y_tr = X.loc[tr_idx], y[train_fit.index.get_indexer(tr_idx)]
    X_va, y_va = X.loc[va_idx], y[train_fit.index.get_indexer(va_idx)]
    
    # LightGBM
    print("  LightGBM eğitiliyor...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, num_leaves=63, random_state=42,
        n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                 callbacks=[lgb.early_stopping(100, verbose=False)])
    lgb_pred = lgb_model.predict(X_va)
    lgb_pred_orig = np.expm1(lgb_pred)
    y_va_orig = np.expm1(y_va)
    lgb_rmspe = rmspe(y_va_orig, lgb_pred_orig)
    lgb_scores.append(lgb_rmspe)
    lgb_models.append(lgb_model)
    print(f"  LightGBM RMSPE: {lgb_rmspe:.4f}")
    
    # XGBoost
    print("  XGBoost eğitiliyor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, max_depth=8, random_state=42,
        n_jobs=-1, tree_method='hist', early_stopping_rounds=100
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_pred = xgb_model.predict(X_va)
    xgb_pred_orig = np.expm1(xgb_pred)
    xgb_rmspe = rmspe(y_va_orig, xgb_pred_orig)
    xgb_scores.append(xgb_rmspe)
    xgb_models.append(xgb_model)
    print(f"  XGBoost RMSPE: {xgb_rmspe:.4f}")
    
    # Ensemble (weighted average)
    ensemble_pred = 0.5 * lgb_pred + 0.5 * xgb_pred
    ensemble_pred_orig = np.expm1(ensemble_pred)
    ensemble_rmspe = rmspe(y_va_orig, ensemble_pred_orig)
    ensemble_scores.append(ensemble_rmspe)
    print(f"  Ensemble RMSPE: {ensemble_rmspe:.4f} ⭐")

print("\n" + "=" * 70)
print("CROSS-VALIDATION SONUÇLARI")
print("=" * 70)
print(f"\nLightGBM:")
print(f"  Foldlar: {[f'{s:.4f}' for s in lgb_scores]}")
print(f"  Ortalama: {np.mean(lgb_scores):.4f} (±{np.std(lgb_scores):.4f})")

print(f"\nXGBoost:")
print(f"  Foldlar: {[f'{s:.4f}' for s in xgb_scores]}")
print(f"  Ortalama: {np.mean(xgb_scores):.4f} (±{np.std(xgb_scores):.4f})")

print(f"\nEnsemble (50/50):")
print(f"  Foldlar: {[f'{s:.4f}' for s in ensemble_scores]}")
print(f"  Ortalama: {np.mean(ensemble_scores):.4f} (±{np.std(ensemble_scores):.4f})")

improvement = (np.mean(lgb_scores) - np.mean(ensemble_scores)) / np.mean(lgb_scores) * 100
print(f"\nİyileşme: {improvement:+.2f}% (LightGBM'e göre)")

# Train final models on full data
print("\n3. Final modeller tüm veri ile eğitiliyor...")
lgb_final = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, num_leaves=63, random_state=42,
    n_jobs=-1, verbose=-1
)
lgb_final.fit(X, y)

xgb_final = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, max_depth=8, random_state=42,
    n_jobs=-1, tree_method='hist'
)
xgb_final.fit(X, y, verbose=False)

# Save models
dump(lgb_final, os.path.join(MODELS_DIR, 'lgb_model.pkl'))
dump(xgb_final, os.path.join(MODELS_DIR, 'xgb_model.pkl'))
with open(os.path.join(MODELS_DIR, 'features.json'), 'w') as f:
    json.dump(feat_cols, f)

# Save ensemble config
ensemble_config = {
    'lgb_weight': 0.5,
    'xgb_weight': 0.5,
    'cv_results': {
        'lgb_mean': float(np.mean(lgb_scores)),
        'xgb_mean': float(np.mean(xgb_scores)),
        'ensemble_mean': float(np.mean(ensemble_scores))
    }
}
with open(os.path.join(MODELS_DIR, 'ensemble_config.json'), 'w') as f:
    json.dump(ensemble_config, f, indent=2)

print("\n✅ Ensemble modeller kaydedildi:")
print(f"  - models/lgb_model.pkl")
print(f"  - models/xgb_model.pkl")
print(f"  - models/ensemble_config.json")
print(f"  - models/features.json")
print("=" * 70)
