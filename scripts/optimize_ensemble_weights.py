"""
Ensemble Ağırlık Optimizasyonu
LightGBM ve XGBoost için optimal ağırlıkları bulur
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
from scipy.optimize import minimize
from joblib import load

from src.data import load_datasets, merge_store
from src.features import build_features, select_feature_columns
from src.metrics import rmspe
from src.validation import time_series_cv_indices

import lightgbm as lgb
import xgboost as xgb

DATA_DIR = os.path.join(ROOT, 'dataset')
MODELS_DIR = os.path.join(ROOT, 'models')

print("=" * 70)
print("ENSEMBLE WEIGHT OPTIMIZATION")
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

# CV folds
folds = time_series_cv_indices(train_fit, n_splits=3, val_weeks=6)

# ============================================================================
# COLLECT PREDICTIONS
# ============================================================================

print("\n2. Model tahminleri toplanıyor...")

lgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'num_leaves': 63,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'max_depth': 8,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
}

all_lgb_preds = []
all_xgb_preds = []
all_y_true = []
all_open_flags = []

for fold_num, (tr_idx, va_idx) in enumerate(folds, 1):
    print(f"  Fold {fold_num}...")
    
    X_tr, y_tr = X.loc[tr_idx], y[train_fit.index.get_indexer(tr_idx)]
    X_va, y_va = X.loc[va_idx], y[train_fit.index.get_indexer(va_idx)]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    lgb_pred = lgb_model.predict(X_va)
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100)
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_pred = xgb_model.predict(X_va)
    
    # Store predictions
    all_lgb_preds.extend(lgb_pred)
    all_xgb_preds.extend(xgb_pred)
    all_y_true.extend(y_va)
    all_open_flags.extend(train_fit.loc[va_idx, 'Open'].values)

# Convert to arrays
all_lgb_preds = np.array(all_lgb_preds)
all_xgb_preds = np.array(all_xgb_preds)
all_y_true = np.array(all_y_true)
all_open_flags = np.array(all_open_flags)

print("✓ Tahminler toplandı")

# ============================================================================
# OPTIMIZE WEIGHTS
# ============================================================================

print("\n3. Optimal ağırlıklar bulunuyor...")

def ensemble_objective(weights):
    """Ensemble RMSPE hesapla"""
    w_lgb = weights[0]
    w_xgb = 1 - w_lgb
    
    # Ensemble prediction (log space)
    ensemble_pred_log = w_lgb * all_lgb_preds + w_xgb * all_xgb_preds
    
    # Convert to original space
    ensemble_pred = np.expm1(ensemble_pred_log)
    y_true_orig = np.expm1(all_y_true)
    
    # Handle Open=0
    ensemble_pred[all_open_flags == 0] = 0
    
    # Calculate RMSPE
    score = rmspe(y_true_orig, ensemble_pred)
    return score

# Test different weight combinations
print("\n  Test ediliyor:")
test_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
test_results = []

for w in test_weights:
    score = ensemble_objective([w])
    test_results.append((w, 1-w, score))
    print(f"    LGB={w:.1f}, XGB={1-w:.1f} → RMSPE: {score:.4f}")

# Find optimal with scipy
result = minimize(
    ensemble_objective,
    x0=[0.5],  # Start from 50/50
    bounds=[(0, 1)],
    method='L-BFGS-B'
)

optimal_lgb_weight = result.x[0]
optimal_xgb_weight = 1 - optimal_lgb_weight
optimal_score = result.fun

print(f"\n✓ Optimal ağırlıklar bulundu!")

# ============================================================================
# COMPARE RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("SONUÇLAR")
print("=" * 70)

# Individual scores
lgb_only_score = ensemble_objective([1.0])
xgb_only_score = ensemble_objective([0.0])
equal_weight_score = ensemble_objective([0.5])

print(f"\nTek Model Skorları:")
print(f"  LightGBM (100%):  {lgb_only_score:.4f} RMSPE")
print(f"  XGBoost (100%):   {xgb_only_score:.4f} RMSPE")

print(f"\nEnsemble Skorları:")
print(f"  Eşit Ağırlık (50/50):  {equal_weight_score:.4f} RMSPE")
print(f"  Optimal Ağırlık:       {optimal_score:.4f} RMSPE")
print(f"    → LightGBM: {optimal_lgb_weight:.1%}")
print(f"    → XGBoost:  {optimal_xgb_weight:.1%}")

improvement = (equal_weight_score - optimal_score) / equal_weight_score * 100
print(f"\n  İyileşme: {improvement:.2f}%")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n4. Sonuçlar kaydediliyor...")

weight_results = {
    'optimal_weights': {
        'lgb_weight': float(optimal_lgb_weight),
        'xgb_weight': float(optimal_xgb_weight)
    },
    'scores': {
        'lgb_only': float(lgb_only_score),
        'xgb_only': float(xgb_only_score),
        'equal_weight': float(equal_weight_score),
        'optimal': float(optimal_score)
    },
    'improvement': {
        'vs_equal_weight_pct': float(improvement),
        'vs_lgb_pct': float((lgb_only_score - optimal_score) / lgb_only_score * 100),
        'vs_xgb_pct': float((xgb_only_score - optimal_score) / xgb_only_score * 100)
    },
    'test_results': [
        {'lgb_weight': w, 'xgb_weight': 1-w, 'rmspe': s}
        for w, _, s in test_results
    ]
}

with open(os.path.join(MODELS_DIR, 'optimal_weights.json'), 'w') as f:
    json.dump(weight_results, f, indent=2)

print(f"✓ Sonuçlar kaydedildi: models/optimal_weights.json")

print("\n💡 Sonraki adım: ensemble_config.json'u güncelle!")
print(f"   lgb_weight: {optimal_lgb_weight:.3f}")
print(f"   xgb_weight: {optimal_xgb_weight:.3f}")
print("=" * 70)
