"""
Ensemble Prediction Script
En iyi modeller (LightGBM + XGBoost) ile final submission üretir
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
from joblib import load

from src.data import load_datasets, merge_store
from src.features import build_features

DATA_DIR = os.path.join(ROOT, 'dataset')
MODELS_DIR = os.path.join(ROOT, 'models')
OUTPUT_FILE = os.path.join(ROOT, 'submission.csv')

print("=" * 70)
print("ENSEMBLE PREDICTION - Final Submission")
print("=" * 70)

# Load ensemble config
print("\n1. Ensemble konfigürasyonu yükleniyor...")
with open(os.path.join(MODELS_DIR, 'ensemble_config.json'), 'r') as f:
    config = json.load(f)

lgb_weight = config['lgb_weight']
xgb_weight = config['xgb_weight']

print(f"✓ LightGBM ağırlığı: {lgb_weight}")
print(f"✓ XGBoost ağırlığı: {xgb_weight}")
print(f"✓ CV RMSPE (Ensemble): {config['cv_results']['ensemble_mean']:.4f}")

# Load models
print("\n2. Modeller yükleniyor...")
lgb_model = load(os.path.join(MODELS_DIR, 'lgb_model.pkl'))
xgb_model = load(os.path.join(MODELS_DIR, 'xgb_model.pkl'))
with open(os.path.join(MODELS_DIR, 'features.json'), 'r') as f:
    feat_cols = json.load(f)

print(f"✓ LightGBM modeli yüklendi")
print(f"✓ XGBoost modeli yüklendi")
print(f"✓ Özellik sayısı: {len(feat_cols)}")

# Load and prepare test data
print("\n3. Test verisi hazırlanıyor...")
train, test, store = load_datasets(DATA_DIR)
train, test = merge_store(train, test, store)
train_fe, test_fe = build_features(train, test)

# Prepare test features
X_test = test_fe[feat_cols].fillna(0)
print(f"✓ Test boyutu: {len(X_test):,}")

# Make predictions
print("\n4. Tahminler oluşturuluyor...")
lgb_pred_log = lgb_model.predict(X_test)
xgb_pred_log = xgb_model.predict(X_test)

# Ensemble predictions (weighted average)
ensemble_pred_log = lgb_weight * lgb_pred_log + xgb_weight * xgb_pred_log

# Inverse log transform
ensemble_pred = np.expm1(ensemble_pred_log)

# Apply business rules
# Rule 1: Open=0 -> Sales=0
if 'Open' in test_fe.columns:
    ensemble_pred[test_fe['Open'] == 0] = 0

# Rule 2: Clip negative predictions to 0
ensemble_pred = np.clip(ensemble_pred, 0, None)

print(f"✓ LightGBM tahminleri tamamlandı")
print(f"✓ XGBoost tahminleri tamamlandı")
print(f"✓ Ensemble tahminleri oluşturuldu")

# Create submission
print("\n5. Submission dosyası oluşturuluyor...")
submission = pd.DataFrame({
    'Id': test_fe['Id'],
    'Sales': ensemble_pred
})

# Statistics
print(f"\nTahmin İstatistikleri:")
print(f"  - Ortalama: {ensemble_pred.mean():.2f}")
print(f"  - Medyan: {np.median(ensemble_pred):.2f}")
print(f"  - Min: {ensemble_pred.min():.2f}")
print(f"  - Max: {ensemble_pred.max():.2f}")
print(f"  - Sıfır tahmin: {(ensemble_pred == 0).sum():,} ({(ensemble_pred == 0).sum()/len(ensemble_pred)*100:.1f}%)")

# Save submission
submission.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Submission kaydedildi: {OUTPUT_FILE}")

# Model comparison
print("\n" + "=" * 70)
print("MODEL KARŞILAŞTIRMASI")
print("=" * 70)
print(f"\nCross-Validation Sonuçları (3-fold):")
print(f"  LightGBM:  {config['cv_results']['lgb_mean']:.4f} RMSPE")
print(f"  XGBoost:   {config['cv_results']['xgb_mean']:.4f} RMSPE")
print(f"  Ensemble:  {config['cv_results']['ensemble_mean']:.4f} RMSPE ⭐")
print(f"\nKullanılan Model: Ensemble ({lgb_weight*100:.0f}% LGB + {xgb_weight*100:.0f}% XGB)")
print("=" * 70)
