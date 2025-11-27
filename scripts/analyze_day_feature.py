"""
"day" Feature Analizi
Ayın günü (1-31) neden en önemli feature? Overfitting var mı?
"""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import load_datasets, merge_store
from src.features import build_features, select_feature_columns
from src.metrics import rmspe
from src.validation import time_series_cv_indices

import lightgbm as lgb

DATA_DIR = os.path.join(ROOT, 'dataset')
FIGURES_DIR = os.path.join(ROOT, 'outputs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 70)
print("'day' FEATURE ANALİZİ")
print("=" * 70)

# Load data
print("\n1. Veri yükleniyor...")
train, test, store = load_datasets(DATA_DIR)
train, test = merge_store(train, test, store)
train_fe, test_fe = build_features(train, test)

train_fit = train_fe[train_fe['Open'] == 1].dropna(subset=['Sales']).copy()

# ============================================================================
# ANALYZE DAY PATTERN
# ============================================================================

print("\n2. 'day' feature pattern analizi...")

# Sales by day of month
day_sales = train_fit.groupby('day')['Sales'].agg(['mean', 'std', 'count'])
print("\nGünlere göre ortalama satış:")
print(day_sales)

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Average sales by day
axes[0, 0].bar(day_sales.index, day_sales['mean'])
axes[0, 0].set_xlabel('Ayın Günü')
axes[0, 0].set_ylabel('Ortalama Satış')
axes[0, 0].set_title('Ayın Gününe Göre Ortalama Satış')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Sales distribution by day groups
day_groups = pd.cut(train_fit['day'], bins=[0, 10, 20, 31], labels=['1-10', '11-20', '21-31'])
train_fit['day_group'] = day_groups
sns.boxplot(data=train_fit, x='day_group', y='Sales', ax=axes[0, 1])
axes[0, 1].set_title('Ay Dönemlerine Göre Satış Dağılımı')
axes[0, 1].set_ylabel('Satış')

# Plot 3: Promo by day
promo_by_day = train_fit.groupby('day')['Promo'].mean()
axes[1, 0].bar(promo_by_day.index, promo_by_day)
axes[1, 0].set_xlabel('Ayın Günü')
axes[1, 0].set_ylabel('Promo Oranı')
axes[1, 0].set_title('Ayın Gününe Göre Promo Oranı')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Day of week vs day of month interaction
pivot = train_fit.pivot_table(values='Sales', index='day', columns='DayOfWeek', aggfunc='mean')
sns.heatmap(pivot, cmap='YlOrRd', ax=axes[1, 1], cbar_kws={'label': 'Ortalama Satış'})
axes[1, 1].set_title('Ayın Günü × Haftanın Günü Etkileşimi')
axes[1, 1].set_xlabel('Haftanın Günü')
axes[1, 1].set_ylabel('Ayın Günü')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'day_feature_analysis.png'), dpi=300, bbox_inches='tight')
print(f"✓ Grafik kaydedildi: day_feature_analysis.png")

# ============================================================================
# TEST WITH/WITHOUT DAY FEATURE
# ============================================================================

print("\n3. 'day' feature ile/siz model karşılaştırması...")

y = np.log1p(train_fit['Sales'].values)
feat_cols = select_feature_columns(train_fit)
X_with_day = train_fit[feat_cols].fillna(0)
X_without_day = X_with_day.drop(columns=['day'])

folds = time_series_cv_indices(train_fit, n_splits=3, val_weeks=6)

params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'num_leaves': 63,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# With day
print("\n  'day' feature ile:")
scores_with = []
for fold_num, (tr_idx, va_idx) in enumerate(folds, 1):
    X_tr = X_with_day.loc[tr_idx]
    y_tr = y[train_fit.index.get_indexer(tr_idx)]
    X_va = X_with_day.loc[va_idx]
    y_va = y[train_fit.index.get_indexer(va_idx)]
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    
    pred = np.expm1(model.predict(X_va))
    y_va_orig = np.expm1(y_va)
    pred[train_fit.loc[va_idx, 'Open'] == 0] = 0
    
    score = rmspe(y_va_orig, pred)
    scores_with.append(score)
    print(f"    Fold {fold_num}: {score:.4f}")

print(f"  Ortalama: {np.mean(scores_with):.4f}")

# Without day
print("\n  'day' feature olmadan:")
scores_without = []
for fold_num, (tr_idx, va_idx) in enumerate(folds, 1):
    X_tr = X_without_day.loc[tr_idx]
    y_tr = y[train_fit.index.get_indexer(tr_idx)]
    X_va = X_without_day.loc[va_idx]
    y_va = y[train_fit.index.get_indexer(va_idx)]
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    
    pred = np.expm1(model.predict(X_va))
    y_va_orig = np.expm1(y_va)
    pred[train_fit.loc[va_idx, 'Open'] == 0] = 0
    
    score = rmspe(y_va_orig, pred)
    scores_without.append(score)
    print(f"    Fold {fold_num}: {score:.4f}")

print(f"  Ortalama: {np.mean(scores_without):.4f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SONUÇ")
print("=" * 70)

improvement = (np.mean(scores_without) - np.mean(scores_with)) / np.mean(scores_without) * 100

print(f"\n'day' feature ile:    {np.mean(scores_with):.4f} RMSPE")
print(f"'day' feature olmadan: {np.mean(scores_without):.4f} RMSPE")
print(f"\nİyileşme: {improvement:.2f}%")

if improvement > 1:
    print("\n✅ 'day' feature gerçekten işe yarıyor!")
    print("   Ayın sonu/başı etkisi var (maaş günleri, kampanyalar)")
elif improvement > 0:
    print("\n⚠️  'day' feature küçük bir etki yapıyor")
    print("   Ama overfitting riski var, dikkatli kullan")
else:
    print("\n❌ 'day' feature overfitting yapıyor!")
    print("   Modelden çıkarmayı düşün")

# Check if end-of-month matters
end_of_month = train_fit[train_fit['day'] >= 25]['Sales'].mean()
mid_month = train_fit[(train_fit['day'] >= 10) & (train_fit['day'] <= 20)]['Sales'].mean()
print(f"\nAy sonu satış (25-31): {end_of_month:.0f}")
print(f"Ay ortası satış (10-20): {mid_month:.0f}")
print(f"Fark: {(end_of_month - mid_month) / mid_month * 100:.1f}%")

print("=" * 70)
