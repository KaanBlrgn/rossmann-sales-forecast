"""
Fold Tutarsızlığı Analizi
Neden Fold 1: 0.1539, Fold 3: 0.1216? (%26 fark)
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
from src.features import build_features
from src.validation import time_series_cv_indices

DATA_DIR = os.path.join(ROOT, 'dataset')
FIGURES_DIR = os.path.join(ROOT, 'outputs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 70)
print("FOLD TUTARSIZLIĞI ANALİZİ")
print("=" * 70)

# Load data
print("\n1. Veri yükleniyor...")
train, test, store = load_datasets(DATA_DIR)
train, test = merge_store(train, test, store)
train_fe, test_fe = build_features(train, test)

train_fit = train_fe[train_fe['Open'] == 1].dropna(subset=['Sales']).copy()

# Get folds
folds = time_series_cv_indices(train_fit, n_splits=3, val_weeks=6)

# ============================================================================
# ANALYZE EACH FOLD
# ============================================================================

print("\n2. Her fold analiz ediliyor...")

fold_stats = []

for fold_num, (tr_idx, va_idx) in enumerate(folds, 1):
    val_data = train_fit.loc[va_idx]
    
    stats = {
        'fold': fold_num,
        'date_range': f"{val_data['Date'].min().strftime('%Y-%m-%d')} → {val_data['Date'].max().strftime('%Y-%m-%d')}",
        'n_samples': len(val_data),
        'sales_mean': val_data['Sales'].mean(),
        'sales_std': val_data['Sales'].std(),
        'sales_cv': val_data['Sales'].std() / val_data['Sales'].mean(),  # Coefficient of variation
        'promo_rate': val_data['Promo'].mean(),
        'school_holiday_rate': val_data['SchoolHoliday'].mean(),
        'unique_stores': val_data['Store'].nunique(),
        'days': val_data['Date'].nunique()
    }
    
    fold_stats.append(stats)
    
    print(f"\nFold {fold_num}:")
    print(f"  Tarih: {stats['date_range']}")
    print(f"  Satış Ort: {stats['sales_mean']:.0f} (Std: {stats['sales_std']:.0f})")
    print(f"  CV: {stats['sales_cv']:.3f}")
    print(f"  Promo: {stats['promo_rate']:.1%}")
    print(f"  Okul Tatili: {stats['school_holiday_rate']:.1%}")

fold_df = pd.DataFrame(fold_stats)

# ============================================================================
# VISUALIZE DIFFERENCES
# ============================================================================

print("\n3. Grafikler oluşturuluyor...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Sales distribution by fold
for fold_num, (_, va_idx) in enumerate(folds, 1):
    val_data = train_fit.loc[va_idx]
    axes[0, 0].hist(val_data['Sales'], bins=50, alpha=0.5, label=f'Fold {fold_num}')
axes[0, 0].set_xlabel('Satış')
axes[0, 0].set_ylabel('Frekans')
axes[0, 0].set_title('Fold\'lara Göre Satış Dağılımı')
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 20000)

# Plot 2: Mean sales by fold
axes[0, 1].bar(fold_df['fold'], fold_df['sales_mean'], color=['red', 'orange', 'green'])
axes[0, 1].set_xlabel('Fold')
axes[0, 1].set_ylabel('Ortalama Satış')
axes[0, 1].set_title('Fold\'lara Göre Ortalama Satış')
axes[0, 1].set_xticks([1, 2, 3])

# Plot 3: Coefficient of Variation
axes[0, 2].bar(fold_df['fold'], fold_df['sales_cv'], color=['red', 'orange', 'green'])
axes[0, 2].set_xlabel('Fold')
axes[0, 2].set_ylabel('CV (Std/Mean)')
axes[0, 2].set_title('Satış Volatilitesi (CV)')
axes[0, 2].set_xticks([1, 2, 3])

# Plot 4: Promo rate
axes[1, 0].bar(fold_df['fold'], fold_df['promo_rate'], color=['red', 'orange', 'green'])
axes[1, 0].set_xlabel('Fold')
axes[1, 0].set_ylabel('Promo Oranı')
axes[1, 0].set_title('Fold\'lara Göre Promo Oranı')
axes[1, 0].set_xticks([1, 2, 3])

# Plot 5: School Holiday rate
axes[1, 1].bar(fold_df['fold'], fold_df['school_holiday_rate'], color=['red', 'orange', 'green'])
axes[1, 1].set_xlabel('Fold')
axes[1, 1].set_ylabel('Okul Tatili Oranı')
axes[1, 1].set_title('Fold\'lara Göre Okul Tatili Oranı')
axes[1, 1].set_xticks([1, 2, 3])

# Plot 6: Time series of all folds
for fold_num, (_, va_idx) in enumerate(folds, 1):
    val_data = train_fit.loc[va_idx]
    daily_sales = val_data.groupby('Date')['Sales'].mean()
    axes[1, 2].plot(daily_sales.index, daily_sales.values, label=f'Fold {fold_num}', alpha=0.7)
axes[1, 2].set_xlabel('Tarih')
axes[1, 2].set_ylabel('Ortalama Satış')
axes[1, 2].set_title('Fold\'ların Zaman Serisi')
axes[1, 2].legend()
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fold_variance_analysis.png'), dpi=300, bbox_inches='tight')
print(f"✓ Grafik kaydedildi: fold_variance_analysis.png")

# ============================================================================
# STATISTICAL TESTS
# ============================================================================

print("\n4. İstatistiksel testler...")

from scipy import stats

# Compare fold 1 vs fold 3 (en farklı olanlar)
fold1_sales = train_fit.loc[folds[0][1], 'Sales']
fold3_sales = train_fit.loc[folds[2][1], 'Sales']

# T-test
t_stat, p_value = stats.ttest_ind(fold1_sales, fold3_sales)
print(f"\nFold 1 vs Fold 3 T-test:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.6f}")

if p_value < 0.001:
    print("  ✅ Fold'lar istatistiksel olarak FARKLI (p < 0.001)")
else:
    print("  ⚠️  Fold'lar benzer")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SONUÇ VE ÖNERİLER")
print("=" * 70)

print("\nFold Karakteristikleri:")
print(fold_df[['fold', 'sales_mean', 'sales_cv', 'promo_rate']].to_string(index=False))

# Identify why fold 3 is easier
fold3_stats = fold_stats[2]
fold1_stats = fold_stats[0]

print(f"\nNeden Fold 3 daha kolay?")
reasons = []

if fold3_stats['sales_cv'] < fold1_stats['sales_cv']:
    reasons.append(f"  - Daha düşük volatilite (CV: {fold3_stats['sales_cv']:.3f} vs {fold1_stats['sales_cv']:.3f})")

if fold3_stats['sales_mean'] < fold1_stats['sales_mean']:
    reasons.append(f"  - Daha düşük ortalama satış ({fold3_stats['sales_mean']:.0f} vs {fold1_stats['sales_mean']:.0f})")

if fold3_stats['promo_rate'] < fold1_stats['promo_rate']:
    reasons.append(f"  - Daha az promo ({fold3_stats['promo_rate']:.1%} vs {fold1_stats['promo_rate']:.1%})")

if fold3_stats['school_holiday_rate'] > fold1_stats['school_holiday_rate']:
    reasons.append(f"  - Daha fazla okul tatili ({fold3_stats['school_holiday_rate']:.1%} vs {fold1_stats['school_holiday_rate']:.1%})")

if reasons:
    print("\n".join(reasons))
else:
    print("  - Mevsimsel etkiler (yaz vs bahar)")

print(f"\nÖNERİLER:")
print("  1. Stratified CV kullan (mevsim bazlı)")
print("  2. Fold ağırlıklandırması yap (zor fold'lara daha fazla ağırlık)")
print("  3. Ensemble'da fold-specific modeller kullan")
print("  4. Test setinin hangi fold'a benzediğini analiz et")

print("=" * 70)
