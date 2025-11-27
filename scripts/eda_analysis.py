"""
EDA (Exploratory Data Analysis) Script
Tez için veri analizi ve görselleştirmeler
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

DATA_DIR = os.path.join(ROOT, 'dataset')
FIGURES_DIR = os.path.join(ROOT, 'outputs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Load data
print("\n1. Veri yükleniyor...")
train, test, store = load_datasets(DATA_DIR)
train_merged = train.merge(store, on='Store', how='left')

print(f"✓ Train: {train.shape}")
print(f"✓ Test: {test.shape}")
print(f"✓ Store: {store.shape}")

# Sales distribution
print("\n2. Sales dağılımı grafikleri oluşturuluyor...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
train['Sales'].hist(bins=100, ax=ax, edgecolor='black', alpha=0.7)
ax.set_title('Sales Dağılımı', fontsize=14, fontweight='bold')
ax.axvline(train['Sales'].mean(), color='red', linestyle='--', label=f'Ort: {train["Sales"].mean():.0f}')
ax.legend()

ax = axes[0, 1]
train[train['Open'] == 1]['Sales'].hist(bins=100, ax=ax, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_title('Sales Dağılımı (Açık Günler)', fontsize=14, fontweight='bold')

ax = axes[1, 0]
np.log1p(train[train['Sales'] > 0]['Sales']).hist(bins=100, ax=ax, edgecolor='black', alpha=0.7, color='coral')
ax.set_title('Log(Sales+1) Dağılımı', fontsize=14, fontweight='bold')

ax = axes[1, 1]
dow_sales = train[train['Open'] == 1].groupby('DayOfWeek')['Sales'].mean()
dow_names = {1:'Pzt',2:'Sal',3:'Çar',4:'Per',5:'Cum',6:'Cmt',7:'Paz'}
labels = [dow_names[d] for d in dow_sales.index]
ax.bar(labels, dow_sales.values, edgecolor='black', alpha=0.7)
ax.set_title('Haftanın Gününe Göre Ort. Satışlar', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'eda_sales_analysis.png'), dpi=150, bbox_inches='tight')
print(f"✓ Kaydedildi: eda_sales_analysis.png")
plt.close()

# Promo effect
print("\n3. Promo etkisi analizi...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

promo_stats = train[train['Open'] == 1].groupby('Promo')['Sales'].mean()
ax = axes[0]
ax.bar(['Promo Yok', 'Promo Var'], promo_stats.values, color=['coral', 'lightgreen'], edgecolor='black', alpha=0.7)
ax.set_title('Promo Etkisi', fontsize=14, fontweight='bold')
ax.set_ylabel('Ortalama Sales')
for i, v in enumerate(promo_stats.values):
    ax.text(i, v+100, f'{v:.0f}', ha='center', fontweight='bold')

ax = axes[1]
storetype_sales = train_merged[train_merged['Open'] == 1].groupby('StoreType')['Sales'].mean()
ax.bar(storetype_sales.index, storetype_sales.values, edgecolor='black', alpha=0.7)
ax.set_title('Mağaza Tipine Göre Ort. Satışlar', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'eda_promo_storetype.png'), dpi=150, bbox_inches='tight')
print(f"✓ Kaydedildi: eda_promo_storetype.png")
plt.close()

# Time series
print("\n4. Zaman serisi analizi...")
daily_sales = train.groupby('Date')['Sales'].sum().reset_index()
daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])

plt.figure(figsize=(16, 6))
plt.plot(daily_sales['Date'], daily_sales['Sales'], linewidth=1, alpha=0.7)
plt.title('Günlük Toplam Satışlar (Tüm Mağazalar)', fontsize=14, fontweight='bold')
plt.xlabel('Tarih')
plt.ylabel('Toplam Sales')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'eda_time_series.png'), dpi=150, bbox_inches='tight')
print(f"✓ Kaydedildi: eda_time_series.png")
plt.close()

# Correlation
print("\n5. Korelasyon matrisi...")
numeric_cols = ['Sales', 'Customers', 'Open', 'Promo', 'DayOfWeek']
corr_matrix = train[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, square=True)
plt.title('Özellik Korelasyon Matrisi', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'eda_correlation.png'), dpi=150, bbox_inches='tight')
print(f"✓ Kaydedildi: eda_correlation.png")
plt.close()

# Summary
print("\n" + "=" * 70)
print("EDA ÖZETİ")
print("=" * 70)
print(f"\nVeri Seti:")
print(f"  - Toplam kayıt: {len(train):,}")
print(f"  - Mağaza sayısı: {train['Store'].nunique()}")
print(f"  - Tarih aralığı: {train['Date'].min()} - {train['Date'].max()}")

open_sales = train[train['Open'] == 1]['Sales']
print(f"\nSatış İstatistikleri (Açık Günler):")
print(f"  - Ortalama: {open_sales.mean():.2f}")
print(f"  - Medyan: {open_sales.median():.2f}")
print(f"  - Std: {open_sales.std():.2f}")

promo_lift = (promo_stats.iloc[1] / promo_stats.iloc[0] - 1) * 100
print(f"\nPromo Etkisi: +{promo_lift:.2f}% satış artışı")

print("\n✅ EDA tamamlandı! Grafikler: outputs/figures/")
print("=" * 70)
