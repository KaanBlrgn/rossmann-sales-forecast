"""
SHAP (SHapley Additive exPlanations) Analizi
Model yorumlama ve özellik etkisi analizi
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
from joblib import load
import json

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️  SHAP kütüphanesi bulunamadı. Yüklemek için: pip install shap")
    exit(1)

from src.data import load_datasets, merge_store
from src.features import build_features

DATA_DIR = os.path.join(ROOT, 'dataset')
MODELS_DIR = os.path.join(ROOT, 'models')
FIGURES_DIR = os.path.join(ROOT, 'outputs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 70)
print("SHAP ANALYSIS - Model Yorumlama")
print("=" * 70)

# Load model and features
print("\n1. Model ve veri yükleniyor...")
# Load ensemble LightGBM model (primary model for SHAP)
model = load(os.path.join(MODELS_DIR, 'lgb_model.pkl'))
print("✓ LightGBM ensemble model yüklendi")
with open(os.path.join(MODELS_DIR, 'features.json'), 'r') as f:
    feat_cols = json.load(f)
print(f"✓ {len(feat_cols)} özellik yüklendi")

train, test, store = load_datasets(DATA_DIR)
train, test = merge_store(train, test, store)
train_fe, test_fe = build_features(train, test)

# Prepare data (Open=1 only)
train_fit = train_fe[train_fe['Open'] == 1].dropna(subset=['Sales']).copy()
X = train_fit[feat_cols].fillna(0)

# Sample for SHAP (computational efficiency)
sample_size = min(1000, len(X))
X_sample = X.sample(sample_size, random_state=42)

print(f"✓ Model yüklendi: {type(model).__name__}")
print(f"✓ Özellik sayısı: {len(feat_cols)}")
print(f"✓ SHAP için örnek boyutu: {sample_size}")

# Create SHAP explainer
print("\n2. SHAP explainer oluşturuluyor...")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    print("✓ SHAP değerleri hesaplandı")
except Exception as e:
    print(f"❌ SHAP hesaplanırken hata: {e}")
    exit(1)

# 1. Summary Plot
print("\n3. SHAP summary plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)
plt.title('SHAP Summary Plot - Özellik Etkileri', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
print(f"✓ Kaydedildi: shap_summary.png")
plt.close()

# 2. Bar Plot (Mean absolute SHAP values)
print("\n4. SHAP bar plot (özellik önemi)...")
plt.figure(figsize=(10, 10))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Ortalama |SHAP değeri|)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'shap_importance.png'), dpi=150, bbox_inches='tight')
print(f"✓ Kaydedildi: shap_importance.png")
plt.close()

# 3. Dependence Plot for top features
print("\n5. SHAP dependence plots (top 4 özellik)...")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_features_idx = np.argsort(mean_abs_shap)[-4:][::-1]
top_features = [feat_cols[i] for i in top_features_idx]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (feat_idx, feat_name) in enumerate(zip(top_features_idx, top_features)):
    ax = axes[idx]
    shap.dependence_plot(feat_idx, shap_values, X_sample, show=False, ax=ax)
    ax.set_title(f'SHAP Dependence: {feat_name}', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'shap_dependence.png'), dpi=150, bbox_inches='tight')
print(f"✓ Kaydedildi: shap_dependence.png")
plt.close()

# 4. Waterfall plot (single prediction explanation)
print("\n6. SHAP waterfall plot (tek tahmin örneği)...")
fig, ax = plt.subplots(figsize=(10, 8))
sample_idx = 0
expected_value = explainer.expected_value
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[sample_idx],
        base_values=expected_value,
        data=X_sample.iloc[sample_idx],
        feature_names=feat_cols
    ),
    show=False
)
plt.title(f'SHAP Waterfall Plot - Tek Tahmin Açıklaması', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'shap_waterfall.png'), dpi=150, bbox_inches='tight')
print(f"✓ Kaydedildi: shap_waterfall.png")
plt.close()

# Save SHAP values summary
print("\n7. SHAP değerleri özeti kaydediliyor...")
shap_summary = pd.DataFrame({
    'feature': feat_cols,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0),
    'mean_shap': shap_values.mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

shap_summary.to_csv(os.path.join(ROOT, 'outputs', 'reports', 'shap_values.csv'), index=False)
print(f"✓ Kaydedildi: shap_values.csv")

print("\n" + "=" * 70)
print("SHAP ANALİZİ TAMAMLANDI")
print("=" * 70)
print(f"\nGrafikler:")
print(f"  - shap_summary.png (özellik etkileri)")
print(f"  - shap_importance.png (özellik önemi)")
print(f"  - shap_dependence.png (top 4 özellik ilişkisi)")
print(f"  - shap_waterfall.png (tek tahmin açıklaması)")
print(f"\nRapor:")
print(f"  - shap_values.csv")

print(f"\nEn etkili 10 özellik:")
print(shap_summary.head(10).to_string(index=False))
print("=" * 70)
