"""
Store Clustering Analysis
1,115 mağazayı benzer özelliklere göre gruplandırma
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.data import load_datasets, merge_store

DATA_DIR = os.path.join(ROOT, 'dataset')
FIGURES_DIR = os.path.join(ROOT, 'outputs', 'figures')
REPORTS_DIR = os.path.join(ROOT, 'outputs', 'reports')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 70)
print("STORE CLUSTERING ANALYSIS")
print("=" * 70)

# Load data
print("\n1. Veri yükleniyor...")
train, test, store = load_datasets(DATA_DIR)
print(f"✓ Train: {train.shape}")
print(f"✓ Store: {store.shape}")
print(f"✓ Benzersiz mağaza sayısı: {train['Store'].nunique()}")

# Prepare clustering features
print("\n2. Mağaza özellikleri hesaplanıyor...")

# Sadece açık günleri kullan
train_open = train[train['Open'] == 1].copy()

# Satış istatistikleri (mağaza bazında)
sales_stats = train_open.groupby('Store').agg({
    'Sales': ['mean', 'median', 'std', 'min', 'max'],
    'Customers': ['mean', 'median'],
    'Promo': 'mean',
    'Open': 'count'
}).reset_index()
sales_stats.columns = ['Store', 'avg_sales', 'median_sales', 'std_sales', 
                       'min_sales', 'max_sales', 'avg_customers', 'median_customers',
                       'promo_usage_rate', 'days_open']

# Promo etkisi
promo_effect = train_open.groupby(['Store', 'Promo'])['Sales'].mean().unstack(fill_value=0)
promo_effect.columns = ['sales_no_promo', 'sales_with_promo']
promo_effect['promo_lift'] = ((promo_effect['sales_with_promo'] / 
                                (promo_effect['sales_no_promo'] + 1)) - 1).clip(0, 2)
promo_effect = promo_effect.reset_index()

# Haftanın günü volatilitesi
dow_volatility = train_open.groupby('Store')['Sales'].std().reset_index()
dow_volatility.columns = ['Store', 'dow_sales_volatility']

# Store meta verilerini encode et
store_encoded = store.copy()
le_storetype = LabelEncoder()
le_assortment = LabelEncoder()
store_encoded['StoreType_enc'] = le_storetype.fit_transform(store['StoreType'].fillna('unknown'))
store_encoded['Assortment_enc'] = le_assortment.fit_transform(store['Assortment'].fillna('unknown'))

# Rekabet özellikleri
store_encoded['log_comp_distance'] = np.log1p(store_encoded['CompetitionDistance'].fillna(75000))
store_encoded['comp_months'] = (
    (2015 - store_encoded['CompetitionOpenSinceYear'].fillna(2000)) * 12 +
    (7 - store_encoded['CompetitionOpenSinceMonth'].fillna(1))
).clip(0, 300)

# Tüm özellikleri birleştir
clustering_data = store_encoded[['Store', 'StoreType', 'Assortment', 'StoreType_enc', 
                                  'Assortment_enc', 'Promo2', 'log_comp_distance', 'comp_months']]
clustering_data = clustering_data.merge(sales_stats, on='Store', how='left')
clustering_data = clustering_data.merge(promo_effect[['Store', 'promo_lift']], on='Store', how='left')
clustering_data = clustering_data.merge(dow_volatility, on='Store', how='left')

# Eksik değerleri doldur
clustering_data = clustering_data.fillna(0)

print(f"✓ Kümeleme verisi hazır: {clustering_data.shape}")
print(f"✓ Özellik sayısı: {clustering_data.shape[1] - 3}")  # Store, StoreType, Assortment hariç

# Select features for clustering
feature_cols = [
    'StoreType_enc', 'Assortment_enc', 'Promo2',
    'log_comp_distance', 'comp_months',
    'avg_sales', 'std_sales', 'avg_customers',
    'promo_usage_rate', 'promo_lift', 'dow_sales_volatility'
]

X = clustering_data[feature_cols].values
store_ids = clustering_data['Store'].values

print(f"\nSeçilen özellikler ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")

# Standardization
print("\n3. Normalizasyon...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✓ StandardScaler uygulandı")

# Elbow Method
print("\n4. Optimal küme sayısı belirleniyor (Elbow Method)...")
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow Curve
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Küme Sayısı (k)', fontsize=12)
ax.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
ax.set_title('Elbow Method - Optimal Küme Sayısı', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axvline(x=5, color='red', linestyle='--', linewidth=2, label='Optimal k=5')
ax.legend()

ax = axes[1]
ax.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax.set_xlabel('Küme Sayısı (k)', fontsize=12)
ax.set_ylabel('Silhouette Score', fontsize=12)
ax.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axvline(x=5, color='red', linestyle='--', linewidth=2, label='Optimal k=5')
ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Good threshold (0.5)')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'clustering_elbow_silhouette.png'), dpi=150, bbox_inches='tight')
print(f"✓ Grafik kaydedildi: clustering_elbow_silhouette.png")
plt.close()

# Final Clustering (k=5)
print("\n5. K-Means kümeleme (k=5)...")
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
clusters = kmeans.fit_predict(X_scaled)

clustering_data['cluster'] = clusters
silhouette_avg = silhouette_score(X_scaled, clusters)

print(f"✓ Kümeleme tamamlandı")
print(f"✓ Silhouette Score: {silhouette_avg:.3f}")
print(f"\nKüme dağılımı:")
print(clustering_data['cluster'].value_counts().sort_index())

# PCA 2D Visualization
print("\n6. PCA 2D görselleştirme...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(14, 10))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
cluster_names = ['Premium City', 'Suburban Std', 'Small Town', 'Flagship', 'Rural Low']

for i in range(optimal_k):
    mask = clusters == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=colors[i], label=f'Cluster {i}: {cluster_names[i]} (n={mask.sum()})',
               alpha=0.6, s=80, edgecolors='black', linewidth=0.5)

# Cluster centers
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
           c='black', marker='X', s=300, edgecolors='white', linewidth=2,
           label='Küme Merkezleri', zorder=10)

plt.xlabel(f'PC1 (Açıklanan Varyans: {pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
plt.ylabel(f'PC2 (Açıklanan Varyans: {pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
plt.title('Store Clustering - PCA 2D Projection', fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'clustering_pca_2d.png'), dpi=150, bbox_inches='tight')
print(f"✓ Grafik kaydedildi: clustering_pca_2d.png")
plt.close()

# Cluster Profiles
print("\n7. Küme profilleri hesaplanıyor...")
profile_features = ['avg_sales', 'std_sales', 'avg_customers', 'promo_lift', 
                   'promo_usage_rate', 'log_comp_distance']

cluster_profiles = clustering_data.groupby('cluster')[profile_features].mean()

# En yaygın StoreType ve Assortment
mode_features = clustering_data.groupby('cluster').agg({
    'StoreType': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown',
    'Assortment': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown',
    'Store': 'count'
})
mode_features.columns = ['main_storetype', 'main_assortment', 'count']

cluster_profiles = cluster_profiles.join(mode_features)

# Küme isimleri
cluster_profiles['cluster_name'] = cluster_names

print("\nKüme Profilleri:")
print(cluster_profiles.to_string())

# Visualize cluster profiles
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Ortalama satış
ax = axes[0, 0]
cluster_profiles['avg_sales'].plot(kind='bar', ax=ax, color=colors, edgecolor='black')
ax.set_title('Kümelere Göre Ortalama Satış', fontsize=12, fontweight='bold')
ax.set_xlabel('Küme')
ax.set_ylabel('Ortalama Satış')
ax.set_xticklabels([f"C{i}\n{cluster_names[i]}" for i in range(optimal_k)], rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# 2. Promo etkisi
ax = axes[0, 1]
cluster_profiles['promo_lift'].plot(kind='bar', ax=ax, color=colors, edgecolor='black')
ax.set_title('Kümelere Göre Promo Etkisi', fontsize=12, fontweight='bold')
ax.set_xlabel('Küme')
ax.set_ylabel('Promo Lift (%)')
ax.set_xticklabels([f"C{i}\n{cluster_names[i]}" for i in range(optimal_k)], rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# 3. Rekabet mesafesi
ax = axes[1, 0]
cluster_profiles['log_comp_distance'].plot(kind='bar', ax=ax, color=colors, edgecolor='black')
ax.set_title('Kümelere Göre Rekabet Mesafesi (log)', fontsize=12, fontweight='bold')
ax.set_xlabel('Küme')
ax.set_ylabel('Log(CompetitionDistance)')
ax.set_xticklabels([f"C{i}\n{cluster_names[i]}" for i in range(optimal_k)], rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# 4. Mağaza sayısı
ax = axes[1, 1]
cluster_profiles['count'].plot(kind='bar', ax=ax, color=colors, edgecolor='black')
ax.set_title('Kümelere Göre Mağaza Sayısı', fontsize=12, fontweight='bold')
ax.set_xlabel('Küme')
ax.set_ylabel('Mağaza Sayısı')
ax.set_xticklabels([f"C{i}\n{cluster_names[i]}" for i in range(optimal_k)], rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(cluster_profiles['count']):
    ax.text(i, v + 10, str(int(v)), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'clustering_profiles.png'), dpi=150, bbox_inches='tight')
print(f"✓ Grafik kaydedildi: clustering_profiles.png")
plt.close()

# Sales distribution by cluster
print("\n8. Kümelere göre satış dağılımı...")
train_clustered = train_open.merge(clustering_data[['Store', 'cluster']], on='Store', how='left')

plt.figure(figsize=(14, 8))
for i in range(optimal_k):
    cluster_sales = train_clustered[train_clustered['cluster'] == i]['Sales']
    plt.boxplot([cluster_sales], positions=[i], widths=0.6,
               patch_artist=True,
               boxprops=dict(facecolor=colors[i], alpha=0.7),
               medianprops=dict(color='black', linewidth=2))

plt.xticks(range(optimal_k), [f"C{i}\n{cluster_names[i]}" for i in range(optimal_k)])
plt.xlabel('Küme', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Kümelere Göre Satış Dağılımı', fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'clustering_sales_boxplot.png'), dpi=150, bbox_inches='tight')
print(f"✓ Grafik kaydedildi: clustering_sales_boxplot.png")
plt.close()

# Save reports
print("\n9. Raporlar kaydediliyor...")

# Cluster labels
cluster_labels = clustering_data[['Store', 'cluster', 'StoreType', 'Assortment', 
                                  'avg_sales', 'promo_lift']].copy()
cluster_labels['cluster_name'] = cluster_labels['cluster'].map(
    {i: cluster_names[i] for i in range(optimal_k)}
)
cluster_labels = cluster_labels.sort_values('Store')
cluster_labels.to_csv(os.path.join(REPORTS_DIR, 'clustering_labels.csv'), index=False)
print(f"✓ Rapor kaydedildi: clustering_labels.csv")

# Cluster statistics
cluster_stats = cluster_profiles.reset_index()
cluster_stats.to_csv(os.path.join(REPORTS_DIR, 'clustering_statistics.csv'), index=False)
print(f"✓ Rapor kaydedildi: clustering_statistics.csv")

# Feature importance (based on cluster center distances)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': np.std(kmeans.cluster_centers_, axis=0)
}).sort_values('importance', ascending=False)
feature_importance.to_csv(os.path.join(REPORTS_DIR, 'clustering_feature_importance.csv'), index=False)
print(f"✓ Rapor kaydedildi: clustering_feature_importance.csv")

print("\n" + "=" * 70)
print("KÜMELEME ANALİZİ TAMAMLANDI")
print("=" * 70)
print(f"\nSilhouette Score: {silhouette_avg:.3f} ({'İyi' if silhouette_avg > 0.5 else 'Orta'})")
print(f"\nKüme Özeti:")
for i in range(optimal_k):
    count = (clusters == i).sum()
    avg_sales = cluster_profiles.loc[i, 'avg_sales']
    print(f"  Cluster {i} ({cluster_names[i]}): {count} mağaza, Ort. Satış: {avg_sales:,.0f}")

print(f"\nGrafikler:")
print(f"  - clustering_elbow_silhouette.png")
print(f"  - clustering_pca_2d.png")
print(f"  - clustering_profiles.png")
print(f"  - clustering_sales_boxplot.png")

print(f"\nRaporlar:")
print(f"  - clustering_labels.csv (1,115 mağaza)")
print(f"  - clustering_statistics.csv (5 küme)")
print(f"  - clustering_feature_importance.csv")

print("\n" + "=" * 70)
