"""
Model performans değerlendirme ve görselleştirme scripti
Tez için kritik: Feature importance, SHAP, hata analizi
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import lightgbm as lgb
    ModelClass = lgb.LGBMRegressor
    model_name = "LightGBM"
except Exception:
    try:
        import xgboost as xgb
        ModelClass = xgb.XGBRegressor
        model_name = "XGBoost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
        ModelClass = HGBR
        model_name = "HistGradientBoosting"

from src.data import load_datasets, merge_store
from src.features import build_features, select_feature_columns
from src.metrics import rmspe
from src.validation import time_series_cv_indices

DATA_DIR = os.path.join(ROOT, "dataset")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def evaluate_cv_performance():
    """3-fold CV ile model performansını değerlendir"""
    print("=" * 70)
    print("MODEL PERFORMANS DEĞERLENDİRMESİ")
    print("=" * 70)
    print(f"Model: {model_name}\n")
    
    # Veri yükle
    train, test, store = load_datasets(DATA_DIR)
    train, test = merge_store(train, test, store)
    train_fe, test_fe = build_features(train, test)
    
    # Eğitim verisi hazırlama
    train_fit = train_fe.copy()
    if "Open" in train_fit.columns:
        train_fit = train_fit[train_fit["Open"] == 1]
    train_fit = train_fit.dropna(subset=["Sales"])
    
    y = np.log1p(train_fit["Sales"].values)
    feat_cols = select_feature_columns(train_fit)
    X = train_fit[feat_cols].copy()
    
    print(f"Eğitim seti boyutu: {len(train_fit):,} satır")
    print(f"Özellik sayısı: {len(feat_cols)}\n")
    
    # CV
    folds = time_series_cv_indices(train_fit, n_splits=3, val_weeks=6)
    
    fold_scores = []
    fold_predictions = []
    all_feature_importance = []
    
    for i, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"Fold {i} eğitiliyor...")
        X_tr, y_tr = X.loc[tr_idx], y[train_fit.index.get_indexer(tr_idx)]
        X_va, y_va = X.loc[va_idx], y[train_fit.index.get_indexer(va_idx)]
        
        # Model eğit
        if model_name == "LightGBM":
            model = ModelClass(n_estimators=500, learning_rate=0.05, subsample=0.8, 
                             colsample_bytree=0.8, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
                     callbacks=[lgb.early_stopping(100, verbose=False)])
            # Feature importance
            importance = model.feature_importances_
        elif model_name == "XGBoost":
            model = ModelClass(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
                     verbose=False, early_stopping_rounds=100)
            importance = model.feature_importances_
        else:
            model = ModelClass(learning_rate=0.05, random_state=42)
            model.fit(X_tr, y_tr)
            importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        
        # Tahmin
        pred = model.predict(X_va)
        
        # Geri dönüşüm
        y_va_orig = np.expm1(y_va)
        y_pred_orig = np.expm1(pred)
        
        # Metrikler
        rmspe_score = rmspe(y_va_orig, y_pred_orig)
        rmse_score = np.sqrt(np.mean((y_va_orig - y_pred_orig) ** 2))
        mae_score = np.mean(np.abs(y_va_orig - y_pred_orig))
        mape_score = np.mean(np.abs((y_va_orig - y_pred_orig) / (y_va_orig + 1))) * 100
        
        fold_scores.append({
            'fold': i,
            'rmspe': rmspe_score,
            'rmse': rmse_score,
            'mae': mae_score,
            'mape': mape_score
        })
        
        # Feature importance kaydet
        if importance is not None:
            all_feature_importance.append(importance)
        
        # Tahminleri kaydet
        fold_df = train_fit.loc[va_idx].copy()
        fold_df['actual'] = y_va_orig
        fold_df['predicted'] = y_pred_orig
        fold_df['error'] = y_pred_orig - y_va_orig
        fold_df['abs_error'] = np.abs(fold_df['error'])
        fold_df['pct_error'] = (fold_df['error'] / (fold_df['actual'] + 1)) * 100
        fold_df['fold'] = i
        fold_predictions.append(fold_df)
        
        print(f"  RMSPE: {rmspe_score:.4f} | RMSE: {rmse_score:.2f} | MAE: {mae_score:.2f}")
    
    # Birleştir
    scores_df = pd.DataFrame(fold_scores)
    all_predictions = pd.concat(fold_predictions, ignore_index=True)
    
    # Ortalama feature importance
    if all_feature_importance:
        avg_importance = np.mean(all_feature_importance, axis=0)
        importance_df = pd.DataFrame({
            'feature': feat_cols,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
    else:
        importance_df = None
    
    print("\n" + "=" * 70)
    print("ÖZET İSTATİSTİKLER")
    print("=" * 70)
    print(f"Ortalama RMSPE: {scores_df['rmspe'].mean():.4f} (±{scores_df['rmspe'].std():.4f})")
    print(f"Ortalama RMSE:  {scores_df['rmse'].mean():.2f} (±{scores_df['rmse'].std():.2f})")
    print(f"Ortalama MAE:   {scores_df['mae'].mean():.2f} (±{scores_df['mae'].std():.2f})")
    print(f"Ortalama MAPE:  {scores_df['mape'].mean():.2f}% (±{scores_df['mape'].std():.2f}%)")
    
    return scores_df, all_predictions, importance_df, feat_cols


def create_performance_plots(scores_df, predictions_df, importance_df):
    """Performans görselleştirmeleri oluştur"""
    print("\n" + "=" * 70)
    print("GRAFİKLER OLUŞTURULUYOR")
    print("=" * 70)
    
    # 1. Feature Importance (Top 20)
    if importance_df is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = importance_df.head(20)
        ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 20 Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, 'feature_importance.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✓ Feature importance: {path}")
        plt.close()
    
    # 2. CV Fold Performans Karşılaştırması
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cross-Validation Performans Analizi', fontsize=16, fontweight='bold')
    
    # RMSPE by fold
    ax = axes[0, 0]
    ax.bar(scores_df['fold'], scores_df['rmspe'], color='coral', edgecolor='black')
    ax.axhline(scores_df['rmspe'].mean(), color='red', linestyle='--', 
               label=f'Ort: {scores_df["rmspe"].mean():.4f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('RMSPE')
    ax.set_title('Fold Bazlı RMSPE Skorları')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # RMSE by fold
    ax = axes[0, 1]
    ax.bar(scores_df['fold'], scores_df['rmse'], color='steelblue', edgecolor='black')
    ax.axhline(scores_df['rmse'].mean(), color='red', linestyle='--',
               label=f'Ort: {scores_df["rmse"].mean():.2f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('RMSE')
    ax.set_title('Fold Bazlı RMSE Skorları')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # MAE by fold
    ax = axes[1, 0]
    ax.bar(scores_df['fold'], scores_df['mae'], color='lightgreen', edgecolor='black')
    ax.axhline(scores_df['mae'].mean(), color='red', linestyle='--',
               label=f'Ort: {scores_df["mae"].mean():.2f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('MAE')
    ax.set_title('Fold Bazlı MAE Skorları')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Metrics summary
    ax = axes[1, 1]
    metrics = ['RMSPE', 'RMSE', 'MAE']
    means = [scores_df['rmspe'].mean(), scores_df['rmse'].mean()/100, scores_df['mae'].mean()/100]
    stds = [scores_df['rmspe'].std(), scores_df['rmse'].std()/100, scores_df['mae'].std()/100]
    x_pos = np.arange(len(metrics))
    ax.bar(x_pos, means, yerr=stds, capsize=5, color=['coral', 'steelblue', 'lightgreen'],
           edgecolor='black', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Normalized Score')
    ax.set_title('Metrics Summary (with Std Dev)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'cv_performance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"✓ CV performans: {path}")
    plt.close()
    
    # 3. Gerçek vs Tahmin + Hata Dağılımı
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Tahmin Kalitesi Analizi', fontsize=16, fontweight='bold')
    
    # Gerçek vs Tahmin scatter
    ax = axes[0]
    sample = predictions_df.sample(min(5000, len(predictions_df)), random_state=42)
    ax.scatter(sample['actual'], sample['predicted'], alpha=0.3, s=10, c=sample['fold'], cmap='viridis')
    max_val = max(sample['actual'].max(), sample['predicted'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Mükemmel Tahmin')
    ax.set_xlabel('Gerçek Satış')
    ax.set_ylabel('Tahmin Edilen Satış')
    ax.set_title('Gerçek vs Tahmin (5000 örnek)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Hata dağılımı
    ax = axes[1]
    errors = predictions_df['error'].clip(-1000, 1000)
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Sıfır Hata')
    ax.set_xlabel('Tahmin Hatası (Predicted - Actual)')
    ax.set_ylabel('Frekans')
    ax.set_title('Hata Dağılımı')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'prediction_quality.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"✓ Tahmin kalitesi: {path}")
    plt.close()
    
    # 4. Hata Analizi (DOW, Promo)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Hata Analizi (Segmentlere Göre)', fontsize=16, fontweight='bold')
    
    # Haftanın gününe göre
    ax = axes[0]
    dow_stats = predictions_df.groupby('DayOfWeek')['abs_error'].mean()
    dow_names = {1:'Pzt', 2:'Sal', 3:'Çar', 4:'Per', 5:'Cum', 6:'Cmt', 7:'Paz'}
    dow_labels = [dow_names.get(d, str(d)) for d in dow_stats.index]
    ax.bar(dow_labels, dow_stats.values, color='coral', edgecolor='black')
    ax.set_xlabel('Haftanın Günü')
    ax.set_ylabel('Ortalama Mutlak Hata')
    ax.set_title('Haftanın Gününe Göre Ortalama Hata')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Promo etkisi
    ax = axes[1]
    if 'Promo' in predictions_df.columns:
        promo_stats = predictions_df.groupby('Promo')['abs_error'].mean()
        promo_labels = ['Promo Yok', 'Promo Var']
        ax.bar(promo_labels, promo_stats.values, color='lightgreen', edgecolor='black')
        ax.set_ylabel('Ortalama Mutlak Hata')
        ax.set_title('Promo Durumuna Göre Ortalama Hata')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'error_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"✓ Hata analizi: {path}")
    plt.close()


def save_reports(scores_df, predictions_df, importance_df):
    """Analiz raporlarını CSV olarak kaydet"""
    print("\n" + "=" * 70)
    print("RAPORLAR KAYDEDILIYOR")
    print("=" * 70)
    
    # CV skorları
    path = os.path.join(REPORTS_DIR, 'cv_scores.csv')
    scores_df.to_csv(path, index=False)
    print(f"✓ CV skorları: {path}")
    
    # Feature importance
    if importance_df is not None:
        path = os.path.join(REPORTS_DIR, 'feature_importance.csv')
        importance_df.to_csv(path, index=False)
        print(f"✓ Feature importance: {path}")
    
    # Tahmin örnekleri (ilk 10000)
    path = os.path.join(REPORTS_DIR, 'validation_predictions.csv')
    predictions_df.head(10000)[['Store', 'Date', 'actual', 'predicted', 'error', 'pct_error', 'fold']].to_csv(
        path, index=False
    )
    print(f"✓ Validasyon tahminleri (örnek): {path}")
    
    # Hata analizi özeti
    error_summary = {
        'metric': ['Mean Error', 'Std Error', 'Mean Abs Error', 'Median Abs Error', 'Max Abs Error'],
        'value': [
            predictions_df['error'].mean(),
            predictions_df['error'].std(),
            predictions_df['abs_error'].mean(),
            predictions_df['abs_error'].median(),
            predictions_df['abs_error'].max()
        ]
    }
    path = os.path.join(REPORTS_DIR, 'error_summary.csv')
    pd.DataFrame(error_summary).to_csv(path, index=False)
    print(f"✓ Hata özeti: {path}")


def main():
    print("\n🚀 Model Değerlendirme Başlatılıyor...\n")
    
    scores_df, predictions_df, importance_df, feat_cols = evaluate_cv_performance()
    create_performance_plots(scores_df, predictions_df, importance_df)
    save_reports(scores_df, predictions_df, importance_df)
    
    print("\n" + "=" * 70)
    print("✅ DEĞERLENDİRME TAMAMLANDI!")
    print("=" * 70)
    print(f"\nGrafikler: {FIGURES_DIR}/")
    print(f"Raporlar: {REPORTS_DIR}/")
    print("\nTez için kullanılabilecek dosyalar:")
    print("  - feature_importance.png")
    print("  - cv_performance.png")
    print("  - prediction_quality.png")
    print("  - error_analysis.png")
    print("  - feature_importance.csv")
    print("  - cv_scores.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
