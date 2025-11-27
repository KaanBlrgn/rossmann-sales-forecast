# Rossmann Store Sales Forecasting

Bitirme tezi projesi - Rossmann mağaza satış tahmini

## 📁 Proje Yapısı

```
sales_forecast/
├── .gitignore            # Git ignore kuralları
├── requirements.txt      # Python bağımlılıkları
├── README.md            # Bu dosya
├── PROJECT_STRUCTURE.md # Detaylı yapı dokümantasyonu
│
├── dataset/             # Veri dosyaları
│   ├── train.csv       # Eğitim verisi (2013-2015)
│   ├── test.csv        # Test verisi (tahmin edilecek)
│   ├── store.csv       # Mağaza meta verisi
│   └── sample_submission.csv
│
├── src/                 # Kaynak modülleri
│   ├── __init__.py
│   ├── data.py         # Veri yükleme ve birleştirme
│   ├── features.py     # Feature engineering
│   ├── metrics.py      # Değerlendirme metrikleri (RMSPE)
│   └── validation.py   # Cross-validation
│
├── scripts/             # Çalıştırılabilir scriptler
│   ├── ensemble_train.py    # Ensemble model eğitimi (AKTİF) ⭐
│   ├── ensemble_predict.py  # Ensemble tahmin (AKTİF) ⭐
│   ├── evaluate.py     # Performans analizi ve grafikler
│   ├── clustering_analysis.py  # Mağaza kümeleme analizi
│   ├── eda_analysis.py # Keşifsel veri analizi
│   ├── shap_analysis.py     # Model yorumlama (SHAP)
│   ├── optimize_ensemble_weights.py  # Ensemble ağırlık optimizasyonu
│   ├── optuna_tuning.py     # Hyperparameter tuning (Optuna)
│   ├── analyze_day_feature.py    # "day" feature analizi
│   └── analyze_fold_variance.py  # Fold tutarlılığı analizi
│
├── models/              # Eğitilmiş model dosyaları
│   ├── lgb_model.pkl   # Ensemble LightGBM (6 MB)
│   ├── xgb_model.pkl   # Ensemble XGBoost (15.8 MB)
│   ├── ensemble_config.json  # Ensemble ağırlıkları
│   └── features.json   # Özellik listesi (45 - cluster + holiday features)
│
├── outputs/             # Analiz çıktıları
│   ├── figures/        # Grafikler (evaluate.py çıktısı)
│   └── reports/        # Raporlar (CSV formatında)
│
├── config.py            # Konfigürasyon ve hiperparametreler
├── submission.csv       # Final tahmin (Ensemble model)
└── archive/             # Eski dosyalar (referans)
```


## 🚀 Kullanım

### 1. Model Eğitimi

#### Ensemble Model (ÖNERİLEN) ⭐

```bash
python scripts/ensemble_train.py
```

**Çıktılar:**
- `models/lgb_model.pkl` - LightGBM model (6 MB)
- `models/xgb_model.pkl` - XGBoost model (15.8 MB)
- `models/ensemble_config.json` - Ensemble konfigürasyonu (optimal ağırlıklar)
- `models/features.json` - Özellik listesi (45 features)
- Ekran: LightGBM, XGBoost ve Ensemble CV skorları

### 2. Keşifsel Veri Analizi (EDA)

```bash
python scripts/eda_analysis.py
```

**Çıktılar:**
- `outputs/figures/eda_sales_analysis.png` - Satış dağılımı
- `outputs/figures/eda_promo_storetype.png` - Promo ve mağaza tipi analizi
- `outputs/figures/eda_time_series.png` - Zaman serisi
- `outputs/figures/eda_correlation.png` - Korelasyon matrisi

### 3. Model Değerlendirme (Tez için önemli!)

```bash
python scripts/evaluate.py
```

**Çıktılar:**
- `outputs/figures/feature_importance.png` - Top 20 özellik önemi
- `outputs/figures/cv_performance.png` - CV fold karşılaştırması
- `outputs/figures/prediction_quality.png` - Gerçek vs Tahmin grafiği
- `outputs/figures/error_analysis.png` - Hata analizi (DOW, Promo)
- `outputs/reports/cv_scores.csv` - Detaylı CV skorları
- `outputs/reports/feature_importance.csv` - Tüm özellikler ve skorları
- `outputs/reports/error_summary.csv` - Hata istatistikleri

### 4. SHAP Analizi (Model Yorumlama)

```bash
python scripts/shap_analysis.py
```

**Çıktılar:**
- `outputs/figures/shap_summary.png` - Özellik etkileri
- `outputs/figures/shap_importance.png` - Özellik önemi
- `outputs/figures/shap_dependence.png` - Top 4 özellik ilişkisi
- `outputs/figures/shap_waterfall.png` - Tek tahmin açıklaması
- `outputs/reports/shap_values.csv` - SHAP değerleri

### 5. Mağaza Kümeleme Analizi

```bash
python scripts/clustering_analysis.py
```

**Çıktılar:**
- `outputs/figures/clustering_elbow_silhouette.png` - Optimal küme sayısı analizi
- `outputs/figures/clustering_pca_2d.png` - PCA 2D küme görselleştirme
- `outputs/figures/clustering_profiles.png` - Küme profilleri
- `outputs/figures/clustering_sales_boxplot.png` - Kümelere göre satış dağılımı
- `outputs/reports/clustering_labels.csv` - Her mağazanın küme etiketi
- `outputs/reports/clustering_statistics.csv` - Küme istatistikleri
- `outputs/reports/clustering_feature_importance.csv` - Kümeleme özellik önemi

### 6. Tahmin Üretimi

#### Ensemble Tahmin (ÖNERİLEN) ⭐

```bash
python scripts/ensemble_predict.py
```

**Çıktı:**
- `submission.csv` - En iyi ensemble model ile tahminler (CV RMSPE: 0.1212)

### 7. İleri Seviye Analizler (YENİ!) 🆕

#### Ensemble Ağırlık Optimizasyonu

```bash
python scripts/optimize_ensemble_weights.py
```

**Çıktı:** `models/optimal_weights.json` - Optimal LightGBM/XGBoost ağırlıkları

#### Hyperparameter Tuning (Optuna)

```bash
python scripts/optuna_tuning.py
```

**Çıktı:** `models/tuning_results.json` - En iyi hiperparametreler (50 trial)

#### "day" Feature Analizi

```bash
python scripts/analyze_day_feature.py
```

**Çıktı:** `outputs/figures/day_feature_analysis.png` - Overfitting kontrolü

#### Fold Tutarsızlığı Analizi

```bash
python scripts/analyze_fold_variance.py
```

**Çıktı:** `outputs/figures/fold_variance_analysis.png` - Fold karakteristikleri

## 📊 Özellik Mühendisliği

### Takvim Özellikleri
- year, month, day, weekofyear, quarter
- is_month_start, is_month_end, is_weekend

### Promo Özellikleri
- Promo2 aktiflik bayrağı
- PromoInterval aktiflik kontrolü

### Rekabet Özellikleri
- competition_open_months
- log_competition_distance

### Kategorik Encoding
- StateHoliday (one-hot)
- StoreType (one-hot)
- Assortment (one-hot)

### Zaman Serisi Özellikleri
- Lag: 7, 14, 28 gün
- Rolling mean: 7, 14, 28 pencere
- Rolling std: 7, 14, 28 pencere
- Sales momentum (trend göstergesi)

### Kümeleme Özellikleri
- cluster_0 to cluster_4 (one-hot encoded)
- 5 store segment: Premium City, Suburban Std, Small Town, Flagship, Rural Low
- K-Means clustering sonuçları model feature'ı olarak kullanıldı

### Tatil Özellikleri (YENİ!)
- is_christmas_week, is_newyear_week, is_easter_week
- days_to_easter (Paskalya yakınlığı)
- school_holiday_tomorrow, school_holiday_yesterday
- Tatil spike'larını yakalamak için eklendi

## 🎯 Modeller

### Ensemble Model (Final - Optimized) ⭐⭐⭐
- **LightGBM + XGBoost** (63.6/36.4 optimal ağırlık)
- **CV RMSPE:** 0.1212 (ortalama %12.12 hata)
- **Fold skorları:** 0.1224, 0.1281, 0.1131
- **Features:** 45 (cluster + holiday features)
- **İyileşme:** +9.9% (baseline'dan), +11.0% (eski ensemble'dan)
- **Optimizasyon:** Ensemble ağırlıkları scipy ile optimize edildi
- **Kullanılan:** `ensemble_predict.py`

### Baseline Model
- **LightGBM** (tek model)
- **CV RMSPE:** 0.1393 (ortalama %13.93 hata)
- **Fold skorları:** 0.1539, 0.1424, 0.1217

**Ortak:**
- **Hedef Dönüşümü:** log1p(Sales)
- **Validasyon:** 3-fold TimeSeriesSplit
- **Metrik:** RMSPE (Root Mean Square Percentage Error)

## 📈 Performans Özeti

| Model | CV RMSPE | Fold 1 | Fold 2 | Fold 3 |
|-------|----------|--------|--------|--------|
| **Ensemble (Optimized)** | **0.1212** | 0.1224 | 0.1281 | **0.1131** |
| LightGBM (Holiday+Cluster) | 0.1218 | 0.1234 | 0.1274 | 0.1146 |
| XGBoost (Holiday+Cluster) | 0.1230 | 0.1241 | 0.1310 | 0.1139 |
| Ensemble (Cluster) | 0.1345 | 0.1469 | 0.1410 | 0.1156 |
| Baseline | 0.1393 | 0.1539 | 0.1424 | 0.1217 |

**En iyi skor:** Ensemble (Optimized) 0.1212 (Top %14-17 Kaggle Rossmann) 🏆  
**İyileşme:** Holiday features + optimal ağırlıklar ile %11.0 daha iyi

## ⚙️ Gereksinimler

```bash
pip install -r requirements.txt
```

**Temel:**
- pandas, numpy, scikit-learn
- lightgbm, xgboost
- joblib

**Analiz & Görselleştirme:**
- matplotlib, seaborn
- shap (model yorumlama)

## 📝 Notlar

- Eğitimde `Open==1` günleri kullanılır
- Testte `Open==0` günler için `Sales=0` yazılır
- Tüm lag/rolling özellikler leakage-safe (sadece geçmiş veri)
- Model artifacts `models/` klasörüne kaydedilir
