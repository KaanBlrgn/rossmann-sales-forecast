# 🚀 MODEL İYİLEŞTİRME PLANI

**Tarih:** 27 Kasım 2025  
**Mevcut Skor:** 0.1345 RMSPE  
**Hedef:** 0.125-0.130 RMSPE (Top %15)

---

## 📊 TESPİT EDİLEN SORUNLAR

### **🔴 Kritik Sorunlar**
1. **Hyperparameter tuning yok** → %2-3 iyileşme potansiyeli
2. **Ensemble ağırlıkları optimize değil** → %0.5 iyileşme
3. **Fold tutarsızlığı** (%26 fark) → Gerçek performans belirsiz

### **🟡 Önemli Sorunlar**
4. **"day" feature aşırı önemli** → Overfitting riski
5. **Tatil features eksik** → Spike'ları yakalayamıyor
6. **Stacking/blending yok** → Basit ensemble

---

## ✅ OLUŞTURULAN ÇÖZÜMLER

### **1. Hyperparameter Tuning (Optuna)** 🔴

**Script:** `scripts/optuna_tuning.py`

**Ne Yapıyor:**
- LightGBM için 50 trial
- XGBoost için 50 trial
- Bayesian optimization (TPE sampler)
- 3-fold CV ile değerlendirme

**Optimize Edilen Parametreler:**
```python
LightGBM:
- learning_rate (0.01-0.1)
- num_leaves (31-127)
- max_depth (5-15)
- min_child_samples (5-100)
- subsample (0.6-0.95)
- colsample_bytree (0.6-0.95)
- reg_alpha, reg_lambda

XGBoost:
- learning_rate (0.01-0.1)
- max_depth (5-15)
- min_child_weight (1-10)
- subsample (0.6-0.95)
- colsample_bytree (0.6-0.95)
- gamma, reg_alpha, reg_lambda
```

**Kullanım:**
```bash
# 1. Tuning yap (1-2 saat)
python scripts/optuna_tuning.py

# 2. Sonuçları kontrol et
cat models/tuning_results.json

# 3. En iyi parametreleri ensemble_train.py'ye kopyala
```

**Beklenen İyileşme:** 0.1345 → 0.130-0.132 (%2-3)

---

### **2. Ensemble Ağırlık Optimizasyonu** 🔴

**Script:** `scripts/optimize_ensemble_weights.py`

**Ne Yapıyor:**
- Her fold için LightGBM ve XGBoost tahminlerini topluyor
- Scipy.optimize ile optimal ağırlıkları buluyor
- 0.0-1.0 arası 11 farklı ağırlık test ediyor

**Kullanım:**
```bash
# 1. Optimal ağırlıkları bul (15-20 dakika)
python scripts/optimize_ensemble_weights.py

# 2. Sonuçları kontrol et
cat models/optimal_weights.json

# 3. ensemble_config.json'u güncelle
```

**Mevcut:**
```json
{
  "lgb_weight": 0.5,
  "xgb_weight": 0.5
}
```

**Muhtemelen Optimal:**
```json
{
  "lgb_weight": 0.3-0.4,
  "xgb_weight": 0.6-0.7
}
```

**Beklenen İyileşme:** 0.1345 → 0.1340-0.1342 (%0.3-0.5)

---

### **3. Tatil Features** 🟡

**Dosya:** `src/features.py` (GÜNCELLENDİ ✅)

**Eklenen Features:**
```python
✅ is_christmas_week (Noel haftası)
✅ is_newyear_week (Yılbaşı haftası)
✅ is_easter_week (Paskalya haftası ±7 gün)
✅ days_to_easter (Paskalya'ya uzaklık)
✅ school_holiday_tomorrow (yarın okul tatili)
✅ school_holiday_yesterday (dün okul tatili)
```

**Kullanım:**
```bash
# Otomatik eklendi, sadece yeniden eğit
python scripts/ensemble_train.py
```

**Beklenen İyileşme:** 0.1345 → 0.1335-0.1340 (%0.5-1)

---

### **4. "day" Feature Analizi** 🟡

**Script:** `scripts/analyze_day_feature.py`

**Ne Yapıyor:**
- Ayın gününe göre satış patternlerini analiz ediyor
- "day" feature ile/siz model karşılaştırması
- Overfitting var mı kontrol ediyor
- Ay sonu/başı etkisini ölçüyor

**Kullanım:**
```bash
python scripts/analyze_day_feature.py
```

**Çıktılar:**
- `outputs/figures/day_feature_analysis.png`
- Konsol: "day" feature gerçekten işe yarıyor mu?

**Karar:**
- Eğer iyileşme > %1 → Tut
- Eğer iyileşme < %0.5 → Çıkar (overfitting riski)

---

### **5. Fold Tutarsızlığı Analizi** 🟡

**Script:** `scripts/analyze_fold_variance.py`

**Ne Yapıyor:**
- Her fold'un karakteristiklerini analiz ediyor
- Neden Fold 3 kolay, Fold 1 zor?
- Mevsimsel etkiler, tatiller, promo oranları
- İstatistiksel testler (t-test)

**Kullanım:**
```bash
python scripts/analyze_fold_variance.py
```

**Çıktılar:**
- `outputs/figures/fold_variance_analysis.png`
- Fold karakteristikleri tablosu
- Öneriler

**Önerilen Çözümler:**
1. Stratified CV (mevsim bazlı)
2. Fold ağırlıklandırması
3. Test setinin hangi fold'a benzediğini analiz et

---

## 🎯 UYGULAMA SIRASI

### **Hızlı İyileşme (2-3 saat)** ⚡

```bash
# 1. Tatil features zaten eklendi ✅
# 2. Ensemble ağırlıklarını optimize et
python scripts/optimize_ensemble_weights.py

# 3. Optimal ağırlıkları uygula
# models/optimal_weights.json'dan kopyala
# ensemble_config.json'u güncelle

# 4. Yeniden eğit
python scripts/ensemble_train.py

# 5. Yeni submission oluştur
python scripts/ensemble_predict.py
```

**Beklenen Sonuç:** 0.1345 → 0.1330-0.1335 (%1-1.5 iyileşme)

---

### **Tam İyileşme (1-2 gün)** 🚀

```bash
# 1. Hyperparameter tuning (1-2 saat)
python scripts/optuna_tuning.py

# 2. En iyi parametreleri ensemble_train.py'ye uygula
# models/tuning_results.json'dan kopyala

# 3. "day" feature analizi (10 dakika)
python scripts/analyze_day_feature.py
# Eğer overfitting varsa çıkar

# 4. Fold variance analizi (10 dakika)
python scripts/analyze_fold_variance.py
# Stratified CV düşün

# 5. Ensemble ağırlık optimizasyonu (20 dakika)
python scripts/optimize_ensemble_weights.py

# 6. Tüm iyileştirmelerle yeniden eğit
python scripts/ensemble_train.py

# 7. Final submission
python scripts/ensemble_predict.py
```

**Beklenen Sonuç:** 0.1345 → 0.125-0.130 (%3-5 iyileşme)

---

## 📊 BEKLENEN SONUÇLAR

### **Mevcut Durum:**
```
RMSPE: 0.1345
Kaggle: ~700-900 / 3,303 (Top %21-27)
```

### **Hızlı İyileşme Sonrası:**
```
RMSPE: 0.1330-0.1335
Kaggle: ~600-700 / 3,303 (Top %18-21)
İyileşme: %1-1.5
```

### **Tam İyileşme Sonrası:**
```
RMSPE: 0.125-0.130
Kaggle: ~450-550 / 3,303 (Top %14-17) ⭐
İyileşme: %3-5
```

---

## 🎓 TEZ İÇİN

### **Mevcut Hali:**
- ✅ Lisans tezi için mükemmel (9/10)
- ✅ Metodoloji sağlam
- ✅ Sonuçlar iyi

### **İyileştirmeler Sonrası:**
- ⭐ Yüksek lisans seviyesi (9.5/10)
- ⭐ Hyperparameter tuning → Bilimsel yaklaşım
- ⭐ Ensemble optimization → İleri seviye
- ⭐ Feature analysis → Derinlemesine analiz

---

## ✅ ÖZET

### **Oluşturulan Scriptler:**
1. ✅ `optuna_tuning.py` - Hyperparameter tuning
2. ✅ `optimize_ensemble_weights.py` - Ağırlık optimizasyonu
3. ✅ `analyze_day_feature.py` - "day" feature analizi
4. ✅ `analyze_fold_variance.py` - Fold tutarsızlığı

### **Güncellenen Dosyalar:**
1. ✅ `src/features.py` - Tatil features eklendi

### **Beklenen İyileşme:**
- **Hızlı:** %1-1.5 (2-3 saat)
- **Tam:** %3-5 (1-2 gün)

### **Kaggle Tahmini:**
- **Mevcut:** Top %21-27
- **Sonrası:** Top %14-17 ⭐

---

**🚀 HAZIR! İyileştirmeleri uygulamaya başlayabilirsin!**
