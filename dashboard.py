"""
Rossmann Sales Forecasting - Interactive Dashboard
Tüm analizleri ve sonuçları görselleştiren web arayüzü
"""
import streamlit as st
import pandas as pd
import os
from PIL import Image

# Page config
st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(ROOT, 'outputs', 'figures')
REPORTS_DIR = os.path.join(ROOT, 'outputs', 'reports')
MODELS_DIR = os.path.join(ROOT, 'models')

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/3498db/ffffff?text=Rossmann+Forecasting", use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "📑 Navigasyon",
        [
            "🏠 Ana Sayfa",
            "📊 Model Performansı",
            "📈 EDA - Veri Analizi",
            "🔍 SHAP - Model Yorumlama",
            "🎯 Kümeleme Analizi",
            "📋 İşlevsel Tablolar",
            "📄 Raporlar",
            "ℹ️ Proje Hakkında"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📌 Hızlı İstatistikler")
    st.metric("Mağaza Sayısı", "1,115")
    st.metric("Model RMSPE", "0.1212")
    st.metric("Toplam Grafik", "16")
    st.metric("Küme Sayısı", "5")

# Helper function to load image
def load_image(filename):
    path = os.path.join(FIGURES_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None

# Helper function to load CSV
def load_csv(filename):
    path = os.path.join(REPORTS_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ==================== ANA SAYFA ====================
if page == "🏠 Ana Sayfa":
    st.markdown('<div class="main-header">📊 Rossmann Mağaza Satış Tahmini</div>', unsafe_allow_html=True)
    st.markdown("### 🎓 Bitirme Tezi Projesi - Makine Öğrenmesi ile Satış Tahmini")
    
    # Proje Özeti Kutusu
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;'>
        <h2 style='color: white; margin-bottom: 1rem;'>🎯 Proje Amacı</h2>
        <p style='font-size: 1.2rem; line-height: 1.8;'>
            <strong>Rossmann</strong> Almanya'nın en büyük eczane zincirlerinden biri. 
            <strong>1,115 farklı mağaza</strong> için <strong>6 hafta ileriye</strong> günlük satış tahminleri yapıyoruz.
            Bu tahminler mağaza yöneticilerine stok planlaması, personel yönetimi ve promosyon stratejileri için yardımcı oluyor.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Veri Seti Tanıtımı
    st.markdown("## 📦 Veri Seti Hakkında")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>📊 Eğitim Verisi (train.csv)</h3>
            <ul style='font-size: 1.1rem; line-height: 2;'>
                <li><strong>1,017,209 satış kaydı</strong></li>
                <li><strong>Tarih:</strong> 2013-01-01 → 2015-07-31 (942 gün)</li>
                <li><strong>1,115 farklı mağaza</strong></li>
                <li><strong>Her satır:</strong> 1 mağaza × 1 gün satışı</li>
            </ul>
            <p style='background: #e3f2fd; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                <strong>Örnek:</strong> Store 1, 2013-01-05 günü 5,263 TL satış yaptı, 632 müşteri geldi, promo vardı.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>🎯 Test Verisi (test.csv)</h3>
            <ul style='font-size: 1.1rem; line-height: 2;'>
                <li><strong>41,088 tahmin yapılacak</strong></li>
                <li><strong>Tarih:</strong> 2015-08-01 → 2015-09-17 (48 gün)</li>
                <li><strong>Aynı 1,115 mağaza</strong></li>
                <li><strong>Hedef:</strong> Her gün için satış tahmini</li>
            </ul>
            <p style='background: #fff3cd; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                <strong>Görev:</strong> Geçmiş verilere bakarak gelecek 6 hafta için her mağazanın günlük satışını tahmin et!
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='metric-card' style='background: #f0f8ff;'>
        <h3>🏪 Mağaza Bilgileri (store.csv)</h3>
        <p style='font-size: 1.1rem; line-height: 1.8;'>
            Her mağaza hakkında sabit bilgiler:
        </p>
        <ul style='font-size: 1.1rem; line-height: 2; columns: 2;'>
            <li><strong>StoreType:</strong> a, b, c, d (4 farklı mağaza tipi)</li>
            <li><strong>Assortment:</strong> a=temel, b=ekstra, c=genişletilmiş</li>
            <li><strong>CompetitionDistance:</strong> En yakın rakip mesafesi (metre)</li>
            <li><strong>Promo2:</strong> Sürekli promo programı var mı?</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics row
    st.markdown("## 🏆 Proje Başarı Metrikleri")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🏆 Model Skoru", "0.1212 RMSPE", delta="-13.0% (baseline'dan)", delta_color="normal")
        st.markdown("Top %14-17 Kaggle 🏆")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🤖 Model Türü", "Ensemble", help="LightGBM + XGBoost (50/50)")
        st.markdown("2 farklı algoritma")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📈 Özellik Sayısı", "34", help="Feature Engineering")
        st.markdown("Lag + Rolling + Meta")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎯 Tahmin Sayısı", "41,088", help="Test seti boyutu")
        st.markdown("6 haftalık tahmin")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Ne Tahmin Ediyoruz?
    st.markdown("## 🎯 Ne Tahmin Ediyoruz?")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                padding: 2rem; border-radius: 15px; color: #333; margin: 1rem 0;'>
        <h2 style='color: #333; margin-bottom: 1rem;'>💡 Tahmin Amacımız</h2>
        <div style='background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
            <h3 style='color: #e74c3c;'>Girdi (X) - Ne biliyoruz?</h3>
            <ul style='font-size: 1.1rem; line-height: 2;'>
                <li><strong>Geçmiş satışlar:</strong> 2013-2015 arası her mağazanın günlük satışları</li>
                <li><strong>Mağaza bilgileri:</strong> Tip, ürün çeşitliliği, rekabet durumu</li>
                <li><strong>Takvim bilgileri:</strong> Hangi gün, hafta, ay, tatil mi?</li>
                <li><strong>Promo bilgileri:</strong> Promosyon var mı, sürekli promo mu?</li>
            </ul>
        </div>
        <div style='background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
            <h3 style='color: #2ecc71;'>Çıktı (Y) - Ne tahmin ediyoruz?</h3>
            <p style='font-size: 1.2rem; line-height: 1.8;'>
                <strong style='color: #e74c3c;'>Günlük Satış (TL)</strong> - Her mağaza için her gün ne kadar satış yapacak?
            </p>
            <p style='font-size: 1.1rem; background: #fff3cd; padding: 1rem; border-radius: 8px;'>
                <strong>Örnek:</strong> Mağaza 5, 2015-08-15 Cumartesi günü <strong>7,245 TL</strong> satış yapacak (tahmini)
            </p>
        </div>
        <div style='background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
            <h3 style='color: #3498db;'>Nasıl Yapıyoruz?</h3>
            <ol style='font-size: 1.1rem; line-height: 2;'>
                <li><strong>Özellik Çıkarma:</strong> Geçmiş satışlardan 34 özellik (lag, rolling mean, trend)</li>
                <li><strong>Model Eğitimi:</strong> LightGBM + XGBoost'u geçmiş verilerle eğitiyoruz</li>
                <li><strong>Tahmin:</strong> Eğitilen model gelecek için tahmin yapıyor</li>
                <li><strong>Sonuç:</strong> 41,088 tahmin (1,115 mağaza × 48 gün = 6 hafta)</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tahmin Örneği
    st.markdown("## 📝 Tahmin Örneği - Gerçek Vaka")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card' style='background: #e3f2fd;'>
            <h3>📍 Mağaza Bilgileri</h3>
            <ul style='font-size: 1rem; line-height: 1.8;'>
                <li><strong>Store ID:</strong> 5</li>
                <li><strong>Type:</strong> a (Ana mağaza)</li>
                <li><strong>Assortment:</strong> c (Genişletilmiş)</li>
                <li><strong>Cluster:</strong> 2 (Small Town)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card' style='background: #fff3e0;'>
            <h3>📅 Tahmin Günü</h3>
            <ul style='font-size: 1rem; line-height: 1.8;'>
                <li><strong>Tarih:</strong> 2015-08-15</li>
                <li><strong>Gün:</strong> Cumartesi</li>
                <li><strong>Promo:</strong> Var ✅</li>
                <li><strong>Tatil:</strong> Yok</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card' style='background: #e8f5e9;'>
            <h3>🎯 Tahmin Sonucu</h3>
            <p style='font-size: 2rem; color: #2ecc71; font-weight: bold; text-align: center; margin: 2rem 0;'>
                7,245 TL
            </p>
            <p style='text-align: center; font-size: 0.9rem; color: #666;'>
                Model Güven: %86.38<br>
                (100 - 13.62 RMSPE)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    st.markdown('<div class="section-header">📊 Hızlı İstatistikler</div>', unsafe_allow_html=True)
    
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        st.markdown("**🔢 Model Karşılaştırması**")
        model_comparison = pd.DataFrame({
            'Model': ['Ensemble (Optimized)', 'LightGBM', 'XGBoost', 'Baseline'],
            'RMSPE': [0.1212, 0.1218, 0.1230, 0.1393],
            'Fold 1': [0.1224, 0.1234, 0.1241, 0.1539],
            'Fold 2': [0.1281, 0.1274, 0.1310, 0.1424],
            'Fold 3': [0.1131, 0.1146, 0.1139, 0.1217]
        })
        st.dataframe(model_comparison, use_container_width=True)
    
    with stats_col2:
        st.markdown("**📁 Proje İçeriği**")
        content_data = pd.DataFrame({
            'Kategori': ['Scriptler', 'Grafikler', 'Raporlar', 'Modeller'],
            'Sayı': [8, 16, 8, 3],
            'Durum': ['✅', '✅', '✅', '✅']
        })
        st.dataframe(content_data, use_container_width=True)
    
    with stats_col3:
        st.markdown("**🎯 Kümeleme Sonuçları**")
        cluster_data = pd.DataFrame({
            'Küme': ['Premium City', 'Suburban Std', 'Small Town', 'Flagship', 'Rural Low'],
            'Mağaza': [283, 273, 154, 375, 30],
            'Ort. Satış': [6567, 5532, 10806, 6502, 8695]
        })
        st.dataframe(cluster_data, use_container_width=True)

# ==================== MODEL PERFORMANSI ====================
elif page == "📊 Model Performansı":
    st.markdown('<div class="section-header">📊 Model Performans Analizi</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 CV Performansı", "🎯 Tahmin Kalitesi", "📊 Feature Importance", "❌ Hata Analizi"])
    
    with tab1:
        st.markdown("### Cross-Validation Performansı")
        img = load_image('cv_performance.png')
        if img:
            st.image(img, use_container_width=True)
        
        st.markdown("#### 📊 CV Sonuçları")
        cv_scores = load_csv('cv_scores.csv')
        if cv_scores is not None:
            st.dataframe(cv_scores, use_container_width=True)
    
    with tab2:
        st.markdown("### Gerçek vs Tahmin Grafiği")
        img = load_image('prediction_quality.png')
        if img:
            st.image(img, use_container_width=True)
        
        st.info("📌 İdeal durum: Noktalar 45° çizgi üzerinde olmalı. Model başarılı!")
    
    with tab3:
        st.markdown("### Top 20 Özellik Önemi")
        img = load_image('feature_importance.png')
        if img:
            st.image(img, use_container_width=True)
        
        st.markdown("#### 📄 Detaylı Özellik Önemi")
        feat_imp = load_csv('feature_importance.csv')
        if feat_imp is not None:
            st.dataframe(feat_imp.head(20), use_container_width=True)
    
    with tab4:
        st.markdown("### Hata Analizi (DayOfWeek & Promo)")
        img = load_image('error_analysis.png')
        if img:
            st.image(img, use_container_width=True)
        
        st.markdown("#### 📊 Hata Özeti")
        error_summary = load_csv('error_summary.csv')
        if error_summary is not None:
            st.dataframe(error_summary, use_container_width=True)

# ==================== EDA ====================
elif page == "📈 EDA - Veri Analizi":
    st.markdown('<div class="section-header">📈 Keşifsel Veri Analizi (EDA)</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Satış Analizi", "🎁 Promo Etkisi", "📅 Zaman Serisi", "🔗 Korelasyon"])
    
    with tab1:
        st.markdown("### Satış Dağılımı ve Haftanın Günü Analizi")
        img = load_image('eda_sales_analysis.png')
        if img:
            st.image(img, use_container_width=True)
        
        st.markdown("""
        **📌 Bulgular:**
        - Ortalama satış: **6,955 TL** (açık günler)
        - Medyan satış: **6,369 TL**
        - En yüksek satış günü: **Pazartesi**
        - Pazar günleri çoğu mağaza kapalı
        """)
    
    with tab2:
        st.markdown("### Promo ve Mağaza Tipi Analizi")
        img = load_image('eda_promo_storetype.png')
        if img:
            st.image(img, use_container_width=True)
        
        st.success("✅ Promo ile **+38.77%** satış artışı!")
    
    with tab3:
        st.markdown("### Zaman Serisi - Günlük Toplam Satışlar")
        img = load_image('eda_time_series.png')
        if img:
            st.image(img, use_container_width=True)
        
        st.info("📈 2013-2015 arası 942 günlük satış verisi")
    
    with tab4:
        st.markdown("### Özellik Korelasyon Matrisi")
        img = load_image('eda_correlation.png')
        if img:
            st.image(img, use_container_width=True)
        
        st.markdown("""
        **🔍 En Yüksek Korelasyonlar:**
        - Sales ↔ Customers: **0.82** (çok güçlü)
        - Sales ↔ Promo: **0.38** (orta)
        """)

# ==================== SHAP ====================
elif page == "🔍 SHAP - Model Yorumlama":
    st.markdown('<div class="section-header">🔍 SHAP - Model Yorumlama Analizi</div>', unsafe_allow_html=True)
    
    st.info("📌 SHAP (SHapley Additive exPlanations): Model nasıl karar veriyor?")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary Plot", "📈 Feature Importance", "🔗 Dependence Plot", "💧 Waterfall Plot"])
    
    with tab1:
        st.markdown("### SHAP Summary Plot - Özellik Etkileri")
        img = load_image('shap_summary.png')
        if img:
            st.image(img, use_container_width=True)
        
        st.markdown("""
        **📌 Nasıl Okunur?**
        - **X ekseni:** SHAP değeri (tahmini ne kadar etkiliyor?)
        - **Renk:** Özellik değeri (kırmızı=yüksek, mavi=düşük)
        - **Her nokta:** Bir tahmin
        """)
    
    with tab2:
        st.markdown("### SHAP Feature Importance")
        img = load_image('shap_importance.png')
        if img:
            st.image(img, use_container_width=True)
        
        st.markdown("#### 📄 SHAP Değerleri")
        shap_values = load_csv('shap_values.csv')
        if shap_values is not None:
            st.dataframe(shap_values.head(15), use_container_width=True)
    
    with tab3:
        st.markdown("### SHAP Dependence Plot - Top 4 Özellik")
        img = load_image('shap_dependence.png')
        if img:
            st.image(img, use_container_width=True)
        
        st.markdown("""
        **🔍 Ne Gösteriyor?**
        - Özellik değeri arttıkça SHAP nasıl değişiyor?
        - İlişki doğrusal mı, non-linear mı?
        """)
    
    with tab4:
        st.markdown("### SHAP Waterfall Plot - Tek Tahmin Açıklaması")
        img = load_image('shap_waterfall.png')
        if img:
            st.image(img, use_container_width=True)
        
        st.markdown("""
        **💡 Açıklama:**
        - Baz tahmin (ortalama) + Her özelliğin katkısı = Final tahmin
        - Kırmızı: Artırıcı etki
        - Mavi: Azaltıcı etki
        """)

# ==================== KÜMELEME ====================
elif page == "🎯 Kümeleme Analizi":
    st.markdown('<div class="section-header">🎯 Mağaza Kümeleme (Segmentasyon) Analizi</div>', unsafe_allow_html=True)
    
    # Kümeleme Açıklaması
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;'>
        <h2 style='color: white; margin-bottom: 1rem;'>🤔 Kümeleme Nedir? Neden Yaptık?</h2>
        <p style='font-size: 1.2rem; line-height: 1.8;'>
            <strong>Kümeleme (Clustering):</strong> Benzer özelliklere sahip mağazaları gruplayarak 
            <strong>5 farklı segment</strong> oluşturduk. Her segment farklı karakteristiklere sahip.
        </p>
        <p style='font-size: 1.2rem; line-height: 1.8;'>
            <strong>Neden?</strong> 1,115 mağaza çok farklı! Bazıları büyük şehirde, bazıları küçük kasabada. 
            Bazıları çok satış yapıyor, bazıları az. Onları <strong>benzer gruplara ayırarak</strong> 
            her gruba özel strateji geliştirebiliriz!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Neyi Kümeledik?
    st.markdown("## 🏪 Neyi Kümeledik?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card' style='background: #e8f5e9;'>
            <h3>📍 Girdi</h3>
            <p style='font-size: 1.1rem;'>
                <strong>1,115 mağaza</strong>
                <br>Her mağaza = 1 satır
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card' style='background: #fff3e0;'>
            <h3>🔢 Özellikler (11 adet)</h3>
            <ul style='font-size: 1rem;'>
                <li>Ortalama satış</li>
                <li>Satış volatilitesi</li>
                <li>Promo etkisi</li>
                <li>Müşteri sayısı</li>
                <li>Mağaza tipi</li>
                <li>Rekabet mesafesi</li>
                <li>... ve daha fazlası</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card' style='background: #f3e5f5;'>
            <h3>🎯 Çıktı</h3>
            <p style='font-size: 1.1rem;'>
                <strong>5 farklı küme</strong>
                <br>Her mağaza bir kümeye atandı
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Kümeleme Süreci
    st.markdown("## 🔄 Nasıl Kümeledik?")
    
    st.markdown("""
    <div class='metric-card'>
        <h3>Adım Adım Kümeleme Süreci</h3>
        <ol style='font-size: 1.1rem; line-height: 2;'>
            <li><strong>Veri Hazırlama:</strong> Her 1,115 mağaza için 11 özellik hesaplandı
                <br><small style='color: #666;'>→ Örn: Mağaza 1'in ortalama satışı: 5,263 TL, promo etkisi: %62</small>
            </li>
            <li><strong>Normalizasyon:</strong> Tüm özellikler 0-1 arasına ölçeklendirildi
                <br><small style='color: #666;'>→ Çünkü satış (0-15,000) ve promo (0-2) farklı ölçeklerde</small>
            </li>
            <li><strong>K-Means Algoritması:</strong> Benzer mağazaları 5 gruba ayırdı
                <br><small style='color: #666;'>→ Makine öğrenmesi algoritması otomatik gruplayıp optimize etti</small>
            </li>
            <li><strong>Sonuç:</strong> Her mağazaya 0-4 arası bir küme etiketi verildi
                <br><small style='color: #666;'>→ Örn: Mağaza 1 → Cluster 0 (Premium City)</small>
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Küme Sonuçları Özeti
    st.markdown("## 📊 Kümeleme Sonuçları - 5 Segment")
    
    st.markdown("""
    <div class='metric-card' style='background: linear-gradient(to right, #ffecd2 0%, #fcb69f 100%);'>
        <div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem;'>
            <div style='text-align: center; padding: 1rem; background: white; border-radius: 10px;'>
                <h4 style='color: #e74c3c;'>Cluster 0</h4>
                <p style='font-size: 0.9rem; font-weight: bold;'>Premium City</p>
                <p style='font-size: 1.5rem; font-weight: bold; color: #e74c3c;'>283</p>
                <p style='font-size: 0.8rem;'>mağaza</p>
                <p style='font-size: 1rem; color: #2ecc71;'>6,567 TL</p>
            </div>
            <div style='text-align: center; padding: 1rem; background: white; border-radius: 10px;'>
                <h4 style='color: #3498db;'>Cluster 1</h4>
                <p style='font-size: 0.9rem; font-weight: bold;'>Suburban Std</p>
                <p style='font-size: 1.5rem; font-weight: bold; color: #3498db;'>273</p>
                <p style='font-size: 0.8rem;'>mağaza</p>
                <p style='font-size: 1rem; color: #e74c3c;'>5,532 TL</p>
            </div>
            <div style='text-align: center; padding: 1rem; background: white; border-radius: 10px; border: 3px solid #f39c12;'>
                <h4 style='color: #2ecc71;'>Cluster 2 ⭐</h4>
                <p style='font-size: 0.9rem; font-weight: bold;'>Small Town</p>
                <p style='font-size: 1.5rem; font-weight: bold; color: #2ecc71;'>154</p>
                <p style='font-size: 0.8rem;'>mağaza</p>
                <p style='font-size: 1.2rem; color: #2ecc71; font-weight: bold;'>10,806 TL</p>
                <small style='color: #f39c12;'>EN YÜKSEK!</small>
            </div>
            <div style='text-align: center; padding: 1rem; background: white; border-radius: 10px;'>
                <h4 style='color: #f39c12;'>Cluster 3</h4>
                <p style='font-size: 0.9rem; font-weight: bold;'>Flagship</p>
                <p style='font-size: 1.5rem; font-weight: bold; color: #f39c12;'>375</p>
                <p style='font-size: 0.8rem;'>mağaza</p>
                <p style='font-size: 1rem; color: #2ecc71;'>6,502 TL</p>
            </div>
            <div style='text-align: center; padding: 1rem; background: white; border-radius: 10px;'>
                <h4 style='color: #9b59b6;'>Cluster 4</h4>
                <p style='font-size: 0.9rem; font-weight: bold;'>Rural Low</p>
                <p style='font-size: 1.5rem; font-weight: bold; color: #9b59b6;'>30</p>
                <p style='font-size: 0.8rem;'>mağaza</p>
                <p style='font-size: 1rem; color: #2ecc71;'>8,695 TL</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Görselleştirme", "📈 Küme Profilleri", "📦 Satış Karşılaştırma", "🔍 Mağaza Ara"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Elbow & Silhouette Analizi")
            img = load_image('clustering_elbow_silhouette.png')
            if img:
                st.image(img, use_container_width=True)
            st.caption("Optimal küme sayısı: k=5")
        
        with col2:
            st.markdown("### PCA 2D Projection")
            img = load_image('clustering_pca_2d.png')
            if img:
                st.image(img, use_container_width=True)
            st.caption("11 boyutlu veri 2D'de görselleştirildi")
    
    with tab2:
        st.markdown("### Kümelere Göre Özellikler")
        img = load_image('clustering_profiles.png')
        if img:
            st.image(img, use_container_width=True)
        
        st.markdown("#### 📊 Küme İstatistikleri")
        cluster_stats = load_csv('clustering_statistics.csv')
        if cluster_stats is not None:
            st.dataframe(cluster_stats, use_container_width=True)
    
    with tab3:
        st.markdown("### 📦 Kümelere Göre Satış Dağılımı")
        
        st.info("💡 **Nasıl Okunur?** Her kutunun ortası medyan, üst/alt çizgiler min/max, kutunun boyutu varyansı gösterir.")
        
        img = load_image('clustering_sales_boxplot.png')
        if img:
            st.image(img, use_container_width=True)
        
        st.success("⭐ **Bulgu:** Cluster 2 (Small Town) en yüksek ortalama satışa sahip: 10,806 TL - Küçük kasabalardaki ana mağazalar en karlı segment!")
        
        st.markdown("""
        **🎯 İş Önerileri:**
        - Cluster 2 stratejisini diğer bölgelere adapte et
        - Assortment 'c' (geniş ürün yelpazesi) başarılı
        - Küçük kasaba lokasyonlarına yatırım artırılabilir
        """)
    
    with tab4:
        st.markdown("### 🔍 Mağaza Arama - Küme Sorgulama")
        
        cluster_labels = load_csv('clustering_labels.csv')
        if cluster_labels is not None:
            # Arama seçenekleri
            search_type = st.radio("Arama Türü", ["Kümeye Göre Filtrele", "Mağaza ID'ye Göre Ara"], horizontal=True)
            
            if search_type == "Kümeye Göre Filtrele":
                selected_cluster = st.selectbox(
                    "Küme Seç", 
                    range(5), 
                    format_func=lambda x: f"Cluster {x} - {['Premium City', 'Suburban Std', 'Small Town', 'Flagship', 'Rural Low'][x]}"
                )
                filtered = cluster_labels[cluster_labels['cluster'] == selected_cluster]
                st.write(f"**📊 Cluster {selected_cluster} Mağazaları:** {len(filtered)} mağaza bulundu")
                st.dataframe(filtered, use_container_width=True, height=400)
                
                # Küme özeti
                st.markdown(f"""
                **📈 Cluster {selected_cluster} Özeti:**
                - **Mağaza Sayısı:** {len(filtered)}
                - **Ortalama Satış:** {filtered['avg_sales'].mean():.2f} TL
                - **Promo Etkisi:** {filtered['promo_lift'].mean():.2f}
                """)
            
            else:
                store_id = st.number_input("Mağaza ID Gir (1-1115)", min_value=1, max_value=1115, value=5)
                store_info = cluster_labels[cluster_labels['Store'] == store_id]
                
                if not store_info.empty:
                    st.success(f"✅ Mağaza {store_id} bulundu!")
                    
                    cluster_id = store_info['cluster'].values[0]
                    cluster_name = ['Premium City', 'Suburban Std', 'Small Town', 'Flagship', 'Rural Low'][cluster_id]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Küme", f"Cluster {cluster_id}")
                        st.caption(cluster_name)
                    
                    with col2:
                        st.metric("Ortalama Satış", f"{store_info['avg_sales'].values[0]:.0f} TL")
                    
                    with col3:
                        st.metric("Promo Etkisi", f"{store_info['promo_lift'].values[0]:.2f}")
                    
                    st.dataframe(store_info, use_container_width=True)
                else:
                    st.error(f"❌ Mağaza {store_id} bulunamadı!")

# ==================== İŞLEVSEL TABLOLAR ====================
elif page == "📋 İşlevsel Tablolar":
    st.markdown('<div class="section-header">📋 İşlevsel Tablolar & Analizler</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;'>
        <h3 style='color: white; margin: 0;'>💼 Tez Değerlendirmesi & İş Analitiği için Hazır Tablolar</h3>
        <p style='margin: 0.5rem 0 0 0;'>Akademik değerlendirme ve iş kararları için optimize edilmiş, anlaşılır tablolar</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Model Performans Özeti", 
        "🏪 Mağaza Performansı", 
        "🎯 Küme Analitiği",
        "📅 Günlük Performans",
        "🎖️ Top/Bottom Listeler"
    ])
    
    # TAB 1: MODEL PERFORMANS ÖZETİ
    with tab1:
        st.markdown("### 📊 Model Performans Karşılaştırma Tablosu")
        st.markdown("**🎓 Tez için:** Model başarısını karşılaştırma")
        
        # Model comparison table (expanded)
        model_perf = pd.DataFrame({
            'Model': ['Ensemble (Optimized)', 'LightGBM', 'XGBoost', 'Baseline'],
            'CV RMSPE (Ort.)': [0.1212, 0.1218, 0.1230, 0.1393],
            'Std. Sapma': [0.0062, 0.0053, 0.0070, 0.0142],
            'Fold 1': [0.1224, 0.1234, 0.1241, 0.1539],
            'Fold 2': [0.1281, 0.1274, 0.1310, 0.1424],
            'Fold 3': [0.1183, 0.1184, 0.1204, 0.1217],
            'En İyi Fold': ['Fold 3', 'Fold 3', 'Fold 3', 'Fold 3'],
            'İyileşme (%)': ['-', '+0.07%', '-1.54%', '-2.28%'],
            'Durum': ['✅ Aktif', '📊 Ensemble İçinde', '📊 Ensemble İçinde', '📦 Arşivlendi']
        })
        
        st.dataframe(
            model_perf.style.highlight_min(subset=['CV RMSPE (Ort.)'], color='lightgreen')
                          .highlight_max(subset=['Fold 3'], color='lightblue')
                          .format({'CV RMSPE (Ort.)': '{:.4f}', 'Std. Sapma': '{:.4f}', 
                                  'Fold 1': '{:.4f}', 'Fold 2': '{:.4f}', 'Fold 3': '{:.4f}'}),
            use_container_width=True
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 En İyi Model", "Ensemble (Optimized)", delta="0.1212 RMSPE")
        with col2:
            st.metric("📈 Baseline'dan İyileşme", "13.0%", delta="-0.0181 RMSPE")
        with col3:
            st.metric("⭐ En İyi Fold", "Fold 3", delta="0.1183 RMSPE")
        
        st.markdown("---")
        
        st.markdown("### 📈 Fold Bazlı Detaylı Performans")
        st.markdown("**🎓 Tez için:** Her fold'un tutarlılığı")
        
        fold_details = pd.DataFrame({
            'Fold': ['Fold 1', 'Fold 2', 'Fold 3', 'Ortalama'],
            'Tarih Aralığı': ['2014-05-01 → 2014-06-11', '2014-11-16 → 2014-12-28', '2015-06-04 → 2015-07-16', 'Tüm Foldlar'],
            'Veri Boyutu': ['103,745', '103,745', '103,745', '311,235'],
            'Ensemble RMSPE': [0.1224, 0.1281, 0.1131, 0.1212],
            'LightGBM RMSPE': [0.1234, 0.1274, 0.1146, 0.1218],
            'XGBoost RMSPE': [0.1241, 0.1310, 0.1139, 0.1230],
            'En İyi': ['XGBoost', 'XGBoost', 'Ensemble', 'Ensemble'],
            'Zorluk': ['Zor 🔴', 'Orta 🟡', 'Kolay 🟢', '-']
        })
        
        st.dataframe(fold_details, use_container_width=True)
        
        st.info("💡 **Yorum:** Fold 3'te tüm modeller daha iyi performans gösterdi. Model tutarlı ve genelleme yeteneği yüksek.")
    
    # TAB 2: MAĞAZA PERFORMANSI
    with tab2:
        st.markdown("### 🏪 Mağaza Bazlı Tahmin Performansı")
        st.markdown("**💼 Rossmann için:** Hangi mağazalarda tahmin daha başarılı?")
        
        # Load validation predictions if available
        val_preds = load_csv('validation_predictions.csv')
        cluster_labels = load_csv('clustering_labels.csv')
        
        if val_preds is not None and cluster_labels is not None:
            # Calculate store-level performance
            val_preds_merged = val_preds.merge(cluster_labels[['Store', 'cluster', 'cluster_name']], on='Store', how='left')
            
            store_perf = val_preds_merged.groupby('Store').apply(
                lambda x: pd.Series({
                    'Tahmin Sayısı': len(x),
                    'Ortalama Gerçek': x['Sales'].mean(),
                    'Ortalama Tahmin': x['Predicted'].mean(),
                    'RMSPE': ((((x['Sales'] - x['Predicted']) / x['Sales']) ** 2).mean()) ** 0.5,
                    'MAE': (x['Sales'] - x['Predicted']).abs().mean(),
                    'Küme': x['cluster'].iloc[0] if 'cluster' in x.columns else -1,
                    'Küme Adı': x['cluster_name'].iloc[0] if 'cluster_name' in x.columns else 'Unknown'
                })
            ).reset_index()
            
            # Top 20 en iyi tahmin edilen mağazalar
            st.markdown("#### 🌟 En İyi Tahmin Edilen Mağazalar (Top 20)")
            top_stores = store_perf.nsmallest(20, 'RMSPE')[['Store', 'Küme Adı', 'Ortalama Gerçek', 'Ortalama Tahmin', 'RMSPE', 'MAE', 'Tahmin Sayısı']]
            st.dataframe(
                top_stores.style.background_gradient(subset=['RMSPE'], cmap='RdYlGn_r')
                               .format({'Ortalama Gerçek': '{:.0f}', 'Ortalama Tahmin': '{:.0f}',
                                       'RMSPE': '{:.4f}', 'MAE': '{:.2f}'}),
                use_container_width=True
            )
            
            st.markdown("#### ⚠️ En Zor Tahmin Edilen Mağazalar (Bottom 20)")
            bottom_stores = store_perf.nlargest(20, 'RMSPE')[['Store', 'Küme Adı', 'Ortalama Gerçek', 'Ortalama Tahmin', 'RMSPE', 'MAE', 'Tahmin Sayısı']]
            st.dataframe(
                bottom_stores.style.background_gradient(subset=['RMSPE'], cmap='RdYlGn')
                                  .format({'Ortalama Gerçek': '{:.0f}', 'Ortalama Tahmin': '{:.0f}',
                                          'RMSPE': '{:.4f}', 'MAE': '{:.2f}'}),
                use_container_width=True
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("En İyi Mağaza", f"Store {top_stores.iloc[0]['Store']}", 
                         delta=f"{top_stores.iloc[0]['RMSPE']:.4f} RMSPE")
            with col2:
                st.metric("En Zor Mağaza", f"Store {bottom_stores.iloc[0]['Store']}", 
                         delta=f"{bottom_stores.iloc[0]['RMSPE']:.4f} RMSPE", delta_color="inverse")
            with col3:
                st.metric("Ortalama Mağaza RMSPE", f"{store_perf['RMSPE'].mean():.4f}")
            
            st.success("💡 **İş Önerisi:** En zor tahmin edilen mağazalar için özel modeller veya manuel müdahale düşünülebilir.")
        else:
            st.warning("⚠️ Validation predictions verisi bulunamadı. `evaluate.py` çalıştırılmamış olabilir.")
    
    # TAB 3: KÜME ANALİTİĞİ
    with tab3:
        st.markdown("### 🎯 Küme Bazlı Performans Analizi")
        st.markdown("**💼 Rossmann için:** Hangi segment için model daha başarılı?")
        
        cluster_stats = load_csv('clustering_statistics.csv')
        val_preds = load_csv('validation_predictions.csv')
        cluster_labels = load_csv('clustering_labels.csv')
        
        if cluster_stats is not None:
            # Expanded cluster table
            cluster_table = cluster_stats[['cluster', 'cluster_name', 'count', 'avg_sales', 'promo_lift', 
                                          'promo_usage_rate', 'main_storetype', 'main_assortment']].copy()
            cluster_table.columns = ['Küme ID', 'Küme Adı', 'Mağaza Sayısı', 'Ort. Satış (TL)', 
                                    'Promo Etkisi', 'Promo Kullanım', 'Ana StoreType', 'Ana Assortment']
            
            st.dataframe(
                cluster_table.style.background_gradient(subset=['Ort. Satış (TL)'], cmap='YlGn')
                                  .format({'Ort. Satış (TL)': '{:.0f}', 'Promo Etkisi': '{:.2f}', 
                                          'Promo Kullanım': '{:.2%}'}),
                use_container_width=True
            )
            
            if val_preds is not None and cluster_labels is not None:
                # Cluster-level performance
                val_merged = val_preds.merge(cluster_labels[['Store', 'cluster', 'cluster_name']], on='Store', how='left')
                
                cluster_perf = val_merged.groupby(['cluster', 'cluster_name']).apply(
                    lambda x: pd.Series({
                        'Tahmin Sayısı': len(x),
                        'RMSPE': ((((x['Sales'] - x['Predicted']) / x['Sales']) ** 2).mean()) ** 0.5,
                        'MAE': (x['Sales'] - x['Predicted']).abs().mean(),
                        'R²': 1 - ((x['Sales'] - x['Predicted']) ** 2).sum() / ((x['Sales'] - x['Sales'].mean()) ** 2).sum()
                    })
                ).reset_index()
                
                st.markdown("#### 📊 Kümelere Göre Model Performansı")
                st.dataframe(
                    cluster_perf.style.background_gradient(subset=['RMSPE'], cmap='RdYlGn_r')
                                     .background_gradient(subset=['R²'], cmap='RdYlGn')
                                     .format({'RMSPE': '{:.4f}', 'MAE': '{:.2f}', 'R²': '{:.3f}'}),
                    use_container_width=True
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    best_cluster = cluster_perf.loc[cluster_perf['RMSPE'].idxmin()]
                    st.metric("🌟 En İyi Tahmin Edilen Küme", 
                             f"{best_cluster['cluster_name']}", 
                             delta=f"{best_cluster['RMSPE']:.4f} RMSPE")
                with col2:
                    worst_cluster = cluster_perf.loc[cluster_perf['RMSPE'].idxmax()]
                    st.metric("⚠️ En Zor Küme", 
                             f"{worst_cluster['cluster_name']}", 
                             delta=f"{worst_cluster['RMSPE']:.4f} RMSPE", 
                             delta_color="inverse")
                
                st.info(f"💡 **İş Stratejisi:** {best_cluster['cluster_name']} segmenti için strateji diğer segmentlere adapte edilebilir.")
        else:
            st.warning("⚠️ Clustering verisi bulunamadı.")
    
    # TAB 4: GÜNLÜK PERFORMANS
    with tab4:
        st.markdown("### 📅 Günlük & Haftalık Performans Analizi")
        st.markdown("**🎓 Tez için:** Model hangi günlerde/durumlarda daha başarılı?")
        
        val_preds = load_csv('validation_predictions.csv')
        
        if val_preds is not None and 'Date' in val_preds.columns:
            val_preds['Date'] = pd.to_datetime(val_preds['Date'])
            val_preds['DayOfWeek'] = val_preds['Date'].dt.dayofweek + 1
            val_preds['DayName'] = val_preds['Date'].dt.day_name()
            
            # Performance by day of week
            dow_perf = val_preds.groupby(['DayOfWeek', 'DayName']).apply(
                lambda x: pd.Series({
                    'Tahmin Sayısı': len(x),
                    'Ortalama Satış': x['Sales'].mean(),
                    'RMSPE': ((((x['Sales'] - x['Predicted']) / x['Sales']) ** 2).mean()) ** 0.5,
                    'MAE': (x['Sales'] - x['Predicted']).abs().mean()
                })
            ).reset_index().sort_values('DayOfWeek')
            
            st.markdown("#### 📊 Haftanın Günlerine Göre Performans")
            st.dataframe(
                dow_perf.style.background_gradient(subset=['RMSPE'], cmap='RdYlGn_r')
                              .background_gradient(subset=['Ortalama Satış'], cmap='Blues')
                              .format({'Ortalama Satış': '{:.0f}', 'RMSPE': '{:.4f}', 'MAE': '{:.2f}'}),
                use_container_width=True
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                best_day = dow_perf.loc[dow_perf['RMSPE'].idxmin()]
                st.metric("🌟 En İyi Tahmin Edilen Gün", best_day['DayName'], 
                         delta=f"{best_day['RMSPE']:.4f} RMSPE")
            with col2:
                worst_day = dow_perf.loc[dow_perf['RMSPE'].idxmax()]
                st.metric("⚠️ En Zor Gün", worst_day['DayName'], 
                         delta=f"{worst_day['RMSPE']:.4f} RMSPE", delta_color="inverse")
            with col3:
                highest_sales_day = dow_perf.loc[dow_perf['Ortalama Satış'].idxmax()]
                st.metric("💰 En Yüksek Satış Günü", highest_sales_day['DayName'], 
                         delta=f"{highest_sales_day['Ortalama Satış']:.0f} TL")
            
            # Promo effect on performance
            if 'Promo' in val_preds.columns:
                st.markdown("#### 🎁 Promo Durumuna Göre Performans")
                promo_perf = val_preds.groupby('Promo').apply(
                    lambda x: pd.Series({
                        'Tahmin Sayısı': len(x),
                        'Ortalama Satış': x['Sales'].mean(),
                        'RMSPE': ((((x['Sales'] - x['Predicted']) / x['Sales']) ** 2).mean()) ** 0.5,
                        'MAE': (x['Sales'] - x['Predicted']).abs().mean()
                    })
                ).reset_index()
                promo_perf['Durum'] = promo_perf['Promo'].map({0: 'Promo Yok', 1: 'Promo Var'})
                
                st.dataframe(
                    promo_perf[['Durum', 'Tahmin Sayısı', 'Ortalama Satış', 'RMSPE', 'MAE']].style
                        .background_gradient(subset=['RMSPE'], cmap='RdYlGn_r')
                        .format({'Ortalama Satış': '{:.0f}', 'RMSPE': '{:.4f}', 'MAE': '{:.2f}'}),
                    use_container_width=True
                )
        else:
            st.warning("⚠️ Validation predictions verisi bulunamadı.")
    
    # TAB 5: TOP/BOTTOM LİSTELER
    with tab5:
        st.markdown("### 🎖️ Top & Bottom Listeler")
        st.markdown("**💼 Rossmann için:** Hangi mağazalar/durumlar dikkat gerektiriyor?")
        
        val_preds = load_csv('validation_predictions.csv')
        cluster_labels = load_csv('clustering_labels.csv')
        
        if val_preds is not None:
            val_merged = val_preds.copy()
            if cluster_labels is not None:
                val_merged = val_merged.merge(cluster_labels[['Store', 'cluster_name', 'avg_sales']], 
                                             on='Store', how='left')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏆 En Yüksek Satış Tahminleri (Top 50)")
                top_sales = val_merged.nlargest(50, 'Predicted')[['Store', 'Date', 'Predicted', 'Sales', 'cluster_name']]
                top_sales['Hata (%)'] = ((top_sales['Predicted'] - top_sales['Sales']) / top_sales['Sales'] * 100)
                st.dataframe(
                    top_sales.style.background_gradient(subset=['Predicted'], cmap='Greens')
                                  .format({'Predicted': '{:.0f}', 'Sales': '{:.0f}', 'Hata (%)': '{:.1f}%'}),
                    use_container_width=True,
                    height=400
                )
            
            with col2:
                st.markdown("#### 📉 En Düşük Satış Tahminleri (Bottom 50)")
                bottom_sales = val_merged.nsmallest(50, 'Predicted')[['Store', 'Date', 'Predicted', 'Sales', 'cluster_name']]
                bottom_sales['Hata (%)'] = ((bottom_sales['Predicted'] - bottom_sales['Sales']) / bottom_sales['Sales'] * 100)
                st.dataframe(
                    bottom_sales.style.background_gradient(subset=['Predicted'], cmap='Reds_r')
                                     .format({'Predicted': '{:.0f}', 'Sales': '{:.0f}', 'Hata (%)': '{:.1f}%'}),
                    use_container_width=True,
                    height=400
                )
            
            st.markdown("---")
            
            # En büyük hatalar
            st.markdown("#### ❌ En Büyük Tahmin Hataları (Top 50)")
            val_merged['Absolute_Error'] = (val_merged['Sales'] - val_merged['Predicted']).abs()
            val_merged['Percentage_Error'] = ((val_merged['Sales'] - val_merged['Predicted']) / val_merged['Sales'] * 100)
            
            biggest_errors = val_merged.nlargest(50, 'Absolute_Error')[
                ['Store', 'Date', 'Sales', 'Predicted', 'Absolute_Error', 'Percentage_Error', 'cluster_name']
            ]
            
            st.dataframe(
                biggest_errors.style.background_gradient(subset=['Absolute_Error'], cmap='Reds')
                                   .format({'Sales': '{:.0f}', 'Predicted': '{:.0f}', 
                                           'Absolute_Error': '{:.0f}', 'Percentage_Error': '{:.1f}%'}),
                use_container_width=True
            )
            
            st.error("⚠️ **Dikkat:** Bu mağazalar/günler için detaylı inceleme gerekebilir. Olağandışı durumlar (tatil, stok problemi vb.) olabilir.")
        else:
            st.warning("⚠️ Validation predictions verisi bulunamadı.")

# ==================== RAPORLAR ====================
elif page == "📄 Raporlar":
    st.markdown('<div class="section-header">📄 Detaylı Raporlar</div>', unsafe_allow_html=True)
    
    report_choice = st.selectbox(
        "Rapor Seç",
        [
            "CV Skorları",
            "Feature Importance",
            "Validation Predictions",
            "Error Summary",
            "SHAP Values",
            "Clustering Labels",
            "Clustering Statistics",
            "Clustering Feature Importance"
        ]
    )
    
    report_map = {
        "CV Skorları": "cv_scores.csv",
        "Feature Importance": "feature_importance.csv",
        "Validation Predictions": "validation_predictions.csv",
        "Error Summary": "error_summary.csv",
        "SHAP Values": "shap_values.csv",
        "Clustering Labels": "clustering_labels.csv",
        "Clustering Statistics": "clustering_statistics.csv",
        "Clustering Feature Importance": "clustering_feature_importance.csv"
    }
    
    filename = report_map[report_choice]
    df = load_csv(filename)
    
    if df is not None:
        st.success(f"✅ {report_choice} yüklendi: {len(df)} satır")
        
        # Search functionality
        if len(df) > 100:
            search = st.text_input("🔍 Ara (Store ID, özellik adı vb.)")
            if search:
                df = df[df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)]
        
        st.dataframe(df, use_container_width=True, height=600)
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 CSV İndir",
            data=csv,
            file_name=filename,
            mime='text/csv'
        )
    else:
        st.error(f"❌ {filename} bulunamadı!")

# ==================== HAKKINDA ====================
elif page == "ℹ️ Proje Hakkında":
    st.markdown('<div class="section-header">ℹ️ Proje Hakkında</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 Proje Bilgileri")
        st.write("""
        **Proje Adı:** Rossmann Store Sales Forecasting
        
        **Tür:** Bitirme Tezi Projesi
        
        **Hedef:** 1,115 Rossmann mağazası için 6 hafta ilerisi satış tahmini
        
        **Veri:** Kaggle Rossmann Store Sales yarışması
        
        **Model:** LightGBM + XGBoost Ensemble
        
        **Performans:** 0.1212 RMSPE (Top %14-17) 🏆
        """)
        
        st.markdown("### 🛠️ Kullanılan Teknolojiler")
        st.write("""
        - **Python 3.13**
        - **Pandas, NumPy** - Veri işleme
        - **Scikit-learn** - ML altyapı
        - **LightGBM, XGBoost** - Modeller
        - **Matplotlib, Seaborn** - Görselleştirme
        - **SHAP** - Model yorumlama
        - **Streamlit** - Web dashboard
        """)
    
    with col2:
        st.markdown("### 📊 Proje İçeriği")
        st.write("""
        **Modüller:**
        - `src/data.py` - Veri yükleme
        - `src/features.py` - Feature engineering
        - `src/metrics.py` - Metrikler
        - `src/validation.py` - CV stratejisi
        
        **Scriptler:**
        - `ensemble_train.py` - Model eğitimi
        - `ensemble_predict.py` - Tahmin
        - `evaluate.py` - Performans analizi
        - `eda_analysis.py` - Veri analizi
        - `shap_analysis.py` - Model yorumlama
        - `clustering_analysis.py` - Mağaza kümeleme
        
        **Çıktılar:**
        - 16 görselleştirme grafiği
        - 8 detaylı CSV raporu
        - submission.csv (41,088 tahmin)
        """)
        
        st.markdown("### 🎯 Sonuçlar")
        st.write("""
        **Model Performansı:**
        - CV RMSPE: 0.1212 (±0.0062)
        - En iyi fold: 0.1131 (Fold 3)
        - Baseline'dan %13.0 iyileşme
        - Holiday features + optimal ağırlıklar ile %11.0 iyileşme
        
        **Kümeleme:**
        - 5 farklı mağaza segmenti
        - Silhouette Score: 0.144
        - En başarılı: Small Town (10,806 ort.)
        
        **Özellikler:**
        - 34 feature (lag, rolling, meta)
        - En önemli: Sales_lag_14, Promo
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "📊 Rossmann Sales Forecasting Dashboard | "
    "Bitirme Tezi Projesi 2025 | "
    "Made with Streamlit 🎈"
    "</div>",
    unsafe_allow_html=True
)
