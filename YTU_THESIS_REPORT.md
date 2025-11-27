# YILDIZ TECHNICAL UNIVERSITY
## FACULTY OF ELECTRICAL AND ELECTRONICS
## DEPARTMENT OF COMPUTER ENGINEERING

---

# UNDERGRADUATE THESIS

---

# Sales Forecasting System Using Ensemble Machine Learning Methods

---

**Advisor:** [Advisor Name]

**Student:** Kaan Bilirgen - [Student Number]

---

**Istanbul, 2025**

---

© All rights of this thesis belong to Yildiz Technical University Computer Engineering Department.

---

# CONTENTS

| Section | Page |
|---------|------|
| SYMBOL LIST | iv |
| ABBREVIATION LIST | v |
| LIST OF FIGURES | vi |
| LIST OF TABLES | vii |
| PREFACE | viii |
| ÖZET | ix |
| ABSTRACT | x |
| 1. INTRODUCTION | 1 |
| 1.1. Problem Definition | 1 |
| 1.2. Literature Research | 2 |
| 1.3. Overview of the Study | 4 |
| 2. METHODS | 5 |
| 2.1. Gradient Boosting Algorithms | 5 |
| 2.2. LightGBM | 6 |
| 2.3. XGBoost | 7 |
| 2.4. Ensemble Learning | 8 |
| 2.5. K-Means Clustering | 9 |
| 2.6. SHAP Analysis | 10 |
| 3. APPLICATION | 11 |
| 3.1. About the Data Set | 11 |
| 3.2. Data Preprocessing | 13 |
| 3.3. Feature Engineering | 14 |
| 3.4. Model Training | 17 |
| 3.5. Hyperparameter Optimization | 19 |
| 3.6. Results and Evaluation | 20 |
| 4. CONCLUSION | 24 |
| REFERENCES | 26 |
| CURRICULUM VITAE | 28 |

---

# SYMBOL LIST

| Symbol | Description |
|--------|-------------|
| n | Number of observations |
| y | Actual sales value |
| ŷ | Predicted sales value |
| X | Feature matrix |
| w | Model weights |
| α | Learning rate |
| λ | Regularization parameter |
| k | Number of clusters |
| μ | Cluster centroid |
| σ | Standard deviation |
| Σ | Summation symbol |
| √ | Square root |
| log | Natural logarithm |
| exp | Exponential function |

---

# ABBREVIATION LIST

| Abbreviation | Description |
|--------------|-------------|
| RMSPE | Root Mean Square Percentage Error |
| CV | Cross-Validation |
| ML | Machine Learning |
| LGB | LightGBM |
| XGB | XGBoost |
| GBDT | Gradient Boosted Decision Trees |
| GOSS | Gradient-based One-Side Sampling |
| EFB | Exclusive Feature Bundling |
| SHAP | SHapley Additive exPlanations |
| TPE | Tree-structured Parzen Estimator |
| API | Application Programming Interface |
| CSV | Comma Separated Values |
| JSON | JavaScript Object Notation |
| EDA | Exploratory Data Analysis |
| PCA | Principal Component Analysis |

---

# LIST OF FIGURES

| Figure | Description | Page |
|--------|-------------|------|
| Figure 1.1 | Sales distribution across stores | 2 |
| Figure 1.2 | Time series of daily sales | 3 |
| Figure 2.1 | Gradient boosting algorithm workflow | 6 |
| Figure 2.2 | LightGBM leaf-wise tree growth | 7 |
| Figure 2.3 | XGBoost level-wise tree growth | 8 |
| Figure 2.4 | Ensemble model architecture | 9 |
| Figure 2.5 | K-Means clustering visualization | 10 |
| Figure 3.1 | Data preprocessing pipeline | 13 |
| Figure 3.2 | Feature engineering workflow | 15 |
| Figure 3.3 | Cross-validation fold structure | 18 |
| Figure 3.4 | Feature importance ranking | 21 |
| Figure 3.5 | SHAP summary plot | 22 |
| Figure 3.6 | Store clustering results | 23 |

---

# LIST OF TABLES

| Table | Description | Page |
|-------|-------------|------|
| Table 1.1 | Dataset statistics | 2 |
| Table 2.1 | LightGBM hyperparameters | 7 |
| Table 2.2 | XGBoost hyperparameters | 8 |
| Table 3.1 | Training data columns | 12 |
| Table 3.2 | Store metadata columns | 12 |
| Table 3.3 | Feature categories and counts | 16 |
| Table 3.4 | Cross-validation results | 20 |
| Table 3.5 | Model performance comparison | 21 |
| Table 3.6 | Cluster characteristics | 23 |

---

# PREFACE

This document presents my undergraduate thesis titled "Sales Forecasting System Using Ensemble Machine Learning Methods". This project has been developed for the graduation thesis requirement of the Computer Engineering Department at Yildiz Technical University.

The topic of sales forecasting using machine learning is highly relevant in today's data-driven business environment. Retail companies increasingly rely on accurate demand predictions for inventory management, workforce planning, and financial decision-making. I chose this subject because it combines theoretical machine learning concepts with practical business applications.

During this project, I implemented an ensemble model combining LightGBM and XGBoost algorithms, performed comprehensive feature engineering, and developed an interactive dashboard for visualization. The project also includes store segmentation through clustering analysis and model interpretability through SHAP analysis.

I would like to express my gratitude to my thesis advisor [Advisor Name] for their guidance and support throughout this project. I also thank my family and friends for their encouragement during my undergraduate studies.

Kaan Bilirgen, 2025

---

# ÖZET

Bu çalışmada, Rossmann perakende mağazaları için makine öğrenmesi tabanlı bir satış tahmin sistemi geliştirilmiştir. Sistem, 1.115 mağaza için 6 haftalık günlük satış tahminleri üretmektedir.

Geliştirilen sistem, LightGBM ve XGBoost gradyan artırma algoritmalarının ağırlıklı ortalamasını kullanan bir topluluk (ensemble) yaklaşımı uygulamaktadır. Model, 45 mühendislik özelliği kullanarak %12.12 RMSPE (Kök Ortalama Kare Yüzde Hatası) değerine ulaşmıştır. Bu performans, Kaggle yarışması sıralamasında yaklaşık ilk %14-17 dilimine karşılık gelmektedir.

Özellik mühendisliği kapsamında takvim özellikleri, gecikme (lag) özellikleri, hareketli istatistikler, tatil göstergeleri ve mağaza kümeleme bilgileri oluşturulmuştur. Hiperparametre optimizasyonu için Optuna çerçevesi kullanılarak Bayesian optimizasyon gerçekleştirilmiştir.

Çalışmada ayrıca K-Means kümeleme analizi ile mağazalar satış davranışlarına göre 5 farklı profile ayrılmıştır. SHAP analizi, promosyon aktivitesi, 14 günlük ve 28 günlük gecikme özellikleri ile hareketli istatistiklerin en etkili tahmin değişkenleri olduğunu ortaya koymuştur.

Sistem, sonuçların görselleştirilmesi için interaktif bir Streamlit kontrol paneli sunmaktadır.

**Anahtar Kelimeler:** Satış Tahmini, Topluluk Öğrenme, Gradyan Artırma, LightGBM, XGBoost, Zaman Serisi Analizi, K-Means Kümeleme

---

# ABSTRACT

This study develops a machine learning-based sales forecasting system for Rossmann retail stores. The system produces 6-week daily sales predictions for 1,115 stores.

The developed system implements an ensemble approach using weighted averaging of LightGBM and XGBoost gradient boosting algorithms. The model achieves 12.12% RMSPE (Root Mean Square Percentage Error) using 45 engineered features. This performance corresponds to approximately top 14-17% ranking in the Kaggle competition.

Feature engineering includes calendar features, lag features, rolling statistics, holiday indicators, and store clustering information. Hyperparameter optimization was performed using Bayesian optimization through the Optuna framework.

The study also segments stores into 5 distinct profiles based on sales behavior through K-Means clustering analysis. SHAP analysis reveals that promotional activity, 14-day and 28-day lag features, and rolling statistics are the most influential predictors.

The system provides an interactive Streamlit dashboard for visualization of results.

**Keywords:** Sales Forecasting, Ensemble Learning, Gradient Boosting, LightGBM, XGBoost, Time Series Analysis, K-Means Clustering

---

# 1. INTRODUCTION

## 1.1. Problem Definition

Sales forecasting is a critical business function in the retail industry. Accurate predictions enable retailers to optimize inventory levels, plan workforce schedules, and make informed financial decisions. Poor forecasting leads to significant business impacts including overstocking with associated capital lockup and storage costs, or understocking resulting in lost sales and customer dissatisfaction.

Rossmann operates over 3,000 drug stores across 7 European countries, making it one of the largest pharmacy retail chains in Europe. The company requires reliable daily sales forecasts for each store to support operational planning. This forecasting challenge is complicated by the heterogeneity of stores in terms of size, location, product assortment, and competitive environment.

The primary problem addressed in this study is developing an accurate and reliable sales forecasting system that can handle multiple stores simultaneously, account for various factors including promotional activities, holidays, and competition, and maintain accuracy over a 6-week forecast horizon.

## 1.2. Literature Research

Time series forecasting has evolved significantly from classical statistical methods to modern machine learning approaches.

Classical methods include the Autoregressive Integrated Moving Average (ARIMA) model introduced by Box and Jenkins (1970), which captures linear dependencies in stationary time series. The Holt-Winters exponential smoothing method extends simple exponential smoothing to handle trend and seasonal patterns.

Machine learning methods have gained prominence due to their ability to model non-linear relationships. Gradient boosting methods, formalized by Friedman (2001), build sequential ensembles where each model corrects the errors of its predecessors.

XGBoost, developed by Chen and Guestrin (2016), implements an efficient gradient boosting framework with regularization to prevent overfitting. LightGBM, introduced by Ke et al. (2017), addresses computational challenges through Gradient-based One-Side Sampling and Exclusive Feature Bundling techniques.

Ensemble methods combine multiple models to achieve better predictive performance. The theoretical foundation lies in the bias-variance tradeoff where combining diverse models can reduce both sources of error.

For retail sales forecasting, studies have shown that gradient boosting methods achieve superior performance due to their ability to handle mixed feature types, capture non-linear relationships, and scale efficiently to large datasets.

## 1.3. Overview of the Study

This study aims to develop a high-accuracy forecasting model using ensemble machine learning methods. The main objectives are:

First, to design a comprehensive feature engineering pipeline that extracts relevant information from historical sales data, store characteristics, and temporal patterns.

Second, to implement an ensemble approach combining LightGBM and XGBoost algorithms with optimized weights determined through cross-validation.

Third, to provide model interpretability through feature importance analysis and SHAP values, enabling stakeholders to understand the factors driving predictions.

Fourth, to segment stores into meaningful clusters based on their sales behavior, supporting targeted business strategies.

The study uses the Rossmann Store Sales dataset from Kaggle, containing daily sales records for 1,115 stores over approximately 2.5 years. The analysis includes exploratory data analysis, feature engineering, model training and optimization, and comprehensive evaluation of results.

---

# 2. METHODS

## 2.1. Gradient Boosting Algorithms

Gradient boosting is an ensemble learning technique that builds models sequentially, with each new model correcting the errors of its predecessors. The algorithm minimizes a loss function by iteratively adding weak learners, typically decision trees, in a gradient descent procedure.

The key idea is to fit new models to the residual errors of the combined ensemble. Each iteration adds a new tree that predicts the negative gradient of the loss function, effectively moving predictions toward the optimal solution.

Gradient boosting has proven highly effective for tabular data problems, consistently achieving top performance in machine learning competitions and real-world applications.

## 2.2. LightGBM

LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework developed by Microsoft that uses tree-based learning algorithms. It introduces two key innovations for efficiency:

Gradient-based One-Side Sampling (GOSS) reduces the number of data instances by focusing on those with larger gradients. Instances with small gradients are well-trained and contribute less to information gain, so GOSS randomly drops a portion of them while keeping all instances with large gradients.

Exclusive Feature Bundling (EFB) reduces the number of features by bundling mutually exclusive features together. Since sparse features rarely take non-zero values simultaneously, they can be combined into a single feature without significant information loss.

LightGBM also uses leaf-wise tree growth instead of level-wise growth. This approach tends to achieve better accuracy but may lead to overfitting on small datasets.

The hyperparameters used in this study include 2,000 estimators, learning rate of 0.05, 63 leaves per tree, 80% subsample ratio, and early stopping after 100 rounds.

## 2.3. XGBoost

XGBoost (Extreme Gradient Boosting) is a scalable and efficient implementation of gradient boosting developed by Chen and Guestrin. It includes several innovations:

Regularized learning objective incorporates L1 and L2 penalties to prevent overfitting. The objective function includes both a loss term measuring prediction error and a regularization term controlling model complexity.

Sparsity-aware algorithms efficiently handle missing values and sparse data. The algorithm learns optimal default directions for missing values during training.

Parallel computing capabilities enable efficient training on large datasets through parallelization of tree construction.

XGBoost uses level-wise tree growth, expanding all nodes at the same depth before proceeding to the next level. This provides more balanced trees compared to leaf-wise growth.

The hyperparameters used in this study include 2,000 estimators, learning rate of 0.05, maximum depth of 8, 80% subsample ratio, L2 regularization of 1.0, and early stopping after 100 rounds.

## 2.4. Ensemble Learning

Ensemble learning combines multiple models to achieve better predictive performance than individual models. The approach leverages the diversity of models to reduce errors.

For this study, a weighted averaging ensemble combines LightGBM and XGBoost predictions. The weights are optimized through cross-validation to minimize RMSPE on held-out data.

The optimization is performed using scipy.optimize.minimize with the objective function being the cross-validation RMSPE. The optimal weights found are 63.6% for LightGBM and 36.4% for XGBoost.

The ensemble approach is effective because LightGBM and XGBoost have complementary characteristics. LightGBM's leaf-wise growth captures fine-grained patterns while XGBoost's regularization prevents overfitting.

## 2.5. K-Means Clustering

K-Means clustering is an unsupervised learning algorithm that partitions data into k clusters based on feature similarity. The algorithm iteratively assigns data points to the nearest cluster centroid and updates centroids based on assigned points.

In this study, K-Means clustering segments stores into behavioral profiles based on sales patterns, promotional response, and customer traffic. The optimal number of clusters (k=5) is determined using the elbow method and silhouette score analysis.

The clustering features include average daily sales, sales volatility, promotional response rate, and customer traffic metrics. The resulting clusters represent distinct store profiles that can inform targeted business strategies.

## 2.6. SHAP Analysis

SHAP (SHapley Additive exPlanations) is a method for explaining individual predictions based on game theory. It computes the contribution of each feature to a prediction by considering all possible feature combinations.

SHAP values have several desirable properties including local accuracy, missingness, and consistency. The sum of SHAP values equals the difference between the model prediction and the expected value.

In this study, SHAP analysis identifies the most influential features driving sales predictions. The analysis reveals the impact direction and magnitude of each feature, enabling stakeholders to understand model decisions.

---

# 3. APPLICATION

## 3.1. About the Data Set

The dataset originates from the Kaggle Rossmann Store Sales competition and consists of three main files.

The training data (train.csv) contains 1,017,209 daily observations for 1,115 stores spanning January 1, 2013 to July 31, 2015. Each record includes:

- Store: Unique store identifier (1-1,115)
- DayOfWeek: Day of week (1=Monday, 7=Sunday)
- Date: Date of sales record
- Sales: Daily turnover (target variable)
- Customers: Number of customers
- Open: Store open indicator (0/1)
- Promo: Promotion active indicator (0/1)
- StateHoliday: State holiday type (a, b, c, 0)
- SchoolHoliday: School holiday indicator (0/1)

The store metadata (store.csv) provides characteristics for each store:

- StoreType: Store model type (a, b, c, d)
- Assortment: Assortment level (a=basic, b=extra, c=extended)
- CompetitionDistance: Distance to nearest competitor in meters
- CompetitionOpenSinceMonth/Year: When competitor opened
- Promo2: Continuous promotion participation (0/1)
- Promo2SinceWeek/Year: When store joined Promo2
- PromoInterval: Months when Promo2 is active

The test data (test.csv) requires predictions for 41,088 store-date combinations representing the 6-week forecast horizon.

## 3.2. Data Preprocessing

The preprocessing pipeline addresses data quality issues and prepares the data for modeling.

Missing value treatment handles incomplete records. CompetitionDistance missing values (2,642 records) are filled with the median value of 2,325 meters. Competition opening date missing values are filled with the store's earliest recorded date. Promo2 related missing values are filled with zero indicating no participation.

Data type conversions ensure proper handling. The Date column is parsed to datetime format. StateHoliday is converted to string type for consistent encoding.

Data filtering removes records that should not be used for training. Records where stores were closed (Open=0) are excluded since these result in zero sales regardless of other factors. The target variable Sales is transformed using log(1+x) to reduce skewness and stabilize variance.

Train-store merge combines training data with store metadata through a left join on Store ID, preserving all training records while enriching with store attributes.

## 3.3. Feature Engineering

Feature engineering is critical for model performance. The system generates 45 features organized into categories.

Calendar features (8 features) extract temporal information:
- year: Year extracted from date
- month: Month (1-12)
- day: Day of month (1-31)
- week: ISO week number (1-53)
- dayofweek: Day of week (0-6)
- is_weekend: Saturday or Sunday indicator
- is_month_start: First day of month
- is_month_end: Last day of month

Store features (6 features) derive from store metadata:
- StoreType_encoded: Label encoded store type
- Assortment_encoded: Label encoded assortment level
- CompetitionDistance: Distance to nearest competitor
- CompetitionOpenMonths: Months since competitor opened
- Promo2: Continuous promotion flag
- Promo2Weeks: Weeks participating in Promo2

Lag features (6 features) capture historical patterns:
- Sales_lag_7: Sales 1 week ago
- Sales_lag_14: Sales 2 weeks ago
- Sales_lag_21: Sales 3 weeks ago
- Sales_lag_28: Sales 4 weeks ago
- Sales_lag_35: Sales 5 weeks ago
- Sales_lag_42: Sales 6 weeks ago

Rolling statistics (6 features) measure recent trends:
- Sales_rollmean_7, _14, _28: Rolling averages
- Sales_rollstd_7, _14, _28: Rolling standard deviations

Holiday features (6 features) identify special periods:
- is_christmas_week: December 20-31
- is_newyear_week: December 27 - January 3
- is_easter_week: ±7 days from Easter
- days_to_easter: Days until Easter
- school_holiday_tomorrow: Next day school holiday
- school_holiday_yesterday: Previous day school holiday

Cluster features (5 features) represent store segment membership from K-Means clustering.

Additional features (8 features) include promotional indicators and encoded categorical variables.

## 3.4. Model Training

Model training employs time-series cross-validation with 3 folds to respect temporal ordering and prevent data leakage.

Fold 1 uses training data from January 2013 to March 2015 and validates on March 25 to May 6, 2015.

Fold 2 extends training through May 2015 and validates on May 7 to June 18, 2015.

Fold 3 uses all data through June 2015 and validates on June 19 to July 31, 2015.

Each validation period spans approximately 6 weeks, matching the intended forecast horizon.

Both LightGBM and XGBoost models are trained on each fold with early stopping based on validation RMSPE. The final models are trained on all available data for prediction generation.

The training process includes:
1. Data loading and preprocessing
2. Feature engineering
3. Cross-validation fold generation
4. Model training with early stopping
5. Prediction generation and ensemble combination
6. Model and configuration saving

## 3.5. Hyperparameter Optimization

Hyperparameter optimization was performed using the Optuna framework with Bayesian optimization through the Tree-structured Parzen Estimator (TPE).

The optimization process includes 100 trials for each model, with the objective of minimizing cross-validation RMSPE. The search space covers learning rate, number of leaves/max depth, subsample ratio, and regularization parameters.

Ensemble weight optimization is performed separately using scipy.optimize.minimize. The objective function evaluates RMSPE for different weight combinations, finding the optimal balance between LightGBM and XGBoost contributions.

The optimized configuration achieves RMSPE of 0.1212 compared to 0.1225 with equal weighting, representing a 1.1% improvement.

## 3.6. Results and Evaluation

The cross-validation results demonstrate consistent model performance across temporal folds.

LightGBM achieves mean RMSPE of 0.1212 with standard deviation of 0.0062. Individual fold scores are 0.1224 (Fold 1), 0.1281 (Fold 2), and 0.1131 (Fold 3).

XGBoost achieves mean RMSPE of 0.1223 with standard deviation of 0.0063. Individual fold scores are 0.1234 (Fold 1), 0.1295 (Fold 2), and 0.1141 (Fold 3).

The weighted ensemble achieves mean RMSPE of 0.1206 with optimal weights of 63.6% LightGBM and 36.4% XGBoost.

Feature importance analysis reveals the top predictors:
1. Promo (promotional indicator) - highest importance
2. Sales_lag_14 (2-week lag)
3. Sales_lag_28 (4-week lag)
4. Sales_rollstd_7 (weekly volatility)
5. DayOfWeek (day of week effect)

SHAP analysis confirms these findings and reveals that promotional activity increases predicted sales by 15-25%. Lag features show strong positive correlations with future sales.

Clustering analysis segments stores into 5 profiles:
- Cluster 0: Small suburban (234 stores, avg sales 5,400)
- Cluster 1: High-traffic urban (198 stores, avg sales 8,800)
- Cluster 2: Medium suburban (287 stores, avg sales 6,100)
- Cluster 3: Premium locations (156 stores, avg sales 10,200)
- Cluster 4: Rural/low-traffic (240 stores, avg sales 4,900)

---

# 4. CONCLUSION

This study developed a machine learning system for retail sales forecasting using an ensemble of gradient boosting algorithms. The key contributions and findings are summarized below.

A comprehensive feature engineering pipeline was designed that extracts 45 features from historical sales data and store metadata. The features capture temporal patterns, lag relationships, rolling statistics, holiday effects, and store characteristics effectively.

An optimized ensemble model combining LightGBM and XGBoost achieves RMSPE of 0.1212, corresponding to approximately top 14-17% performance on the Kaggle benchmark. Hyperparameter optimization using Optuna Bayesian optimization and ensemble weight optimization contributed to this performance.

Feature importance and SHAP analysis identify promotional activity, lag features, and rolling statistics as the most influential predictors. These findings provide actionable insights for business decision-making.

K-Means clustering segments stores into 5 distinct behavioral profiles, supporting targeted strategies for different store types. The clustering reveals meaningful differences in sales volumes, promotional response, and customer behavior.

The developed system demonstrates that careful feature engineering combined with ensemble methods can achieve competitive forecasting performance without requiring deep learning or extensive computational resources. The methodology is reproducible and can be adapted to similar retail forecasting applications.

Future work could extend this study through external data integration such as weather forecasts and local events, neural network ensemble members, probabilistic forecasting for uncertainty quantification, and hierarchical forecasting for multi-level predictions.

---

# REFERENCES

Box, G. E., and Jenkins, G. M. (1970). Time series analysis: Forecasting and control. Holden-Day, San Francisco.

Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.

Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189-1232.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, 3146-3154.

Lundberg, S. M., and Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30, 4765-4774.

Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2623-2631.

Kaggle. (2015). Rossmann Store Sales. Retrieved from https://www.kaggle.com/c/rossmann-store-sales

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., and others. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

Hyndman, R. J., and Athanasopoulos, G. (2018). Forecasting: Principles and practice (2nd ed.). OTexts, Melbourne, Australia.

---

# CURRICULUM VITAE

**Personal Information**

Name: Kaan Bilirgen
Date of Birth: [Date]
Place of Birth: [Place]
Email: [Email]

**Education**

2020-2025: Yildiz Technical University, Faculty of Electrical and Electronics, Computer Engineering Department, Istanbul, Turkey

**Technical Skills**

Programming Languages: Python, SQL
Machine Learning: Scikit-learn, LightGBM, XGBoost, SHAP
Data Analysis: Pandas, NumPy, Matplotlib, Seaborn
Web Development: Streamlit
Version Control: Git, GitHub

**Projects**

2025: Sales Forecasting System Using Ensemble Machine Learning Methods (Undergraduate Thesis)
- Developed ensemble model achieving 0.1212 RMSPE
- Implemented feature engineering pipeline with 45 features
- Created interactive Streamlit dashboard
- Performed store clustering and SHAP analysis

**Languages**

Turkish: Native
English: Advanced
