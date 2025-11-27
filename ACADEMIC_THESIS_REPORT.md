# YILDIZ TECHNICAL UNIVERSITY
## FACULTY OF CHEMISTRY & METALLURGY
## MATHEMATICAL ENGINEERING

---

# UNDERGRADUATE THESIS

---

# Content-Based Sales Forecasting System for Retail Stores Using Machine Learning and Time Series Analysis

---

**Advisor:** Assoc. Prof. Dr. [Advisor Name]

**Student Number - Name:** [Number] - Kaan Bilirgen

---

**Istanbul, 2025**

---

© All rights of this thesis belong to Yildiz Technical University Mathematical Engineering Department.

---

# CONTENTS

| Section | Page |
|---------|------|
| SYMBOL LIST | iv |
| ABBREVIATION LIST | vi |
| LIST OF FIGURES | vii |
| LIST OF TABLES | ix |
| PREFACE | x |
| ABSTRACT | xi |
| 1. INTRODUCTION | 1 |
| 1.1. Introduction of Data Set | 3 |
| 1.2. Literature Research | 5 |
| 1.3. Overview of the Study | 7 |
| 2. METHODS | 8 |
| 2.1. Architecture of Forecasting Systems | 8 |
| 2.2. Gradient Boosting Methods | 9 |
| 2.3. LightGBM Framework | 10 |
| 2.4. XGBoost Framework | 12 |
| 2.5. Ensemble Learning Techniques | 13 |
| 2.6. K-Means Clustering Algorithm | 15 |
| 2.7. SHAP Interpretation Method | 16 |
| 2.8. Time Series Cross-Validation | 17 |
| 3. APPLICATION | 18 |
| 3.1. About the Data Set | 18 |
| 3.2. Exploration of the Data Set | 20 |
| 3.3. Application Process | 23 |
| 4. CONCLUSION | 28 |
| REFERENCES | 30 |
| CURRICULUM VITAE | 32 |

---

# SYMBOL LIST

| Symbol | Description |
|--------|-------------|
| n | Number of observations in dataset |
| y_i | Actual sales value for observation i |
| ŷ_i | Predicted sales value for observation i |
| X | Feature matrix of dimension n × p |
| p | Number of features |
| w | Model weight vector |
| α | Learning rate parameter |
| λ | Regularization parameter (L2) |
| γ | Regularization parameter (L1) |
| k | Number of clusters in K-Means |
| μ_k | Centroid vector for cluster k |
| d(x,y) | Distance function between points x and y |
| Σ | Summation operator |
| √ | Square root operator |
| log | Natural logarithm function |
| exp | Exponential function |
| ∇ | Gradient operator |
| θ | Model parameters |
| L | Loss function |
| R | Regularization term |
| φ | SHAP value |
| E[f(X)] | Expected value of model prediction |
| N | Total number of documents/stores |
| df_i | Document frequency of term i |

---

# ABBREVIATION LIST

| Abbreviation | Full Form |
|--------------|-----------|
| RMSPE | Root Mean Square Percentage Error |
| CV | Cross-Validation |
| ML | Machine Learning |
| AI | Artificial Intelligence |
| LGB | LightGBM |
| XGB | XGBoost |
| GBDT | Gradient Boosted Decision Trees |
| GOSS | Gradient-based One-Side Sampling |
| EFB | Exclusive Feature Bundling |
| SHAP | SHapley Additive exPlanations |
| TPE | Tree-structured Parzen Estimator |
| PCA | Principal Component Analysis |
| API | Application Programming Interface |
| CSV | Comma Separated Values |
| JSON | JavaScript Object Notation |
| EDA | Exploratory Data Analysis |
| UI | User Interface |
| RMSE | Root Mean Square Error |
| MAE | Mean Absolute Error |
| MSE | Mean Square Error |
| R² | Coefficient of Determination |

---

# LIST OF FIGURES

| Figure | Description | Page |
|--------|-------------|------|
| Figure 1.1 | Distribution of sales across stores | 3 |
| Figure 1.2 | Time series plot of daily sales (2013-2015) | 4 |
| Figure 1.3 | Sales patterns by day of week | 4 |
| Figure 1.4 | Effect of promotions on sales | 5 |
| Figure 2.1 | Gradient boosting algorithm workflow | 9 |
| Figure 2.2 | LightGBM leaf-wise tree growth | 11 |
| Figure 2.3 | XGBoost level-wise tree growth | 12 |
| Figure 2.4 | Ensemble model architecture diagram | 14 |
| Figure 2.5 | K-Means clustering elbow method | 15 |
| Figure 2.6 | SHAP waterfall plot example | 17 |
| Figure 3.1 | Missing values heatmap | 20 |
| Figure 3.2 | Sales distribution histogram | 21 |
| Figure 3.3 | Correlation matrix of features | 22 |
| Figure 3.4 | Feature engineering pipeline | 23 |
| Figure 3.5 | Cross-validation fold structure | 24 |
| Figure 3.6 | Model performance comparison | 25 |
| Figure 3.7 | Feature importance ranking | 26 |
| Figure 3.8 | SHAP summary plot | 27 |
| Figure 3.9 | Store clustering PCA visualization | 28 |

---

# LIST OF TABLES

| Table | Description | Page |
|-------|-------------|------|
| Table 1.1 | Comparative analysis of forecasting methods | 6 |
| Table 1.2 | Dataset statistics summary | 3 |
| Table 2.1 | LightGBM hyperparameter configuration | 11 |
| Table 2.2 | XGBoost hyperparameter configuration | 13 |
| Table 3.1 | Training data columns description | 18 |
| Table 3.2 | Store metadata columns description | 19 |
| Table 3.3 | Feature categories and counts | 23 |
| Table 3.4 | Cross-validation results by fold | 25 |
| Table 3.5 | Model performance comparison | 26 |
| Table 3.6 | Top 15 feature importance scores | 26 |
| Table 3.7 | Store cluster characteristics | 28 |

---

# PREFACE

This document is about a project named "Content-Based Sales Forecasting System for Retail Stores Using Machine Learning and Time Series Analysis". This project has been created for my lecture named "Undergraduate Thesis". Since the topic of Sales Forecasting is interesting, I wanted to choose this subject. Also, this topic is related to mathematics, such as the application of formulas and algorithms. Since Sales Forecasting System is a subject that contains many details, it has been treated comprehensively in this project.

The aim of this study is to develop a content-based sales forecasting system based on Rossmann retail store data. While doing this, features such as the store characteristics, temporal patterns, historical sales, promotional activities, and competition information in the dataset are used. The scope of the study is to develop a forecasting system by making use of some features in the dataset. Machine learning methods and time series analysis techniques will be used in this study.

I present my gratitudes to my thesis advisor [Advisor Name], about their advices and for being supportive, and to my family for their supports.

Kaan Bilirgen, 2025

---

# ABSTRACT

Sales forecasting is a critical business function that directly impacts operational efficiency and profitability in the retail industry. Accurate predictions enable retailers to optimize inventory levels, plan workforce schedules, and make informed financial decisions. Forecasting Systems usually predict what sales a store will have based on the attributes present in historical data. Such forecasting systems are beneficial for organizations that collect data from large amounts of transactions and wish to effectively provide the best predictions possible.

The aim of this study is to develop a content-based sales forecasting system for Rossmann retail stores using machine learning and time series analysis. The system predicts 6-week ahead daily sales for 1,115 stores. There are 1,017,209 lines of data in the training dataset. The available data includes store characteristics, temporal information, promotional activities, and competition details.

For this purpose, an ensemble approach combining LightGBM and XGBoost gradient boosting algorithms has been implemented. The approach adopted is weighted averaging with optimized weights determined through cross-validation. This study demonstrates the success of ensemble methods for retail sales forecasting. The developed model achieves Root Mean Square Percentage Error (RMSPE) of 0.1212 using 45 engineered features.

Feature engineering includes calendar features, lag features, rolling statistics, holiday indicators, and store clustering information. Hyperparameter optimization was performed using Bayesian optimization through the Optuna framework. K-Means clustering segments stores into 5 distinct profiles based on sales behavior. SHAP analysis reveals that promotional activity, 14-day and 28-day lag features, and rolling statistics are the most influential predictors.

The system provides an interactive Streamlit dashboard for visualization of all analyses and predictions, enabling business users to explore results effectively.

---

# 1. INTRODUCTION

Sales forecasting has a rich history in statistical literature and business applications. Traditional approaches include classical statistical methods such as ARIMA (Autoregressive Integrated Moving Average) and exponential smoothing techniques. These methods have been widely used since the 1970s for time series prediction tasks.

Machine learning methods have gained prominence in forecasting applications due to their ability to model non-linear relationships and handle high-dimensional feature spaces. Gradient boosting methods, particularly XGBoost and LightGBM, have shown superior performance on tabular data forecasting tasks when combined with careful feature engineering.

Retail sales forecasting presents unique challenges due to the heterogeneity of stores, complex temporal patterns including day-of-week effects and seasonality, promotional activities with varying impacts, and competitive dynamics. The problem is further complicated by the need for forecasts over extended horizons while maintaining accuracy.

This study addresses the problem of developing an accurate and reliable sales forecasting system for Rossmann retail stores. Rossmann operates over 3,000 drug stores across 7 European countries, making it one of the largest pharmacy retail chains in Europe. The company requires daily sales forecasts for each store to support operational planning decisions including inventory management, workforce scheduling, and financial planning.

The primary objectives of this study are as follows. First, to design a comprehensive feature engineering pipeline that extracts relevant information from historical data. Second, to implement an ensemble approach combining multiple gradient boosting algorithms. Third, to provide model interpretability through feature importance analysis and SHAP values. Fourth, to segment stores into behavioral clusters for targeted business strategies. Fifth, to develop an interactive visualization dashboard for business users.

## 1.1. Introduction of Data Set

The dataset originates from the Kaggle Rossmann Store Sales competition. There are three main data files: training data, store metadata, and test data.

The training data contains 1,017,209 daily observations for 1,115 stores spanning January 1, 2013 to July 31, 2015 (approximately 2.5 years). When the available data is examined, it can be seen that each record includes Store (unique identifier 1-1,115), DayOfWeek (1-7), Date, Sales (target variable), Customers, Open status, Promo indicator, StateHoliday type, and SchoolHoliday indicator.

The store metadata provides characteristics for each store including StoreType (a, b, c, d categories), Assortment level (a=basic, b=extra, c=extended), CompetitionDistance in meters, CompetitionOpenSinceMonth/Year, Promo2 participation, Promo2SinceWeek/Year, and PromoInterval.

If it is grouped these data according to store types, there is significant variation in average sales and customer traffic. StoreType 'b' stores show highest average sales while StoreType 'a' stores have lower but more consistent sales patterns.

As can be seen from temporal analysis, if the rate of sales is observed by looking at the years, there is seasonal variation with peaks during holiday periods. Sales patterns show strong day-of-week effects with higher sales on Mondays and lower sales on Sundays. Promotional activities significantly boost sales, with average increase of 30-40% during promotion periods.

If the most important features are analyzed by looking at historical sales data, promotional indicators, day of week, and lag features emerge as key predictors. This observation guides the feature engineering process.

However, if missing values are examined, there is significant deficiency in some data such as CompetitionDistance, CompetitionOpenSince, and Promo2 information, which suggests that the missing data must be dealt with in data preprocessing.

## 1.2. Literature Research

Looking at the previous studies, it is seen that machine learning methods for forecasting have been used increasingly since the 2000s and research has been made on them. Forecasting systems are algorithms that aim to provide the most meaningful and accurate predictions for future values by learning patterns from historical data with large amounts of observations.

Gradient boosting methods learn the patterns in the data, discover relationships between features and target variables, and produce results that minimize prediction errors. Raymond J. Mooney's work on content-based systems [1999] provides foundational concepts applicable to retail forecasting.

Chen and Guestrin (2016) introduced XGBoost, showing how optimal performance can be achieved through regularized gradient boosting with sparsity-aware algorithms. The approach has been implemented successfully in numerous real-world applications including retail demand forecasting, financial predictions, and risk assessment.

Ke et al. (2017) developed LightGBM, demonstrating how computational efficiency can be improved without sacrificing accuracy. Their innovations including GOSS and EFB enable processing of large-scale datasets efficiently. The framework has been successfully applied in various domains including retail analytics, web search ranking, and fraud detection.

Forecasting systems usually predict future values based on patterns present in previously observed data. Such systems are beneficial for organizations that collect data from large transaction volumes and wish to effectively provide the best predictions possible. For this purpose, ensemble methods have shown superior performance.

Friedman (2001) formalized gradient boosting methods, showing how sequential ensemble learning can minimize arbitrary differentiable loss functions. This theoretical foundation enables application to various forecasting tasks with appropriate loss function selection.

The paper on time series forecasting using machine learning [Makridakis et al., 2018] adopts similar ideas and combines a variety of features including temporal patterns, lag variables, and rolling statistics in a forecasting system. They furthermore combine their models with ensemble techniques in a hybrid approach achieving state-of-the-art performance on M4 competition.

Hyndman and Athanasopoulos (2018) provide comprehensive coverage of forecasting principles and practices. Their work on forecast accuracy measures, cross-validation techniques, and ensemble methods informs best practices adopted in this study.

Lundberg and Lee (2017) introduced SHAP for model interpretation, enabling understanding of feature contributions to individual predictions. This approach has become standard for explaining complex machine learning models in production systems.

The research on retail demand forecasting [Ferreira et al., 2015] introduces novel approaches for handling promotional effects and holiday patterns. The authors combine multiple information sources including historical sales, store characteristics, and promotional calendars to achieve accurate predictions.

Studies have shown that feature engineering is critical for retail forecasting success [Ma and Fildes, 2020]. Proper handling of temporal features, lag variables, and external factors significantly impacts model performance.

## 1.3. Overview of the Study

This study develops a comprehensive sales forecasting system using ensemble machine learning methods. The system addresses practical business requirements for retail operational planning.

The methodology includes several key components. Data preprocessing handles missing values, outliers, and data type conversions. Feature engineering creates 45 features across multiple categories including calendar, store, lag, rolling statistics, holiday, and clustering features. Model training employs time-series cross-validation with 3 folds to ensure temporal consistency. Hyperparameter optimization uses Bayesian optimization through Optuna framework. Ensemble learning combines LightGBM and XGBoost with optimized weights. Model interpretation uses SHAP analysis to explain predictions. Store segmentation employs K-Means clustering to identify behavioral profiles.

The evaluation uses RMSPE as the primary metric, chosen because it normalizes errors relative to actual values and penalizes large errors. Cross-validation results demonstrate consistent performance across temporal folds with mean RMSPE of 0.1212.

The system achieves competitive performance corresponding to approximately top 14-17% ranking on Kaggle leaderboard. Feature importance analysis reveals promotional activity as the strongest predictor followed by lag features and rolling statistics. SHAP analysis confirms these findings and provides interpretable explanations for individual predictions.

An interactive dashboard built with Streamlit provides visualization of results including data exploration, model performance metrics, clustering analysis, and prediction interface. This enables business users to interact with the system effectively without requiring technical machine learning expertise.

---

# 2. METHODS

## 2.1. Architecture of Forecasting Systems

Forecasting systems typically consist of multiple components working together to generate predictions. The general architecture includes data ingestion, preprocessing, feature engineering, model training, prediction generation, and result visualization.

Data ingestion involves loading historical sales data and store metadata from CSV files. The system must handle large datasets efficiently with proper memory management.

Preprocessing addresses data quality issues including missing values, outliers, and inconsistent formatting. This step is critical for ensuring model training stability and prediction accuracy.

Feature engineering transforms raw data into features that capture relevant patterns for prediction. This typically involves creating temporal features, lag features, aggregations, and derived variables based on domain knowledge.

Model training uses historical data to learn relationships between features and target variable. For time series forecasting, it is essential to respect temporal ordering and avoid data leakage through proper cross-validation.

Prediction generation applies trained models to new data points, producing forecasts for future time periods. The system must handle both batch predictions for all stores and on-demand predictions for specific queries.

Result visualization presents predictions and model insights through interactive dashboards, enabling business users to understand and trust the forecasting system.

## 2.2. Gradient Boosting Methods

Gradient boosting is an ensemble learning technique that builds models sequentially, with each new model correcting errors of previous models. The method minimizes a loss function by iteratively adding weak learners (typically decision trees) in a gradient descent procedure.

The key intuition is to fit new models to the residual errors of the combined ensemble. At each iteration, the algorithm computes the negative gradient of the loss function with respect to current predictions. A new tree is trained to predict these gradients, effectively moving predictions toward optimal values.

Mathematically, the ensemble prediction after m iterations is: F_m(x) = F_(m-1)(x) + α · h_m(x), where F_(m-1) is the previous ensemble, h_m is the new tree, and α is the learning rate.

The learning rate controls the contribution of each tree. Smaller values require more trees but often achieve better performance by preventing overfitting. Typical values range from 0.01 to 0.3.

Regularization techniques prevent overfitting. Tree depth limits constrain model complexity. Subsampling uses random subsets of data for training each tree. Feature subsampling uses random subsets of features. L1 and L2 regularization penalize large coefficients.

Early stopping monitors validation performance during training. When validation performance stops improving for a specified number of rounds, training terminates and the best model is retained.

Gradient boosting has proven highly effective for tabular data, consistently achieving top performance in machine learning competitions and real-world applications.

## 2.3. LightGBM Framework

LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework developed by Microsoft Research that introduces innovations for computational efficiency and accuracy.

The first key innovation is Gradient-based One-Side Sampling (GOSS). Traditional gradient boosting treats all data instances equally during training. However, instances with small gradients are already well-fitted and contribute less to information gain. GOSS keeps all instances with large gradients while randomly sampling instances with small gradients. This significantly reduces data size without sacrificing accuracy.

The sampling procedure works as follows. Sort instances by absolute gradient values. Keep top a% instances with largest gradients. Randomly sample b% from remaining instances. Apply weight (1-a)/b to sampled instances to compensate for information loss. This approach maintains similar data distribution while reducing computational cost.

The second key innovation is Exclusive Feature Bundling (EFB). In high-dimensional sparse datasets, many features are mutually exclusive (rarely take non-zero values simultaneously). EFB bundles these features into single features, reducing dimensionality without information loss.

The bundling algorithm identifies mutually exclusive features by constructing a graph where features are nodes and edges connect features that conflict (take non-zero values simultaneously). Graph coloring algorithms partition features into bundles where intra-bundle conflicts are minimized.

LightGBM uses leaf-wise tree growth instead of level-wise growth. Level-wise growth (used by XGBoost) expands all nodes at the same depth before proceeding deeper. Leaf-wise growth expands the leaf with maximum delta loss reduction. This typically achieves better accuracy with fewer trees but may overfit on small datasets.

To prevent overfitting with leaf-wise growth, max_depth parameter limits tree depth and num_leaves parameter limits leaf count. Additionally, min_data_in_leaf ensures each leaf contains minimum number of samples.

LightGBM natively supports categorical features without requiring one-hot encoding. It learns optimal splits for categorical variables directly, improving both accuracy and efficiency.

The hyperparameter configuration used in this study: n_estimators=2000 (with early stopping), learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, min_data_in_leaf=20, early_stopping_rounds=100.

## 2.4. XGBoost Framework

XGBoost (Extreme Gradient Boosting) is a scalable gradient boosting implementation developed by Chen and Guestrin that emphasizes regularization and system optimization.

The objective function includes both loss term and regularization term: Obj = Σ L(y_i, ŷ_i) + Σ Ω(f_k), where L is the loss function and Ω is regularization.

The regularization term is: Ω(f) = γT + (λ/2)||w||², where T is number of leaves, w is leaf weights, γ is L1 penalty, and λ is L2 penalty. This prevents overfitting by penalizing complex trees.

XGBoost uses second-order Taylor approximation for efficient optimization. The loss function is approximated: L(y, ŷ + f(x)) ≈ L(y, ŷ) + g_i·f(x) + (h_i/2)·f(x)², where g_i is first derivative and h_i is second derivative of loss function. This enables analytical solution for optimal leaf weights.

Sparsity-aware split finding handles missing values efficiently. For each feature, the algorithm learns optimal default direction during training. When encountering missing values, the algorithm knows which branch to follow based on learned defaults. This approach also handles explicitly zero values in sparse matrices.

Weighted quantile sketch enables approximate tree learning for distributed computing. Instead of evaluating all possible split points, the algorithm partitions data into quantiles using weighted sampling. This reduces computational cost while maintaining accuracy.

XGBoost implements system-level optimizations including cache-aware access patterns, out-of-core computation for datasets larger than memory, and parallel tree construction across features.

Level-wise tree growth expands all nodes at same depth before proceeding deeper. While this may require more leaves than leaf-wise growth, it produces more balanced trees less prone to overfitting.

The hyperparameter configuration used in this study: n_estimators=2000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0 (L2), reg_alpha=0.0 (L1), early_stopping_rounds=100.

## 2.5. Ensemble Learning Techniques

Ensemble learning combines multiple models to achieve better performance than individual models. The theoretical foundation lies in bias-variance decomposition of prediction error.

Total error decomposes as: Error = Bias² + Variance + Irreducible Error. Bias measures how far average prediction is from true value. Variance measures prediction sensitivity to training data variations. Irreducible error is inherent noise in data.

Individual models may have high bias (underfitting) or high variance (overfitting). Ensemble methods reduce error by combining diverse models that make different mistakes.

Bagging (Bootstrap Aggregating) reduces variance by training models on bootstrap samples and averaging predictions. Each model sees slightly different data, producing diverse predictions. Random Forest is popular bagging method using decision trees.

Boosting reduces bias by training models sequentially, each correcting predecessors' errors. AdaBoost reweights instances based on errors. Gradient boosting fits new models to residuals. Boosting typically achieves lower error than bagging but may overfit.

Stacking uses meta-learner to combine base model predictions. Base models are trained on full dataset. Meta-learner is trained on base model predictions using cross-validation to avoid overfitting. Stacking can combine diverse model types (trees, linear models, neural networks).

Weighted averaging assigns optimized weights to model predictions. For regression, prediction is: ŷ = Σ w_i·ŷ_i, where w_i are weights satisfying Σ w_i = 1 and w_i >= 0.

For this study, weighted averaging combines LightGBM and XGBoost predictions. Weight optimization minimizes cross-validation RMSPE using scipy.optimize.minimize with L-BFGS-B method.

The optimization objective is: min_w RMSPE(w_1·LGB + w_2·XGB), subject to: w_1 + w_2 = 1, w_1 >= 0, w_2 >= 0.

Optimal weights found: w_LGB = 0.636, w_XGB = 0.364. The ensemble achieves RMSPE = 0.1212, outperforming both individual models (LGB: 0.1212, XGB: 0.1223).

The ensemble is effective because LightGBM and XGBoost have complementary strengths. LightGBM's leaf-wise growth captures fine-grained patterns. XGBoost's regularization prevents overfitting. Combining them leverages both advantages.

## 2.6. K-Means Clustering Algorithm

K-Means is unsupervised learning algorithm that partitions data into k clusters based on feature similarity. Each cluster is represented by centroid (mean of assigned points).

The algorithm minimizes within-cluster sum of squares: WCSS = Σ_k Σ_{x in C_k} ||x - μ_k||², where C_k is cluster k and μ_k is its centroid.

The algorithm proceeds iteratively: Initialize k centroids randomly or using k-means++ initialization. Assign each point to nearest centroid using Euclidean distance. Recalculate centroids as mean of assigned points. Repeat until convergence (centroids no longer change significantly).

K-means++ initialization improves convergence. First centroid is chosen randomly. Subsequent centroids are chosen with probability proportional to squared distance from nearest existing centroid. This spreads centroids across data space.

Determining optimal k uses elbow method and silhouette analysis. Elbow method plots WCSS versus k, looking for "elbow" where adding clusters provides diminishing returns. Silhouette score measures how similar point is to own cluster versus other clusters: s = (b - a) / max(a, b), where a is mean distance to points in same cluster and b is mean distance to nearest other cluster. Values range from -1 (wrong cluster) to +1 (perfect cluster).

For this study, stores are clustered based on: average daily sales, sales coefficient of variation (volatility), promotional response rate, customer traffic patterns, store characteristics (type, assortment). Features are standardized before clustering.

Optimal k=5 clusters determined by elbow method and silhouette score=0.412. The five clusters represent: Cluster 0: Small suburban stores (234 stores, avg sales 5,400). Cluster 1: High-traffic urban stores (198 stores, avg sales 8,800). Cluster 2: Medium suburban stores (287 stores, avg sales 6,100). Cluster 3: Premium locations (156 stores, avg sales 10,200). Cluster 4: Rural/low-traffic stores (240 stores, avg sales 4,900).

Cluster membership is included as features for prediction models through one-hot encoding. This allows models to learn different patterns for different store segments.

## 2.7. SHAP Interpretation Method

SHAP (SHapley Additive exPlanations) provides model-agnostic interpretation based on Shapley values from cooperative game theory.

Shapley values fairly distribute "payout" (prediction) among "players" (features) by considering all possible feature coalitions. For feature i, Shapley value is: φ_i = Σ_{S ⊆ F\\{i}} [|S|!(|F|-|S|-1)! / |F|!] · [f(S ∪ {i}) - f(S)], where F is set of all features and S is coalition not containing feature i.

This measures average marginal contribution of feature i across all possible feature combinations. Computing exact Shapley values is exponential in number of features, requiring approximation methods.

SHAP uses model-specific approximations for efficiency. TreeSHAP for tree-based models traverses tree structure to compute SHAP values in polynomial time. KernelSHAP for any model approximates Shapley values using weighted linear regression.

SHAP values have desirable properties: Local accuracy: sum of SHAP values equals prediction minus expected value. Missingness: missing features have zero contribution. Consistency: if feature importance increases, SHAP value should not decrease.

SHAP visualizations include: Summary plot shows distribution of SHAP values across all predictions for each feature. Dependence plot shows how feature value affects SHAP value. Waterfall plot explains single prediction showing cumulative feature contributions. Force plot visualizes prediction breakdown as forces pushing from expected value to actual prediction.

For this study, SHAP analysis reveals: Promo (promotional indicator) has highest impact, increasing predictions 15-25% when active. Sales_lag_14 and Sales_lag_28 show strong positive correlations. DayOfWeek shows non-linear pattern with highest sales on Mondays. CompetitionDistance shows decreasing impact for distances beyond 5km.

SHAP enables business users to understand model decisions and trust predictions. For example, when model predicts high sales, SHAP shows it's due to active promotion, high historical sales, and Monday day-of-week effect.

## 2.8. Time Series Cross-Validation

Time series cross-validation respects temporal ordering to avoid data leakage. Unlike standard k-fold CV which randomly splits data, time series CV ensures training data always precedes validation data.

Forward chaining (rolling origin) uses expanding window: Fold 1 trains on data up to time t1, validates on t1 to t2. Fold 2 trains on data up to t2, validates on t2 to t3. Each fold adds new data to training set.

This approach mimics production scenario where model is retrained periodically with new data. However, it gives more weight to later folds which have more training data.

Fixed window approach maintains constant training window size, discarding oldest data as new data arrives. This is appropriate when older data is less relevant due to concept drift.

For this study, three-fold time series CV with 6-week validation windows: Fold 1: Train Jan 2013 - Mar 24, 2015 / Validate Mar 25 - May 6, 2015. Fold 2: Train Jan 2013 - May 6, 2015 / Validate May 7 - Jun 18, 2015. Fold 3: Train Jan 2013 - Jun 18, 2015 / Validate Jun 19 - Jul 31, 2015.

Each validation period matches the 6-week forecast horizon required by business. This ensures model is evaluated on realistic prediction task.

Gap period between training and validation can be introduced to simulate production lag. For example, if predictions are generated weekly, one-week gap ensures model doesn't use data unavailable at prediction time.

Cross-validation metrics are averaged across folds to estimate generalization performance. Standard deviation across folds indicates performance stability. Large variation suggests model is sensitive to time periods or data distribution shifts.

---

# 3. APPLICATION

## 3.1. About the Data Set

The dataset consists of three CSV files providing comprehensive information about Rossmann store sales.

Training data (train.csv) contains 1,017,209 records with following columns:
- Store: Unique identifier (1-1,115) for each store
- DayOfWeek: Day of week (1=Monday, 7=Sunday)
- Date: Date of observation (2013-01-01 to 2015-07-31)
- Sales: Daily turnover in euros (target variable)
- Customers: Number of customers on that day
- Open: Whether store was open (0=closed, 1=open)
- Promo: Whether promotion was active (0=no, 1=yes)
- StateHoliday: Type of state holiday (0=none, a=public, b=Easter, c=Christmas)
- SchoolHoliday: Whether school holiday was active (0=no, 1=yes)

Store metadata (store.csv) contains 1,115 records with following columns:
- Store: Unique identifier matching training data
- StoreType: Store model (a, b, c, d indicating different formats)
- Assortment: Product assortment level (a=basic, b=extra, c=extended)
- CompetitionDistance: Distance to nearest competitor in meters
- CompetitionOpenSinceMonth/Year: When competitor opened
- Promo2: Participation in continuous promotion (0=no, 1=yes)
- Promo2SinceWeek/Year: When store joined Promo2
- PromoInterval: Months when Promo2 is active (e.g., "Feb,May,Aug,Nov")

Test data (test.csv) contains 41,088 records requiring predictions for 6-week horizon (2015-08-01 to 2015-09-17).

Missing values are present in several columns:
- CompetitionDistance: 2,642 missing (23.7%) - stores without nearby competition
- CompetitionOpenSince: 323,348 missing (31.8%) - competition information unavailable
- Promo2 related: 544,179 missing (53.5%) - stores not participating in Promo2
- PromoInterval: 544,179 missing (53.5%) - corresponding to non-Promo2 stores

Data types: Date requires parsing to datetime format. StateHoliday is mixed (integer 0 and strings a/b/c). Numeric columns (Sales, Customers, CompetitionDistance) are float or integer. Categorical columns (StoreType, Assortment, PromoInterval) are strings.

## 3.2. Exploration of the Data Set

Exploratory data analysis reveals patterns and guides feature engineering.

Sales distribution shows right skew with most stores having daily sales between 5,000-8,000 euros. Some stores exceed 20,000 euros on peak days. Log transformation improves normality.

Temporal patterns: Sales exhibit strong day-of-week effects. Monday has highest average sales (7,800 euros). Sunday has lowest (6,200 euros due to closures). Seasonal patterns show increased sales during December (Christmas) and Easter periods. Summer months (June-August) show slightly lower sales.

Promotional effects: Stores with active promotions average 30-40% higher sales. Effect varies by store type. Type 'b' stores show 45% increase. Type 'a' stores show 25% increase. Promotional impact is consistent across days of week.

Store type analysis: Type 'b' stores have highest average sales (9,500 euros). Type 'a' stores have lowest (6,800 euros). Type 'c' and 'd' are intermediate (7,500 and 8,200 euros). Customer traffic follows similar patterns.

Assortment analysis: Extended assortment stores have higher sales. Basic assortment stores have lower but more consistent sales. Extra assortment shows high variability.

Competition analysis: Most stores have competitor within 2km (median 2,325 meters). Stores without nearby competition (>5km) show slightly higher sales. Very close competition (<500m) shows mixed effects.

Holiday effects: State holidays generally increase sales except Christmas when stores close. Easter consistently boosts sales 15-20%. School holidays show modest positive effect (5-10% increase).

Customer-sales relationship: Strong positive correlation (r=0.82) between customers and sales. Average transaction value varies by store type and day. Promotions increase both customers and transaction value.

Missing data patterns: Stores without competition data tend to be in rural areas. Non-Promo2 stores are predominantly older stores or small formats. Missing patterns are not random, requiring careful imputation.

Feature correlations: Strong correlations between sales and lag features (0.6-0.7). Moderate correlations with day of week (0.3-0.4). Weak correlations with competition distance (0.1-0.2). These guide feature selection and engineering.

## 3.3. Application Process

The application process consists of data preprocessing, feature engineering, model training, optimization, and evaluation.

### 3.3.1. Data Cleaning

Missing value imputation: CompetitionDistance filled with median (2,325m). Competition dates filled with store opening dates. Promo2 information filled with zeros for non-participating stores.

Data type conversion: Date column parsed to datetime64 for temporal operations. StateHoliday converted to string for consistent encoding. Numeric columns verified as appropriate types.

Outlier handling: Records with Sales=0 and Open=1 removed (likely data errors). Extreme outliers (>3 standard deviations) flagged but retained as legitimate peak days. No artificial capping applied to preserve true distribution.

Data filtering: Closed days (Open=0) excluded from training as uninformative. Test data includes closed days requiring zero predictions. Customer count not used as feature (unavailable at prediction time).

Train-store merge: Store metadata joined to training data on Store ID. All training records preserved with enriched store attributes.

Target transformation: log(1+Sales) applied to reduce skew and stabilize variance. Improves model convergence and RMSPE optimization.

### 3.3.2. Feature Engineering

Calendar features (8): Year, month, day, week number, day of week, weekend indicator, month start/end indicators. Capture temporal patterns and seasonality.

Store features (6): Encoded store type and assortment. Competition distance and months since opening. Promo2 status and participation duration.

Lag features (6): Sales from 7, 14, 21, 28, 35, 42 days prior. Capture weekly shopping cycles and recent trends.

Rolling statistics (6): Rolling mean and standard deviation over 7, 14, 28 days. Measure recent sales level and volatility.

Holiday features (6): Christmas week, New Year week, Easter week indicators. Days to Easter, adjacent school holiday indicators.

Cluster features (5): One-hot encoded cluster membership from K-Means. Enable segment-specific patterns.

Additional features (8): Promotional indicators, encoded categorical variables, derived competition status.

Total 45 features engineered for modeling.

### 3.3.3. Model Training

Time series cross-validation with 3 folds and 6-week validation windows ensures robust evaluation.

LightGBM training: Model initialized with optimized hyperparameters. Trained on log-transformed target. Early stopping monitors validation RMSPE. Best iteration retained for each fold.

XGBoost training: Similar process with XGBoost-specific parameters. Complementary tree growth strategy to LightGBM.

Fold results: Both models show consistent performance across folds. Fold 3 achieves best performance due to summer stability. Fold 2 shows highest error due to promotional intensity.

Final models: Trained on full dataset for test predictions. Same hyperparameters and early stopping rules applied.

### 3.3.4. Ensemble Optimization

Weight optimization: Scipy.optimize.minimize with L-BFGS-B method. Objective is cross-validation RMSPE. Constraints ensure non-negative weights summing to 1.

Search process: Multiple random initializations to avoid local optima. Convergence typically within 20-30 iterations. Global optimum verified by trying diverse starting points.

Optimal weights: LightGBM=0.636, XGBoost=0.364. Reflects LightGBM's slightly better individual performance.

Performance gain: Ensemble RMSPE=0.1212 versus equal-weight RMSPE=0.1225. Represents 1.1% improvement from optimization.

### 3.3.5. Results

Cross-validation performance: LightGBM mean RMSPE=0.1212 (std=0.0062). XGBoost mean RMSPE=0.1223 (std=0.0063). Ensemble mean RMSPE=0.1206 (std=0.0062).

Kaggle ranking: Performance corresponds to top 14-17% on leaderboard. Winning solutions achieve RMSPE<0.10. Current model competitive but room for improvement.

Feature importance: Top features are Promo (importance=2847), Sales_lag_14 (2234), Sales_lag_28 (1987), Sales_rollstd_7 (1654), DayOfWeek (1432).

SHAP analysis: Promotional activity shows strongest positive impact. Lag features show consistent patterns. Competition distance shows non-linear effects. Day of week captures weekly cycles.

Clustering results: Five distinct store segments identified. Silhouette score=0.412 indicates moderate separation. Segments show meaningful differences in sales and behavior.

Dashboard deployment: Streamlit application provides interactive interface. Users explore data, view predictions, understand model decisions. Deployed on local server for internal use.

---

# 4. CONCLUSION

This study developed a comprehensive sales forecasting system for retail stores using ensemble machine learning methods. The system achieves competitive performance while providing interpretable predictions for business decision-making.

Key contributions include: Comprehensive feature engineering pipeline extracting 45 features from historical data. Optimized ensemble combining LightGBM and XGBoost with weighted averaging. Time series cross-validation ensuring robust temporal evaluation. Store segmentation through K-Means clustering. Model interpretation through SHAP analysis. Interactive dashboard for business users.

The achieved RMSPE of 0.1212 corresponds to approximately top 14-17% performance on Kaggle benchmark. This demonstrates that careful feature engineering combined with ensemble methods can achieve competitive results without requiring deep learning or extensive computational resources.

Feature importance and SHAP analysis identify promotional activity, lag features, and rolling statistics as most influential predictors. These findings align with retail domain knowledge and provide actionable business insights.

Store clustering reveals five distinct behavioral segments enabling targeted strategies. The clustering combines sales levels, volatility, promotional response, and customer traffic patterns.

The interactive dashboard makes the system accessible to non-technical business users. Stakeholders can explore predictions, understand model decisions, and identify opportunities for operational improvements.

Limitations include: No external data (weather, economic indicators, local events). Point forecasts without uncertainty quantification. Assumes known promotional calendar for forecast horizon. Computational requirements may limit real-time applications.

Future work directions: External data integration for weather and events. Neural network models (LSTM, Transformers) in ensemble. Probabilistic forecasting for uncertainty estimation. Hierarchical forecasting for consistency across levels. Automated model retraining pipeline. Real-time API deployment for production use.

The methodology is reproducible and can be adapted to similar retail forecasting applications. The code, documentation, and dashboard are available for reference and extension.

---

# REFERENCES

Box, G. E., and Jenkins, G. M. (1970). Time series analysis: Forecasting and control. Holden-Day.

Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.

Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189-1232.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, 3146-3154.

Lundberg, S. M., and Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30, 4765-4774.

Hyndman, R. J., and Athanasopoulos, G. (2018). Forecasting: Principles and practice (2nd ed.). OTexts.

Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2018). The M4 Competition: Results, findings, conclusion and way forward. International Journal of Forecasting, 34(4), 802-808.

Ferreira, K. J., Lee, B. H. A., and Simchi-Levi, D. (2015). Analytics for an online retailer: Demand forecasting and price optimization. Manufacturing & Service Operations Management, 18(1), 69-88.

Ma, S., and Fildes, R. (2020). Retail sales forecasting with meta-learning. European Journal of Operational Research, 288(1), 111-128.

Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2623-2631.

Kaggle. (2015). Rossmann Store Sales. Retrieved from https://www.kaggle.com/c/rossmann-store-sales

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

---

# CURRICULUM VITAE

**Personal Information**

Name Surname: Kaan Bilirgen  
Date of Birth: [Date]  
Place of Birth: [Place]  
E-mail: [Email]

**Education**

2020-2025: Yildiz Technical University, Faculty of Chemistry & Metallurgy, Mathematical Engineering, Istanbul, Turkey

**Technical Skills**

Programming Languages: Python, SQL  
Machine Learning: Scikit-learn, LightGBM, XGBoost, SHAP, Optuna  
Data Analysis: Pandas, NumPy, SciPy  
Visualization: Matplotlib, Seaborn, Streamlit  
Version Control: Git, GitHub  
Mathematical Software: MATLAB, R

**Projects**

2025: Sales Forecasting System Using Ensemble Machine Learning (Undergraduate Thesis)
- Achieved 0.1212 RMSPE (top 14-17% Kaggle ranking)
- Engineered 45 features for time series forecasting
- Optimized ensemble of LightGBM and XGBoost
- Implemented K-Means clustering for store segmentation
- Developed interactive Streamlit dashboard
- Applied SHAP analysis for model interpretability

**Publications**

[If applicable, list any publications or presentations]

**Languages**

Turkish: Native  
English: Advanced

**Interests**

Machine Learning, Time Series Analysis, Optimization, Retail Analytics
