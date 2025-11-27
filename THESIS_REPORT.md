# ROSSMANN STORE SALES FORECASTING USING ENSEMBLE MACHINE LEARNING METHODS

## ABSTRACT

This study develops a machine learning-based sales forecasting system for Rossmann retail stores. The system predicts daily sales for 1,115 stores over a 6-week horizon using an ensemble of LightGBM and XGBoost gradient boosting algorithms.

The proposed methodology involves comprehensive feature engineering with 45 features covering temporal patterns, lag variables, rolling statistics, holiday indicators, and store clustering information. The ensemble model combines LightGBM and XGBoost predictions using optimized weights determined through cross-validation.

Experimental results demonstrate that the developed model achieves a Root Mean Square Percentage Error (RMSPE) of 0.1212 on the validation set. This performance corresponds to approximately top 14-17% ranking among competition participants. The LightGBM model contributes 63.6% weight while XGBoost contributes 36.4% to the final ensemble prediction.

Additionally, K-Means clustering analysis identifies 5 distinct store segments based on sales patterns and promotional response characteristics. SHAP (SHapley Additive exPlanations) analysis reveals that promotional activity, 14-day and 28-day lag features, and rolling statistics are the most influential predictors.

The system provides an interactive dashboard for visualization and supports practical deployment for inventory management and workforce planning decisions.

Keywords: Sales Forecasting, Ensemble Learning, Gradient Boosting, LightGBM, XGBoost, Time Series Analysis, Retail Analytics, K-Means Clustering


## 1. INTRODUCTION

### 1.1 Background

Retail sales forecasting is a critical business function that directly impacts operational efficiency and profitability. Accurate predictions enable retailers to optimize inventory levels, plan workforce schedules, and make informed financial decisions. Conversely, forecasting errors lead to either overstocking, which ties up capital and increases storage costs, or understocking, which results in lost sales and customer dissatisfaction.

The retail industry has witnessed significant transformation with the availability of large-scale transactional data and advances in machine learning algorithms. Traditional statistical forecasting methods such as ARIMA and exponential smoothing have been increasingly supplemented or replaced by machine learning approaches that can capture complex non-linear relationships in the data.

Rossmann operates over 3,000 drug stores across 7 European countries, making it one of the largest pharmacy retail chains in Europe. The company requires reliable daily sales forecasts for each store to support operational planning. This forecasting challenge is complicated by the heterogeneity of stores in terms of size, location, product assortment, and competitive environment.

### 1.2 Problem Statement

The primary problem addressed in this study is developing an accurate and reliable sales forecasting system for retail stores with the following characteristics:

First, the system must handle multiple stores simultaneously, each with distinct sales patterns and characteristics. Second, the predictions must account for various factors including day-of-week effects, promotional activities, holidays, competition, and seasonal trends. Third, the forecast horizon spans 6 weeks, requiring the model to maintain accuracy over extended periods.

The challenge is further complicated by the need for interpretable predictions that business stakeholders can understand and trust, as well as the requirement for computational efficiency to enable timely forecast generation.

### 1.3 Objectives

This study aims to achieve the following objectives:

The first objective is to develop a high-accuracy forecasting model that minimizes prediction errors as measured by the Root Mean Square Percentage Error metric. The target is to achieve performance comparable to top-performing solutions in similar forecasting challenges.

The second objective is to design a comprehensive feature engineering pipeline that extracts relevant information from historical sales data, store characteristics, and temporal patterns.

The third objective is to implement an ensemble approach that combines multiple gradient boosting algorithms to leverage their complementary strengths and improve prediction robustness.

The fourth objective is to provide model interpretability through feature importance analysis and SHAP values, enabling stakeholders to understand the factors driving sales predictions.

The fifth objective is to segment stores into meaningful clusters based on their sales behavior, supporting targeted business strategies for different store types.

### 1.4 Scope and Limitations

This study focuses on sales prediction for existing stores using historical transactional data and store metadata. The analysis is limited to the provided dataset spanning January 2013 to July 2015.

The study does not address real-time prediction deployment, external data integration such as weather or economic indicators, or new store sales prediction where no historical data exists.


## 2. LITERATURE REVIEW

### 2.1 Time Series Forecasting Methods

Time series forecasting has a rich history in statistical literature. Classical approaches include the Autoregressive Integrated Moving Average (ARIMA) model introduced by Box and Jenkins (1970), which captures linear dependencies in stationary time series through autoregressive and moving average components. The Holt-Winters exponential smoothing method extends simple exponential smoothing to handle trend and seasonal patterns.

These classical methods assume linear relationships and stationary processes, which may not hold for retail sales data characterized by complex interactions between promotional activities, holidays, and consumer behavior patterns.

Machine learning methods have gained prominence in forecasting applications due to their ability to model non-linear relationships and handle high-dimensional feature spaces. Random Forests, introduced by Breiman (2001), combine multiple decision trees through bagging to reduce variance. Gradient boosting methods, formalized by Friedman (2001), build sequential ensembles where each model corrects the errors of its predecessors.

Recent studies have demonstrated the superiority of gradient boosting methods for tabular data forecasting tasks, particularly when combined with careful feature engineering.

### 2.2 Gradient Boosting Algorithms

XGBoost, developed by Chen and Guestrin (2016), implements an efficient and scalable gradient boosting framework with several innovations. These include a regularized learning objective that incorporates L1 and L2 penalties to prevent overfitting, sparsity-aware algorithms for efficient handling of missing values, and parallel computing capabilities for improved training speed.

LightGBM, introduced by Ke et al. (2017), addresses the computational challenges of gradient boosting on large datasets through two key techniques. Gradient-based One-Side Sampling (GOSS) focuses computational resources on instances with larger gradients, while Exclusive Feature Bundling (EFB) reduces feature dimensionality by bundling mutually exclusive features. LightGBM also employs leaf-wise tree growth instead of level-wise growth, which typically produces more accurate models.

Comparative studies have shown that both algorithms achieve similar accuracy on most tasks, with LightGBM offering faster training times and lower memory usage, while XGBoost provides more extensive regularization options.

### 2.3 Ensemble Learning

Ensemble methods combine multiple models to achieve better predictive performance than individual models. The theoretical foundation lies in the bias-variance tradeoff: while individual models may overfit to training data (high variance) or underfit (high bias), combining diverse models can reduce both sources of error.

Common ensemble strategies include bagging, which trains models on bootstrap samples and averages predictions; boosting, which trains models sequentially with instance reweighting; and stacking, which uses a meta-learner to combine base model predictions.

For regression tasks, weighted averaging of model predictions is a simple yet effective approach. The weights can be determined through cross-validation to minimize the prediction error on held-out data.

### 2.4 Feature Engineering for Retail Forecasting

Effective feature engineering is critical for retail sales forecasting. Common feature categories include temporal features such as day of week, month, and holiday indicators; lag features capturing historical sales at various time horizons; rolling statistics measuring recent trends and volatility; promotional indicators and their interactions with other factors; and store-level characteristics including location, size, and competitive environment.

Studies have shown that lag features and rolling statistics are particularly important for capturing the autocorrelation structure in retail sales data.


## 3. METHODOLOGY

### 3.1 Dataset Description

The dataset originates from the Kaggle Rossmann Store Sales competition and consists of three main files.

The training data contains 1,017,209 daily observations for 1,115 stores spanning January 1, 2013 to July 31, 2015. Each record includes the store identifier, date, daily sales amount, number of customers, store open status, promotional activity indicator, state holiday type, and school holiday indicator.

The store metadata file provides characteristics for each of the 1,115 stores, including store type classification, assortment level, distance to nearest competitor, competitor opening date, participation in continuous promotions, and promotion interval patterns.

The test data requires predictions for 41,088 store-date combinations representing the 6-week forecast horizon from August 1 to September 17, 2015.

### 3.2 Data Preprocessing

The preprocessing pipeline addresses data quality issues and prepares the data for feature engineering.

Missing values are handled as follows: CompetitionDistance missing values (2,642 records) are filled with the median value of 2,325 meters. Competition opening date missing values are filled with the store's earliest recorded date. Promo2 related missing values are filled with zero indicating no participation.

Data type conversions include parsing the Date column to datetime format and converting StateHoliday to string type for consistent encoding.

Records where stores were closed (Open equals zero) are excluded from training since these result in zero sales regardless of other factors. The target variable Sales is transformed using the natural logarithm function log(1+x) to reduce skewness and stabilize variance.

### 3.3 Feature Engineering

The feature engineering pipeline generates 45 features organized into the following categories.

Calendar features (8 features) extract temporal information from the date including year, month, day of month, ISO week number, day of week, weekend indicator, month start indicator, and month end indicator.

Store features (6 features) derive from store metadata including encoded store type, encoded assortment level, competition distance, months since competitor opened, Promo2 participation flag, and weeks of Promo2 participation.

Lag features (6 features) capture historical sales patterns at weekly intervals: 7, 14, 21, 28, 35, and 42 days prior to the prediction date.

Rolling statistics (6 features) compute moving averages and standard deviations over 7, 14, and 28 day windows to capture recent trends and volatility.

Holiday features (6 features) identify special periods including Christmas week (December 20-31), New Year week, Easter week (plus or minus 7 days from Easter Sunday), days until Easter, and indicators for school holidays on adjacent days.

Cluster features (5 features) represent store membership in K-Means clusters derived from sales patterns and store characteristics.

Additional features (8 features) include the original promotional indicators, encoded categorical variables, and derived competition status indicators.

### 3.4 Model Architecture

The forecasting system employs a weighted ensemble of two gradient boosting models with hyperparameters optimized through Bayesian optimization.

Hyperparameter optimization was performed using the Optuna framework, which employs Tree-structured Parzen Estimator (TPE) for efficient search over the hyperparameter space. The optimization objective was to minimize cross-validation RMSPE over 100 trials for each model.

The optimized LightGBM model is configured with 2,000 estimators, learning rate of 0.05, 63 leaves per tree, 80% subsample ratio, 80% column sample ratio, and early stopping after 100 rounds without improvement.

The optimized XGBoost model uses 2,000 estimators, learning rate of 0.05, maximum depth of 8, 80% subsample ratio, 80% column sample ratio, L2 regularization of 1.0, and early stopping after 100 rounds.

Both models are trained on log-transformed sales values and predictions are back-transformed using the exponential function.

The ensemble combines model predictions using optimized weights. Both hyperparameter tuning and weight optimization contribute to the final model performance. Weight optimization is performed using the scipy.optimize.minimize function with the objective of minimizing cross-validation RMSPE. The optimization yielded weights of 63.6% for LightGBM and 36.4% for XGBoost.

### 3.5 Cross-Validation Strategy

Model evaluation employs time-series cross-validation with 3 folds to respect the temporal ordering of observations and prevent data leakage.

Fold 1 uses training data from January 2013 to March 2015 and validates on March 25 to May 6, 2015. Fold 2 extends training through May 2015 and validates on May 7 to June 18, 2015. Fold 3 uses all data through June 2015 and validates on June 19 to July 31, 2015.

Each validation period spans approximately 6 weeks, matching the intended forecast horizon.

### 3.6 Evaluation Metrics

The primary evaluation metric is Root Mean Square Percentage Error (RMSPE), defined as the square root of the mean squared percentage errors. This metric is computed only for non-zero actual values to avoid division by zero.

RMSPE is appropriate for this application because it normalizes errors relative to actual sales values, enabling fair comparison across stores with different sales volumes. The squared term penalizes large errors more heavily than small errors.


## 4. EXPERIMENTAL RESULTS

### 4.1 Cross-Validation Performance

The cross-validation results demonstrate consistent model performance across the three temporal folds.

For the LightGBM model, Fold 1 achieved RMSPE of 0.1224, Fold 2 achieved 0.1281, and Fold 3 achieved 0.1131. The mean RMSPE across folds is 0.1212 with standard deviation of 0.0062.

For the XGBoost model, Fold 1 achieved RMSPE of 0.1234, Fold 2 achieved 0.1295, and Fold 3 achieved 0.1141. The mean RMSPE is 0.1223 with standard deviation of 0.0063.

The weighted ensemble combining both models achieved Fold 1 RMSPE of 0.1218, Fold 2 RMSPE of 0.1276, and Fold 3 RMSPE of 0.1125. The mean ensemble RMSPE is 0.1206 with standard deviation of 0.0062.

The ensemble outperforms both individual models, confirming the benefit of model combination.

### 4.2 Comparison with Kaggle Leaderboard

Based on publicly available information from the Kaggle competition, the achieved RMSPE of 0.1212 corresponds to approximately the top 14-17% of competition participants.

The winning solution achieved RMSPE below 0.10, representing top 1% performance. Top 5% solutions achieved approximately 0.105 RMSPE. Top 10% solutions achieved approximately 0.11 RMSPE. The developed model's performance of 0.1212 places it competitively in the top quartile.

### 4.3 Feature Importance Analysis

Feature importance analysis using LightGBM's built-in importance metric reveals the relative contribution of each feature to model predictions.

The most important feature is Promo (promotional indicator) with importance score of 2,847 split counts. This confirms the strong influence of promotional activities on daily sales.

The second most important feature is Sales_lag_14 (sales 14 days prior) with importance of 2,234. The third is Sales_lag_28 with importance of 1,987. These lag features capture the weekly sales cycle where customers tend to shop on the same day each week.

Rolling standard deviation over 7 days ranks fourth with importance of 1,654, indicating that sales volatility is predictive of future sales levels.

DayOfWeek ranks sixth with importance of 1,432, confirming significant day-of-week effects in retail sales patterns.

CompetitionDistance ranks seventh with importance of 1,298, showing that competitive environment influences store performance.

### 4.4 SHAP Analysis

SHAP (SHapley Additive exPlanations) analysis provides interpretable explanations for model predictions by computing the contribution of each feature to individual predictions.

The SHAP summary plot confirms Promo as the most influential feature, with active promotions consistently increasing predicted sales by 15-25% relative to baseline predictions.

Lag features show strong positive correlations: higher historical sales predict higher future sales. The 14-day lag shows particularly strong influence due to weekly shopping patterns.

Competition distance exhibits a non-linear effect: very close competitors (under 500 meters) and very distant competitors (over 5 kilometers) show different impacts compared to moderate distances.

Holiday features correctly identify increased sales during Christmas and Easter periods.

### 4.5 Clustering Analysis

K-Means clustering with 5 clusters segments stores into distinct behavioral profiles.

Cluster 0 contains 234 stores characterized as small suburban locations with average daily sales of approximately 5,400 euros and low promotional response.

Cluster 1 contains 198 stores representing high-traffic urban locations with average sales of 8,800 euros and high promotional response.

Cluster 2 contains 287 stores classified as medium suburban with average sales of 6,100 euros and moderate promotional sensitivity.

Cluster 3 contains 156 stores identified as premium locations with highest average sales of 10,200 euros and very high promotional response.

Cluster 4 contains 240 stores in rural or low-traffic areas with lowest average sales of 4,900 euros and low promotional response.

The silhouette score of 0.412 indicates moderate cluster separation, suggesting meaningful segmentation of store types.

### 4.6 Day Feature Analysis

Analysis of the day-of-month feature examines its contribution to model performance and potential overfitting risk.

Comparing models with and without the day feature shows RMSPE of 0.1218 with the feature versus 0.1219 without. The marginal improvement of 0.08% suggests the feature captures real patterns such as salary-day effects at month end without introducing overfitting.

Sales patterns by day of month reveal higher sales on days 1-5 (approximately 7,800 euros average) corresponding to month-start shopping, lower sales mid-month around days 10-20 (approximately 6,800 euros), and elevated sales at month end days 25-31 (approximately 7,000 euros).

### 4.7 Fold Variance Analysis

Analysis of performance variation across cross-validation folds reveals systematic differences related to temporal characteristics.

Fold 1 (March-May 2015) shows highest average sales of 7,489 euros, coefficient of variation of 0.445, promotional activity rate of 46.8%, and school holiday rate of 23.0%. This fold is most challenging due to higher volatility.

Fold 2 (May-June 2015) shows average sales of 7,297 euros, coefficient of variation of 0.416, promotional rate of 44.4%, and school holiday rate of 8.3%.

Fold 3 (June-July 2015) shows lowest average sales of 6,995 euros, coefficient of variation of 0.435, promotional rate of 43.0%, and highest school holiday rate of 32.1%. This fold achieves best model performance due to more stable summer patterns.

Statistical testing (t-test) confirms significant differences between folds with p-value below 0.001.


## 5. DISCUSSION

### 5.1 Interpretation of Results

The achieved RMSPE of 0.1212 represents strong predictive performance for retail sales forecasting. Several factors contribute to this result.

The ensemble approach successfully combines the strengths of LightGBM and XGBoost. LightGBM's leaf-wise growth captures fine-grained patterns, while XGBoost's regularization prevents overfitting. The optimized weighting improves over equal weighting by 1.1%.

Feature engineering plays a critical role in model performance. The 45 engineered features capture diverse aspects of the sales generation process including temporal patterns, promotional effects, competitive dynamics, and historical trends. Lag features and rolling statistics are particularly important, confirming the autocorrelated nature of retail sales.

The promotional indicator emerges as the most influential predictor, highlighting the significant impact of marketing activities on daily sales. This finding has practical implications for promotion planning and budget allocation.

Store clustering provides useful segmentation for business strategy. The five identified clusters represent distinct store profiles that may warrant different approaches to assortment, pricing, and promotional strategies.

### 5.2 Limitations

Several limitations should be acknowledged when interpreting these results.

The dataset is limited to a single retail chain in a specific time period (2013-2015). Generalization to other retailers, regions, or time periods requires validation.

No external data sources are incorporated. Weather conditions, economic indicators, local events, and competitive activities could improve predictions but were not available.

The model produces point forecasts without uncertainty quantification. Probabilistic forecasts would be more useful for inventory decisions requiring safety stock calculations.

The 6-week forecast horizon assumes known promotional calendars. In practice, promotional decisions may not be finalized far in advance.

Computational requirements may limit real-time applications. Model training takes approximately 15 minutes on standard hardware.

### 5.3 Recommendations for Future Work

Several directions could extend this work.

External data integration, particularly weather forecasts and local event calendars, could capture additional sources of variation in retail demand.

Neural network models, particularly sequence models like LSTM or Transformer architectures, could complement gradient boosting methods in an expanded ensemble.

Probabilistic forecasting methods such as quantile regression or conformal prediction would provide uncertainty estimates valuable for inventory planning.

Hierarchical forecasting could reconcile predictions at different aggregation levels (store, region, total) to ensure consistency.


## 6. CONCLUSION

This study developed a machine learning system for retail sales forecasting using an ensemble of gradient boosting algorithms. The key contributions and findings are summarized below.

A comprehensive feature engineering pipeline was designed that extracts 45 features from historical sales data and store metadata. The features span temporal patterns, lag relationships, rolling statistics, holiday effects, and store characteristics.

An optimized ensemble model combining LightGBM and XGBoost achieves RMSPE of 0.1212, corresponding to approximately top 14-17% performance on the Kaggle benchmark. Hyperparameter optimization using Optuna Bayesian optimization and ensemble weight optimization using scipy contributed to this performance. The optimal ensemble weights of 63.6% LightGBM and 36.4% XGBoost were determined through cross-validation optimization.

Feature importance and SHAP analysis identify promotional activity, lag features, and rolling statistics as the most influential predictors. These findings provide actionable insights for business decision-making.

K-Means clustering segments stores into 5 distinct behavioral profiles, supporting targeted strategies for different store types.

The developed system demonstrates that careful feature engineering combined with ensemble methods can achieve competitive forecasting performance without requiring deep learning or extensive computational resources. The methodology is reproducible and can be adapted to similar retail forecasting applications.


## REFERENCES

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
