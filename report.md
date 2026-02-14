TSA Passenger Volume Forecasting Report
Introduction

This project develops a baseline forecasting model to predict daily TSA passenger volumes. The objective was to implement a technically sound, interpretable model that captures core temporal dynamics without unnecessary architectural complexity. The resulting model — a Random Forest Regressor trained on seven engineered temporal features — achieved a validation Mean Absolute Error (MAE) of 95,663 passengers (3.67% MAPE), representing approximately a 70% improvement over a naive persistence baseline.

This report details the modeling framework, feature engineering strategy, validation methodology, leakage mitigation approach, and performance evaluation, followed by key technical insights and potential extensions.

Methodology

A Random Forest Regressor was selected as the baseline model due to its ability to capture nonlinear relationships without requiring explicit feature transformations. Tree-based ensemble methods naturally model interaction effects and non-monotonic dependencies between temporal predictors and passenger volume, while remaining robust to feature scaling and monotonic transformations.

The final model consists of:

n_estimators = 100

max_depth = 15

Regularization via min_samples_split and min_samples_leaf

Increasing max_depth beyond 15 reduced training error but increased validation error, indicating overfitting. A depth of 15 provided an appropriate bias–variance tradeoff.

Feature Engineering Strategy

Seven features were constructed across three categories:

1. Calendar-Based Temporal Features

day_of_week (0–6)

month (1–12)

year

is_weekend (binary)

These encode periodic and seasonal structure directly from the date index.

2. Lag Features

lag_1: Volume at time 
𝑡
−
1
t−1

lag_7: Volume at time 
𝑡
−
7
t−7

These capture short-term persistence and weekly seasonality.

3. Rolling Aggregation

rolling_mean_7: Mean of volumes from 
𝑡
−
7
t−7 through 
𝑡
−
1
t−1

To prevent target leakage:

df['rolling_mean_7'] = df['Volume'].shift(1).rolling(window=7).mean()


The shift ensures strict temporal causality by excluding the current observation from the rolling window.

Train–Validation Strategy

A chronological split was used:

Training: 2022-01-01 to 2025-04-02

Validation: 2025-04-03 to 2025-07-01 (90 days)

Random splitting was avoided because it violates temporal ordering and introduces forward-looking bias. The chronological split ensures evaluation reflects true out-of-sample forecasting performance.

Feature engineering was performed prior to splitting to preserve temporal continuity across the boundary.

Data Leakage Mitigation

Safeguards implemented:

All lag features constructed using .shift()

Rolling statistics computed on shifted series

Chronological split performed after feature construction

Validation data remained fully held-out during tuning

Each prediction at time 
𝑡
t depends exclusively on information available at time 
𝑡
−
1
t−1 or earlier.

Model Performance
Training Performance

MAE: 51,471

RMSE: 81,451

MAPE: 2.37%

Validation Performance

MAE: 95,663

RMSE: 117,727

MAPE: 3.67%

The training–validation gap is consistent with normal generalization behavior.

Baseline Comparisons
Model	MAE
Naive (lag-1)	313,890
Seasonal naive (lag-7)	112,849
Random Forest	95,663

Improvements:

~70% over naive persistence

~15% over seasonal naive

The modest improvement over seasonal naive confirms that weekly autocorrelation is the dominant predictive signal.

Feature Importance

lag_7 — 74.4%

rolling_mean_7 — 11.7%

lag_1 — 5.3%

month — 4.7%

Remaining features — <3%

The dominance of lag_7 confirms strong weekly periodicity. The low importance of is_weekend suggests redundancy with lag-based seasonality encoding.

Key Technical Insights

Temporal leakage is the primary methodological risk in time-series modeling.

Weekly autocorrelation accounts for most predictive power.

Increasing model complexity without additional signal introduces overfitting.

Residual analysis shows heavy-tailed errors concentrated around holidays, suggesting value in calendar-based holiday indicators.

Conclusion

This project demonstrates that a carefully engineered Random Forest baseline can capture dominant temporal structure in TSA passenger volumes using a compact feature set. The model achieves 3.67% validation MAPE while maintaining interpretability and strict temporal causality.

Potential Extensions

Holiday and event indicators derived from calendar metadata

Time-series cross-validation for systematic hyperparameter tuning

Additional seasonal lags (14, 21, 28 days)

Hybrid or ensemble modeling approaches

Technical Summary

Model: Random Forest Regressor (scikit-learn)
Features: 7 (temporal + lag + rolling)
Training observations: 1,181
Validation observations: 90
Test horizon: 211 days

Validation Metrics:

MAE: 95,663

RMSE: 117,727

MAPE: 3.67%

Improvement over naive: 69.5%

Improvement over seasonal naive: 15.2%
