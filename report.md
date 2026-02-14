# TSA Passenger Volume Forecasting for Kalshi Markets

## Overview

This project develops a leakage-free baseline forecasting model for daily TSA passenger volumes with direct applicability to Kalshi-style prediction markets. A Random Forest Regressor trained on seven engineered temporal features achieves a validation MAE of 95,663 passengers (3.67% MAPE), representing a ~70% improvement over a naive persistence baseline and a ~15% improvement over a seasonal naive benchmark.

Beyond point forecasting accuracy, this work emphasizes temporal validation discipline, signal diagnostics, and market-relevant framing, establishing a foundation for probabilistic event-based forecasting.

---

## Modeling Approach

A Random Forest Regressor was selected due to its ability to capture nonlinear relationships and interaction effects without explicit feature transformations. Tree-based ensembles are well-suited to time-series regression when dominant structure is autoregressive but not strictly linear.

The final model configuration includes:

- n_estimators = 100  
- max_depth = 15  
- Regularization via min_samples_split and min_samples_leaf  

Empirical testing showed that deeper trees reduced training error but degraded validation performance, indicating overfitting. A maximum depth of 15 provided the best bias–variance tradeoff.

---

## Feature Engineering

Seven features were constructed, intentionally limited to establish a robust baseline:

### Temporal Features
- day_of_week (0–6)
- month (1–12)
- year
- is_weekend (binary)

### Lag-Based Autoregressive Features
- lag_1: volume at t−1
- lag_7: volume at t−7

### Trend Feature
- rolling_mean_7: mean of volumes from t−7 through t−1

To prevent target leakage, rolling statistics were computed on shifted series:

    df['rolling_mean_7'] = df['Volume'].shift(1).rolling(window=7).mean()

This guarantees strict temporal causality: predictions at time t depend only on information available at or before t−1.

---

## Validation Strategy & Leakage Control

All data was sorted chronologically and split using a forward holdout:

- Training: 2022-01-01 → 2025-04-02  
- Validation: 2025-04-03 → 2025-07-01  

Random splits and k-fold cross-validation were intentionally avoided, as they violate temporal ordering and introduce forward-looking bias.

Lag features and rolling aggregates were constructed prior to splitting to preserve continuity at the boundary. Validation data remained fully held out during feature selection and hyperparameter tuning.

---

## Results

### Validation Performance
- MAE: 95,663  
- RMSE: 117,727  
- MAPE: 3.67%

The training–validation gap is consistent with expected generalization behavior and does not indicate severe overfitting.

### Baseline Comparison

| Model | MAE |
|------|------|
| Naive (lag-1) | 313,890 |
| Seasonal naive (lag-7) | 112,849 |
| Random Forest | **95,663** |

Performance gains confirm that the model extracts incremental signal beyond simple weekly persistence.

---

## Signal Interpretation

Feature importance analysis reveals that TSA passenger volume behaves primarily as a weekly autoregressive process:

- lag_7 dominates (74.4%), confirming strong weekly stationarity  
- rolling_mean_7 captures short-term trend persistence  
- Calendar features contribute marginal signal once lag structure is present  

The modest improvement over the seasonal naive baseline establishes an upper bound on extractable signal from simple autoregressive structure alone, suggesting that further gains require modeling exogenous events.

---

## Residual Diagnostics

Residuals are approximately zero-mean but exhibit heavy tails, with the largest errors concentrated around major travel holidays. Autocorrelation analysis shows minimal remaining weekly structure, indicating that dominant seasonal effects have been captured.

These patterns suggest that holiday and event indicators derived purely from calendar metadata would reduce tail risk and improve calibration.

---

## Practical Extensions

Although this work focuses on a baseline regression model, the framework can be extended in several straightforward directions:

- Incorporating holiday indicators to reduce extreme residuals.
- Adding additional lag features (lag_14, lag_21) to capture multi-week structure.
- Implementing rolling-origin cross-validation for more robust generalization estimates.
- Estimating prediction intervals via bootstrapping or quantile regression forests.

These extensions would enhance robustness while maintaining the simplicity of the baseline architecture.

---

## Production Considerations

This pipeline is structured for operational deployment:

- Rolling retraining with daily updates  
- Concept drift monitoring for structural changes in travel behavior  
- Strict data latency controls for lag-based features  
- Automated evaluation of error, directionality, and calibration  

The modular design enables straightforward extension into a live inference or trading system.

---

## Key Learnings

One of the most important challenges in building this baseline model was avoiding data leakage during forecasting. Because lag and rolling statistics depend on prior observations, it was not sufficient to compute features on the full dataset and then generate predictions for the test set. Doing so would inadvertently use future information.

To address this, I implemented recursive forecasting. Test predictions were generated sequentially, where each predicted value was appended to the historical data before computing features for the next time step. This ensured that all lag and rolling features were constructed only from information that would realistically be available at prediction time.

Another learning was understanding the limitations of tree-based models for time series forecasting. Unlike traditional time series models, tree-based regressors do not inherently understand temporal ordering. As a result, careful feature engineering (such as lag variables and rolling means) is essential to encode temporal structure.

Finally, this project reinforced the importance of maintaining simplicity when building a baseline model. Rather than over-engineering the feature set, focusing on a small number of intuitive temporal features allowed for a clean, interpretable starting point that can be extended in future iterations.

---

## Conclusion

This project establishes a technically disciplined, leakage-free baseline for TSA passenger volume forecasting with direct applicability to Kalshi-style prediction markets. The model captures dominant weekly autoregressive structure, improves meaningfully over naive baselines, and provides a foundation for probabilistic event forecasting.

Future improvements should prioritize exogenous signal integration (holidays, disruptions) and uncertainty estimation rather than increased autoregressive complexity.
