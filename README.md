# TSA-Kalshi-Prisha-Jain

## Overview

This project implements a time-series forecasting model to predict TSA passenger volumes, generating forecasts for `tsa_test.csv` beginning July 2, 2025.

The solution is built using a Random Forest Regressor and a structured forecasting pipeline designed for clarity, reproducibility, and proper temporal validation. The approach incorporates fundamental time-based features and lagged observations to capture seasonality and short-term autocorrelation patterns inherent in passenger traffic data.

The modeling process includes:

- Chronological data splitting to reflect real-world forecasting conditions  
- Feature construction using calendar signals and lag variables  
- Evaluation using MAE and RMSE  
- Visualization of predicted versus actual values  
- Modular, production-conscious code structure  

---

## Project Structure

tsa-forecasting-baseline/
│
├── data/
├── src/
│   └── forecasting.py
├── outputs/
├── report.md
├── requirements.txt
└── README.md

---

## Methodology

- **Model:** Random Forest Regressor  
- **Features:** Day of Week, Month, Year, Weekend indicator, Lag 1, Lag 7, Rolling 7-day mean  
- **Validation:** Chronological split to prevent temporal leakage  
- **Metrics:** MAE, RMSE  

---

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt

