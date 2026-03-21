# Time Series Forecasting for Employee Attrition

A comprehensive forecasting pipeline for predicting monthly employee attrition rates in a multi-department organization. Built to demonstrate expertise in time series analysis, seasonality modeling, and production-grade forecasting workflows — ideal for workforce planning and predictive HR analytics.

## Project Overview

This project tackles employee attrition forecasting — a core capability for HCM/HR software companies. We generate realistic synthetic attrition data with trend and seasonality, then apply multiple modern forecasting approaches (SARIMA, Facebook Prophet) to predict future departures and staffing needs. The pipeline includes full exploratory analysis, preprocessing, model development, and cross-model comparison.

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `00_data_collection/` | Generate synthetic 6-year monthly HR dataset with seasonal and trend components |
| `01_eda/` | Time series exploratory analysis: trends, seasonality, autocorrelation, decomposition |
| `02_preprocessing/` | Stationarity testing, differencing, feature engineering, train/test splitting |
| `03_sarima/` | SARIMA model development with grid search, diagnostics, and evaluation |
| `04_prophet/` | Facebook Prophet model with Optuna hyperparameter tuning |
| `05_comparison/` | Side-by-side evaluation of both models: metrics, residuals, forecast overlay |

## Dataset

| Attribute | Value |
|-----------|-------|
| **Time Span** | 72 months (6 years), monthly granularity |
| **Company Size** | ~5,000 employees across 6 departments |
| **Departments** | Engineering, Sales, Marketing, HR, Finance, Operations |
| **Features (per month per dept)** | headcount, new_hires, departures, attrition_rate, avg_tenure_months, avg_satisfaction_score |
| **Patterns Included** | Upward headcount trend, seasonal attrition spikes (Jan/Feb/Sep), summer dips, random noise, restructuring events |
| **Storage** | AWS S3 (time-series-forecasting-demo-repo) as CSV |

## Methodology

### 00: Data Collection
Generate realistic synthetic employee data with:
- Overall company growth trend
- Seasonal departures (higher in January post-bonus, lower in summer)
- Department-specific base attrition rates
- Occasional event months (restructuring, layoffs)
- Aggregate to company-wide monthly attrition for forecasting

### 01: Exploratory Data Analysis
- Import and summarize raw data
- Visualize attrition trends over time, by department
- Seasonal decomposition (trend, seasonal, residual components)
- Autocorrelation analysis (ACF/PACF) to inform model selection
- Rolling statistics to assess stationarity
- Monthly boxplots to highlight seasonal patterns

### 02: Preprocessing
- Aggregate multi-department data to company level
- Test stationarity with ADF and KPSS tests
- Apply differencing if needed
- Engineer lag features and rolling averages for alternative approaches
- Temporal train/test split (70% train, last 12 months test)
- Save splits for consistent model training

### 03: SARIMA Modeling
- Grid search over (p,d,q)(P,D,Q,s) parameter space
- Select best model by AIC/BIC
- Diagnostic checks: residual ACF, Ljung-Box test, QQ plot
- Generate 12-month forecast with 95% confidence intervals
- Evaluate on test set: RMSE, MAE, MAPE

### 04: Facebook Prophet
- Optuna hyperparameter tuning (changepoint prior scale, seasonality settings)
- Fit final model with optimized parameters
- Decompose forecast into trend and seasonal components
- Generate probabilistic forecast with uncertainty bands
- Evaluate on test set

### 05: Model Comparison
- Load both SARIMA and Prophet models
- Side-by-side metrics table (RMSE, MAE, MAPE)
- Overlay forecast plots for visual comparison
- Residual analysis and statistical error comparison
- Recommendations for production deployment

## Results

| Metric | SARIMA | Prophet |
|--------|--------|---------|
| **RMSE** | 0.087 | 0.095 |
| **MAE** | 0.062 | 0.070 |
| **MAPE (%)** | 4.2 | 4.8 |

*Note: Exact results depend on synthetic data generation seed. See `05_comparison/output/` for latest evaluation plots.*

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Synthetic Data** | Allows full control over seasonality, trend, and anomalies. Demonstrates ability to design realistic scenarios. |
| **Company-Level Aggregation** | Focuses forecasting on total attrition, the most actionable metric for workforce planning. |
| **SARIMA + Prophet** | Captures both statistical rigor (SARIMA) and business interpretability (Prophet trend/seasonality components). |
| **12-Month Test Window** | Realistic forecast horizon for quarterly planning cycles in HCM. |
| **Confidence Intervals** | Production systems need uncertainty quantification for risk assessment. |
| **S3 Storage** | Demonstrates cloud data pipeline and scalability — relevant for enterprise HCM deployment. |

## Getting Started

### Prerequisites
- Python 3.8+
- AWS credentials configured (for S3 access)
- Jupyter notebook environment

### Installation
```bash
pip install -r requirements.txt
```

### Execution Order
1. **00_data_collection/notebook.ipynb** — Generate and upload synthetic data to S3
2. **01_eda/notebook.ipynb** — Explore data patterns and characteristics
3. **02_preprocessing/notebook.ipynb** — Prepare data for modeling
4. **03_sarima/notebook.ipynb** — Train SARIMA model
5. **04_prophet/notebook.ipynb** — Train Prophet model
6. **05_comparison/notebook.ipynb** — Compare and evaluate both models

Each notebook is self-contained and can be run independently after 00 and 02.

### Output
- Plots saved to `*/output/` directories (PNG format)
- Models serialized to S3
- Metrics and comparison tables printed to notebook

## Key Insights

- **Seasonality is strong:** Month-of-year accounts for 40%+ of attrition variance
- **Trend is modest but real:** ~2% decline over 6 years after accounting for growth
- **Events matter:** Restructuring months show 2-3x normal departures
- **Both models perform well:** SARIMA slightly more accurate, Prophet more interpretable

## Production Considerations

- **Retraining cadence:** Monthly with new departures data
- **Monitoring:** Compare forecasts to actuals; trigger alerts if MAPE > 10%
- **Feature drift:** Track satisfaction scores and tenure distributions for model degradation
- **Scalability:** Current pipeline handles single company. Multi-tenant deployment would shard by company ID
- **Integration:** Forecast output feeds into workforce planning workflows (headcount budgets, recruitment planning)

## Contact

For questions or collaboration on workforce analytics, reach out to Aaron England.