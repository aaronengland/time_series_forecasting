# Project Overview

Time Series Forecasting of employee attrition — a portfolio project for Aaron England's personal website (aaron-england.com). This project is specifically designed to demonstrate skills relevant to a **Staff Data Scientist role at Paylocity**, an HCM/HR software company that provides workforce planning tools, schedule forecasting, and predictive workforce insights.

## Why This Project Exists

Paylocity's data science team builds time series forecasting models for workforce planning — predicting staffing needs, headcount trends, and attrition patterns. This project demonstrates end-to-end time series forecasting on synthetic (but realistic) monthly employee attrition data.

## S3 Bucket

`time-series-forecasting-demo-repo`

All data is read from and written to this bucket. Generated data goes in `00_data_collection/`, preprocessed data in `02_preprocessing/`.

## Dataset

Synthetic monthly employee attrition data generated in notebook 00. The data simulates 6 years of monthly headcount and attrition across 6 departments, with realistic patterns including trend, seasonality (Q1 spikes after bonuses, summer dips), and noise. This mirrors what a real company's workforce planning data looks like.

## Project Structure

```
time_series_forecasting/
├── 00_data_collection/notebook.ipynb     # Generate synthetic monthly attrition data, upload to S3
├── 01_eda/notebook.ipynb                 # Time series EDA: trends, seasonality, decomposition
├── 02_preprocessing/notebook.ipynb       # Stationarity tests, differencing, train/test split
├── 03_sarima/notebook.ipynb              # SARIMA model with Optuna tuning
├── 04_prophet/notebook.ipynb             # Facebook Prophet model with Optuna tuning
├── 05_comparison/notebook.ipynb          # SARIMA vs Prophet comparison
├── requirements.txt
└── README.md
```

## Execution Order

Run notebooks sequentially: 00 → 01 → 02 → 03 → 04 → 05. Each notebook reads from S3 and writes outputs to `./output/`.

## Key Technical Details

- **Synthetic data generation**: Realistic monthly attrition with trend, seasonality (period=12), departmental variation, and random noise
- **Stationarity testing**: ADF test, KPSS test, differencing as needed
- **SARIMA**: Uses Optuna (20 trials) to search over (p,d,q)(P,D,Q) with s=12, minimizing AIC
- **Prophet**: Uses Optuna (20 trials) to tune changepoint_prior_scale, seasonality_prior_scale, seasonality_mode
- **Evaluation**: RMSE, MAE, MAPE on held-out test period; forecast with confidence intervals
- **Comparison notebook**: Side-by-side metrics, overlaid forecasts, residual analysis

## Coding Conventions

- **Classes**: `SarimaModel`, `ProphetModel`, `ModelComparison` (and EDA/preprocessing classes)
- **Hungarian notation**: `str_`, `int_`, `flt_`, `cls_`, `df_`, `list_`, `dict_`, `bool_`
- **Constants section** at top of each notebook: `str_bucket`, `str_task`, `str_dirname_output`
- **Plots**: Saved to `./output/` with `dpi=150`, `bbox_inches='tight'`
- **S3 loading**: Try S3 first, fall back to local path

## What Comes After This Repo

Once all notebooks are run, the results will be integrated into Aaron's personal website as a dedicated `/time-series` page with an interactive forecast visualization (Plotly chart with historical trend + future predictions + confidence intervals) and full methodology write-up.
