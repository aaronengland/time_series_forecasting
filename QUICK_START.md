# Quick Start Guide

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set AWS credentials (for S3 access)
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

## Running the Pipeline

### Local Execution (Jupyter)
```bash
# 1. Generate synthetic data and upload to S3
jupyter notebook 00_data_collection/notebook.ipynb

# 2. Explore data patterns
jupyter notebook 01_eda/notebook.ipynb

# 3. Prepare for modeling
jupyter notebook 02_preprocessing/notebook.ipynb

# 4. Train SARIMA
jupyter notebook 03_sarima/notebook.ipynb

# 5. Train Prophet
jupyter notebook 04_prophet/notebook.ipynb

# 6. Compare models
jupyter notebook 05_comparison/notebook.ipynb
```

### SageMaker Execution
1. Upload `requirements.txt` and all notebooks to SageMaker
2. Create Jupyter kernel with dependencies installed
3. Run notebooks in order (1-6)
4. Models and plots automatically saved to S3

## Expected Runtime
- Data Collection: 2-3 minutes
- EDA: 2-3 minutes
- Preprocessing: 1-2 minutes
- SARIMA (grid search): 10-15 minutes
- Prophet (Optuna tuning): 5-10 minutes
- Comparison: 2-3 minutes
- **Total: ~25-35 minutes**

## Output Artifacts
```
00_data_collection/output/
  - 01_generated_data_overview.png
  - employee_attrition_data.csv (local copy)

01_eda/output/
  - 02_attrition_over_time.png
  - 03_attrition_by_department.png
  - 04_seasonal_decomposition.png
  - 05_headcount_trend.png
  - 06_monthly_boxplots.png
  - 07_autocorrelation.png
  - 08_rolling_statistics.png

02_preprocessing/output/
  - 09_differencing.png
  - 10_train_test_split.png
  - train_data.csv (local copy)
  - test_data.csv (local copy)

03_sarima/output/
  - 11_sarima_diagnostics.png
  - 12_sarima_forecast.png
  - sarima_model.pkl (local copy)

04_prophet/output/
  - 13_prophet_forecast_default.png
  - 14_prophet_forecast_custom.png
  - 15_prophet_components.png
  - prophet_model.pkl (local copy)

05_comparison/output/
  - 16_metrics_comparison.png
  - 17_forecast_overlay.png
  - 18_residuals_comparison.png
```

## S3 Bucket Structure
```
s3://time-series-forecasting-demo-repo/
  00_data_collection/
    └── employee_attrition_data.csv
  02_preprocessing/
    ├── train_data.csv
    └── test_data.csv
  03_sarima/
    └── sarima_model.pkl
  04_prophet/
    └── prophet_model.pkl
```

## Key Classes and Methods

### DataCollectionManager
```python
mgr = DataCollectionManager(str_bucket, str_task, str_dirname_output)
df = mgr.generate_data(int_months=72)
mgr.upload_to_s3(df, 'employee_attrition_data.csv')
mgr.plot_generated_data(df)
```

### TimeSeriesEDA
```python
eda = TimeSeriesEDA(str_bucket, str_dirname_output)
df = eda.import_data()
eda.plot_seasonal_decomposition()
eda.plot_autocorrelation()
```

### TimeSeriesPreprocessor
```python
prep = TimeSeriesPreprocessor(str_bucket, str_dirname_output)
df = prep.import_data()
int_d = prep.make_stationary()
df_train, df_test = prep.split_data(df_company, int_test_months=12)
```

### SarimaModel
```python
sarima = SarimaModel(str_bucket, str_dirname_output)
df_train, df_test = sarima.import_data()
df_results = sarima.grid_search_order()
sarima.fit_model()
df_forecast = sarima.forecast(int_steps=12)
metrics = sarima.evaluate(df_forecast)
```

### ProphetModel
```python
prophet = ProphetModel(str_bucket, str_dirname_output)
df_train, df_test = prophet.import_data()
study = prophet.tune_model(int_n_trials=50)
prophet.fit_model()
df_forecast = prophet.forecast(int_periods=12)
metrics = prophet.evaluate(df_forecast)
```

### ModelComparison
```python
comp = ModelComparison(str_bucket, str_dirname_output)
comp.import_splits()
comp.load_models()
comp.generate_forecasts()
df_metrics = comp.plot_metrics_comparison()
comp.plot_forecast_overlay()
```

## Troubleshooting

**AWS Credentials Error**
- Ensure AWS credentials are set via environment variables
- Check bucket name matches: `time-series-forecasting-demo-repo`
- Verify IAM permissions: S3 read/write access

**Memory Issues**
- Grid search uses ~2GB RAM during SARIMA tuning
- Reduce grid search space if memory-constrained
- Prophet requires ~1GB for Optuna tuning with 50 trials

**SARIMA Convergence**
- Some parameter combinations may not converge
- This is normal; grid search skips and continues
- Check final model diagnostics for convergence

**Prophet Warnings**
- "Peak at a boundary" warnings are benign
- Filter with `warnings.filterwarnings('ignore')`

## Next Steps

1. **Customization**: Modify synthetic data parameters in DataCollectionManager
2. **Production**: Deploy best model using the serialized `.pkl` file
3. **Monitoring**: Implement retraining cadence (monthly recommended)
4. **Integration**: Connect forecast output to workforce planning workflows

## References

- [SARIMA Theory](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)
- [Facebook Prophet Docs](https://facebook.github.io/prophet/)
- [Statsmodels Documentation](https://www.statsmodels.org/stable/tsa.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)

