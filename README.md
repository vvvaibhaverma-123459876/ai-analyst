# AI Analyst v0.1 🚀  
An evidence-backed AI analytics assistant that turns raw CSV data into:
- KPI trends
- anomaly detection
- driver attribution (what caused the change)
- executive-ready summaries

## Features
- **CSV Upload**: Load any structured dataset.
- **Auto Profiling**: Nulls, dtypes, unique counts, numeric stats.
- **KPI Trend Viewer**: Choose a date column + KPI + time grain.
- **Anomaly Detection**: Rolling baseline + z-score threshold.
- **Driver Attribution**: Compares last N days vs previous N days across dimensions.
- **Executive Summary**: Leadership-style narrative using evidence only.

## Demo Screenshots
### KPI Trend + Anomalies
![Trend](assets/trend_anomalies.png)

### Drivers + Executive Summary
![Drivers](assets/drivers_summary.png)

## Quickstart (Local)
### 1) Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt