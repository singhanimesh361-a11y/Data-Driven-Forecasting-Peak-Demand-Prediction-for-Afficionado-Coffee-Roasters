# ADIP - Afficionado Demand Intelligence Platform
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

The Afficionado Demand Intelligence Platform (ADIP) is a specialized machine learning pipeline and interactive dashboard designed to forecast store-level, category-specific demand for Afficionado Coffee Roasters. By moving from reactive scheduling to data-driven proactive planning, ADIP empowers regional managers to optimize staffing and reduce inventory stock-outs during peak hours.

### Key Metrics
* **149,116** transactions analyzed
* **3** NYC retail locations
* **$698,812** total revenue modeled
* **9** distinct product categories

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the data pipeline:**
   ```bash
   python -m src.ingestion.xlsx_loader --xlsx data/raw/Afficionado_Coffee_Roasters.xlsx
   ```
3. **Launch the dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

## Architecture

```text
+----------------+      +-------------------+      +-------------------+
|  Excel Source  | ---> | xlsx_loader.py    | ---> | SQLite Database   |
| (Transactions) |      | (Parse & Verify)  |      | (daily_store, etc)|
+----------------+      +-------------------+      +-------------------+
                                                           |
                                                           v
+----------------+      +-------------------+      +-------------------+
| MLflow Server  | <--- | registry.py       | <--- | feature_pipeline  |
| (Model Reg.)   |      | (Train & Eval)    |      | (Lags, Rolling)   |
+----------------+      +-------------------+      +-------------------+
        |                                                  |
        v                                                  v
+----------------+      +-------------------+      +-------------------+
| Streamlit App  | <--- | forecast_engine   | <--- | Forecasters       |
| (Dashboard)    |      | (Nightly Refresh) |      | (XGBoost, Prophet)|
+----------------+      +-------------------+      +-------------------+
```

## Features

The Streamlit dashboard includes 6 core pages:
* **Executive Overview**: High-level KPIs, 30-day forecast chart, and peak demand alerts.
* **Store Forecast**: Deep-dive per location, revenue breakdown, and accuracy metrics.
* **Hourly Heatmap**: Interactive visualization of the 10 AM peak window.
* **Category Intelligence**: Product mix trends and market basket insights.
* **Model Comparison**: Leaderboard, residual distribution, and walk-forward validation charts.
* **Scenario Planner**: What-if analysis for demand shocks and P&L impact.

## Models

| Model | Type | MAPE Target | Best Use Case |
|-------|------|-------------|---------------|
| Naive | Baseline | > 25% | Sanity checking and simple comparisons. |
| Seasonal Naive | Baseline | ~ 20% | Capturing strict day-of-week patterns. |
| SARIMA | Statistical | < 20% | Interpretable autoregressive components. |
| Prophet | Statistical | < 18% | Multiplicative seasonality and trend shifts. |
| XGBoost | Machine Learning| < 15% | Complex non-linear feature interactions. |
| Ensemble | Meta-Learner | < 12% | Highest accuracy, robust production choice. |

## Deployment

* **Local**: Run via `docker-compose up -d` to spin up the dashboard, MLflow, and the scheduler.
* **Render**: Connect the repo to Render and use the provided `Dockerfile`.
* **AWS ECS**: Push image to ECR and deploy via Fargate.
* **Antigravity**: Deploy as a Managed Agent on GCP Vertex AI using `deploy.sh` and the included agent manifest.

## Contributing

1. Create a feature branch (`feature/your-feature-name`).
2. Write tests for any new logic.
3. Run `flake8` and `pytest` before committing.
4. Submit a Pull Request using the standard template.

## License

MIT License. See LICENSE for details.
