# ADIP User Guide

Welcome to the Afficionado Demand Intelligence Platform (ADIP). This guide is designed to help regional managers and operations staff leverage the platform for proactive decision-making.

## Dashboard Navigation

### 1. Executive Overview
**Purpose:** Your daily starting point for high-level health checks.
* Focus on the **Peak Demand Alert Panel** to identify upcoming high-volume hours.
* Use the KPI cards to monitor the freshness of the forecast data.

### 2. Store Forecast
**Purpose:** Detailed planning for a specific location.
* Select a store from the sidebar to view its unique revenue breakdown and transaction volume trend.

### 3. Hourly Heatmap
**Purpose:** Visualizing the daily rush.
* The heatmap clearly shows the consistent 10 AM peak. Use the "Historical vs Forecasted" toggle to see if upcoming days deviate from the norm.

### 4. Category Intelligence
**Purpose:** Inventory and purchasing decisions.
* Track which categories (e.g., Coffee vs. Bakery) are driving revenue and adjust your procurement orders accordingly.

### 5. Model Comparison
**Purpose:** Transparency into the AI's performance.
* See which model is currently the most accurate (lowest MAPE) and review the residual charts to understand where models might over- or under-predict.

### 6. Scenario Planner
**Purpose:** Stress-testing and what-if analysis.
* Use the **Demand Shock (%)** slider to simulate events (e.g., a local festival boosting traffic by 20%). The P&L table will update dynamically to show the impact on net margin.

## Interpreting Confidence Intervals
On the forecast charts, you'll see shaded bands around the main prediction line:
* **The solid line:** The single most likely outcome (the base case).
* **The darker band (80% CI):** We are 80% confident the actual value will fall within this range.
* **The lighter band (95% CI):** A wider, more conservative range capturing almost all plausible outcomes (the best/worst case scenarios).

## Scenario Planning for Staffing
To optimize staffing:
1. Open the **Scenario Planner** page.
2. Review the **Worst Case** column in the P&L table. This assumes demand hits the upper bound of the 95% confidence interval.
3. Check the **Staff Cost** estimation. The system assumes you need 1 FTE per 150 transactions per hour.
4. Schedule staff based on the base case, but keep on-call staff ready if the worst-case scenario predicts a significant shortfall.

## FAQ

**Why does the 10 AM peak matter?**
Data shows the 08:00–10:00 AM window generates over 35% of daily volume. Staffing and inventory must be heavily concentrated here to avoid bottlenecks.

**What does MAPE mean?**
Mean Absolute Percentage Error. A MAPE of 15% means the model's predictions are, on average, within 15% of the actual revenue. Lower is better.

**How often do forecasts refresh?**
The system runs a nightly batch job at 02:00 AM to generate fresh 30-day daily and 72-hour hourly forecasts.

**What if the data badge says STALE?**
This means the underlying database hasn't received new transaction data in over 25 hours. Contact the IT team to ensure the POS data pipeline is flowing.

## Glossary
* **Base Case:** The median forecast prediction.
* **Confidence Interval (CI):** A range of values indicating the uncertainty of a forecast.
* **Meta-Learner:** A model (like our Ensemble) that learns how to best combine the predictions of other models.
* **Residual:** The difference between an actual observed value and the model's predicted value.
