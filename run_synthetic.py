import pandas as pd
import numpy as np
from src.models.registry import ModelRegistry, train_all_models
from src.features.feature_pipeline import FeaturePipeline
import sqlite3
import os

# Create dummy db
db_path = 'data/processed/adip.db'
os.makedirs(os.path.dirname(db_path), exist_ok=True)

print("Creating dummy dataset...")
rng = np.random.default_rng(42)
n_days = 60
dates = pd.date_range("2025-01-01", periods=n_days, freq="D")

rows = []
for sid in [3, 5]: # Valid store ids
    trend = np.linspace(800, 1200, n_days)
    weekly = 150 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    noise = rng.normal(0, 40, n_days)
    revenue = trend + weekly + noise + sid * 50
    for d, r in zip(dates, revenue):
        rows.append({"date": d, "store_id": sid, "total_revenue": r, "transaction_count": int(r/5)})

df = pd.DataFrame(rows)
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

with sqlite3.connect(db_path) as conn:
    df.to_sql('daily_store', conn, if_exists='replace', index=False)

print("Running pipeline...")
registry = ModelRegistry()
train_all_models(db_path, registry)
print("Pipeline complete!")
