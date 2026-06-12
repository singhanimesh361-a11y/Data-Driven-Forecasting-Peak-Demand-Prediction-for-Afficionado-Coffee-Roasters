import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

print("Generating synthetic data.xlsx...")
os.makedirs('data/raw', exist_ok=True)

rng = np.random.default_rng(42)
n_transactions = 5000

categories = ['Coffee', 'Tea', 'Bakery', 'Drinking Chocolate', 'Coffee beans', 'Branded', 'Loose Tea', 'Flavours', 'Packaged Chocolate']
store_ids = [3, 5, 8]

# Generate transactions over a 100 day period
start_date = datetime(2025, 1, 1)
end_date = start_date + timedelta(days=100)

rows = []
for i in range(n_transactions):
    dt = start_date + timedelta(days=rng.uniform(0, 100))
    year = dt.year
    
    # Calculate fractional year time
    start_of_year = datetime(year, 1, 1)
    fractional_time = (dt - start_of_year).total_seconds() / (365 * 86400)
    
    rows.append({
        'transaction_id': f"TXN-{i+1:06d}",
        'year': year,
        'transaction_time': fractional_time,
        'transaction_qty': rng.integers(1, 5),
        'unit_price': round(rng.uniform(2.5, 12.0), 2),
        'store_id': rng.choice(store_ids),
        'store_location': 'NYC',
        'product_id': f"PRD-{rng.integers(1, 100):03d}",
        'product_category': rng.choice(categories),
        'product_type': 'Type A',
        'product_detail': 'Detail B'
    })

df = pd.DataFrame(rows)
df.to_excel('data/raw/data.xlsx', sheet_name='Transactions', index=False)
print("Saved 5,000 transactions to data/raw/data.xlsx")
