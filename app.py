import requests
import pandas as pd
import numpy as np
from scipy import stats

# 1. Fetch Data
URL = "https://workspace-production-9fae.up.railway.app/history"
try:
    response = requests.get(URL)
    data = response.json()
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

# 2. Process Data
df = pd.DataFrame(data)

# Ensure PnL is numeric
if 'pnl' not in df.columns:
    print("Error: 'pnl' column not found.")
    exit()

pnl = df['pnl']

# 3. Calculate Distribution Metrics
metrics = {
    "Count (Trades)": len(pnl),
    "Mean PnL": pnl.mean(),
    "Median PnL": pnl.median(),
    "Std Deviation": pnl.std(),
    "Skewness": pnl.skew(),
    "Kurtosis": pnl.kurtosis(),  # Fisher (excess) kurtosis
    "Min PnL": pnl.min(),
    "Max PnL": pnl.max(),
    "Total PnL": pnl.sum(),
    "Win Rate (%)": (len(pnl[pnl > 0]) / len(pnl)) * 100,
}

# 4. Calculate Max Drawdown (Time-Series Metric)
# Sort by time first to ensure drawdown is accurate
if 'time' in df.columns:
    df['dt'] = pd.to_datetime(df['time'])
    df = df.sort_values('dt')
    
    cumulative = df['pnl'].cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    metrics["Max Drawdown"] = drawdown.max()

# 5. Print Report
print("="*40)
print("       PnL STATISTICAL REPORT       ")
print("="*40)

for key, value in metrics.items():
    print(f"{key:<20}: {value: .4f}")

print("\nINTERPRETATION:")
if metrics['Skewness'] < 0:
    print("- Skewness < 0: The tail is on the left. Frequent small wins, occasional large losses.")
else:
    print("- Skewness > 0: The tail is on the right. Frequent small losses, occasional large wins.")

if metrics['Kurtosis'] > 0:
    print("- Kurtosis > 0: Heavy tails (Leptokurtic). Higher risk of extreme outcomes (outliers).")
else:
    print("- Kurtosis < 0: Light tails (Platykurtic). Fewer extreme outliers than a normal distribution.")
