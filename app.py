import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime, timedelta
from flask import Flask, send_file
import os

# 1. SETUP & CONFIGURATION
# ------------------------
symbol = 'BTC/USDT'
timeframe = '1d'
start_date_str = '2018-01-01 00:00:00'

# Define the choppy market ranges as requested
# Format: (Start Date, End Date)
choppy_ranges = [
    ('2019-11-22', '2020-01-18'),
    ('2020-05-06', '2020-07-23'),
    ('2021-05-19', '2021-07-30'),
    ('2022-01-24', '2022-04-06'),
    ('2023-03-19', '2023-10-25'),
    ('2024-04-01', '2024-11-01'), # Assumed start/end of month for "apr24-nov24"
    ('2025-03-01', '2025-04-20'), # Assumed start of month for "mar25"
    ('2025-11-13', datetime.now().strftime('%Y-%m-%d')) # "today"
]

def fetch_binance_history(symbol, start_str):
    """Fetches full OHLCV history from Binance with pagination."""
    print(f"Fetching data for {symbol} starting from {start_str}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(start_str)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Move to next timestamp
            
            # Break if we reached the current time
            if since > exchange.milliseconds():
                break
                
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Remove duplicates just in case
    df = df[~df.index.duplicated(keep='first')]
    print(f"Fetched {len(df)} days of data.")
    return df

# 2. DATA PREPARATION
# -------------------
df = fetch_binance_history(symbol, start_date_str)

# Labeling the Data (0 = Trending/Normal, 1 = Choppy)
df['target'] = 0
for start, end in choppy_ranges:
    # Use string slicing to handle data
    mask = (df.index >= start) & (df.index <= end)
    df.loc[mask, 'target'] = 1

# 3. FEATURE ENGINEERING
# ----------------------
# We create 30 columns representing the last 30 days of price action.
# IMPORTANT: We cannot use raw prices because $6,000 (2018) is not comparable to $100,000 (2025).
# We use % change relative to the specific day's close.
feature_cols = []
window_size = 30

for i in range(1, window_size + 1):
    col_name = f'lag_{i}_pct'
    # Formula: (Close[t-i] - Close[t]) / Close[t]
    # This captures the shape of the last 30 days regardless of absolute price
    df[col_name] = (df['close'].shift(i) - df['close']) / df['close']
    feature_cols.append(col_name)

# Add 365-day Simple Moving Average (SMA) feature normalized in the same way
# Calculate 365-day SMA of close price
df['sma_365'] = df['close'].rolling(window=365).mean()
# Normalize: (SMA[t] - Close[t]) / Close[t]
df['sma_365_norm'] = (df['sma_365'] - df['close']) / df['close']
feature_cols.append('sma_365_norm')

# Drop NaN values created by the shifting (first 30 days) and SMA calculation (first 365 days)
# Ensure we drop rows where ANY feature is NaN to maintain consistent array lengths
df.dropna(subset=feature_cols, inplace=True)

# 4. MODEL TRAINING
# -----------------
X = df[feature_cols]
y = df['target']

# Split train/test (80/20) - shuffle=False to preserve time order for validity check
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# 5. EVALUATION
# -------------
predictions = model.predict(X_test)
print("\n--- Model Evaluation (Test Set) ---")
print(classification_report(y_test, predictions))

# 6. VISUALIZATION
# ----------------
# We will predict on the WHOLE dataset to visualize the "Choppy" zones identified
df['prediction'] = model.predict(X)

plt.figure(figsize=(14, 7))

# Plot price
plt.plot(df.index, df['close'], label='BTC Price', color='black', alpha=0.6, linewidth=1)

# Highlight ACTUAL labels (Green zones)
for start, end in choppy_ranges:
    plt.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='green', alpha=0.1, label='Actual Choppy' if start==choppy_ranges[0][0] else "")

# Highlight PREDICTED labels (Red dots where model thinks it's choppy)
# We filter for where prediction == 1
choppy_preds = df[df['prediction'] == 1]
plt.scatter(choppy_preds.index, choppy_preds['close'], color='red', s=5, label='Predicted Choppy', zorder=3)

plt.title('Logistic Regression: Choppy Market Detection (BTC/USDT)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# Output the coefficients to see which days in history matter most
print("\n--- Feature Importance (Last 30 Days) ---")
importance = pd.DataFrame({'lag': range(1, 31), 'coefficient': model.coef_[0]})
print(importance.sort_values(by='coefficient', ascending=False).head(5))

# Save the plot to a file
import os
plot_dir = '/app/static'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
plot_path = os.path.join(plot_dir, 'plot.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")

# Create Flask app
app = Flask(__name__)

@app.route('/')
def serve_plot():
    return send_file(plot_path, mimetype='image/png')

@app.route('/health')
def health_check():
    return 'OK', 200

# Run evaluation and visualization before starting server
if __name__ == '__main__':
    print("Starting web server on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)
