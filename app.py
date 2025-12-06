import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime
from flask import Flask, send_file
import os

# 1. SETUP & CONFIGURATION
# ------------------------
symbol = 'BTC/USDT'
timeframe = '1d'
start_date_str = '2018-01-01 00:00:00'

choppy_ranges = [
    ('2021-05-26', '2021-07-23'),
    ('2022-01-24', '2022-04-06'),
    ('2023-03-19', '2023-10-18'),
    ('2024-03-18', '2024-10-01'),
    ('2025-03-08', '2025-04-13')
]

def fetch_binance_history(symbol, start_str):
    print(f"Fetching data for {symbol} starting from {start_str}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(start_str)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if since > exchange.milliseconds(): break
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    print(f"Fetched {len(df)} days of data.")
    return df

# 2. DATA PREPARATION
# -------------------
df = fetch_binance_history(symbol, start_date_str)

# Labeling
df['target'] = 0
for start, end in choppy_ranges:
    mask = (df.index >= start) & (df.index <= end)
    df.loc[mask, 'target'] = 1

# 3. ADVANCED FEATURE ENGINEERING
# -------------------------------
feature_cols = []

# A. Lags removed - using only technical indicators
# Log returns still calculated for potential future use
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

# B. Kaufman Efficiency Ratio (KER) - The Chop Detector
# KER = Direction / Volatility
n_er = 13 # Standard setting
# Net change over period
change = df['close'].diff(n_er).abs()
# Sum of absolute changes over period (Volatility)
volatility = df['close'].diff(1).abs().rolling(n_er).sum()
df['ker'] = change / volatility
# feature_cols.append('ker')  # Removed from features

# C. Bollinger Band Width (Volatility Squeeze)
# Low width usually indicates chop/consolidation
window = 20
sma = df['close'].rolling(window).mean()
std = df['close'].rolling(window).std()
df['bb_width'] = ( (sma + 2*std) - (sma - 2*std) ) / sma
# feature_cols.append('bb_width')  # Removed from features

# D. Distance from Medium Term Trend (119 SMA)
# In chop, price whipsaws AROUND the SMA. In trend, it stays above/below.
df['sma_50_dist'] = (df['close'] - df['close'].rolling(119).mean()) / df['close'].rolling(119).mean()
# We take absolute value because chop can be above or below, we just care that it's CLOSE to the average
df['sma_50_abs_dist'] = df['sma_50_dist'].abs() 
# feature_cols.append('sma_50_abs_dist')  # Removed as per user request

# G. Distance from Long-Term Trend (365 SMA)
# Similar logic to 50 SMA but over a longer period to capture broader consolidation
df['sma_365_dist'] = (df['close'] - df['close'].rolling(365).mean()) / df['close'].rolling(365).mean()
df['sma_365_abs_dist'] = df['sma_365_dist'].abs()

# E. Distance to 365 SMA with abs dist * 1.2
df['sma_365_abs_dist_1_2'] = df['sma_365_abs_dist'] * 1.2
feature_cols.append('sma_365_abs_dist')
feature_cols.append('sma_365_abs_dist_1_2')

# H. Distance from 120 SMA
# Added as per user request for additional medium-term trend analysis
df['sma_120_dist'] = (df['close'] - df['close'].rolling(120).mean()) / df['close'].rolling(120).mean()
df['sma_120_abs_dist'] = df['sma_120_dist'].abs()

# I. Distance to 120 SMA with abs dist * 1.2
df['sma_120_abs_dist_1_2'] = df['sma_120_abs_dist'] * 1.2
feature_cols.append('sma_120_abs_dist')
feature_cols.append('sma_120_abs_dist_1_2')

# F. Distance to 120 SMA with 1.15 and 0.85 multipliers (Normalized) - Removed as per user request
df['sma_120_1_15'] = df['close'].rolling(120).mean() * 1.15
df['sma_120_0_85'] = df['close'].rolling(120).mean() * 0.85
df['dist_sma_120_1_15'] = (df['close'] - df['sma_120_1_15']) / df['close']
df['dist_sma_120_0_85'] = (df['close'] - df['sma_120_0_85']) / df['close']
# feature_cols.append('dist_sma_120_1_15')  # Removed
# feature_cols.append('dist_sma_120_0_85')  # Removed

df.dropna(subset=feature_cols, inplace=True)

# 4. SCALING & TRAINING
# ---------------------
X = df[feature_cols]
y = df['target']

# Important: Logistic Regression works best with Scaled data
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Increased C (inverse regularization) slightly to allow more complexity
model = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)
model.fit(X_train, y_train)

# 5. EVALUATION
# -------------
predictions = model.predict(X_test)
print("\n--- Model Evaluation (Test Set) ---")
print(classification_report(y_test, predictions))

# Feature Importance
print("\n--- Feature Importance ---")
importance = pd.DataFrame({'feature': feature_cols, 'coef': model.coef_[0]})
# Sort by absolute value to see impact regardless of direction
importance['abs_coef'] = importance['coef'].abs()
print(importance.sort_values(by='abs_coef', ascending=False).head(10))

# 6. VISUALIZATION
# ----------------
df['prediction'] = model.predict(X_scaled)
# Create a probability score (confidence)
df['pred_prob'] = model.predict_proba(X_scaled)[:, 1]

plt.figure(figsize=(14, 8))

# Subplot 1: Price and Zones
ax1 = plt.subplot(2, 1, 1)
ax1.plot(df.index, df['close'], color='black', alpha=0.6, linewidth=1, label='BTC Price')
# Add 365-day SMA
ax1.plot(df.index, df['close'].rolling(365).mean(), color='blue', alpha=0.7, linewidth=1, label='365 SMA')

# Actual Labels
for start, end in choppy_ranges:
    ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='green', alpha=0.15)

# Predicted Labels (Red dots)
choppy_preds = df[df['prediction'] == 1]
ax1.scatter(choppy_preds.index, choppy_preds['close'], color='red', s=3, zorder=3, alpha=0.6, label='Pred Chop')
ax1.set_title('Choppy Market Detection (Green=Actual, Red=Predicted)')
ax1.legend()

# Subplot 2: Absolute Distance from 365 SMA * 1.2
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax2.plot(df.index, df['sma_365_abs_dist_1_2'], color='purple', linewidth=1, label='365 SMA Abs Dist * 1.2')
ax2.set_title('Absolute Distance from 365 SMA Multiplied by 1.2')
ax2.legend()

import os
plot_dir = '/app/static'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
plot_path = os.path.join(plot_dir, 'plot.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Flask App
app = Flask(__name__)
@app.route('/')
def serve_plot(): return send_file(plot_path, mimetype='image/png')
@app.route('/health')
def health(): return 'OK', 200

if __name__ == '__main__':
    print("Starting web server on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)
