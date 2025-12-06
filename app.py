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

# Defined Choppy Zones
choppy_ranges = [
    ('2019-11-22', '2020-01-18'),
    ('2020-05-06', '2020-07-23'),
    ('2021-05-19', '2021-07-30'),
    ('2022-01-24', '2022-04-06'),
    ('2023-03-19', '2023-10-25'),
    ('2024-04-01', '2024-11-01'),
    ('2025-03-01', '2025-04-20'),
    ('2025-11-13', datetime.now().strftime('%Y-%m-%d'))
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

# 3. FEATURE ENGINEERING (YOUR FORMULA)
# -------------------------------------
window_size = 60

# Calculate Log Returns: ln(Pt / Pt-1)
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

# Numerator: Sum of Absolute Log Returns (Total Volatility/Path Length)
df['sum_abs_log_ret'] = df['log_ret'].abs().rolling(window_size).mean() # Using mean to keep scale manageable

# Denominator: Absolute Sum of Log Returns (Net Directional Change)
df['abs_sum_log_ret'] = df['log_ret'].rolling(window_size).mean().abs()

# The "Inefficiency Index" (Your Formula)
# Add small epsilon to denominator to prevent division by zero
epsilon = 1e-8
df['inefficiency_index'] = df['sum_abs_log_ret'] / (df['abs_sum_log_ret'] + epsilon)

# Add Volatility (Std Dev) as a secondary context feature
# (Sometimes high volatility is chop, sometimes it's a breakout)
df['volatility'] = df['log_ret'].rolling(window_size).std()

# Drop NaNs
feature_cols = ['inefficiency_index', 'volatility']
df.dropna(subset=feature_cols, inplace=True)

# Clip outliers: If price doesn't move at all, index explodes to infinity.
# We cap it at 20 (meaning path was 20x longer than the net move)
df['inefficiency_index'] = df['inefficiency_index'].clip(upper=20)

# 4. TRAINING
# -----------
X = df[feature_cols]
y = df['target']

# Scale features (Critical for regression)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

model = LogisticRegression(class_weight='balanced', C=0.5)
model.fit(X_train, y_train)

# 5. VISUALIZATION
# ----------------
df['prediction'] = model.predict(X_scaled)
df['prob_chop'] = model.predict_proba(X_scaled)[:, 1]

plt.figure(figsize=(14, 10))

# Plot 1: Price and Chop Zones
ax1 = plt.subplot(3, 1, 1)
ax1.plot(df.index, df['close'], color='black', alpha=0.6, linewidth=1, label='BTC Price')
# Highlight Actual Choppy Zones (Green)
for start, end in choppy_ranges:
    ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='green', alpha=0.15, label='Actual Chop' if start==choppy_ranges[0][0] else "")
# Highlight Predicted Chop (Red Dots)
chop_preds = df[df['prediction'] == 1]
ax1.scatter(chop_preds.index, chop_preds['close'], color='red', s=2, label='Predicted Chop', zorder=3)
ax1.set_title('BTC Price vs Choppy Detection')
ax1.legend(loc='upper left')

# Plot 2: Your Formula (Inefficiency Index)
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.plot(df.index, df['inefficiency_index'], color='purple', linewidth=1)
ax2.set_title('Inefficiency Index (Your Formula)')
ax2.set_ylabel('Ratio (Vol / Direction)')
ax2.axhline(df['inefficiency_index'].mean(), color='gray', linestyle='--', label='Average')
ax2.grid(True, alpha=0.3)

# Plot 3: Probability Output
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
ax3.plot(df.index, df['prob_chop'], color='blue', alpha=0.8, linewidth=1)
ax3.fill_between(df.index, df['prob_chop'], color='blue', alpha=0.1)
ax3.axhline(0.5, color='red', linestyle='--')
ax3.set_title('Model Probability of Chop')
ax3.set_ylabel('Probability')

plt.tight_layout()

# Save plot
plot_dir = '/app/static'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
plot_path = os.path.join(plot_dir, 'plot.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Flask
app = Flask(__name__)
@app.route('/')
def serve_plot(): return send_file(plot_path, mimetype='image/png')
@app.route('/health')
def health(): return 'OK', 200

if __name__ == '__main__':
    print("Feature Importance:")
    print(pd.DataFrame({'Feature': feature_cols, 'Coeff': model.coef_[0]}))
    print("\nStarting Web Server...")
    app.run(host='0.0.0.0', port=8080, debug=False)
