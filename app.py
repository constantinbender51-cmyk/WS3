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

# The specific "Losing Periods" (Drawdown > 15%) you identified
# Format: (Start Date, End Date)
losing_ranges = [
    ('2018-08-27', '2018-09-06'),
    ('2018-09-13', '2018-09-16'),
    ('2018-09-19', '2018-11-18'),
    ('2018-11-29', '2018-11-29'),
    ('2019-08-20', '2019-09-23'),
    ('2020-03-25', '2020-03-26'),
    ('2020-04-03', '2020-05-06'),
    ('2020-05-11', '2020-05-11'),
    ('2020-07-09', '2020-07-22'),
    ('2020-07-24', '2020-07-24'),
    ('2021-01-19', '2021-02-18'),
    ('2021-02-20', '2021-02-20'),
    ('2021-02-23', '2021-03-08'),
    ('2021-03-10', '2021-03-12'),
    ('2021-03-14', '2021-11-07'),
    ('2021-11-09', '2022-01-21'),
    ('2022-01-23', '2022-06-10'),
    ('2022-07-06', '2022-07-10'),
    ('2022-07-15', '2022-09-05'),
    ('2022-09-07', '2022-09-19'),
    ('2022-09-21', '2022-11-20'),
    ('2022-11-22', '2023-01-13'),
    ('2023-01-15', '2023-01-15'),
    ('2023-01-18', '2023-01-18'),
    ('2024-07-21', '2024-11-09'),
    ('2025-01-28', '2025-01-28'),
    ('2025-02-01', '2025-02-25'),
    ('2025-03-01', '2025-03-02'),
    ('2025-03-06', '2025-03-06'),
    ('2025-09-07', '2025-09-11'),
    ('2025-09-14', '2025-09-15'),
    ('2025-09-21', '2025-10-05'),
    ('2025-10-07', '2025-10-07'),
    ('2025-10-09', '2025-10-15'),
    ('2025-10-19', '2025-11-16'),
    ('2025-11-18', '2025-11-18')
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

# Labeling Target: 1 if date is in a Losing Period, 0 otherwise
df['target'] = 0
for start, end in losing_ranges:
    mask = (df.index >= start) & (df.index <= end)
    df.loc[mask, 'target'] = 1

# 3. FEATURE ENGINEERING
# ----------------------
# Calculate Inefficiency Index with window=14
window_size = 14
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
df['sum_abs_log_ret'] = df['log_ret'].abs().rolling(window_size).mean()
df['abs_sum_log_ret'] = df['log_ret'].rolling(window_size).mean().abs()
epsilon = 1e-8
df['inefficiency_index'] = df['sum_abs_log_ret'] / (df['abs_sum_log_ret'] + epsilon)
df['inefficiency_index'] = df['inefficiency_index'].clip(upper=20)

# Create 14 Lagged Features of the Index
# "14 days of inefficiency index"
feature_cols = []
for i in range(14):
    col_name = f'ineff_lag_{i}'
    df[col_name] = df['inefficiency_index'].shift(i)
    feature_cols.append(col_name)

# Drop NaNs
df.dropna(subset=feature_cols, inplace=True)

# 4. TRAINING
# -----------
X = df[feature_cols]
y = df['target']

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# Split for validation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

model = LogisticRegression(class_weight='balanced', C=0.5, max_iter=1000)
model.fit(X_train, y_train)

# 5. EVALUATION
# -------------
predictions = model.predict(X_test)
print("\n--- Model Evaluation (Test Set) ---")
print(classification_report(y_test, predictions))

# Feature Importance
print("\n--- Feature Importance (Correlation to Losing Periods) ---")
importance = pd.DataFrame({'Lag': range(14), 'Coefficient': model.coef_[0]})
print(importance.sort_values(by='Coefficient', ascending=False))

# Predict on full dataset for plotting
df['prediction'] = model.predict(X_scaled)
df['prob_loss'] = model.predict_proba(X_scaled)[:, 1]

# 6. VISUALIZATION
# ----------------
plt.figure(figsize=(14, 12))

# Subplot 1: Price & Actual Losing Zones
ax1 = plt.subplot(3, 1, 1)
ax1.plot(df.index, df['close'], color='black', alpha=0.7, label='BTC Price')
ax1.set_yscale('log')
# Highlight Actual Losing Periods (Red)
for start, end in losing_ranges:
    ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='red', alpha=0.2, label='Actual Drawdown' if start==losing_ranges[0][0] else "")
# Highlight periods where predicted probability of loss is > 0.5 (Yellow)
ax1.fill_between(df.index, ax1.get_ylim()[0], ax1.get_ylim()[1],
                 where=(df['prob_loss'] > 0.5),
                 color='yellow', alpha=0.3, label='Predicted > 0.5 Prob')
ax1.set_title('BTC Price vs Actual Losing Periods (>15% DD)')
ax1.legend()
ax1.grid(True, which='both', linestyle='--', alpha=0.3)

# Subplot 2: The Inefficiency Index
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.plot(df.index, df['inefficiency_index'], color='purple', linewidth=1)
ax2.set_title('Inefficiency Index (Window=14)')
ax2.set_ylabel('Index Value')
ax2.grid(True, alpha=0.3)

# Subplot 3: Predicted Probability of Loss
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
ax3.plot(df.index, df['prob_loss'], color='orange', linewidth=1, label='Prob. of Losing Period')
ax3.axhline(0.5, color='gray', linestyle='--')
ax3.fill_between(df.index, 0, df['prob_loss'], color='orange', alpha=0.2)
ax3.set_title('Model Prediction: Probability of being in a Losing Period')
ax3.set_ylabel('Probability')
ax3.grid(True, alpha=0.3)

plt.tight_layout()

# Save
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
    print("\nStarting Web Server...")
    app.run(host='0.0.0.0', port=8080, debug=False)
