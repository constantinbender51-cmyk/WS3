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

# 3. FEATURE ENGINEERING (Updated per request)
# -------------------------------------
# Window increased to 60 days
window_size = 60

# Calculate Log Returns
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

# Numerator: Sum of Absolute Log Returns
df['sum_abs_log_ret'] = df['log_ret'].abs().rolling(window_size).mean()

# Denominator: Absolute Sum of Log Returns
df['abs_sum_log_ret'] = df['log_ret'].rolling(window_size).mean().abs()

# Inefficiency Index
epsilon = 1e-8
df['inefficiency_index'] = df['sum_abs_log_ret'] / (df['abs_sum_log_ret'] + epsilon)

# Volatility feature REMOVED as requested

# Clip outliers
df['inefficiency_index'] = df['inefficiency_index'].clip(upper=20)

# Drop NaNs for training
train_df = df.dropna(subset=['inefficiency_index']).copy()

# 4. TRAINING
# -----------
X = train_df[['inefficiency_index']]
y = train_df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# No split, training on full history to align with your "backtest" perspective 
# (Note: In a rigorous scientific context, this is lookahead bias, but standard for strategy prototyping)
model = LogisticRegression(class_weight='balanced', C=0.5)
model.fit(X_scaled, y)

# Generate Predictions for the whole DF
# We need to scale the original DF's features
df_features = df[['inefficiency_index']].fillna(0) # Fillna temp just to run predict, though rows are dropped in backtest
df_scaled = scaler.transform(df_features)
df['prediction'] = model.predict(df_scaled)

# 5. STRATEGY BACKTEST
# --------------------
# Technical Indicators for Strategy
df['sma_40'] = df['close'].rolling(40).mean()
df['sma_120'] = df['close'].rolling(120).mean()

# Strategy Parameters
SL_PCT = 0.02
TP_PCT = 0.16

# We'll iterate to handle the path-dependent SL/TP logic accurately
# Initialize results
df['strategy_equity'] = 1.0
df['buy_hold_equity'] = 1.0
df['position'] = 'CASH' # LONG, SHORT, CASH
df['daily_return'] = 0.0

current_equity = 1.0
hold_equity = 1.0

# Start loop after indicators are valid
start_idx = max(120, window_size)

for i in range(start_idx, len(df)):
    today = df.index[i]
    yesterday = df.index[i-1]
    
    # Yesterday's data for signal generation
    prev_close = df['close'].iloc[i-1]
    prev_sma40 = df['sma_40'].iloc[i-1]
    prev_sma120 = df['sma_120'].iloc[i-1]
    
    # Today's data for execution
    open_price = df['open'].iloc[i]
    high_price = df['high'].iloc[i]
    low_price = df['low'].iloc[i]
    close_price = df['close'].iloc[i]
    
    # Model Filter (Prediction from yesterday/today morning)
    # If model says 1 (Choppy), we hold CASH.
    is_choppy = df['prediction'].iloc[i] == 1
    
    daily_ret = 0.0
    position = 'CASH'
    
    # Buy & Hold Reference (Close to Close)
    bh_ret = (close_price - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
    hold_equity *= (1 + bh_ret)
    
    if not is_choppy:
        # Check Long Condition
        if prev_close > prev_sma40 and prev_close > prev_sma120:
            position = 'LONG'
            # Execution Logic
            entry = open_price
            stop_loss = entry * (1 - SL_PCT)
            take_profit = entry * (1 + TP_PCT)
            
            # Check stops based on Low/High
            if low_price <= stop_loss:
                # Stopped out
                daily_ret = -SL_PCT
            elif high_price >= take_profit:
                # Take profit
                daily_ret = TP_PCT
            else:
                # Held to close
                daily_ret = (close_price - entry) / entry
                
        # Check Short Condition
        elif prev_close < prev_sma40 and prev_close < prev_sma120:
            position = 'SHORT'
            entry = open_price
            stop_loss = entry * (1 + SL_PCT)
            take_profit = entry * (1 - TP_PCT)
            
            if high_price >= stop_loss:
                daily_ret = -SL_PCT
            elif low_price <= take_profit:
                daily_ret = TP_PCT
            else:
                daily_ret = (entry - close_price) / entry
    
    # Update Equity
    current_equity *= (1 + daily_ret)
    
    df.at[today, 'strategy_equity'] = current_equity
    df.at[today, 'buy_hold_equity'] = hold_equity
    df.at[today, 'position'] = position

# 6. VISUALIZATION
# ----------------
plt.figure(figsize=(14, 10))

# Plot 1: Equity Curves (Log Scale for readability)
ax1 = plt.subplot(2, 1, 1)
# Filter out the pre-start data for plotting
plot_data = df.iloc[start_idx:]
ax1.plot(plot_data.index, plot_data['strategy_equity'], label='Strategy (Trend + Anti-Chop)', color='blue', linewidth=2)
ax1.plot(plot_data.index, plot_data['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.5)
ax1.set_yscale('log')
ax1.set_title('Strategy Equity Curve (Log Scale)')
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend()

# Plot 2: Drawdown & Regime
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
# Calculate Drawdown
rolling_max = plot_data['strategy_equity'].cummax()
drawdown = (plot_data['strategy_equity'] - rolling_max) / rolling_max
ax2.plot(plot_data.index, drawdown, color='red', alpha=0.6, label='Drawdown')
ax2.fill_between(plot_data.index, drawdown, 0, color='red', alpha=0.1)

# Overlay Choppy Signal background
# Rescale signal to fit on the drawdown chart for visibility
ax2.set_ylabel('Drawdown')
ax2.set_title('Strategy Drawdown')
ax2.grid(True, alpha=0.3)

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
    print(f"Final Strategy Equity: {current_equity:.2f}x")
    print(f"Final Buy & Hold Equity: {hold_equity:.2f}x")
    print("\nStarting Web Server...")
    app.run(host='0.0.0.0', port=8080, debug=False)
