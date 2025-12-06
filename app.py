import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from flask import Flask, send_file
import os

# 1. SETUP & CONFIGURATION
# ------------------------
symbol = 'BTC/USDT'
timeframe = '1d'
start_date_str = '2018-01-01 00:00:00'

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

df['target'] = 0
for start, end in choppy_ranges:
    mask = (df.index >= start) & (df.index <= end)
    df.loc[mask, 'target'] = 1

# 3. FEATURE ENGINEERING
# ----------------------
window_size = 60
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

# Numerator: Sum of Absolute Log Returns
df['sum_abs_log_ret'] = df['log_ret'].abs().rolling(window_size).mean()

# Denominator: Absolute Sum of Log Returns
df['abs_sum_log_ret'] = df['log_ret'].rolling(window_size).mean().abs()

# Inefficiency Index
epsilon = 1e-8
df['inefficiency_index'] = df['sum_abs_log_ret'] / (df['abs_sum_log_ret'] + epsilon)
df['inefficiency_index'] = df['inefficiency_index'].clip(upper=20)

train_df = df.dropna(subset=['inefficiency_index']).copy()

# 4. TRAINING
# -----------
X = train_df[['inefficiency_index']]
y = train_df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(class_weight='balanced', C=0.5)
model.fit(X_scaled, y)

# Predict on whole DF
df_features = df[['inefficiency_index']].fillna(0)
df_scaled = scaler.transform(df_features)
df['prediction'] = model.predict(df_scaled)

# 5. STRATEGY BACKTEST (Fixed Causality)
# --------------------
df['sma_40'] = df['close'].rolling(40).mean()
df['sma_120'] = df['close'].rolling(120).mean()

SL_PCT = 0.02
TP_PCT = 0.16

df['strategy_equity'] = 1.0
df['buy_hold_equity'] = 1.0
df['position'] = 'CASH'
df['daily_return'] = 0.0
df['causal_drawdown'] = 0.0 # New column to store causal drawdown values

current_equity = 1.0
hold_equity = 1.0
max_equity_observed = 1.0 # This will track the highest `current_equity` reached so far.

# Start loop
start_idx = max(120, window_size)

for i in range(start_idx, len(df)):
    today = df.index[i]
    
    # Yesterday's data (Day T-1) for decision making on Day T
    prev_close = df['close'].iloc[i-1]
    prev_sma40 = df['sma_40'].iloc[i-1]
    prev_sma120 = df['sma_120'].iloc[i-1]
    prev_prediction_is_choppy = df['prediction'].iloc[i-1] == 1 # Model's choppy prediction from yesterday
    
    # Calculate yesterday's drawdown based on the equity at the end of day i-1
    # `current_equity` at this point holds the value for day i-1's close.
    # `max_equity_observed` at this point holds the max equity up to day i-1's close.
    yesterday_drawdown_value = (current_equity - max_equity_observed) / max_equity_observed
    
    # Combine conditions to decide if we trade today
    is_market_restricted_today = False
    
    if prev_prediction_is_choppy: # If model predicted choppy yesterday
        is_market_restricted_today = True
    
    if yesterday_drawdown_value < -0.15: # If yesterday's drawdown was > 15%
        is_market_restricted_today = True
        
    # Execution Data (Day T)
    open_price = df['open'].iloc[i]
    high_price = df['high'].iloc[i]
    low_price = df['low'].iloc[i]
    close_price = df['close'].iloc[i]
    
    daily_ret = 0.0
    position = 'CASH'
    
    # Buy & Hold Logic (always trades)
    bh_ret = (close_price - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
    hold_equity *= (1 + bh_ret)
    
    # Strategy Logic
    if not is_market_restricted_today: # Only trade if not restricted
        # Long Logic
        if prev_close > prev_sma40 and prev_close > prev_sma120:
            position = 'LONG'
            entry = open_price
            stop_loss = entry * (1 - SL_PCT)
            take_profit = entry * (1 + TP_PCT)
            
            if low_price <= stop_loss:
                daily_ret = -SL_PCT
            elif high_price >= take_profit:
                daily_ret = TP_PCT
            else:
                daily_ret = (close_price - entry) / entry
                
        # Short Logic
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
    
    current_equity *= (1 + daily_ret) # Update current_equity for day `i`
    
    # Update max_equity_observed *after* `current_equity` has been updated for day `i`
    max_equity_observed = max(max_equity_observed, current_equity)
    
    df.at[today, 'strategy_equity'] = current_equity
    df.at[today, 'buy_hold_equity'] = hold_equity
    df.at[today, 'position'] = position
    df.at[today, 'causal_drawdown'] = (current_equity - max_equity_observed) / max_equity_observed

# 6. VISUALIZATION
# ----------------
plt.figure(figsize=(14, 20))

plot_data = df.iloc[start_idx:]

# Calculate drawdown for visualization
rolling_max = plot_data['strategy_equity'].cummax()
drawdown = (plot_data['strategy_equity'] - rolling_max) / rolling_max

# Plot 1: Equity
ax1 = plt.subplot(4, 1, 1)
ax1.plot(plot_data.index, plot_data['strategy_equity'], label='Strategy (Strictly Causal)', color='blue', linewidth=2)
ax1.plot(plot_data.index, plot_data['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.5)
# Mark periods with drawdown > 15%
y_min, y_max = ax1.get_ylim()
ax1.fill_between(plot_data.index, y_min, y_max,
                 where=(drawdown < -0.15), facecolor='purple', alpha=0.2, label='Drawdown > 15%')
ax1.set_yscale('log')
ax1.set_title('Strategy Equity Curve (Log Scale)')
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend()

# Plot 2: Drawdown
ax2 = plt.subplot(4, 1, 2, sharex=ax1)
ax2.plot(plot_data.index, drawdown, color='red', alpha=0.6)
ax2.fill_between(plot_data.index, drawdown, 0, color='red', alpha=0.1)
ax2.set_ylabel('Drawdown')
ax2.set_title('Strategy Drawdown')
ax2.grid(True, alpha=0.3)

# Plot 3: Price + Chop Flags
ax3 = plt.subplot(4, 1, 3, sharex=ax1)
ax3.plot(plot_data.index, plot_data['close'], label='Close Price', color='green', linewidth=1)
ax3.set_title('Price Action vs Model Choppy Flags (Red)')
ax3.set_ylabel('Price')
ax3.grid(True, alpha=0.3)
# Overlay Choppy Signal (using i-1 prediction effectively shifted to today)
# Note: For visualization, we just show where prediction was 1
ax3.fill_between(plot_data.index, ax3.get_ylim()[0], ax3.get_ylim()[1],
                 where=plot_data['prediction'] == 1, facecolor='red', alpha=0.15, label='Choppy Regime')
ax3.legend()

# Plot 4: Inefficiency Index
ax4 = plt.subplot(4, 1, 4, sharex=ax1)
ax4.plot(plot_data.index, plot_data['inefficiency_index'], label='Inefficiency Index', color='purple', linewidth=1)
ax4.set_title('Inefficiency Index')
ax4.set_ylabel('Index Value')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()

plot_dir = '/app/static'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
plot_path = os.path.join(plot_dir, 'plot.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

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
