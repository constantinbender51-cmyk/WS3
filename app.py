import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, send_file
import os

# 1. PERMANENT CONFIGURATION (Statically Applied)
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'  # This is the variable the error was referring to
START_DATE = '2018-01-01 00:00:00'

# Strategy Parameters
SMA_FAST = 32
SMA_SLOW = 114
III_WINDOW = 27
TP_PCT = 0.126
SL_PCT = 0.043

# Regime / Filter Parameters
U_FLAT_THRESH = 0.356  # Flat regime threshold
Y_BAND_WIDTH = 0.077   # 7.7% Band width for SMA filters

# Leverage Tier Parameters
LEV_THRESH_LOW = 0.058
LEV_THRESH_HIGH = 0.259
LEV_LOW = 0.079
LEV_MID = 4.327
LEV_HIGH = 3.868

def fetch_binance_history(symbol, start_str):
    print(f"Fetching data for {symbol} starting from {start_str}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(start_str)
    all_ohlcv = []
    while True:
        try:
            # Fixed: Using the global TIMEFRAME variable
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if since > exchange.milliseconds(): break
        except Exception as e:
            print(f"Error fetching: {e}")
            break
    
    if not all_ohlcv:
        raise Exception("No data fetched from Binance. Check your symbol or connection.")
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

# 2. DATA PREPARATION
df = fetch_binance_history(SYMBOL, START_DATE)

# Indicators
df['sma_fast'] = df['close'].rolling(SMA_FAST).mean()
df['sma_slow'] = df['close'].rolling(SMA_SLOW).mean()

# Intrinsic Intensity Index (III)
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
df['net_direction'] = df['log_ret'].rolling(III_WINDOW).sum().abs()
df['path_length'] = df['log_ret'].abs().rolling(III_WINDOW).sum()
df['iii'] = df['net_direction'] / (df['path_length'] + 1e-8)

# 3. BACKTEST LOGIC
print("Running backtest with optimized parameters...")
start_idx = max(SMA_SLOW, III_WINDOW)

# Pre-calculate Tier-based Leverages
iii_vals = df['iii'].shift(1).fillna(0).values
lev_assignments = np.zeros(len(df))

for i in range(len(df)):
    val = iii_vals[i]
    if val < LEV_THRESH_LOW:
        lev_assignments[i] = LEV_LOW
    elif val < LEV_THRESH_HIGH:
        lev_assignments[i] = LEV_MID
    else:
        lev_assignments[i] = LEV_HIGH

df['base_ret'] = 0.0
df['strategy_ret'] = 0.0
df['equity'] = 1.0
df['active_leverage'] = 0.0

equity = 1.0
is_busted = False

for i in range(start_idx, len(df)):
    prev_close = df['close'].iloc[i-1]
    prev_fast = df['sma_fast'].iloc[i-1]
    prev_slow = df['sma_slow'].iloc[i-1]
    prev_iii = df['iii'].iloc[i-1]
    
    open_p = df['open'].iloc[i]
    high_p = df['high'].iloc[i]
    low_p = df['low'].iloc[i]
    close_p = df['close'].iloc[i]
    
    daily_base_ret = 0.0
    
    # 1. Band Filter Logic (using y)
    upper_band_fast = prev_fast * (1 + Y_BAND_WIDTH)
    lower_band_fast = prev_fast * (1 - Y_BAND_WIDTH)
    upper_band_slow = prev_slow * (1 + Y_BAND_WIDTH)
    lower_band_slow = prev_slow * (1 - Y_BAND_WIDTH)
    
    # 2. Flat Regime Filter (using u)
    is_flat_regime = prev_iii < U_FLAT_THRESH
    
    # Execution Logic
    if not is_flat_regime:
        # Bullish Trend
        if prev_close > upper_band_fast and prev_close > upper_band_slow:
            entry = open_p
            sl = entry * (1 - SL_PCT)
            tp = entry * (1 + TP_PCT)
            if low_p <= sl: daily_base_ret = -SL_PCT
            elif high_p >= tp: daily_base_ret = TP_PCT
            else: daily_base_ret = (close_p - entry) / entry
            
        # Bearish Trend
        elif prev_close < lower_band_fast and prev_close < lower_band_slow:
            entry = open_p
            sl = entry * (1 + SL_PCT)
            tp = entry * (1 - TP_PCT)
            if high_p >= sl: daily_base_ret = -SL_PCT
            elif low_p <= tp: daily_base_ret = TP_PCT
            else: daily_base_ret = (entry - close_p) / entry
    
    # Apply Leverage Tiers
    leverage = lev_assignments[i]
    strategy_ret = daily_base_ret * leverage
    
    if not is_busted:
        equity *= (1 + strategy_ret)
        if equity <= 0.01:
            equity = 0
            is_busted = True
            
    df.at[df.index[i], 'base_ret'] = daily_base_ret
    df.at[df.index[i], 'strategy_ret'] = strategy_ret
    df.at[df.index[i], 'equity'] = equity
    df.at[df.index[i], 'active_leverage'] = leverage if daily_base_ret != 0 else 0

# 4. METRICS CALCULATION
plot_df = df.iloc[start_idx:].copy()
days = (plot_df.index[-1] - plot_df.index[0]).days
cagr = (plot_df['equity'].iloc[-1]**(365/days)) - 1 if plot_df['equity'].iloc[-1] > 0 else -1

# Sharpe
active_rets = plot_df['strategy_ret'][plot_df['strategy_ret'] != 0]
sharpe = (active_rets.mean() / active_rets.std() * np.sqrt(365)) if len(active_rets) > 1 else 0

# Max Drawdown
roll_max = plot_df['equity'].cummax()
drawdown = (plot_df['equity'] - roll_max) / roll_max
max_dd = drawdown.min()

print("\n" + "="*50)
print("FINAL STATIC BACKTEST RESULTS")
print("="*50)
print(f"Total Return:    {(plot_df['equity'].iloc[-1]-1)*100:.2f}%")
print(f"CAGR:            {cagr*100:.2f}%")
print(f"Max Drawdown:    {max_dd*100:.2f}%")
print(f"Sharpe Ratio:    {sharpe:.2f}")
print("-" * 50)
print(f"SMA Fast/Slow:   {SMA_FAST} / {SMA_SLOW}")
print(f"TP/SL %:         {TP_PCT*100}% / {SL_PCT*100}%")
print(f"Leverage Tiers:  {LEV_LOW}x / {LEV_MID}x / {LEV_HIGH}x")
print("="*50 + "\n")

# 5. VISUALIZATION
plt.figure(figsize=(12, 12))

# Equity Curve
ax1 = plt.subplot(3, 1, 1)
ax1.plot(plot_df.index, plot_df['equity'], label='Strategy Equity', color='#2ecc71', linewidth=2)
ax1.set_yscale('log')
ax1.set_title(f'Optimized Strategy Performance (CAGR: {cagr*100:.1f}%)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Leverage and III
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.fill_between(plot_df.index, 0, plot_df['active_leverage'], color='#9b59b6', alpha=0.3, label='Active Leverage')
ax2.plot(plot_df.index, plot_df['iii'], color='#3498db', alpha=0.5, linewidth=0.5, label='III Value')
ax2.axhline(U_FLAT_THRESH, color='red', linestyle='--', alpha=0.6, label='Flat Thresh (u)')
ax2.set_title('Leverage Exposure & Intensity Index')
ax2.legend(loc='upper left')

# Drawdown
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
ax3.fill_between(plot_df.index, drawdown, 0, color='#e74c3c', alpha=0.3)
ax3.plot(plot_df.index, drawdown, color='#c0392b', linewidth=1)
ax3.set_title(f'Drawdown Profile (Max: {max_dd*100:.1f}%)')
ax3.set_ylabel('DD %')

plt.tight_layout()
os.makedirs('/app/static', exist_ok=True)
plot_path = '/app/static/plot.png'
plt.savefig(plot_path, dpi=300)

app = Flask(__name__)
@app.route('/')
def serve_plot(): return send_file(plot_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
