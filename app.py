import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, send_file
import os

# 1. CONFIGURATION
# ----------------
symbol = 'BTC/USDT'
timeframe = '1d'
start_date_str = '2018-01-01 00:00:00'

# Strategy Params
SMA_FAST = 40
SMA_SLOW = 120
SL_PCT = 0.02
TP_PCT = 0.16

# Circuit Breaker Params
DRAWDOWN_LIMIT = -0.15  # -15%
COOLDOWN_DAYS = 20      # Stay out for ~3 weeks

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

# 2. PREPARE DATA
# ---------------
df = fetch_binance_history(symbol, start_date_str)

# Calculate Indicators
df['sma_fast'] = df['close'].rolling(SMA_FAST).mean()
df['sma_slow'] = df['close'].rolling(SMA_SLOW).mean()

# 3. BACKTEST ENGINE
# ------------------
df['strategy_equity'] = 1.0
df['buy_hold_equity'] = 1.0
df['drawdown'] = 0.0
df['position'] = 'CASH'
df['state'] = 'ACTIVE' # ACTIVE or COOLDOWN

equity = 1.0
peak_equity = 1.0
hold_equity = 1.0

# Circuit Breaker State
cooldown_counter = 0
last_trigger_equity = 0.0 # To prevent immediate re-triggering

start_idx = SMA_SLOW

for i in range(start_idx, len(df)):
    today = df.index[i]
    
    # Yesterday's Data (Signal Generation)
    prev_close = df['close'].iloc[i-1]
    prev_fast = df['sma_fast'].iloc[i-1]
    prev_slow = df['sma_slow'].iloc[i-1]
    
    # Today's Data (Execution)
    open_p = df['open'].iloc[i]
    high_p = df['high'].iloc[i]
    low_p = df['low'].iloc[i]
    close_p = df['close'].iloc[i]
    
    # 1. Update Buy & Hold
    bh_ret = (close_p - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
    hold_equity *= (1 + bh_ret)
    
    # 2. Check Circuit Breaker Status
    # Update Peak Equity
    if equity > peak_equity:
        peak_equity = equity
        
    current_dd = (equity - peak_equity) / peak_equity
    
    # Cooldown Logic
    if cooldown_counter > 0:
        # We are cooling down
        cooldown_counter -= 1
        position = 'CASH'
        daily_ret = 0.0
        df.at[today, 'state'] = 'COOLDOWN'
        
    else:
        # We are active
        # Check if we need to Trigger Circuit Breaker
        # Condition: DD < -15% AND we are below the equity level where we last triggered
        # (This ensures we don't trigger daily if we stay flat at -15%)
        if current_dd < DRAWDOWN_LIMIT and equity < (last_trigger_equity if last_trigger_equity > 0 else equity + 1):
            cooldown_counter = COOLDOWN_DAYS
            last_trigger_equity = equity # Set new floor
            position = 'CASH'
            daily_ret = 0.0
            df.at[today, 'state'] = 'TRIGGERED'
            print(f"[{today.date()}] Circuit Breaker Triggered! DD: {current_dd:.2%}. Cooling down for {COOLDOWN_DAYS} days.")
        else:
            # Normal Trading Logic
            df.at[today, 'state'] = 'ACTIVE'
            position = 'CASH'
            daily_ret = 0.0
            
            # Long Signal
            if prev_close > prev_fast and prev_close > prev_slow:
                position = 'LONG'
                entry = open_p
                sl = entry * (1 - SL_PCT)
                tp = entry * (1 + TP_PCT)
                
                if low_p <= sl: daily_ret = -SL_PCT
                elif high_p >= tp: daily_ret = TP_PCT
                else: daily_ret = (close_p - entry) / entry
            
            # Short Signal
            elif prev_close < prev_fast and prev_close < prev_slow:
                position = 'SHORT'
                entry = open_p
                sl = entry * (1 + SL_PCT)
                tp = entry * (1 - TP_PCT)
                
                if high_p >= sl: daily_ret = -SL_PCT
                elif low_p <= tp: daily_ret = TP_PCT
                else: daily_ret = (entry - close_p) / entry

            equity *= (1 + daily_ret)

    df.at[today, 'strategy_equity'] = equity
    df.at[today, 'buy_hold_equity'] = hold_equity
    df.at[today, 'drawdown'] = current_dd
    df.at[today, 'position'] = position

# 4. VISUALIZATION
# ----------------
plt.figure(figsize=(14, 12))

# Equity Curve
ax1 = plt.subplot(3, 1, 1)
plot_data = df.iloc[start_idx:]
ax1.plot(plot_data.index, plot_data['strategy_equity'], label='Strategy w/ Circuit Breaker', color='blue', linewidth=2)
ax1.plot(plot_data.index, plot_data['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.5)
ax1.set_yscale('log')
ax1.set_title(f'Strategy Equity (Circuit Breaker: 15% DD -> {COOLDOWN_DAYS} Days Cash)')
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend()

# Drawdown Curve
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.plot(plot_data.index, plot_data['drawdown'], color='red', linewidth=1)
ax2.fill_between(plot_data.index, plot_data['drawdown'], 0, color='red', alpha=0.1)
ax2.axhline(DRAWDOWN_LIMIT, color='black', linestyle='--', label='-15% Trigger')
ax2.set_title('Strategy Drawdown')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Circuit Breaker Active Zones
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
# Create a numerical series for state: 1 for active, 0 for cooldown
state_series = plot_data['state'].apply(lambda x: 1 if x in ['COOLDOWN', 'TRIGGERED'] else 0)
ax3.fill_between(plot_data.index, 0, 1, where=state_series==1, color='orange', alpha=0.5, label='Circuit Breaker Active (Cash)')
ax3.plot(plot_data.index, plot_data['close'], color='green', linewidth=1, label='Price')
ax3.set_title('Circuit Breaker Activation Periods (Orange = Forced Cash)')
ax3.legend()

plt.tight_layout()

# Save
plot_dir = '/app/static'
if not os.path.exists(plot_dir): os.makedirs(plot_dir)
plot_path = os.path.join(plot_dir, 'plot.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Flask
app = Flask(__name__)
@app.route('/')
def serve_plot(): return send_file(plot_path, mimetype='image/png')
@app.route('/health')
def health(): return 'OK', 200

if __name__ == '__main__':
    print(f"Final Strategy Equity: {equity:.2f}x")
    print("\nStarting Web Server...")
    app.run(host='0.0.0.0', port=8080, debug=False)
