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
DRAWDOWN_LIMIT = -0.15   # -15%

# ⚠️ DANGER ZONE ⚠️
LEVERAGE = 2.0 

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

# 3. BACKTEST ENGINE (SHADOW VS LEVERAGED REAL)
# ---------------------------------------------
df['shadow_equity'] = 1.0
df['real_equity'] = 1.0
df['buy_hold_equity'] = 1.0
df['shadow_dd'] = 0.0
df['state'] = 'ACTIVE' 

shadow_equity = 1.0
real_equity = 1.0
hold_equity = 1.0
shadow_peak = 1.0

is_busted = False
state = 'ACTIVE' # Simple State Machine: ACTIVE or STOPPED

start_idx = SMA_SLOW

for i in range(start_idx, len(df)):
    today = df.index[i]
    
    # Yesterday's Data (Signal)
    prev_close = df['close'].iloc[i-1]
    prev_fast = df['sma_fast'].iloc[i-1]
    prev_slow = df['sma_slow'].iloc[i-1]
    
    # Two days ago (For Crossover Detection)
    prev2_close = df['close'].iloc[i-2]
    prev2_fast = df['sma_fast'].iloc[i-2]
    prev2_slow = df['sma_slow'].iloc[i-2]
    
    # Execution Data
    open_p = df['open'].iloc[i]
    high_p = df['high'].iloc[i]
    low_p = df['low'].iloc[i]
    close_p = df['close'].iloc[i]
    
    # --- A. BASE STRATEGY RETURN (1x SHADOW) ---
    # We always run this to calculate the "True Market Condition"
    raw_strategy_ret = 0.0
    
    # Long Logic
    if prev_close > prev_fast and prev_close > prev_slow:
        entry = open_p
        sl = entry * (1 - SL_PCT)
        tp = entry * (1 + TP_PCT)
        if low_p <= sl: raw_strategy_ret = -SL_PCT
        elif high_p >= tp: raw_strategy_ret = TP_PCT
        else: raw_strategy_ret = (close_p - entry) / entry
        
    # Short Logic
    elif prev_close < prev_fast and prev_close < prev_slow:
        entry = open_p
        sl = entry * (1 + SL_PCT)
        tp = entry * (1 - TP_PCT)
        if high_p >= sl: raw_strategy_ret = -SL_PCT
        elif low_p <= tp: raw_strategy_ret = TP_PCT
        else: raw_strategy_ret = (entry - close_p) / entry
        
    # Update Shadow
    shadow_equity *= (1 + raw_strategy_ret)
    if shadow_equity > shadow_peak: shadow_peak = shadow_equity
    current_shadow_dd = (shadow_equity - shadow_peak) / shadow_peak
    
    df.at[today, 'shadow_equity'] = shadow_equity
    df.at[today, 'shadow_dd'] = current_shadow_dd
    
    # --- B. BUY & HOLD ---
    bh_ret = (close_p - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
    hold_equity *= (1 + bh_ret)
    
    # --- C. LEVERAGED REAL PORTFOLIO ---
    if is_busted:
        real_daily_ret = 0.0
        df.at[today, 'state'] = 'BUSTED'
    else:
        # 1. CHECK FOR RESET SIGNAL (SMA CROSS)
        # Did Price Cross SMA 40 or SMA 120 Yesterday?
        cross_fast = (prev2_close < prev2_fast and prev_close > prev_fast) or \
                     (prev2_close > prev2_fast and prev_close < prev_fast)
        cross_slow = (prev2_close < prev2_slow and prev_close > prev_slow) or \
                     (prev2_close > prev2_slow and prev_close < prev_slow)
        
        any_cross = cross_fast or cross_slow
        
        # 2. STATE LOGIC
        # Priority: If Cross -> ACTIVE. Else Check Limit.
        
        if any_cross:
            state = 'ACTIVE'
            # If we wake up, we trade immediately
            real_daily_ret = raw_strategy_ret * LEVERAGE
            df.at[today, 'state'] = 'RESET' # Visual marker for wakeup
            
        elif state == 'STOPPED':
            # Still stopped, no cross yet
            real_daily_ret = 0.0
            df.at[today, 'state'] = 'STOPPED'
            
        else: # currently ACTIVE and no cross today
            # Check if we need to STOP based on Shadow DD
            # Using i-1 DD to be strictly causal
            prev_shadow_dd = df['shadow_dd'].iloc[i-1] if i > 0 else 0
            
            if prev_shadow_dd < DRAWDOWN_LIMIT:
                state = 'STOPPED'
                real_daily_ret = 0.0
                df.at[today, 'state'] = 'STOPPED'
            else:
                state = 'ACTIVE'
                real_daily_ret = raw_strategy_ret * LEVERAGE
                df.at[today, 'state'] = 'ACTIVE'

        # Apply Return
        real_equity *= (1 + real_daily_ret)
        
        # Check Bankruptcy
        if real_equity <= 0.05:
            real_equity = 0
            is_busted = True
            print(f"💀 ACCOUNT LIQUIDATED ON {today.date()} 💀")

    df.at[today, 'real_equity'] = real_equity
    df.at[today, 'buy_hold_equity'] = hold_equity

# 4. VISUALIZATION
# ----------------
plt.figure(figsize=(14, 12))

# Plot 1: Equity
ax1 = plt.subplot(3, 1, 1)
plot_data = df.iloc[start_idx:]
ax1.plot(plot_data.index, plot_data['real_equity'], label=f'Real Equity ({LEVERAGE}x)', color='blue', linewidth=2)
ax1.plot(plot_data.index, plot_data['shadow_equity'], label='Shadow Equity (1x)', color='red', alpha=0.5, linestyle='--', linewidth=1)
ax1.plot(plot_data.index, plot_data['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.3)
ax1.set_yscale('log')
ax1.set_title(f'Leveraged Strategy ({LEVERAGE}x) w/ Hard Stop & Cross Reset')
ax1.legend()
ax1.grid(True, which='both', linestyle='--', alpha=0.3)

# Plot 2: Shadow Drawdown
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.plot(plot_data.index, plot_data['shadow_dd'], color='black', alpha=0.6, label='Shadow Drawdown')
ax2.axhline(DRAWDOWN_LIMIT, color='red', linestyle='--', label='Hard Stop Limit')
ax2.set_title('Signal Drawdown (Shadow Account)')
ax2.grid(True, alpha=0.3)

# Plot 3: States
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
# Mapping for visualization
# ACTIVE=0, STOPPED=1, RESET=2, BUSTED=3
state_map = {'ACTIVE': 0, 'STOPPED': 1, 'RESET': 2, 'BUSTED': 3}
num_state = plot_data['state'].map(state_map)

ax3.fill_between(plot_data.index, 0, 1, where=num_state==1, color='red', alpha=0.3, label='STOPPED (Cash)')
ax3.fill_between(plot_data.index, 0, 1, where=num_state==2, color='purple', alpha=1.0, label='SMA Cross (Wake Up)')
ax3.fill_between(plot_data.index, 0, 1, where=num_state==3, color='black', alpha=1.0, label='LIQUIDATED')

ax3.plot(plot_data.index, plot_data['close'], color='green', linewidth=1, label='BTC Price')
ax3.set_title('Trading State (Red = Stopped until next SMA Cross)')
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
    print(f"Final Real Equity ({LEVERAGE}x): {real_equity:.2f}x")
    print(f"Final Shadow Equity (1x): {shadow_equity:.2f}x")
    print("\nStarting Web Server...")
    app.run(host='0.0.0.0', port=8080, debug=False)
