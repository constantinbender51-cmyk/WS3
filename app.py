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
INITIAL_COOLDOWN = 20    # Initial pause
EXTENSION_COOLDOWN = 7   # Extend if still dropping

# ⚠️ DANGER ZONE ⚠️
LEVERAGE = 2.0  # Try 2.0, 3.0, etc.

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

cooldown_counter = 0
is_busted = False

start_idx = SMA_SLOW

for i in range(start_idx, len(df)):
    today = df.index[i]
    
    # Yesterday's Data (Signal)
    prev_close = df['close'].iloc[i-1]
    prev_fast = df['sma_fast'].iloc[i-1]
    prev_slow = df['sma_slow'].iloc[i-1]
    
    # Today's Execution Data
    open_p = df['open'].iloc[i]
    high_p = df['high'].iloc[i]
    low_p = df['low'].iloc[i]
    close_p = df['close'].iloc[i]
    
    # --- A. BASE STRATEGY RETURN (1x) ---
    raw_strategy_ret = 0.0
    
    # Long
    if prev_close > prev_fast and prev_close > prev_slow:
        entry = open_p
        sl = entry * (1 - SL_PCT)
        tp = entry * (1 + TP_PCT)
        if low_p <= sl: raw_strategy_ret = -SL_PCT
        elif high_p >= tp: raw_strategy_ret = TP_PCT
        else: raw_strategy_ret = (close_p - entry) / entry
        
    # Short
    elif prev_close < prev_fast and prev_close < prev_slow:
        entry = open_p
        sl = entry * (1 + SL_PCT)
        tp = entry * (1 - TP_PCT)
        if high_p >= sl: raw_strategy_ret = -SL_PCT
        elif low_p <= tp: raw_strategy_ret = TP_PCT
        else: raw_strategy_ret = (entry - close_p) / entry
        
    # Update Shadow (1x Signal Source)
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
        # 1. Handle Cooldown / State
        if cooldown_counter > 0:
            cooldown_counter -= 1
            real_daily_ret = 0.0
            
            # Extension Logic (Look at Shadow 1x Performance)
            if cooldown_counter == 0:
                idx_7d = i - 7
                if idx_7d >= 0:
                    past_shadow = df['shadow_equity'].iloc[idx_7d]
                    if past_shadow > 0:
                        recent_perf = (shadow_equity - past_shadow) / past_shadow
                        if recent_perf < 0:
                            cooldown_counter = EXTENSION_COOLDOWN
                            df.at[today, 'state'] = 'EXTENDED'
                        else:
                            df.at[today, 'state'] = 'ACTIVE'
                    else:
                         df.at[today, 'state'] = 'ACTIVE'
                else:
                    df.at[today, 'state'] = 'ACTIVE'
            else:
                if df.at[today, 'state'] == 'ACTIVE': df.at[today, 'state'] = 'COOLDOWN'
                
        else:
            # Active Trading
            # Check Trigger (Based on Shadow DD)
            prev_shadow_dd = df['shadow_dd'].iloc[i-1] if i > 0 else 0
            
            if prev_shadow_dd < DRAWDOWN_LIMIT:
                cooldown_counter = INITIAL_COOLDOWN
                real_daily_ret = 0.0
                df.at[today, 'state'] = 'TRIGGERED'
            else:
                # APPLY LEVERAGE HERE
                # Note: Costs of borrowing are ignored for simplicity, but spread is deadly here.
                real_daily_ret = raw_strategy_ret * LEVERAGE
                df.at[today, 'state'] = 'ACTIVE'

        real_equity *= (1 + real_daily_ret)
        
        # Check Bankruptcy
        if real_equity <= 0.05: # Effectively busted at 95% loss
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
ax1.plot(plot_data.index, plot_data['real_equity'], label=f'Real Equity ({LEVERAGE}x Lev)', color='blue', linewidth=2)
ax1.plot(plot_data.index, plot_data['shadow_equity'], label='Shadow Equity (1x Signal)', color='red', alpha=0.5, linestyle='--', linewidth=1)
ax1.plot(plot_data.index, plot_data['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.3)
ax1.set_yscale('log')
ax1.set_title(f'Leveraged Strategy ({LEVERAGE}x) vs Shadow vs Buy&Hold')
ax1.legend()
ax1.grid(True, which='both', linestyle='--', alpha=0.3)

# Plot 2: Shadow Drawdown
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.plot(plot_data.index, plot_data['shadow_dd'], color='black', alpha=0.6, label='Shadow Drawdown (1x)')
ax2.axhline(DRAWDOWN_LIMIT, color='red', linestyle='--', label='Trigger Level')
ax2.set_title('Signal Drawdown (Shadow)')
ax2.grid(True, alpha=0.3)

# Plot 3: States
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
state_map = {'ACTIVE': 0, 'TRIGGERED': 1, 'COOLDOWN': 1, 'EXTENDED': 2, 'BUSTED': 3}
num_state = plot_data['state'].map(state_map)

ax3.fill_between(plot_data.index, 0, 1, where=num_state==1, color='orange', alpha=0.5, label='Cooldown')
ax3.fill_between(plot_data.index, 0, 1, where=num_state==2, color='darkred', alpha=0.6, label='Extended')
ax3.fill_between(plot_data.index, 0, 1, where=num_state==3, color='black', alpha=1.0, label='LIQUIDATED')
ax3.plot(plot_data.index, plot_data['close'], color='green', linewidth=1, label='BTC Price')
ax3.set_title(f'Trading State (Leverage: {LEVERAGE}x)')
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
