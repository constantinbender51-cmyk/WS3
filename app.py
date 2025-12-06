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

# 3. BACKTEST ENGINE (SHADOW VS REAL)
# -----------------------------------
# "Shadow" tracks the strategy performance if we NEVER stopped trading.
# "Real" tracks our equity with the circuit breaker applied.

df['shadow_equity'] = 1.0
df['real_equity'] = 1.0
df['buy_hold_equity'] = 1.0
df['shadow_dd'] = 0.0
df['state'] = 'ACTIVE' # ACTIVE, COOLDOWN, EXTENDED

shadow_equity = 1.0
real_equity = 1.0
hold_equity = 1.0
shadow_peak = 1.0

cooldown_counter = 0

start_idx = SMA_SLOW

for i in range(start_idx, len(df)):
    today = df.index[i]
    
    # Yesterday's Data (Signal)
    prev_close = df['close'].iloc[i-1]
    prev_fast = df['sma_fast'].iloc[i-1]
    prev_slow = df['sma_slow'].iloc[i-1]
    
    # Today's Data (Execution)
    open_p = df['open'].iloc[i]
    high_p = df['high'].iloc[i]
    low_p = df['low'].iloc[i]
    close_p = df['close'].iloc[i]
    
    # --- A. CALCULATE RAW STRATEGY RETURN (SHADOW) ---
    strategy_daily_ret = 0.0
    
    # Check Long
    if prev_close > prev_fast and prev_close > prev_slow:
        entry = open_p
        sl = entry * (1 - SL_PCT)
        tp = entry * (1 + TP_PCT)
        if low_p <= sl: strategy_daily_ret = -SL_PCT
        elif high_p >= tp: strategy_daily_ret = TP_PCT
        else: strategy_daily_ret = (close_p - entry) / entry
        
    # Check Short
    elif prev_close < prev_fast and prev_close < prev_slow:
        entry = open_p
        sl = entry * (1 + SL_PCT)
        tp = entry * (1 - TP_PCT)
        if high_p >= sl: strategy_daily_ret = -SL_PCT
        elif low_p <= tp: strategy_daily_ret = TP_PCT
        else: strategy_daily_ret = (entry - close_p) / entry
        
    # Update Shadow Equity (The "Ghost" Strategy)
    shadow_equity *= (1 + strategy_daily_ret)
    if shadow_equity > shadow_peak:
        shadow_peak = shadow_equity
    
    current_shadow_dd = (shadow_equity - shadow_peak) / shadow_peak
    
    # Store Shadow Data (Needed for lookback checks)
    df.at[today, 'shadow_equity'] = shadow_equity
    df.at[today, 'shadow_dd'] = current_shadow_dd
    
    # --- B. UPDATE BUY & HOLD ---
    bh_ret = (close_p - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
    hold_equity *= (1 + bh_ret)
    
    # --- C. REAL PORTFOLIO LOGIC (CIRCUIT BREAKER) ---
    real_daily_ret = 0.0
    
    if cooldown_counter > 0:
        # We are in Cooldown (Cash)
        cooldown_counter -= 1
        real_daily_ret = 0.0
        
        # If counter hits 0, we perform the "Extension Check"
        if cooldown_counter == 0:
            # Look back 7 days at SHADOW equity
            # logic: If shadow strategy lost money in last 7 days, it's unsafe to re-enter.
            idx_7d = i - 7
            if idx_7d >= 0:
                past_shadow = df['shadow_equity'].iloc[idx_7d]
                # Avoid division by zero
                if past_shadow > 0:
                    recent_perf = (shadow_equity - past_shadow) / past_shadow
                else:
                    recent_perf = 0
                
                if recent_perf < 0:
                    # Strategy is still bleeding. Extend.
                    cooldown_counter = EXTENSION_COOLDOWN
                    df.at[today, 'state'] = 'EXTENDED'
                else:
                    # Strategy stabilized. Resume.
                    df.at[today, 'state'] = 'ACTIVE'
            else:
                df.at[today, 'state'] = 'ACTIVE'
        else:
            # Just verify state label
            if df.at[today, 'state'] == 'ACTIVE': # Should not happen if counter > 0, but good for safety
                df.at[today, 'state'] = 'COOLDOWN'
                
    else:
        # We are Active
        # Check Trigger Condition based on SHADOW DD
        # Use a small buffer to avoid immediate re-trigger if we just resumed at -15.1%
        # But generally, if shadow is deep in DD, we rely on the 7-day slope check to keep us out.
        # Here we check if we fall below limit.
        
        if current_shadow_dd < DRAWDOWN_LIMIT:
            # Trigger!
            cooldown_counter = INITIAL_COOLDOWN
            real_daily_ret = 0.0 # Go to cash immediately (simulate close at open?) 
            # In backtest we assume we don't take the trade today if triggered at yesterday's close/today's open
            # Note: Ideally trigger is based on Yesterday's DD. 
            # Let's be strict: If YESTERDAY's Shadow DD < -15%, we don't trade today.
            
            # Re-check causality:
            # current_shadow_dd is calculated using TODAY's close. We can't use it to stop TODAY's trade.
            # We must use YESTERDAY's Shadow DD.
            
            prev_shadow_dd = df['shadow_dd'].iloc[i-1] if i > 0 else 0
            if prev_shadow_dd < DRAWDOWN_LIMIT:
                 cooldown_counter = INITIAL_COOLDOWN
                 real_daily_ret = 0.0
                 df.at[today, 'state'] = 'TRIGGERED'
            else:
                # Safe to trade today
                real_daily_ret = strategy_daily_ret
                df.at[today, 'state'] = 'ACTIVE'
        else:
             real_daily_ret = strategy_daily_ret
             df.at[today, 'state'] = 'ACTIVE'

    real_equity *= (1 + real_daily_ret)
    
    df.at[today, 'real_equity'] = real_equity
    df.at[today, 'buy_hold_equity'] = hold_equity

# 4. VISUALIZATION
# ----------------
plt.figure(figsize=(14, 12))

# Plot 1: Equity Curves
ax1 = plt.subplot(3, 1, 1)
plot_data = df.iloc[start_idx:]
ax1.plot(plot_data.index, plot_data['real_equity'], label='Real Equity (Protected)', color='blue', linewidth=2)
ax1.plot(plot_data.index, plot_data['shadow_equity'], label='Shadow Equity (Unprotected)', color='red', alpha=0.5, linestyle='--', linewidth=1)
ax1.plot(plot_data.index, plot_data['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.3)
ax1.set_yscale('log')
ax1.set_title('Real vs Shadow Equity')
ax1.legend()
ax1.grid(True, which='both', linestyle='--', alpha=0.3)

# Plot 2: Shadow Drawdown & Triggers
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.plot(plot_data.index, plot_data['shadow_dd'], color='black', alpha=0.6, label='Shadow Drawdown')
ax2.axhline(DRAWDOWN_LIMIT, color='red', linestyle='--', label='-15% Limit')
ax2.fill_between(plot_data.index, plot_data['shadow_dd'], DRAWDOWN_LIMIT, 
                 where=plot_data['shadow_dd'] < DRAWDOWN_LIMIT, color='red', alpha=0.1)
ax2.set_title('Shadow Drawdown (The "Trigger" Source)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Active vs Cooldown States
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
# 0 = Active, 1 = Cooldown/Triggered, 2 = Extended
state_map = {'ACTIVE': 0, 'TRIGGERED': 1, 'COOLDOWN': 1, 'EXTENDED': 2}
num_state = plot_data['state'].map(state_map)

ax3.fill_between(plot_data.index, 0, 1, where=num_state>=1, color='orange', alpha=0.5, label='Initial Pause (20 Days)')
ax3.fill_between(plot_data.index, 0, 1, where=num_state==2, color='darkred', alpha=0.6, label='Extended Pause (Strategy still losing)')
ax3.plot(plot_data.index, plot_data['close'], color='green', linewidth=1, label='BTC Price')
ax3.set_title('Circuit Breaker States (Orange=20d, Red=7d Extension)')
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
    print(f"Final Real Equity: {real_equity:.2f}x")
    print(f"Final Shadow Equity: {shadow_equity:.2f}x")
    print("\nStarting Web Server...")
    app.run(host='0.0.0.0', port=8080, debug=False)
