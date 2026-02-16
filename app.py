import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
OUPUT_TRADES = 3             # How many trades to visualize
MIN_WINDOW = 10
MAX_WINDOW = 100
THRESHOLD_PCT = 0.10
exchange = ccxt.binance()

def fit_ols(x, y):
    """Standard OLS Regression"""
    if len(x) < 2: return None, None
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def fetch_data():
    print(f"Fetching 2000 candles for {SYMBOL}...")
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1000)
    # Fetch a bit more to ensure we get trades
    since = ohlcv[0][0] - (1000 * 60 * 60 * 1000)
    prev = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
    data = prev + ohlcv
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def plot_trade_forensics(df, trade, trade_id):
    """Generates a detailed chart for a specific completed trade"""
    entry_idx = trade['entry_index']
    exit_idx = trade['exit_index']
    window = trade['window']
    
    # Zoom in: Show 50 candles before entry and 20 after exit
    start_plot = max(0, entry_idx - window - 20)
    end_plot = min(len(df), exit_idx + 20)
    df_zoom = df.iloc[start_plot:end_plot].reset_index(drop=True)
    
    # Re-calculate the specific OLS lines that triggered the ENTRY
    # Note: We fit on [entry_idx - window - 1 : entry_idx - 1] 
    # This proves we did NOT use the breakout candle for the fit
    abs_entry_idx = entry_idx
    
    fit_slice = slice(abs_entry_idx - window - 1, abs_entry_idx - 1)
    
    # We need to map these absolute indices to our zoomed dataframe indices
    # relative_entry_idx = df_zoom[df_zoom['timestamp'] == df.iloc[entry_idx]['timestamp']].index[0]
    
    plt.figure(figsize=(14, 8))
    plt.style.use('dark_background')
    
    # 1. Plot Candles
    up = df_zoom[df_zoom.close >= df_zoom.open]
    down = df_zoom[df_zoom.close < df_zoom.open]
    plt.bar(up.index, up.close - up.open, 0.6, bottom=up.open, color='green', alpha=0.6)
    plt.bar(up.index, up.high - up.close, 0.1, bottom=up.close, color='green', alpha=0.6)
    plt.bar(up.index, up.open - up.low, 0.1, bottom=up.low, color='green', alpha=0.6)
    
    plt.bar(down.index, down.close - down.open, 0.6, bottom=down.open, color='red', alpha=0.6)
    plt.bar(down.index, down.high - down.open, 0.1, bottom=down.open, color='red', alpha=0.6)
    plt.bar(down.index, down.close - down.low, 0.1, bottom=down.low, color='red', alpha=0.6)

    # 2. Re-Construct the Signal Channel (The "Why we entered" lines)
    # We fit on the "Lagged" window (Original DF indices)
    x_fit_abs = np.arange(abs_entry_idx - window - 1, abs_entry_idx - 1)
    y_fit = df['close'].iloc[fit_slice].values
    yh_fit = df['high'].iloc[fit_slice].values
    yl_fit = df['low'].iloc[fit_slice].values
    
    mm, cm = fit_ols(x_fit_abs, y_fit)
    
    # Map lines to the Zoomed Plot X-Axis
    # Find offset between absolute index and zoom index
    # We need to find where x_fit_abs[0] is in df_zoom
    
    # Simplified plotting: We just project the lines based on the visual chart indices
    # We find the relative index of the entry candle
    rel_entry = df_zoom[df_zoom.timestamp == df.iloc[entry_idx].timestamp].index[0]
    
    # Create the x-array for the channel relative to the entry
    x_channel = np.arange(rel_entry - window - 1, rel_entry + 5) # Draw past entry a bit
    
    # Recalculate coefficients on the zoomed data values (math is invariant to x-shift if slope is preserved)
    # Actually, simpler to just re-fit on the exact same values using relative X
    y_vals = df_zoom.loc[rel_entry - window - 1 : rel_entry - 2, 'close'].values
    h_vals = df_zoom.loc[rel_entry - window - 1 : rel_entry - 2, 'high'].values
    l_vals = df_zoom.loc[rel_entry - window - 1 : rel_entry - 2, 'low'].values
    x_vals = np.arange(rel_entry - window - 1, rel_entry - 1)
    
    m, c = fit_ols(x_vals, y_vals)
    yt = m * x_vals + c
    mu, cu = fit_ols(x_vals[h_vals > yt], h_vals[h_vals > yt])
    ml, cl = fit_ols(x_vals[l_vals < yt], l_vals[l_vals < yt])
    
    # Plot Channel
    x_proj = np.arange(rel_entry - window - 1, rel_entry + 2)
    upper_line = mu * x_proj + cu
    lower_line = ml * x_proj + cl
    
    plt.plot(x_proj, upper_line, color='white', linestyle='--', linewidth=1, label="Upper Band (At Entry)")
    plt.plot(x_proj, lower_line, color='white', linestyle='--', linewidth=1, label="Lower Band (At Entry)")
    
    # 3. Plot Thresholds
    dist = upper_line[-1] - lower_line[-1]
    thresh = dist * THRESHOLD_PCT
    
    # 4. Markers
    # Entry
    plt.plot(rel_entry, df.iloc[entry_idx]['close'], marker='^', color='yellow', markersize=12, label='ENTRY')
    # Exit
    rel_exit = df_zoom[df_zoom.timestamp == df.iloc[exit_idx].timestamp].index[0]
    plt.plot(rel_exit, df.iloc[exit_idx]['close'], marker='X', color='orange', markersize=12, label='EXIT')
    
    # Annotations
    pnl = trade['pnl']
    res = "WIN" if pnl > 0 else "LOSS"
    plt.title(f"Trade #{trade_id} | {trade['type'].upper()} | PnL: {pnl:.2f} ({res}) | Window: {window}")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # Save or Show
    plt.show()

def run_forensic_backtest():
    df = fetch_data()
    print(f"Data Loaded: {len(df)} candles.")
    
    active_trade = None
    completed_trades = []
    
    # Standard Logic Loop
    for i in range(MAX_WINDOW + 1, len(df)):
        price = df.iloc[i]['close']
        
        # 1. Check Exit
        if active_trade:
            t = active_trade
            closed = False
            pnl = 0
            
            # Dynamic Stop Calculation (Recalculate band on current window)
            w = t['window']
            # Fit on [i-w : i] (Lagged fit relative to NOW)
            x_fit = np.arange(i - w, i)
            yc = df['close'].values[i-w:i]; yh = df['high'].values[i-w:i]; yl = df['low'].values[i-w:i]
            
            mm, cm = fit_ols(x_fit, yc)
            if mm is not None:
                yt = mm * x_fit + cm
                mu, cu = fit_ols(x_fit[yh > yt], yh[yh > yt])
                ml, cl = fit_ols(x_fit[yl < yt], yl[yl < yt])
                
                if mu and ml:
                    # Dynamic Stop Levels
                    stop_long = ml * i + cl
                    stop_short = mu * i + cu
                    
                    if t['type'] == 'long':
                        if price <= stop_long or price >= t['target']:
                            pnl = price - t['entry']; closed = True
                    elif t['type'] == 'short':
                        if price >= stop_short or price <= t['target']:
                            pnl = t['entry'] - price; closed = True
            
            if closed:
                t['exit_index'] = i
                t['exit_price'] = price
                t['pnl'] = pnl
                completed_trades.append(t)
                active_trade = None
                print(f"Trade Closed. PnL: {pnl:.2f}")
                if len(completed_trades) >= OUPUT_TRADES: break
            continue # Only 1 trade at a time
            
        # 2. Check Entry
        # Fit on [i-w-1 : i-1] (Exclude current candle i)
        last_c = df.iloc[i-1]['close'] # The breakout candle
        
        # Optimize: Scan largest window first
        for w in range(MAX_WINDOW, MIN_WINDOW - 1, -1):
            x_fit = np.arange(i - w - 1, i - 1)
            yc = df['close'].values[i-w-1:i-1]
            yh = df['high'].values[i-w-1:i-1]
            yl = df['low'].values[i-w-1:i-1]
            
            mm, cm = fit_ols(x_fit, yc)
            if mm is None: continue
            
            yt = mm * x_fit + cm
            mu, cu = fit_ols(x_fit[yh > yt], yh[yh > yt])
            ml, cl = fit_ols(x_fit[yl < yt], yl[yl < yt])
            
            if mu and ml:
                # Project to breakout candle (i-1)
                proj_idx = i - 1
                proj_u = mu * proj_idx + cu
                proj_l = ml * proj_idx + cl
                dist = proj_u - proj_l
                th = dist * THRESHOLD_PCT
                
                # Check Breakout
                if last_c > (proj_u + th):
                    active_trade = {
                        'type': 'long', 'entry': df.iloc[i]['open'], # Enter on Open of current
                        'target': proj_u + dist, 'window': w, 'entry_index': i
                    }
                    print(f"Long Entry at {active_trade['entry']} (Window {w})")
                    break
                elif last_c < (proj_l - th):
                    active_trade = {
                        'type': 'short', 'entry': df.iloc[i]['open'],
                        'target': proj_l - dist, 'window': w, 'entry_index': i
                    }
                    print(f"Short Entry at {active_trade['entry']} (Window {w})")
                    break

    # 3. Visualize
    for idx, t in enumerate(completed_trades):
        plot_trade_forensics(df, t, idx+1)

if __name__ == "__main__":
    run_forensic_backtest()
