import ccxt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import http.server
import socketserver
import threading
import time
from datetime import datetime, timedelta

# Configuration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LIMIT = 1000 # Binance fetch limit per request
LOOKBACK_DAYS = 365
OUTPUT_DIR = 'trade_plots'

def fetch_data():
    exchange = ccxt.binance()
    now = exchange.milliseconds()
    since = now - (LOOKBACK_DAYS * 24 * 60 * 60 * 1000)
    all_candles = []
    
    print(f"Fetching {LOOKBACK_DAYS} days of {SYMBOL} {TIMEFRAME} data...")
    
    while since < now:
        try:
            candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since, LIMIT)
            if not candles:
                break
            
            since = candles[-1][0] + 1
            all_candles += candles
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Drop unfinished candle
    df = df.iloc[:-1]
    
    # Ensure unique index
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    return df

def fit_line(x, y):
    # Add constant for intercept
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
    return results

def get_line_point(results, x_val):
    return results.params.iloc[0] + results.params.iloc[1] * x_val

def simulate_trade(df, entry_idx, direction, stop_distance_pct, plot_id=None, context_data=None):
    entry_price = df.iloc[entry_idx]['close']
    current_idx = entry_idx + 1
    
    highest_price = entry_price
    lowest_price = entry_price
    
    exit_price = None
    exit_idx = None
    
    while current_idx < len(df):
        candle = df.iloc[current_idx]
        
        if direction == 'long':
            # Update high water mark
            if candle['high'] > highest_price:
                highest_price = candle['high']
            
            # Check trailing stop
            stop_price = highest_price * (1 - stop_distance_pct)
            if candle['low'] <= stop_price:
                exit_price = stop_price
                exit_idx = current_idx
                break
                
        elif direction == 'short':
            # Update low water mark
            if candle['low'] < lowest_price:
                lowest_price = candle['low']
            
            # Check trailing stop
            stop_price = lowest_price * (1 + stop_distance_pct)
            if candle['high'] >= stop_price:
                exit_price = stop_price
                exit_idx = current_idx
                break
        
        current_idx += 1
        
    if exit_idx is None:
        # Force close at end of data
        exit_idx = len(df) - 1
        exit_price = df.iloc[exit_idx]['close']

    pnl_pct = (exit_price - entry_price) / entry_price if direction == 'long' else (entry_price - exit_price) / entry_price
    
    # Plotting if requested
    if plot_id is not None and context_data is not None:
        plot_trade_setup(df, context_data, entry_idx, exit_idx, entry_price, exit_price, direction, plot_id)
        
    return {
        'entry_idx': entry_idx,
        'exit_idx': exit_idx,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'direction': direction,
        'pnl': pnl_pct,
        'stop_dist': stop_distance_pct
    }

def plot_trade_setup(df, context, entry_idx, exit_idx, entry_price, exit_price, direction, plot_id):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    window_df = context['window_df']
    upper_params = context['upper_params']
    lower_params = context['lower_params']
    mid_params = context['mid_params']
    
    # Expand plot range to include trade duration
    plot_end = min(exit_idx + 10, len(df))
    plot_start = max(0, window_df.index[0] - 5)
    plot_data = df.iloc[plot_start:plot_end]
    
    plt.figure(figsize=(12, 6))
    
    # Plot Candles
    up = plot_data[plot_data.close >= plot_data.open]
    down = plot_data[plot_data.close < plot_data.open]
    
    col1 = 'green'
    col2 = 'red'
    
    width = .6
    width2 = .1
    
    plt.bar(up.index, up.close-up.open, width, bottom=up.open, color=col1)
    plt.bar(up.index, up.high-up.close, width2, bottom=up.close, color=col1)
    plt.bar(up.index, up.low-up.open, width2, bottom=up.open, color=col1)
    
    plt.bar(down.index, down.close-down.open, width, bottom=down.open, color=col2)
    plt.bar(down.index, down.high-down.open, width2, bottom=down.open, color=col2)
    plt.bar(down.index, down.low-down.close, width2, bottom=down.close, color=col2)
    
    # Plot Regression Lines
    x_range = np.arange(window_df.index[0], entry_idx + 1) # Extend to breakout candle
    X_range = sm.add_constant(x_range)
    
    # Mid Line
    y_mid = mid_params.predict(X_range)
    plt.plot(x_range, y_mid, 'b--', alpha=0.5, label='Mid Line')
    
    # Upper Line
    if upper_params:
        y_upper = upper_params.predict(X_range)
        plt.plot(x_range, y_upper, 'g-', label='Upper Line')
        
    # Lower Line
    if lower_params:
        y_lower = lower_params.predict(X_range)
        plt.plot(x_range, y_lower, 'r-', label='Lower Line')
        
    # Markers
    plt.scatter(entry_idx, entry_price, marker='^' if direction == 'long' else 'v', color='blue', s=100, zorder=5, label='Entry')
    plt.scatter(exit_idx, exit_price, marker='x', color='black', s=100, zorder=5, label='Exit')
    
    plt.title(f"Trade #{plot_id} - {direction.upper()} - PnL: {context['pnl']:.2%}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"{OUTPUT_DIR}/trade_{plot_id:03d}.png"
    plt.savefig(filename)
    plt.close()

def run_strategy():
    df = fetch_data()
    # Reset index to be strictly numeric for OLS
    df = df.reset_index(drop=True)
    
    trades = []
    i = 100 # Start index allowing for max window
    
    print("Starting backtest...")
    
    while i < len(df) - 1:
        trade_found = False
        
        # Shrinking window loop: 100 down to 10
        for window_size in range(100, 9, -1):
            start_idx = i - window_size
            end_idx = i # Exclusive in python slice, so this is index 0 to 99 relative to current i
            
            window_df = df.iloc[start_idx:end_idx]
            current_candle = df.iloc[i]
            
            # 1. Main Trendline (OLS on Close)
            # Use relative indices for calculation to keep numbers small, but map back later
            x = window_df.index.values
            y = window_df['close'].values
            
            mid_model = fit_line(x, y)
            mid_preds = mid_model.predict(sm.add_constant(x))
            
            # 2. Split Sets
            # Note: Align boolean mask with array indices
            upper_mask = y > mid_preds
            lower_mask = y < mid_preds
            
            upper_set_x = x[upper_mask]
            upper_set_y = window_df.iloc[upper_mask]['high'].values # Fit to Highs
            
            lower_set_x = x[lower_mask]
            lower_set_y = window_df.iloc[lower_mask]['low'].values # Fit to Lows
            
            # 3. Fit 2nd and 3rd lines
            upper_model = None
            lower_model = None
            
            breakout_long = False
            breakout_short = False
            upper_val_now = 0
            lower_val_now = 0
            
            # Needs enough points to fit
            if len(upper_set_x) > 2:
                upper_model = fit_line(upper_set_x, upper_set_y)
                # Check breakout candle (i)
                upper_val_now = get_line_point(upper_model, i)
                if current_candle['close'] > upper_val_now:
                    breakout_long = True
                    
            if len(lower_set_x) > 2:
                lower_model = fit_line(lower_set_x, lower_set_y)
                # Check breakout candle (i)
                lower_val_now = get_line_point(lower_model, i)
                if current_candle['close'] < lower_val_now:
                    breakout_short = True
            
            # Logic: If both break (rare expanding volatility), prefer the one with stronger momentum? 
            # Or just take the first detected. Prompt implies distinct states.
            # We will prioritize the loop order (Long then Short or based on magnitude).
            # Let's take the first valid signal found.
            
            if breakout_long or breakout_short:
                direction = 'long' if breakout_long else 'short'
                breakout_level = upper_val_now if breakout_long else lower_val_now
                
                # Trailing stop distance
                entry_price = current_candle['close']
                dist_pct = abs(entry_price - breakout_level) / entry_price
                
                # Minimum viable stop to avoid noise (optional, but good practice. strict adherence to prompt: raw calc)
                if dist_pct <= 0: dist_pct = 0.005 # Fallback if line equals close
                
                # Execute Trade
                context = {
                    'window_df': window_df,
                    'upper_params': upper_model,
                    'lower_params': lower_model,
                    'mid_params': mid_model,
                    'pnl': 0 # placeholder
                }
                
                plot_id = len(trades) + 1 if len(trades) < 10 else None
                
                result = simulate_trade(df, i, direction, dist_pct, plot_id, context)
                
                # Update context pnl for plot label
                context['pnl'] = result['pnl']
                # Re-plot if needed to add correct PnL title (hacky but works)
                if plot_id:
                     plot_trade_setup(df, context, result['entry_idx'], result['exit_idx'], 
                                      result['entry_price'], result['exit_price'], direction, plot_id)

                trades.append(result)
                print(f"Trade {len(trades)}: {direction} | Entry: {entry_price:.2f} | PnL: {result['pnl']:.2%} | Window: {window_size}")
                
                # Advance index to end of trade + 1 to restart search
                i = result['exit_idx'] + 1
                trade_found = True
                break # Break inner loop (window shrinking)
        
        if not trade_found:
            i += 1 # Shift window + 1

    # Generate index.html
    generate_html(trades)
    return trades

def generate_html(trades):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    html = """
    <html>
    <head><title>Strategy Backtest Results</title>
    <style>body{font-family:monospace; padding:20px;} img{max-width:100%; border:1px solid #ccc; margin-bottom:20px;}</style>
    </head>
    <body>
    <h1>Backtest Results</h1>
    <p>Total Trades: """ + str(len(trades)) + """</p>
    """
    
    total_pnl = sum([t['pnl'] for t in trades])
    html += f"<h3>Cumulative PnL (Uncompounded): {total_pnl:.2%}</h3>"
    html += "<hr>"
    
    # List images
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    for f in files:
        html += f"<h3>{f}</h3><img src='{f}' /><br>"
        
    html += "</body></html>"
    
    with open(f"{OUTPUT_DIR}/index.html", "w") as f:
        f.write(html)

def start_server():
    os.chdir(OUTPUT_DIR)
    PORT = 8000
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at port {PORT}")
        print(f"Open http://localhost:{PORT} in your browser")
        httpd.serve_forever()

if __name__ == "__main__":
    trades = run_strategy()
    
    # Start server in main thread or daemon? User wants code that works. 
    # Usually blocking server at end is best.
    if len(trades) > 0:
        start_server()
    else:
        print("No trades found.")
