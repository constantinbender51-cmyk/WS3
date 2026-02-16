import ccxt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import http.server
import socketserver
import time
from datetime import datetime

# Configuration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LIMIT = 1000 
LOOKBACK_DAYS = 60 # Reduced to 2 months
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
    
    # Ensure unique index and reset
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    return df

def fit_line(x, y):
    # Add constant for intercept
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
    return results

def get_line_point(results, x_val):
    # FIXED: Use bracket indexing for numpy array
    return results.params[0] + results.params[1] * x_val

def simulate_trade(df, entry_idx, direction, stop_distance_pct, plot_id=None, context=None):
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
        exit_idx = len(df) - 1
        exit_price = df.iloc[exit_idx]['close']

    # Calculate PnL
    pnl_pct = (exit_price - entry_price) / entry_price if direction == 'long' else (entry_price - exit_price) / entry_price
    
    # Update context with result for plotting
    if context:
        context['pnl'] = pnl_pct
    
    # Plotting if requested
    if plot_id is not None and context is not None:
        plot_trade_setup(df, context, entry_idx, exit_idx, entry_price, exit_price, direction, plot_id)
        
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
    
    # Expand plot range
    plot_end = min(exit_idx + 15, len(df))
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
    x_range = np.arange(window_df.index[0], entry_idx + 2) 
    X_range = sm.add_constant(x_range)
    
    # Mid Line
    y_mid = mid_params.predict(X_range)
    plt.plot(x_range, y_mid, 'b--', alpha=0.5, label='Mid Line')
    
    # Upper Line
    if upper_params:
        X_upper = sm.add_constant(x_range)
        y_upper = upper_params.predict(X_upper)
        plt.plot(x_range, y_upper, 'g-', label='Upper Line')
        
    # Lower Line
    if lower_params:
        X_lower = sm.add_constant(x_range)
        y_lower = lower_params.predict(X_lower)
        plt.plot(x_range, y_lower, 'r-', label='Lower Line')
        
    # Markers
    marker = '^' if direction == 'long' else 'v'
    plt.scatter(entry_idx, entry_price, marker=marker, color='blue', s=100, zorder=10, label='Entry')
    plt.scatter(exit_idx, exit_price, marker='x', color='black', s=100, zorder=10, label='Exit')
    
    plt.title(f"Trade #{plot_id} - {direction.upper()} - PnL: {context['pnl']:.2%}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"{OUTPUT_DIR}/trade_{plot_id:03d}.png"
    plt.savefig(filename)
    plt.close()

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
    
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    for f in files:
        html += f"<h3>{f}</h3><img src='{f}' /><br>"
        
    html += "</body></html>"
    
    with open(f"{OUTPUT_DIR}/index.html", "w") as f:
        f.write(html)

def run_strategy():
    df = fetch_data()
    df = df.reset_index(drop=True)
    
    trades = []
    i = 100 
    
    print("Starting backtest...")
    
    while i < len(df) - 1:
        trade_found = False
        
        # Shrinking window loop: 100 down to 10
        for window_size in range(100, 9, -1):
            start_idx = i - window_size
            end_idx = i 
            
            window_df = df.iloc[start_idx:end_idx]
            current_candle = df.iloc[i]
            
            x = window_df.index.values
            y = window_df['close'].values
            
            # 1. Main Trendline
            mid_model = fit_line(x, y)
            mid_preds = mid_model.predict(sm.add_constant(x))
            
            # 2. Split Sets
            upper_mask = y > mid_preds
            lower_mask = y < mid_preds
            
            upper_set_x = x[upper_mask]
            upper_set_y = window_df.iloc[upper_mask]['high'].values 
            
            lower_set_x = x[lower_mask]
            lower_set_y = window_df.iloc[lower_mask]['low'].values 
            
            # 3. Fit 2nd and 3rd lines
            upper_model = None
            lower_model = None
            
            breakout_long = False
            breakout_short = False
            upper_val_now = 0
            lower_val_now = 0
            
            if len(upper_set_x) > 2:
                upper_model = fit_line(upper_set_x, upper_set_y)
                upper_val_now = get_line_point(upper_model, i)
                if current_candle['close'] > upper_val_now:
                    breakout_long = True
                    
            if len(lower_set_x) > 2:
                lower_model = fit_line(lower_set_x, lower_set_y)
                lower_val_now = get_line_point(lower_model, i)
                if current_candle['close'] < lower_val_now:
                    breakout_short = True
            
            if breakout_long or breakout_short:
                direction = 'long' if breakout_long else 'short'
                breakout_level = upper_val_now if breakout_long else lower_val_now
                
                entry_price = current_candle['close']
                dist_pct = abs(entry_price - breakout_level) / entry_price
                
                if dist_pct <= 0: dist_pct = 0.005
                
                context = {
                    'window_df': window_df,
                    'upper_params': upper_model,
                    'lower_params': lower_model,
                    'mid_params': mid_model,
                    'pnl': 0
                }
                
                plot_id = len(trades) + 1 if len(trades) < 10 else None
                
                result = simulate_trade(df, i, direction, dist_pct, plot_id, context)
                trades.append(result)
                
                print(f"Trade {len(trades)}: {direction} | Entry: {entry_price:.2f} | PnL: {result['pnl']:.2%} | Window: {window_size}")
                
                i = result['exit_idx'] + 1
                trade_found = True
                break 
        
        if not trade_found:
            i += 1 

    generate_html(trades)
    return trades

def start_server():
    # Switch to output directory
    os.chdir(OUTPUT_DIR)
    
    # Railway provides PORT env var, defaults to 8000
    PORT = int(os.environ.get("PORT", 8000))
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Bind to 0.0.0.0 for external access
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"Serving at 0.0.0.0:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    trades = run_strategy()
    
    if len(trades) > 0:
        start_server()
    else:
        print("No trades found. Exiting.")
