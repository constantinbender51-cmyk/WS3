import ccxt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import http.server
import socketserver
import time
import threading
from datetime import datetime, timedelta

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
FEES = 0.0004 # 0.04%
OUTPUT_DIR = 'trade_plots'
PORT = int(os.environ.get("PORT", 8000))

# --- Data Management ---
def fetch_data(days, limit=1000):
    exchange = ccxt.binance()
    now = exchange.milliseconds()
    since = now - (days * 24 * 60 * 60 * 1000) - (100 * 3600 * 1000) # Buffer for lookback
    all_candles = []
    
    while since < now:
        try:
            candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since, limit)
            if not candles: break
            since = candles[-1][0] + 1
            all_candles += candles
            time.sleep(exchange.rateLimit / 1000)
        except Exception: break

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    
    # Drop the last candle as it is likely unfinished (unless fetched exactly at close, but better safe)
    if len(df) > 0:
        df = df.iloc[:-1]
        
    return df

# --- Strategy Core ---
def fit_line(x, y):
    X = sm.add_constant(x)
    return sm.OLS(y, X).fit()

def get_line_point(results, x_val):
    return results.params[0] + results.params[1] * x_val

def detect_breakout(df, i, window_size):
    start_idx = i - window_size
    window_df = df.iloc[start_idx:i]
    current_candle = df.iloc[i]
    
    x = window_df.index.values
    y = window_df['close'].values
    
    # 1. Mid Line
    mid_model = fit_line(x, y)
    mid_preds = mid_model.predict(sm.add_constant(x))
    
    # 2. Split
    upper_mask = y > mid_preds
    lower_mask = y < mid_preds
    
    upper_x, upper_y = x[upper_mask], window_df.iloc[upper_mask]['high'].values
    lower_x, lower_y = x[lower_mask], window_df.iloc[lower_mask]['low'].values
    
    res = {'signal': None, 'stop_dist': 0, 'params': (None, None, mid_model)}
    
    # 3. Fit Bounds
    upper_model = None
    lower_model = None
    
    bullish = False
    bearish = False
    
    if len(upper_x) > 2:
        upper_model = fit_line(upper_x, upper_y)
        if current_candle['close'] > get_line_point(upper_model, i):
            bullish = True
            res['breakout_level'] = get_line_point(upper_model, i)
            
    if len(lower_x) > 2:
        lower_model = fit_line(lower_x, lower_y)
        if current_candle['close'] < get_line_point(lower_model, i):
            bearish = True
            res['breakout_level'] = get_line_point(lower_model, i)
    
    res['params'] = (upper_model, lower_model, mid_model)
    res['window_df'] = window_df
    
    if bullish or bearish:
        res['signal'] = 'long' if bullish else 'short'
        entry = current_candle['close']
        # Dynamic stop distance calculation
        dist = abs(entry - res['breakout_level']) / entry
        res['stop_dist'] = max(dist, 0.001) # Min 0.1% stop
        
    return res

# --- Backtesting Engine ---
def run_simulation(df, stop_config, save_plots=False):
    # stop_config: 'dynamic', 0.005, 0.01, 0.02
    trades = []
    equity = 100.0
    equity_curve = [equity]
    
    i = 100
    while i < len(df) - 1:
        trade_found = False
        
        # Window shrinking loop 100 -> 10
        for window_size in range(100, 9, -1):
            detection = detect_breakout(df, i, window_size)
            
            if detection['signal']:
                direction = detection['signal']
                entry_price = df.iloc[i]['close']
                
                # Determine Stop Distance
                if stop_config == 'dynamic':
                    stop_pct = detection['stop_dist']
                else:
                    stop_pct = float(stop_config)
                
                # Execute Trade
                exit_data = simulate_trade_execution(df, i, direction, stop_pct)
                
                # Calculate Result (Compounding + Fees)
                # Entry Fee
                position_size = equity * (1 - FEES)
                
                # Price Movement
                raw_return = (exit_data['price'] - entry_price) / entry_price
                if direction == 'short': raw_return *= -1
                
                position_val = position_size * (1 + raw_return)
                
                # Exit Fee
                equity = position_val * (1 - FEES)
                equity_curve.append(equity)
                
                trade_record = {
                    'index': len(trades),
                    'entry_idx': i,
                    'exit_idx': exit_data['idx'],
                    'entry_price': entry_price,
                    'exit_price': exit_data['price'],
                    'direction': direction,
                    'pnl_pct': raw_return,
                    'equity': equity,
                    'stop_dist': stop_pct
                }
                trades.append(trade_record)
                
                if save_plots and len(trades) <= 10:
                    context = {
                        'window_df': detection['window_df'],
                        'upper_params': detection['params'][0],
                        'lower_params': detection['params'][1],
                        'mid_params': detection['params'][2],
                        'pnl': raw_return
                    }
                    plot_trade(df, context, i, exit_data['idx'], entry_price, exit_data['price'], direction, len(trades))

                i = exit_data['idx'] + 1
                trade_found = True
                break
        
        if not trade_found:
            i += 1
            
    return trades, equity_curve

def simulate_trade_execution(df, entry_idx, direction, stop_pct):
    entry_price = df.iloc[entry_idx]['close']
    curr = entry_idx + 1
    
    highest = entry_price
    lowest = entry_price
    
    while curr < len(df):
        candle = df.iloc[curr]
        
        if direction == 'long':
            highest = max(highest, candle['high'])
            stop = highest * (1 - stop_pct)
            if candle['low'] <= stop:
                return {'price': stop, 'idx': curr}
        else:
            lowest = min(lowest, candle['low'])
            stop = lowest * (1 + stop_pct)
            if candle['high'] >= stop:
                return {'price': stop, 'idx': curr}
        curr += 1
        
    return {'price': df.iloc[-1]['close'], 'idx': len(df)-1}

# --- Plotting ---
def plot_trade(df, context, entry_idx, exit_idx, entry_price, exit_price, direction, plot_id):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    window_df = context['window_df']
    x_range = np.arange(window_df.index[0], entry_idx + 5)
    X_range = sm.add_constant(x_range)
    
    plt.figure(figsize=(10, 5))
    
    # Plot Candles (Subset)
    plot_start = max(0, window_df.index[0] - 5)
    plot_end = min(exit_idx + 10, len(df))
    sub = df.iloc[plot_start:plot_end]
    
    up = sub[sub.close >= sub.open]
    down = sub[sub.close < sub.open]
    
    plt.bar(up.index, up.close-up.open, 0.6, bottom=up.open, color='green')
    plt.bar(up.index, up.high-up.close, 0.1, bottom=up.close, color='green')
    plt.bar(up.index, up.low-up.open, 0.1, bottom=up.open, color='green')
    
    plt.bar(down.index, down.close-down.open, 0.6, bottom=down.open, color='red')
    plt.bar(down.index, down.high-down.open, 0.1, bottom=down.open, color='red')
    plt.bar(down.index, down.low-down.close, 0.1, bottom=down.close, color='red')
    
    # Lines
    mid = context['mid_params'].predict(X_range)
    plt.plot(x_range, mid, 'b--', alpha=0.5)
    
    if context['upper_params']:
        plt.plot(x_range, context['upper_params'].predict(X_range), 'g-')
    if context['lower_params']:
        plt.plot(x_range, context['lower_params'].predict(X_range), 'r-')
        
    plt.scatter(entry_idx, entry_price, marker='^', color='blue', s=100, zorder=10)
    plt.scatter(exit_idx, exit_price, marker='x', color='black', s=100, zorder=10)
    
    plt.title(f"Trade {plot_id} ({direction}) PnL: {context['pnl']:.2%}")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/trade_{plot_id}.png")
    plt.close()

def plot_equity(curve, title="Equity Curve"):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    plt.figure(figsize=(10, 5))
    plt.plot(curve)
    plt.title(title)
    plt.ylabel('Equity ($)')
    plt.xlabel('Trades')
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/equity_curve.png")
    plt.close()

def update_html(trades, best_config, is_live=False):
    html = f"""<html><body>
    <h1>System Status: {'LIVE' if is_live else 'BACKTEST COMPLETE'}</h1>
    <h2>Best Config: {best_config}</h2>
    <h3>Equity Curve</h3>
    <img src="equity_curve.png" width="800"><br>
    <h3>Recent Trades</h3>
    <table border="1">
    <tr><th>ID</th><th>Dir</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Eq</th></tr>
    """
    for t in reversed(trades[-20:]):
        html += f"<tr><td>{t['index']}</td><td>{t['direction']}</td><td>{t['entry_price']:.2f}</td><td>{t['exit_price']:.2f}</td><td>{t['pnl_pct']:.2%}</td><td>{t['equity']:.2f}</td></tr>"
    html += "</table><hr><h3>Setup Plots</h3>"
    
    images = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith('trade_')])
    for img in images:
        html += f"<img src='{img}' width='400'>"
        
    with open(f"{OUTPUT_DIR}/index.html", "w") as f:
        f.write(html)

# --- Workflow Control ---
def optimize_and_run():
    print("Fetching optimization data (20 days)...")
    df_opt = fetch_data(20)
    
    configs = ['dynamic', 0.005, 0.01, 0.02]
    results = {}
    
    print("Running optimization...")
    for cfg in configs:
        _, curve = run_simulation(df_opt, cfg)
        final_eq = curve[-1]
        results[cfg] = final_eq
        print(f"Config {cfg}: ${final_eq:.2f}")
        
    best_config = max(results, key=results.get)
    print(f"Winner: {best_config}")
    
    print("Fetching full data (60 days)...")
    df_full = fetch_data(60)
    
    print(f"Running full backtest with {best_config}...")
    trades, curve = run_simulation(df_full, best_config, save_plots=True)
    
    plot_equity(curve, f"Equity Curve ({best_config})")
    update_html(trades, best_config)
    
    return best_config, df_full, trades, curve

# --- Live Loop ---
def live_trader(best_config, initial_equity):
    print("Starting Live Trader...")
    equity = initial_equity
    trades = [] # Keep local track
    
    while True:
        now = datetime.utcnow()
        # Sleep until 5s past next hour
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        sleep_sec = (next_hour - now).total_seconds() + 5
        
        print(f"Live: Sleeping {sleep_sec:.1f}s until {next_hour.strftime('%H:%M:%S')} + 5s...")
        time.sleep(sleep_sec)
        
        print("Live: Waking up. Fetching data...")
        # Fetch enough history for 100 candle window
        df = fetch_data(5) # 5 days is plenty for 100h
        
        # Latest completed candle is at index -1
        i = len(df) - 1
        current_candle = df.iloc[i]
        
        print(f"Live: Analyzing candle {current_candle['timestamp']} Close: {current_candle['close']}")
        
        trade_found = False
        for window_size in range(100, 9, -1):
            det = detect_breakout(df, i, window_size)
            if det['signal']:
                print(f"Live: SIGNAL FOUND! {det['signal']}")
                
                # In live mode, we don't have the future, so we can't 'simulate' the exit immediately
                # We just log the entry signal. 
                # Real implementation would need a websocket or ticker loop to manage the open trade.
                # For this scope, we record the signal and hypothetically enter.
                
                direction = det['signal']
                stop_pct = det['stop_dist'] if best_config == 'dynamic' else float(best_config)
                
                trade_record = {
                    'index': len(trades) + 999, # Offset ID
                    'entry_idx': i,
                    'exit_idx': -1, # Unknown
                    'entry_price': current_candle['close'],
                    'exit_price': 0,
                    'direction': direction,
                    'pnl_pct': 0,
                    'equity': equity,
                    'stop_dist': stop_pct
                }
                trades.append(trade_record)
                
                # Append to HTML (simplified)
                with open(f"{OUTPUT_DIR}/index.html", "a") as f:
                    f.write(f"<p>LIVE SIGNAL: {direction} @ {current_candle['close']} (Stop: {stop_pct:.2%})</p>")
                
                trade_found = True
                break
        
        if not trade_found:
            print("Live: No signal.")

def start_server():
    os.chdir(OUTPUT_DIR)
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"Serving at 0.0.0.0:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # 1. Optimize & Backtest
    best_cfg, _, trades, curve = optimize_and_run()
    
    # 2. Start Web Server (Thread)
    t_server = threading.Thread(target=start_server)
    t_server.daemon = True
    t_server.start()
    
    # 3. Enter Live Loop
    # Pass final equity from backtest as starting point
    live_trader(best_cfg, curve[-1])
