import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
import io
import base64
import http.server
import socketserver
import time
from datetime import datetime
import sys

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
WINDOW_SIZE = 50  # W parameter
PORT = 8080
DAYS_BACK = 365

# ==========================================
# DATA FETCHING
# ==========================================
def fetch_data():
    print(f"[SYSTEM] Fetching {DAYS_BACK} days of {SYMBOL} {TIMEFRAME} data...")
    exchange = ccxt.binance()
    limit = 1000
    since = exchange.parse8601((datetime.now() - pd.Timedelta(days=DAYS_BACK)).isoformat())
    
    all_candles = []
    while True:
        try:
            candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since, limit)
            if not candles:
                break
            since = candles[-1][0] + 1
            all_candles += candles
            print(f"\r[DATA] Fetched {len(all_candles)} candles...", end="")
            if len(candles) < limit:
                break
            time.sleep(0.1) # Rate limit respect
        except Exception as e:
            print(f"\n[ERROR] Data fetch failed: {e}")
            break
            
    print("\n[SYSTEM] Data fetch complete.")
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# ==========================================
# MATH & STRATEGY LOGIC
# ==========================================
def get_ols_prediction(x, y, target_x):
    """Fits y = mx + c and returns prediction for target_x"""
    if len(x) < 2: return None, None, None # Need at least 2 points
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return (m * target_x) + c, m, c

def run_backtest(df):
    print("[SYSTEM] Starting Forensic Backtest...")
    trades = []
    equity = [10000.0] # Starting capital
    active_trade = None
    
    # Pre-calculate arrays for speed where possible, but loop is needed for logic
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    times = df['timestamp'].values
    
    total_candles = len(df)
    
    # Store history for visualization
    history = []

    for i in range(WINDOW_SIZE + 1, total_candles):
        # Progress bar
        if i % 100 == 0:
            sys.stdout.write(f"\r[SIMULATION] Processing candle {i}/{total_candles}")
            sys.stdout.flush()

        current_close = closes[i]
        current_high = highs[i]
        current_low = lows[i]
        current_time = times[i]
        
        # 1. Manage Active Trade (Intra-bar Exit)
        if active_trade:
            entry_price = active_trade['entry_price']
            stop_loss = active_trade['stop_loss']
            take_profit = active_trade['take_profit']
            direction = active_trade['direction']
            
            exit_price = None
            exit_reason = None
            
            # Check High/Low for execution
            if direction == 'long':
                if current_low <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'Stop Loss'
                elif current_high >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'Take Profit'
            elif direction == 'short':
                if current_high >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'Stop Loss'
                elif current_low <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'Take Profit'
            
            if exit_price:
                pnl = (exit_price - entry_price) / entry_price if direction == 'long' else (entry_price - exit_price) / entry_price
                capital = equity[-1] * (1 + pnl)
                equity.append(capital)
                
                active_trade['exit_price'] = exit_price
                active_trade['exit_time'] = current_time
                active_trade['pnl'] = pnl
                active_trade['reason'] = exit_reason
                active_trade['end_idx'] = i
                trades.append(active_trade)
                active_trade = None
                continue # Trade closed, wait for next setup

        # 2. Strategy Calculation (Lagged Window)
        # Window: [i - W - 1] to [i - 1]
        start_idx = i - WINDOW_SIZE - 1
        end_idx = i - 1 # Inclusive in python slicing logic implies end_idx is limit
        
        # Slice data
        w_closes = closes[start_idx:end_idx]
        w_highs = highs[start_idx:end_idx]
        w_lows = lows[start_idx:end_idx]
        x_axis = np.arange(len(w_closes))
        
        # Center Line (OLS on Close)
        center_pred_curr, m_c, c_c = get_ols_prediction(x_axis, w_closes, len(w_closes)) # Predict for "current" relative index (which is W)
        
        # Calculate fit lines values over the window to split regressions
        center_line_values = (m_c * x_axis) + c_c
        
        # Identify Upper/Lower cohorts
        mask_upper = w_highs > center_line_values
        mask_lower = w_lows < center_line_values
        
        # Upper Line (OLS on Highs where High > Center)
        if np.sum(mask_upper) < 2: continue
        upper_pred_curr, m_u, c_u = get_ols_prediction(x_axis[mask_upper], w_highs[mask_upper], len(w_closes))
        
        # Lower Line (OLS on Lows where Low < Center)
        if np.sum(mask_lower) < 2: continue
        lower_pred_curr, m_l, c_l = get_ols_prediction(x_axis[mask_lower], w_lows[mask_lower], len(w_closes))

        # Breakout Trigger Logic
        dist = upper_pred_curr - lower_pred_curr
        if dist <= 0: continue # Crossed lines, invalid wedge
        
        threshold_long = upper_pred_curr + (dist * 0.10)
        threshold_short = lower_pred_curr - (dist * 0.10)
        
        # Store data for visualization if a trade happens
        current_setup = {
            'idx': i,
            'time': current_time,
            'window_indices': range(start_idx, end_idx),
            'center_params': (m_c, c_c),
            'upper_params': (m_u, c_u),
            'lower_params': (m_l, c_l),
            'threshold_long': threshold_long,
            'threshold_short': threshold_short,
            'dist': dist,
            'upper_val': upper_pred_curr,
            'lower_val': lower_pred_curr
        }

        # Entry Logic
        if current_close > threshold_long:
            stop_price = upper_pred_curr
            target_price = stop_price + dist
            active_trade = {
                'entry_time': current_time,
                'entry_price': current_close,
                'direction': 'long',
                'stop_loss': stop_price,
                'take_profit': target_price,
                'start_idx': i,
                'setup': current_setup
            }
        elif current_close < threshold_short:
            stop_price = lower_pred_curr
            target_price = stop_price - dist
            active_trade = {
                'entry_time': current_time,
                'entry_price': current_close,
                'direction': 'short',
                'stop_loss': stop_price,
                'take_profit': target_price,
                'start_idx': i,
                'setup': current_setup
            }

    print("\n[SYSTEM] Backtest complete.")
    return trades, equity, df

# ==========================================
# VISUALIZATION
# ==========================================
def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1e1e1e')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def generate_trade_charts(trades, df):
    print("[SYSTEM] Generating forensic charts...")
    charts = []
    
    # Only visualize first 5 completed trades
    viz_trades = trades[:5]
    
    for t_idx, trade in enumerate(viz_trades):
        setup = trade['setup']
        start_idx = setup['window_indices'][0]
        # Show window + execution + some buffer
        end_idx = min(trade['end_idx'] + 10, len(df)-1)
        
        subset = df.iloc[start_idx : end_idx + 1].reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.grid(color='#333333')
        
        # Plot Candles
        col_up = '#2ebd85'
        col_down = '#d13030'
        
        for idx, row in subset.iterrows():
            color = col_up if row['close'] >= row['open'] else col_down
            ax.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=1)
            ax.plot([idx, idx], [row['open'], row['close']], color=color, linewidth=4)
            
        # Helper for Line Drawing
        # Map global indices to local subset indices
        window_len = len(setup['window_indices'])
        x_window = np.arange(window_len)
        
        # Reconstruct Lines
        m_c, c_c = setup['center_params']
        m_u, c_u = setup['upper_params']
        m_l, c_l = setup['lower_params']
        
        # Plot OLS Lines (Lagged Window only)
        # They exist from local index 0 to window_len (excluded)
        ax.plot(x_window, (m_u * x_window) + c_u, color='white', linestyle='--', alpha=0.5, label='Upper OLS')
        ax.plot(x_window, (m_l * x_window) + c_l, color='white', linestyle='--', alpha=0.5, label='Lower OLS')
        
        # Plot Trigger Point (The "gap") at index window_len (The entry candle)
        entry_idx = window_len 
        
        # Breakout Levels
        ax.axhline(setup['threshold_long'], color='yellow', linestyle=':', alpha=0.7, label='Long Trigger')
        ax.axhline(setup['threshold_short'], color='yellow', linestyle=':', alpha=0.7, label='Short Trigger')
        
        # Stop and Target
        ax.axhline(trade['stop_loss'], color='red', linewidth=1.5, label='Stop Loss')
        ax.axhline(trade['take_profit'], color='#00ff00', linewidth=1.5, label='Target')
        
        # Entry Marker
        ax.plot(entry_idx, trade['entry_price'], marker='o', color='cyan', markersize=8)
        
        # Exit Marker
        exit_local_idx = entry_idx + (trade['end_idx'] - trade['start_idx'])
        ax.plot(exit_local_idx, trade['exit_price'], marker='x', color='orange', markersize=8)

        title = f"Trade #{t_idx+1}: {trade['direction'].upper()} | PnL: {trade['pnl']*100:.2f}% | Reason: {trade['reason']}"
        ax.set_title(title, color='white', fontsize=14)
        
        charts.append(plot_to_base64(fig))
        
    return charts

def generate_equity_curve(equity):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    ax.plot(equity, color='#00ff00', linewidth=2)
    ax.set_title('Equity Curve', color='white')
    ax.tick_params(colors='white')
    ax.grid(color='#333333')
    return plot_to_base64(fig)

# ==========================================
# WEB SERVER
# ==========================================
def serve_report(charts, equity_chart, trades):
    html_content = f"""
    <html>
    <head>
        <title>Bro's Algo Forensic Report</title>
        <style>
            body {{ background-color: #121212; color: #e0e0e0; font-family: monospace; padding: 20px; }}
            .chart-container {{ margin-bottom: 40px; border: 1px solid #333; padding: 10px; }}
            h1, h2 {{ color: #ffffff; }}
            .metric {{ font-size: 1.2em; color: #4caf50; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Forensic Strategy Report: OLS Breakout</h1>
        <div class="metric">Total Trades: {len(trades)}</div>
        <div class="metric">Final Equity: ${trades[-1]['exit_price'] if not trades else 'N/A'} (approx)</div>
        
        <h2>Equity Curve</h2>
        <div class="chart-container">
            <img src="data:image/png;base64,{equity_chart}" />
        </div>
        
        <h2>Forensic Trade Analysis (First 5)</h2>
    """
    
    for chart in charts:
        html_content += f"""
        <div class="chart-container">
            <img src="data:image/png;base64,{chart}" />
        </div>
        """
        
    html_content += "</body></html>"
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html_content.encode('utf-8'))
            
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"[SERVER] Dashboard active at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Get Data
    df = fetch_data()
    
    # 2. Run Algo
    trades, equity, processed_df = run_backtest(df)
    
    if not trades:
        print("[WARNING] No trades triggered. Check parameters.")
        sys.exit()
    
    # 3. Generate Visuals
    trade_charts = generate_trade_charts(trades, processed_df)
    equity_chart = generate_equity_curve(equity)
    
    # 4. Serve
    serve_report(trade_charts, equity_chart, trades)
