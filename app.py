import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for web server plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import io
import base64
import http.server
import socketserver
import itertools
import socket

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
DAYS_BACK = 30
STARTING_BALANCE = 10000
PORT = 8000

# --- Grid Search Parameters ---
PARAM_GRID = {
    'vol_mult': [2.0, 3.0, 4.0, 5.0],           
    'sl_trigger': [1, 2, 3, 4],                 
    'sl_pct': [0.005, 0.01, 0.015, 0.02, 0.03]  
}

def get_local_ip():
    """Helper to find the local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def fetch_binance_data(symbol, timeframe, days):
    print(f"Fetching {days} days of {timeframe} data for {symbol}...")
    exchange = ccxt.binance()
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Pre-calculate indicators ONCE
    df['vol_5h_avg'] = df['volume'].shift(1).rolling(window=5).mean()
    df['color'] = np.where(df['close'] >= df['open'], 1, -1)
    
    return df

def run_backtest(df, vol_mult, sl_trigger, sl_pct):
    """Runs a single backtest and returns equity and history."""
    state = 2             
    position = 0          
    entry_price = 0.0
    consecutive_sl = 0    
    
    balance = STARTING_BALANCE
    
    # History tracking arrays (padded for the first 6 moving-average hours)
    equity_curve = [balance] * 6  
    state_history = [2] * 6
    sl_history = [False] * 6
    
    for i in range(6, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # --- 1. STATE TRANSITIONS ---
        if prev['volume'] > (vol_mult * prev['vol_5h_avg']):
            state = 1
            consecutive_sl = 0 
            
        if consecutive_sl >= sl_trigger:
            state = 2
            
        sl_hit = False
            
        # --- 2. STATE LOGIC ---
        if state == 2:
            if position != 0:
                pnl_pct = (current['open'] - entry_price) / entry_price if position == 1 else (entry_price - current['open']) / entry_price
                balance *= (1 + pnl_pct)
                position = 0
            equity_curve.append(balance)
            state_history.append(state)
            sl_history.append(sl_hit)
            continue
            
        if state == 1:
            target_position = 1 if prev['color'] == 1 else -1
            
            if position != target_position:
                if position != 0:
                    pnl_pct = (current['open'] - entry_price) / entry_price if position == 1 else (entry_price - current['open']) / entry_price
                    balance *= (1 + pnl_pct)
                position = target_position
                entry_price = current['open']
                
            # Stop Loss Check
            sl_price = entry_price * (1 - sl_pct) if position == 1 else entry_price * (1 + sl_pct)
            
            if position == 1 and current['low'] <= sl_price:
                sl_hit = True
                pnl_pct = -sl_pct
            elif position == -1 and current['high'] >= sl_price:
                sl_hit = True
                pnl_pct = -sl_pct
                
            if sl_hit:
                balance *= (1 + pnl_pct) 
                position = 0             
                consecutive_sl += 1      
            else:
                consecutive_sl = 0        
                
            if position != 0:
                unrealized_pnl = (current['close'] - entry_price) / entry_price if position == 1 else (entry_price - current['close']) / entry_price
                equity_curve.append(balance * (1 + unrealized_pnl))
            else:
                equity_curve.append(balance)
                
            state_history.append(state)
            sl_history.append(sl_hit)
                
    # Safeguard padding
    if len(equity_curve) < len(df):
        pad_len = len(df) - len(equity_curve)
        equity_curve += [balance] * pad_len
        state_history += [2] * pad_len
        sl_history += [False] * pad_len
        
    return equity_curve, balance, state_history, sl_history

def grid_search(df):
    print("\nStarting Grid Search Optimization...")
    best_roi = -float('inf')
    best_params = {}
    
    # Store history variables for the best run
    best_equity = None
    best_states = None
    best_sls = None
    best_balance = 0
    results = []
    
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Testing {len(combinations)} parameter combinations. Please wait...\n")
    
    for params in combinations:
        equity, final_balance, states, sls = run_backtest(
            df, 
            vol_mult=params['vol_mult'], 
            sl_trigger=params['sl_trigger'], 
            sl_pct=params['sl_pct']
        )
        
        roi = ((final_balance - STARTING_BALANCE) / STARTING_BALANCE) * 100
        
        results.append({
            'vol_mult': params['vol_mult'],
            'sl_trigger': params['sl_trigger'],
            'sl_pct': f"{params['sl_pct']*100:.2f}%",
            'roi': roi,
            'balance': final_balance
        })
        
        if roi > best_roi:
            best_roi = roi
            best_params = params
            best_equity = equity
            best_states = states
            best_sls = sls
            best_balance = final_balance

    top_5_results = sorted(results, key=lambda x: x['roi'], reverse=True)[:5]
    return best_params, best_balance, best_roi, best_equity, best_states, best_sls, top_5_results

def generate_html_report(df, best_params, final_balance, roi, top_5):
    print("Generating advanced chart for web display...")
    plt.figure(figsize=(16, 10))
    
    # --- Plot 1: Candlesticks & States ---
    ax1 = plt.subplot(2, 1, 1)
    
    # Filter up and down candles
    up = df[df['close'] >= df['open']]
    down = df[df['close'] < df['open']]
    
    # Width of candlestick elements (approx 40 minutes converted to days)
    width = 0.03 
    
    # Plot Candlesticks
    ax1.bar(up['timestamp'], up['close'] - up['open'], bottom=up['open'], color='green', width=width, zorder=3)
    ax1.vlines(up['timestamp'], up['low'], up['high'], color='green', linewidth=1, zorder=3)
    
    ax1.bar(down['timestamp'], down['open'] - down['close'], bottom=down['close'], color='red', width=width, zorder=3)
    ax1.vlines(down['timestamp'], down['low'], down['high'], color='red', linewidth=1, zorder=3)

    # Plot Background States
    # State 1 (Active) = Blue tint
    ax1.fill_between(df['timestamp'], df['high'].max() * 1.05, df['low'].min() * 0.95, 
                     where=(df['state'] == 1), color='blue', alpha=0.1, label='State 1 (Active Trading)', zorder=1)
    # State 2 (Inactive) = Gray tint
    ax1.fill_between(df['timestamp'], df['high'].max() * 1.05, df['low'].min() * 0.95, 
                     where=(df['state'] == 2), color='gray', alpha=0.15, label='State 2 (Cool-down)', zorder=1)

    # Plot Stop Loss Hits
    sl_data = df[df['sl_hit'] == True]
    if not sl_data.empty:
        ax1.scatter(sl_data['timestamp'], sl_data['close'], marker='X', color='black', s=120, label='Stop Loss Hit', zorder=5)

    ax1.set_ylim(df['low'].min() * 0.98, df['high'].max() * 1.02)
    ax1.set_title(f'{SYMBOL} {TIMEFRAME} Price - States & Stop Losses')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, zorder=0)
    
    # --- Plot 2: Equity Curve ---
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(df['timestamp'], df['equity'], label='Optimized Strategy Equity', color='#27ae60', linewidth=2)
    ax2.set_title(f"Optimized Equity Curve (Vol: {best_params['vol_mult']}x | SL Trigger: {best_params['sl_trigger']} | SL: {best_params['sl_pct']*100}%)")
    ax2.set_ylabel('Balance (USDT)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Format dates nicely
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # Generate HTML Table
    table_rows = ""
    for idx, row in enumerate(top_5):
        table_rows += f"<tr><td>{idx+1}</td><td>{row['vol_mult']}x</td><td>{row['sl_trigger']}</td><td>{row['sl_pct']}</td><td style='color:{'green' if row['roi']>0 else 'red'}; font-weight:bold;'>{row['roi']:.2f}%</td></tr>"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f4f9; color: #333; text-align: center; padding: 20px; }}
            .stats-container {{ background: #fff; border-radius: 8px; padding: 20px; margin: 0 auto 20px; width: 80%; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: flex; justify-content: space-around; }}
            .stat-box {{ margin: 0 10px; }}
            h1 {{ color: #2c3e50; }}
            .value {{ font-size: 22px; font-weight: bold; color: #2980b9; margin-top: 5px; }}
            .chart-container img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px;}}
            table {{ margin: 0 auto; border-collapse: collapse; width: 60%; background: #fff; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            th, td {{ padding: 12px; border: 1px solid #ddd; text-align: center; }}
            th {{ background-color: #2c3e50; color: white; }}
        </style>
    </head>
    <body>
        <h1>Optimized Algorithmic Backtest Report</h1>
        
        <h2>Best Parameters Found</h2>
        <div class="stats-container">
            <div class="stat-box"><div>Volume Multiplier</div><div class="value">{best_params['vol_mult']}x</div></div>
            <div class="stat-box"><div>State 2 Trigger</div><div class="value">{best_params['sl_trigger']} Cons. SLs</div></div>
            <div class="stat-box"><div>Stop Loss %</div><div class="value">{best_params['sl_pct']*100:.2f}%</div></div>
        </div>

        <h2>Best Results</h2>
        <div class="stats-container" style="background-color: #eafaf1; border: 1px solid #27ae60;">
            <div class="stat-box"><div>Starting Balance</div><div class="value" style="color:#27ae60;">${STARTING_BALANCE:,.2f}</div></div>
            <div class="stat-box"><div>Final Balance</div><div class="value" style="color:#27ae60;">${final_balance:,.2f}</div></div>
            <div class="stat-box"><div>Net ROI</div><div class="value" style="color:#27ae60;">{roi:.2f}%</div></div>
        </div>
        
        <div class="chart-container">
            <img src="data:image/png;base64,{image_base64}" alt="Backtest Chart">
        </div>

        <h2>Top 5 Parameter Combinations</h2>
        <table>
            <tr>
                <th>Rank</th><th>Volume Mult.</th><th>State 2 Trigger</th><th>Stop Loss</th><th>ROI</th>
            </tr>
            {table_rows}
        </table>
        <br><br>
    </body>
    </html>
    """
    return html

class BacktestServer(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(self.server.html_content.encode('utf-8'))

if __name__ == "__main__":
    # 1. Fetch Data
    df = fetch_binance_data(SYMBOL, TIMEFRAME, DAYS_BACK)
    
    # 2. Run Grid Search
    best_params, best_balance, best_roi, best_equity, best_states, best_sls, top_5 = grid_search(df)
    
    # Add best results back to DataFrame for plotting
    df['equity'] = best_equity
    df['state'] = best_states
    df['sl_hit'] = best_sls
    
    # 3. Generate HTML Report
    html_report = generate_html_report(df, best_params, best_balance, best_roi, top_5)
    
    # 4. Start HTTP Server
    local_ip = get_local_ip()
    handler = BacktestServer
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        httpd.html_content = html_report  
        print(f"\n✅ Grid Search & Backtest Complete!")
        print(f"🏆 Best ROI: {best_roi:.2f}%")
        print(f"🌐 Server running! Click the link below to view the results:")
        print(f"➡️  http://{local_ip}:{PORT}  (or http://localhost:{PORT})")
        print("Press Ctrl+C to stop the server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")