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
import urllib.parse
import socket

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
DAYS_BACK = 30
STARTING_BALANCE = 10000
PORT = 8000

# Global variable to hold market data so we only fetch it once
GLOBAL_DF = None

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
    
    # History tracking arrays
    equity_curve = [balance] * 6  
    state_history = [2] * 6
    sl_history = [False] * 6
    position_history = [0] * 6  # 1 for Long, -1 for Short, 0 for Flat
    
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
            position_history.append(0)
            continue
            
        if state == 1:
            target_position = 1 if prev['color'] == 1 else -1
            
            if position != target_position:
                if position != 0:
                    pnl_pct = (current['open'] - entry_price) / entry_price if position == 1 else (entry_price - current['open']) / entry_price
                    balance *= (1 + pnl_pct)
                position = target_position
                entry_price = current['open']
            
            # Store the position held during this candle for the chart
            chart_position = position
                
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
            position_history.append(chart_position)
                
    # Safeguard padding
    if len(equity_curve) < len(df):
        pad_len = len(df) - len(equity_curve)
        equity_curve += [balance] * pad_len
        state_history += [2] * pad_len
        sl_history += [False] * pad_len
        position_history += [0] * pad_len
        
    return equity_curve, balance, state_history, sl_history, position_history

def generate_html_report(df, params, final_balance, roi):
    print(f"Generating charts for web display...")
    plt.figure(figsize=(16, 14))
    
    width = 0.03 # Candlestick width
    
    # =========================================================
    # --- Plot 1: Full 30-Day Candlesticks & States ---
    # =========================================================
    ax1 = plt.subplot(3, 1, 1)
    up = df[df['close'] >= df['open']]
    down = df[df['close'] < df['open']]
    
    ax1.bar(up['timestamp'], up['close'] - up['open'], bottom=up['open'], color='green', width=width, zorder=3)
    ax1.vlines(up['timestamp'], up['low'], up['high'], color='green', linewidth=1, zorder=3)
    
    ax1.bar(down['timestamp'], down['open'] - down['close'], bottom=down['close'], color='red', width=width, zorder=3)
    ax1.vlines(down['timestamp'], down['low'], down['high'], color='red', linewidth=1, zorder=3)

    ax1.fill_between(df['timestamp'], df['high'].max() * 1.05, df['low'].min() * 0.95, 
                     where=(df['state'] == 1), color='blue', alpha=0.1, label='State 1 (Active Trading)', zorder=1)
    ax1.fill_between(df['timestamp'], df['high'].max() * 1.05, df['low'].min() * 0.95, 
                     where=(df['state'] == 2), color='gray', alpha=0.15, label='State 2 (Cool-down)', zorder=1)

    sl_data = df[df['sl_hit'] == True]
    if not sl_data.empty:
        ax1.scatter(sl_data['timestamp'], sl_data['close'], marker='X', color='black', s=100, label='Stop Loss Hit', zorder=5)

    ax1.set_ylim(df['low'].min() * 0.98, df['high'].max() * 1.02)
    ax1.set_title(f'30-DAY VIEW: {SYMBOL} {TIMEFRAME} Price & States')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, zorder=0)
    
    # =========================================================
    # --- Plot 2: 2-Day Zoomed In (Positions & Stop Loss) ---
    # =========================================================
    ax2 = plt.subplot(3, 1, 2)
    
    # Get only the last 48 hours (2 days)
    df_2d = df.tail(48).copy()
    
    up_2d = df_2d[df_2d['close'] >= df_2d['open']]
    down_2d = df_2d[df_2d['close'] < df_2d['open']]
    
    ax2.bar(up_2d['timestamp'], up_2d['close'] - up_2d['open'], bottom=up_2d['open'], color='green', width=width, zorder=3)
    ax2.vlines(up_2d['timestamp'], up_2d['low'], up_2d['high'], color='green', linewidth=1, zorder=3)
    
    ax2.bar(down_2d['timestamp'], down_2d['open'] - down_2d['close'], bottom=down_2d['close'], color='red', width=width, zorder=3)
    ax2.vlines(down_2d['timestamp'], down_2d['low'], down_2d['high'], color='red', linewidth=1, zorder=3)

    # Shaded Backgrounds for Positions
    y_max = df_2d['high'].max() * 1.02
    y_min = df_2d['low'].min() * 0.98
    
    ax2.fill_between(df_2d['timestamp'], y_max, y_min, where=(df_2d['position'] == 1), color='green', alpha=0.15, label='Long Position', zorder=1)
    ax2.fill_between(df_2d['timestamp'], y_max, y_min, where=(df_2d['position'] == -1), color='red', alpha=0.15, label='Short Position', zorder=1)

    sl_data_2d = df_2d[df_2d['sl_hit'] == True]
    if not sl_data_2d.empty:
        ax2.scatter(sl_data_2d['timestamp'], sl_data_2d['close'], marker='X', color='black', s=120, label='Stop Loss Hit', zorder=5)

    ax2.set_ylim(y_min, y_max)
    ax2.set_xlim(df_2d['timestamp'].min() - pd.Timedelta(hours=1), df_2d['timestamp'].max() + pd.Timedelta(hours=1))
    ax2.set_title(f'LAST 2 DAYS (48H): Active Positions')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, zorder=0)

    # =========================================================
    # --- Plot 3: 30-Day Equity Curve ---
    # =========================================================
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(df['timestamp'], df['equity'], label='Strategy Equity', color='#27ae60', linewidth=2)
    ax3.set_title(f"30-DAY VIEW: Equity Curve")
    ax3.set_ylabel('Balance (USDT)')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive Backtest</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f4f9; color: #333; text-align: center; padding: 20px; }}
            .form-container {{ background: #2c3e50; color: white; border-radius: 8px; padding: 20px; margin: 0 auto 20px; width: 80%; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .form-container input {{ margin: 0 10px; padding: 5px; width: 80px; text-align: center; border-radius: 4px; border: none; }}
            .form-container button {{ padding: 8px 15px; background-color: #27ae60; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 14px;}}
            .form-container button:hover {{ background-color: #219150; }}
            .stats-container {{ background: #fff; border-radius: 8px; padding: 20px; margin: 0 auto 20px; width: 80%; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: flex; justify-content: space-around; }}
            .stat-box {{ margin: 0 10px; }}
            h1 {{ color: #2c3e50; }}
            .value {{ font-size: 22px; font-weight: bold; color: #2980b9; margin-top: 5px; }}
            .chart-container img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px;}}
        </style>
    </head>
    <body>
        <h1>Interactive Algorithmic Backtester</h1>
        
        <div class="form-container">
            <form method="POST">
                <label>Volume Multiplier: 
                    <input type="number" step="0.1" name="vol_mult" value="{params['vol_mult']}"> x
                </label>
                &nbsp;&nbsp;&nbsp;
                <label>Consecutive SLs to State 2: 
                    <input type="number" step="1" name="sl_trigger" value="{params['sl_trigger']}">
                </label>
                &nbsp;&nbsp;&nbsp;
                <label>Stop Loss: 
                    <input type="number" step="0.1" name="sl_pct" value="{params['sl_pct']}"> %
                </label>
                &nbsp;&nbsp;&nbsp;
                <button type="submit">Run Backtest</button>
            </form>
        </div>

        <div class="stats-container">
            <div class="stat-box"><div>Starting Balance</div><div class="value">${STARTING_BALANCE:,.2f}</div></div>
            <div class="stat-box"><div>Final Balance</div><div class="value">${final_balance:,.2f}</div></div>
            <div class="stat-box"><div>Net ROI</div><div class="value" style="color:{'#27ae60' if roi >= 0 else '#c0392b'};">{roi:.2f}%</div></div>
        </div>
        
        <div class="chart-container">
            <img src="data:image/png;base64,{image_base64}" alt="Backtest Chart">
        </div>
    </body>
    </html>
    """
    return html

def execute_run(params):
    """Helper function to run the backtest and return the HTML."""
    df_run = GLOBAL_DF.copy()
    
    sl_pct_decimal = params['sl_pct'] / 100.0
    
    equity, final_balance, states, sls, positions = run_backtest(
        df_run, 
        vol_mult=params['vol_mult'], 
        sl_trigger=params['sl_trigger'], 
        sl_pct=sl_pct_decimal
    )
    
    df_run['equity'] = equity
    df_run['state'] = states
    df_run['sl_hit'] = sls
    df_run['position'] = positions
    
    roi = ((final_balance - STARTING_BALANCE) / STARTING_BALANCE) * 100
    
    return generate_html_report(df_run, params, final_balance, roi)

class BacktestServer(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        params = {'vol_mult': 3.0, 'sl_trigger': 3, 'sl_pct': 1.0}
        html_content = execute_run(params)
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        parsed_data = urllib.parse.parse_qs(post_data)

        params = {
            'vol_mult': float(parsed_data.get('vol_mult', ['3.0'])[0]),
            'sl_trigger': int(parsed_data.get('sl_trigger', ['3'])[0]),
            'sl_pct': float(parsed_data.get('sl_pct', ['1.0'])[0])
        }
        
        html_content = execute_run(params)
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))

if __name__ == "__main__":
    # 1. Fetch Data ONCE at startup
    GLOBAL_DF = fetch_binance_data(SYMBOL, TIMEFRAME, DAYS_BACK)
    
    # 2. Start HTTP Server
    local_ip = get_local_ip()
    handler = BacktestServer
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"\n✅ Data Fetched! Web server is starting...")
        print(f"🌐 Open this link in your browser to interact with the backtester:")
        print(f"➡️  http://{local_ip}:{PORT}  (or http://localhost:{PORT})")
        print("\nPress Ctrl+C to stop the server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")