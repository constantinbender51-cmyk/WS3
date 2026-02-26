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
SYMBOL = 'PEPE/USDT'     
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
    return df

def run_backtest(df, sl_pct, tsl_pct):
    """
    Runs backtest where EVERY HOUR is a new, independent 100% exposure trade.
    Incorporates Initial Stop Loss and Trailing Stop Loss (c%).
    """
    balance = STARTING_BALANCE
    
    # History tracking arrays (pad the first index since we use it to look back)
    sl_history = [False]  
    tsl_history = [False]
    position_history = [0] 
    exit_price_history = [0.0]
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # --- 1. DETERMINE DIRECTION ---
        position = 1 if prev['close'] >= prev['open'] else -1
        
        # --- 2. SETUP VARIABLES ---
        entry_price = current['open']
        sl_price = entry_price * (1 - sl_pct) if position == 1 else entry_price * (1 + sl_pct)
        
        sl_hit = False
        tsl_hit = False
        exit_price = current['close'] # Default exit is the close of the hour
        pnl_pct = 0.0
        
        # --- 3. EVALUATE LONG POSITION ---
        if position == 1:
            # Check Initial SL (Conservative: Assume worst case if low hits SL)
            if current['low'] <= sl_price:
                sl_hit = True
                exit_price = sl_price
                pnl_pct = -sl_pct
            else:
                # Check Trailing SL Activation (Did it reach c% profit?)
                activation_price = entry_price * (1 + tsl_pct)
                if current['high'] >= activation_price:
                    trailing_sl = current['high'] * (1 - tsl_pct)
                    # Because it must travel from High -> Close, if close is lower than TSL, it hit it
                    if current['close'] <= trailing_sl:
                        tsl_hit = True
                        exit_price = trailing_sl
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (current['close'] - entry_price) / entry_price
                else:
                    pnl_pct = (current['close'] - entry_price) / entry_price

        # --- 4. EVALUATE SHORT POSITION ---
        elif position == -1:
            # Check Initial SL
            if current['high'] >= sl_price:
                sl_hit = True
                exit_price = sl_price
                pnl_pct = -sl_pct
            else:
                # Check Trailing SL Activation
                activation_price = entry_price * (1 - tsl_pct)
                if current['low'] <= activation_price:
                    trailing_sl = current['low'] * (1 + tsl_pct)
                    # Because it must travel from Low -> Close, if close is higher than TSL, it hit it
                    if current['close'] >= trailing_sl:
                        tsl_hit = True
                        exit_price = trailing_sl
                        pnl_pct = (entry_price - exit_price) / entry_price
                    else:
                        pnl_pct = (entry_price - current['close']) / entry_price
                else:
                    pnl_pct = (entry_price - current['close']) / entry_price
                
        # Update Balance
        balance *= (1 + pnl_pct) 
            
        sl_history.append(sl_hit)
        tsl_history.append(tsl_hit)
        position_history.append(position)
        exit_price_history.append(exit_price)
                
    return balance, sl_history, tsl_history, position_history, exit_price_history

def generate_html_report(df, sl_pct, tsl_pct, final_balance, roi):
    print(f"Generating chart for web display (Initial SL: {sl_pct*100}%, Trailing: {tsl_pct*100}%)...")
    plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(1, 1, 1)
    
    # Filter to only the last 48 hours (2 Days)
    df_2d = df.tail(48).copy()
    
    # Plot Candlesticks
    up = df_2d[df_2d['close'] >= df_2d['open']]
    down = df_2d[df_2d['close'] < df_2d['open']]
    width = 0.03 
    
    ax1.bar(up['timestamp'], up['close'] - up['open'], bottom=up['open'], color='green', width=width, zorder=3)
    ax1.vlines(up['timestamp'], up['low'], up['high'], color='green', linewidth=1, zorder=3)
    
    ax1.bar(down['timestamp'], down['open'] - down['close'], bottom=down['close'], color='red', width=width, zorder=3)
    ax1.vlines(down['timestamp'], down['low'], down['high'], color='red', linewidth=1, zorder=3)

    # Plot Background Position Shading
    y_max = df_2d['high'].max() * 1.01
    y_min = df_2d['low'].min() * 0.99
    
    ax1.fill_between(df_2d['timestamp'], y_max, y_min, where=(df_2d['position'] == 1), color='green', alpha=0.15, label='Long Trade', zorder=1)
    ax1.fill_between(df_2d['timestamp'], y_max, y_min, where=(df_2d['position'] == -1), color='red', alpha=0.15, label='Short Trade', zorder=1)

    # Plot Initial Stop Loss Hits
    sl_data = df_2d[df_2d['sl_hit'] == True]
    if not sl_data.empty:
        ax1.scatter(sl_data['timestamp'], sl_data['exit_price'], marker='X', color='black', s=150, label='Initial SL Hit', zorder=5)

    # Plot Trailing Stop Loss Hits
    tsl_data = df_2d[df_2d['tsl_hit'] == True]
    if not tsl_data.empty:
        ax1.scatter(tsl_data['timestamp'], tsl_data['exit_price'], marker='o', color='darkorange', s=120, label='Trailing SL Hit', zorder=5)

    ax1.set_ylim(y_min, y_max)
    ax1.set_xlim(df_2d['timestamp'].min() - pd.Timedelta(hours=1), df_2d['timestamp'].max() + pd.Timedelta(hours=1))
    ax1.set_title(f'LAST 48 HOURS: Hourly Trades ({SYMBOL} | Initial SL: {sl_pct*100:.1f}% | Trailing c: {tsl_pct*100:.1f}%)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, zorder=0)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert plot to Base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hourly Reversal Backtest</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f4f9; color: #333; text-align: center; padding: 20px; }}
            .form-container {{ background: #2c3e50; color: white; border-radius: 8px; padding: 20px; margin: 0 auto 20px; width: 70%; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .form-container input {{ margin: 0 10px; padding: 5px; width: 70px; text-align: center; border-radius: 4px; border: none; font-size: 16px;}}
            .form-container button {{ padding: 8px 15px; background-color: #27ae60; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 16px; margin-left: 20px;}}
            .form-container button:hover {{ background-color: #219150; }}
            .stats-container {{ background: #fff; border-radius: 8px; padding: 20px; margin: 0 auto 20px; width: 80%; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: flex; justify-content: space-around; }}
            .stat-box {{ margin: 0 10px; }}
            h1 {{ color: #2c3e50; margin-bottom: 5px;}}
            p.subtitle {{ color: #7f8c8d; margin-top: 0; margin-bottom: 20px; }}
            .value {{ font-size: 22px; font-weight: bold; color: #2980b9; margin-top: 5px; }}
            .chart-container img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px;}}
        </style>
    </head>
    <body>
        <h1>Per-Hour Independent Trades ({SYMBOL})</h1>
        <p class="subtitle">100% Exposure | Fresh Position Every Hour</p>
        
        <div class="form-container">
            <form method="POST">
                <label>Initial Stop Loss: 
                    <input type="number" step="0.1" name="sl_pct" value="{sl_pct*100}"> %
                </label>
                &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
                <label>Trailing Activation & Distance (c): 
                    <input type="number" step="0.1" name="tsl_pct" value="{tsl_pct*100}"> %
                </label>
                <button type="submit">Run Backtest</button>
            </form>
        </div>

        <div class="stats-container">
            <div class="stat-box"><div>Starting Balance</div><div class="value">${STARTING_BALANCE:,.2f}</div></div>
            <div class="stat-box"><div>Final Balance (30 Days)</div><div class="value">${final_balance:,.2f}</div></div>
            <div class="stat-box"><div>30-Day Net ROI</div><div class="value" style="color:{'#27ae60' if roi >= 0 else '#c0392b'};">{roi:.2f}%</div></div>
        </div>
        
        <div class="chart-container">
            <img src="data:image/png;base64,{image_base64}" alt="Backtest Chart">
        </div>
    </body>
    </html>
    """
    return html

def execute_run(sl_pct_input, tsl_pct_input):
    """Helper function to run the backtest and return the HTML."""
    df_run = GLOBAL_DF.copy()
    
    sl_pct_decimal = sl_pct_input / 100.0
    tsl_pct_decimal = tsl_pct_input / 100.0
    
    final_balance, sls, tsls, positions, exit_prices = run_backtest(df_run, sl_pct_decimal, tsl_pct_decimal)
    
    df_run['sl_hit'] = sls
    df_run['tsl_hit'] = tsls
    df_run['position'] = positions
    df_run['exit_price'] = exit_prices
    
    roi = ((final_balance - STARTING_BALANCE) / STARTING_BALANCE) * 100
    
    return generate_html_report(df_run, sl_pct_decimal, tsl_pct_decimal, final_balance, roi)

class BacktestServer(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        # Default parameters when you open the page (Initial SL: 2.0%, Trailing C: 2.0%)
        html_content = execute_run(sl_pct_input=2.0, tsl_pct_input=2.0)
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        parsed_data = urllib.parse.parse_qs(post_data)

        # Get inputs from form
        sl_pct_input = float(parsed_data.get('sl_pct', ['2.0'])[0])
        tsl_pct_input = float(parsed_data.get('tsl_pct', ['2.0'])[0])
        
        html_content = execute_run(sl_pct_input, tsl_pct_input)
        
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