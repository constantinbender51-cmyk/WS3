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
    
    # Pre-calculate Wick and Body metrics for State 1 conditions
    df['body'] = abs(df['close'] - df['open'])
    df['top_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['bot_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['max_wick'] = df[['top_wick', 'bot_wick']].max(axis=1)
    
    df['wick_pct'] = df['max_wick'] / df['open']
    # If body is 0, wick/body ratio is infinity
    df['wick_body_ratio'] = np.where(df['body'] == 0, np.inf, df['max_wick'] / df['body'])
    
    return df

def run_backtest(df, sl_pct, tsl_pct, f_pct, g_pct, u_pct):
    balance = STARTING_BALANCE
    
    sl_history = [False]  
    tsl_history = [False]
    position_history = [0] 
    exit_price_history = [0.0]
    state_history = [2]
    
    state = 2
    low_profit_count = 0
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # --- 1. STATE TRANSITIONS ---
        # Trigger State 1: Wick > f% OR Wick/Body > g%
        if prev['wick_pct'] > f_pct or prev['wick_body_ratio'] > g_pct:
            state = 1
            low_profit_count = 0  # Reset counter when newly triggered
            
        # Trigger State 2: 3 consecutive hours of profit < u%
        elif low_profit_count >= 3:
            state = 2
            
        # --- 2. EXECUTE BASED ON STATE ---
        if state == 2:
            # No trades taken in State 2
            sl_history.append(False)
            tsl_history.append(False)
            position_history.append(0)
            exit_price_history.append(current['close'])
            state_history.append(2)
            continue
            
        if state == 1:
            # Green prev candle -> Long (1), Red prev candle -> Short (-1)
            position = 1 if prev['close'] >= prev['open'] else -1
            
            entry_price = current['open']
            sl_price = entry_price * (1 - sl_pct) if position == 1 else entry_price * (1 + sl_pct)
            
            sl_hit = False
            tsl_hit = False
            exit_price = current['close']
            pnl_pct = 0.0
            
            # --- EVALUATE LONG POSITION ---
            if position == 1:
                if current['low'] <= sl_price:
                    sl_hit = True
                    exit_price = sl_price
                    pnl_pct = -sl_pct
                else:
                    activation_price = entry_price * (1 + tsl_pct)
                    if current['high'] >= activation_price:
                        trailing_sl = current['high'] * (1 - tsl_pct)
                        if current['close'] <= trailing_sl:
                            tsl_hit = True
                            exit_price = trailing_sl
                            pnl_pct = (exit_price - entry_price) / entry_price
                        else:
                            pnl_pct = (current['close'] - entry_price) / entry_price
                    else:
                        pnl_pct = (current['close'] - entry_price) / entry_price

            # --- EVALUATE SHORT POSITION ---
            elif position == -1:
                if current['high'] >= sl_price:
                    sl_hit = True
                    exit_price = sl_price
                    pnl_pct = -sl_pct
                else:
                    activation_price = entry_price * (1 - tsl_pct)
                    if current['low'] <= activation_price:
                        trailing_sl = current['low'] * (1 + tsl_pct)
                        if current['close'] >= trailing_sl:
                            tsl_hit = True
                            exit_price = trailing_sl
                            pnl_pct = (entry_price - exit_price) / entry_price
                        else:
                            pnl_pct = (entry_price - current['close']) / entry_price
                    else:
                        pnl_pct = (entry_price - current['close']) / entry_price
                    
            # --- UPDATE STATE 2 TRIGGERS ---
            if pnl_pct < u_pct:
                low_profit_count += 1
            else:
                low_profit_count = 0
                    
            # Update Balance
            balance *= (1 + pnl_pct) 
                
            sl_history.append(sl_hit)
            tsl_history.append(tsl_hit)
            position_history.append(position)
            exit_price_history.append(exit_price)
            state_history.append(1)
                
    return balance, sl_history, tsl_history, position_history, exit_price_history, state_history

def generate_html_report(df, params, final_balance, roi):
    print(f"Generating chart for web display...")
    plt.figure(figsize=(16, 9))
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

    # Plot Background Position Shading (Only active in State 1)
    y_max = df_2d['high'].max() * 1.01
    y_min = df_2d['low'].min() * 0.99
    
    ax1.fill_between(df_2d['timestamp'], y_max, y_min, where=(df_2d['position'] == 1), color='green', alpha=0.15, label='State 1: Long', zorder=1)
    ax1.fill_between(df_2d['timestamp'], y_max, y_min, where=(df_2d['position'] == -1), color='red', alpha=0.15, label='State 1: Short', zorder=1)
    ax1.fill_between(df_2d['timestamp'], y_max, y_min, where=(df_2d['state'] == 2), color='gray', alpha=0.10, label='State 2: Flat', zorder=1)

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
    ax1.set_title(f"LAST 48 HOURS ({SYMBOL}) | f: {params['f']}% | g: {params['g']}% | u: {params['u']}%")
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
            .form-container {{ background: #2c3e50; color: white; border-radius: 8px; padding: 20px; margin: 0 auto 20px; width: 85%; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .form-container input {{ margin: 0 5px; padding: 5px; width: 60px; text-align: center; border-radius: 4px; border: none; font-size: 15px;}}
            .form-container button {{ padding: 8px 15px; background-color: #27ae60; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 16px; margin-top: 15px; width: 200px;}}
            .form-container button:hover {{ background-color: #219150; }}
            .stats-container {{ background: #fff; border-radius: 8px; padding: 20px; margin: 0 auto 20px; width: 85%; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: flex; justify-content: space-around; }}
            .stat-box {{ margin: 0 10px; }}
            h1 {{ color: #2c3e50; margin-bottom: 5px;}}
            p.subtitle {{ color: #7f8c8d; margin-top: 0; margin-bottom: 20px; }}
            .value {{ font-size: 22px; font-weight: bold; color: #2980b9; margin-top: 5px; }}
            .chart-container img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px;}}
            .input-group {{ display: inline-block; margin: 5px 15px; text-align: right;}}
        </style>
    </head>
    <body>
        <h1>Advanced Hourly Reversal ({SYMBOL})</h1>
        <p class="subtitle">State 1/2 Logic | 100% Exposure</p>
        
        <div class="form-container">
            <form method="POST">
                <div class="input-group"><label>Wick Size > (f): <input type="number" step="0.1" name="f" value="{params['f']}"> %</label></div>
                <div class="input-group"><label>Wick/Body > (g): <input type="number" step="1" name="g" value="{params['g']}"> %</label></div>
                <div class="input-group"><label>Profit Threshold (u): <input type="number" step="0.1" name="u" value="{params['u']}"> %</label></div>
                <br>
                <div class="input-group"><label>Initial SL: <input type="number" step="0.1" name="sl" value="{params['sl']}"> %</label></div>
                <div class="input-group"><label>Trailing Act/Dist (c): <input type="number" step="0.1" name="tsl" value="{params['tsl']}"> %</label></div>
                <br>
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

def execute_run(params):
    df_run = GLOBAL_DF.copy()
    
    # Convert form percentages to decimals
    sl_pct = params['sl'] / 100.0
    tsl_pct = params['tsl'] / 100.0
    f_pct = params['f'] / 100.0
    g_pct = params['g'] / 100.0
    u_pct = params['u'] / 100.0
    
    final_balance, sls, tsls, positions, exit_prices, states = run_backtest(
        df_run, sl_pct, tsl_pct, f_pct, g_pct, u_pct
    )
    
    df_run['sl_hit'] = sls
    df_run['tsl_hit'] = tsls
    df_run['position'] = positions
    df_run['exit_price'] = exit_prices
    df_run['state'] = states
    
    roi = ((final_balance - STARTING_BALANCE) / STARTING_BALANCE) * 100
    
    return generate_html_report(df_run, params, final_balance, roi)

class BacktestServer(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        # Default Parameters
        default_params = {
            'f': 1.0,   # Wick is > 1% of price
            'g': 150.0, # Wick is > 1.5x the body
            'u': 0.5,   # State 2 triggers if profit < 0.5% for 3 hrs
            'sl': 2.0,  # 2% Initial SL
            'tsl': 2.0  # 2% Trailing SL
        }
        html_content = execute_run(default_params)
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        parsed_data = urllib.parse.parse_qs(post_data)

        # Get inputs from form
        params = {
            'f': float(parsed_data.get('f', ['1.0'])[0]),
            'g': float(parsed_data.get('g', ['150.0'])[0]),
            'u': float(parsed_data.get('u', ['0.5'])[0]),
            'sl': float(parsed_data.get('sl', ['2.0'])[0]),
            'tsl': float(parsed_data.get('tsl', ['2.0'])[0]),
        }
        
        html_content = execute_run(params)
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))

if __name__ == "__main__":
    GLOBAL_DF = fetch_binance_data(SYMBOL, TIMEFRAME, DAYS_BACK)
    
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