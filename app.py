import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use headless backend for web server plotting
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64
import http.server
import socketserver

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
DAYS_BACK = 30
STOP_LOSS_PCT = 0.01      # Updated to 1% Stop Loss
STARTING_BALANCE = 10000
PORT = 8000               # Port for the Web Server

def fetch_binance_data(symbol, timeframe, days):
    """Fetches historical data from Binance using CCXT."""
    print(f"Fetching {days} days of {timeframe} data for {symbol}...")
    exchange = ccxt.binance()
    
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def run_backtest(df):
    print("Running Backtest...")
    
    # Pre-calculate indicators
    df['vol_5h_avg'] = df['volume'].shift(1).rolling(window=5).mean()
    df['color'] = np.where(df['close'] > df['open'], 1, -1) # 1=Green, -1=Red
    
    state = 2             
    position = 0          
    entry_price = 0.0
    consecutive_sl = 0    
    
    balance = STARTING_BALANCE
    equity_curve = [balance] * 6  
    
    for i in range(6, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # --- 1. STATE TRANSITIONS ---
        if prev['volume'] > (3 * prev['vol_5h_avg']):
            state = 1
            consecutive_sl = 0 
            
        if consecutive_sl >= 3:
            state = 2
            
        # --- 2. STATE LOGIC ---
        if state == 2:
            if position != 0:
                pnl_pct = (current['open'] - entry_price) / entry_price if position == 1 else (entry_price - current['open']) / entry_price
                balance *= (1 + pnl_pct)
                position = 0
            equity_curve.append(balance)
            continue
            
        if state == 1:
            target_position = 1 if prev['color'] == 1 else -1
            
            # Flip / Open
            if position != target_position:
                if position != 0:
                    pnl_pct = (current['open'] - entry_price) / entry_price if position == 1 else (entry_price - current['open']) / entry_price
                    balance *= (1 + pnl_pct)
                position = target_position
                entry_price = current['open']
                
            # Stop Loss Check (1%)
            sl_price = entry_price * (1 - STOP_LOSS_PCT) if position == 1 else entry_price * (1 + STOP_LOSS_PCT)
            
            sl_hit = False
            if position == 1 and current['low'] <= sl_price:
                sl_hit = True
                pnl_pct = -STOP_LOSS_PCT
            elif position == -1 and current['high'] >= sl_price:
                sl_hit = True
                pnl_pct = -STOP_LOSS_PCT
                
            if sl_hit:
                balance *= (1 + pnl_pct) 
                position = 0             
                consecutive_sl += 1      
            else:
                consecutive_sl = 0        
                
            # Equity tracking
            if position != 0:
                unrealized_pnl = (current['close'] - entry_price) / entry_price if position == 1 else (entry_price - current['close']) / entry_price
                equity_curve.append(balance * (1 + unrealized_pnl))
            else:
                equity_curve.append(balance)
                
    df['equity'] = equity_curve + [balance] * (len(df) - len(equity_curve))
    return df, balance

def generate_html_report(df, final_balance):
    """Generates the plot and embeds it into an HTML string via base64."""
    print("Generating chart for web display...")
    plt.figure(figsize=(14, 8))
    
    # Plot Price
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df['timestamp'], df['close'], label='BTC Price', color='#2c3e50')
    ax1.set_title(f'{SYMBOL} {TIMEFRAME} Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Equity Curve
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(df['timestamp'], df['equity'], label='Strategy Equity', color='#27ae60')
    ax2.set_title('Backtest Equity Curve')
    ax2.set_ylabel('Balance (USDT)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Encode to base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    roi = ((final_balance - STARTING_BALANCE) / STARTING_BALANCE) * 100
    
    # Build HTML string
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f4f9; color: #333; text-align: center; padding: 20px; }}
            .stats-container {{ background: #fff; border-radius: 8px; padding: 20px; margin: 0 auto 20px; width: 60%; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .stat-box {{ display: inline-block; margin: 0 20px; }}
            h1 {{ color: #2c3e50; }}
            .value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
            .chart-container img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <h1>Algorithmic Backtest Report</h1>
        <div class="stats-container">
            <div class="stat-box"><div>Starting Balance</div><div class="value">${STARTING_BALANCE:,.2f}</div></div>
            <div class="stat-box"><div>Final Balance</div><div class="value">${final_balance:,.2f}</div></div>
            <div class="stat-box"><div>Net ROI</div><div class="value">{roi:.2f}%</div></div>
        </div>
        <div class="chart-container">
            <img src="data:image/png;base64,{image_base64}" alt="Backtest Chart">
        </div>
    </body>
    </html>
    """
    return html

class BacktestServer(http.server.BaseHTTPRequestHandler):
    # Retrieve the pre-rendered HTML content
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(self.server.html_content.encode('utf-8'))

if __name__ == "__main__":
    # 1. Fetch & Backtest
    df = fetch_binance_data(SYMBOL, TIMEFRAME, DAYS_BACK)
    results_df, final_balance = run_backtest(df)
    
    # 2. Generate HTML Report
    html_report = generate_html_report(results_df, final_balance)
    
    # 3. Start HTTP Server
    handler = BacktestServer
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        httpd.html_content = html_report  # Attach HTML to server instance
        print(f"\n✅ Backtest Complete!")
        print(f"🌐 Server running. Open your browser to: http://localhost:{PORT}")
        print("Press Ctrl+C to stop the server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")