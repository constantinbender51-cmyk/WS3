import http.server
import socketserver
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

# ==========================================
# 1. Data Fetching (Binance API via requests)
# ==========================================
def get_binance_data():
    base_url = "https://api.binance.com/api/v3/klines"
    # Fetch approx 30 days of 1h data (24 * 30 = 720 candles). Limit 1000 is sufficient.
    params = {
        'symbol': 'ETHUSDT',
        'interval': '1h',
        'limit': 1000
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'q_vol', 'num_trades', 'tbb_base', 'tbb_quote', 'ignore'
    ])
    
    # Convert to floats
    cols = ['open', 'high', 'low', 'close']
    df[cols] = df[cols].astype(float)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    
    # Filter last 30 days
    cutoff = datetime.now() - timedelta(days=30)
    df = df[df['timestamp'] > cutoff].reset_index(drop=True)
    
    return df

# ==========================================
# 2. Vectorized Backtesting Engine
# ==========================================
def backtest(df, sl_pct, tp_pct):
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values
    
    # -- Long --
    long_tp = open_arr * (1 + tp_pct)
    long_sl = open_arr * (1 - sl_pct)
    
    # Logic: Assume SL hit first if both in range (Conservative)
    l_hit_sl = low_arr <= long_sl
    l_hit_tp = high_arr >= long_tp
    
    # PnL logic
    l_pnl = (close_arr - open_arr) / open_arr # Default: Close
    l_pnl = np.where(l_hit_sl, -sl_pct, l_pnl) # SL overrides Close
    l_pnl = np.where(l_hit_tp & ~l_hit_sl, tp_pct, l_pnl) # TP overrides Close if no SL
    
    # -- Short --
    short_tp = open_arr * (1 - tp_pct)
    short_sl = open_arr * (1 + sl_pct)
    
    s_hit_sl = high_arr >= short_sl
    s_hit_tp = low_arr <= short_tp
    
    s_pnl = (open_arr - close_arr) / open_arr
    s_pnl = np.where(s_hit_sl, -sl_pct, s_pnl)
    s_pnl = np.where(s_hit_tp & ~s_hit_sl, tp_pct, s_pnl)
    
    # Combined returns
    return l_pnl + s_pnl

# ==========================================
# 3. Grid Search Optimization
# ==========================================
def optimize_strategy(df):
    best_sharpe = -999
    best_params = (0, 0)
    best_curve = []
    
    # Search space: 0.1% to 5.0%
    range_pct = np.linspace(0.001, 0.05, 50) 
    
    # Iterate grid
    for sl in range_pct:
        for tp in range_pct:
            returns = backtest(df, sl, tp)
            
            if np.std(returns) == 0:
                continue
                
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(24 * 365)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (sl, tp)
                best_curve = np.cumsum(returns)
                
    return best_params, best_sharpe, best_curve

# ==========================================
# 4. HTTP Server Handler
# ==========================================
class BacktestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Run Logic
            df = get_binance_data()
            (best_sl, best_tp), sharpe, curve = optimize_strategy(df)
            
            # Generate Plot
            plt.figure(figsize=(10, 6))
            plt.plot(curve, label=f'Best Strategy (SL={best_sl*100:.1f}%, TP={best_tp*100:.1f}%)')
            plt.title(f'ETH/USDT 30D Hourly Volatility Strategy\nSharpe: {sharpe:.2f}')
            plt.xlabel('Hours')
            plt.ylabel('Cumulative Return (Unleveraged)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            # Generate HTML
            html = f"""
            <html>
            <head><title>Bot Results</title></head>
            <body style="font-family: monospace; padding: 20px; background: #f0f0f0;">
                <h1>Optimization Results</h1>
                <div style="background: white; padding: 20px; border-radius: 5px;">
                    <h3>Parameters</h3>
                    <ul>
                        <li><b>Stop Loss:</b> {best_sl*100:.2f}%</li>
                        <li><b>Take Profit:</b> {best_tp*100:.2f}%</li>
                        <li><b>Max Sharpe Ratio:</b> {sharpe:.4f}</li>
                    </ul>
                    <h3>Visualization</h3>
                    <img src="data:image/png;base64,{img_base64}" />
                </div>
            </body>
            </html>
            """
            
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
            
        except Exception as e:
            self.send_response(500)
            self.wfile.write(str(e).encode('utf-8'))

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    PORT = 8080
    Handler = BacktestHandler
    
    print(f"Serving visualization on http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()
