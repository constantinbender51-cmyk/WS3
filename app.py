import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

# --- CONFIGURATION ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LIMIT = 2000 
PORT = 8080

exchange = ccxt.binance()
report_html = "<h1>Simulating... Refresh in 10s.</h1>"

def fit_ols(x, y):
    if len(x) < 2: return None, None
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def run_backtest():
    global report_html
    print("Fetching data...")
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1000)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
    
    # State variables
    active_side = None # 'long', 'short', or None
    stop_level = 0
    target_dist = 0
    trade_log = []
    equity = [0]
    
    # Logic Parameters
    W = 50 # Fixed window for simplified version
    BUFF = 0.10

    print("Running Simulation...")
    for i in range(W + 1, len(df)):
        row = df.iloc[i]
        prev_close = df.iloc[i-1]['close']
        
        # 1. ENTRY LOGIC (Only if no active breakout)
        if active_side is None:
            # Fit Lagged OLS (exclude current candle i)
            x = np.arange(i-W, i)
            y = df['close'].values[i-W:i]
            yh = df['high'].values[i-W:i]
            yl = df['low'].values[i-W:i]
            
            m, c = fit_ols(x, y)
            yt = m * x + c
            mu, cu = fit_ols(x[yh > yt], yh[yh > yt])
            ml, cl = fit_ols(x[yl < yt], yl[yl < yt])
            
            if mu and ml:
                top = mu * i + cu
                bot = ml * i + cl
                dist = top - bot
                
                if prev_close > top + (dist * BUFF):
                    active_side = 'long'
                    stop_level = top
                    target_dist = dist
                elif prev_close < bot - (dist * BUFF):
                    active_side = 'short'
                    stop_level = bot
                    target_dist = dist

        # 2. PNL & EXIT LOGIC
        hour_pnl = 0
        if active_side == 'long':
            # Check for Stop (Support Touch) or Target (Distance Floor)
            if row['low'] <= stop_level:
                hour_pnl = stop_level - row['open']
                active_side = None
            elif row['high'] >= (stop_level + target_dist):
                hour_pnl = (stop_level + target_dist) - row['open']
                active_side = None
            else:
                hour_pnl = row['close'] - row['open']
                
        elif active_side == 'short':
            # Check for Stop (Resistance Touch) or Target (Distance Floor)
            if row['high'] >= stop_level:
                hour_pnl = row['open'] - stop_level
                active_side = None
            elif row['low'] <= (stop_level - target_dist):
                hour_pnl = row['open'] - (stop_level - target_dist)
                active_side = None
            else:
                hour_pnl = row['open'] - row['close']
        
        equity.append(equity[-1] + hour_pnl)
        if hour_pnl != 0 or active_side is not None:
            trade_log.append({'idx': i, 'pnl': hour_pnl, 'side': active_side})

    # Generate Output
    plt.figure(figsize=(10, 5))
    plt.style.use('dark_background')
    plt.plot(equity, color='cyan')
    plt.title("Simplified Equity Curve")
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    img_str = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    report_html = f"""
    <html><body style="background:#111; color:#eee; font-family:sans-serif; text-align:center;">
        <h1>Simplified Backtest Report</h1>
        <img src="data:image/png;base64,{img_str}">
        <p>Total Trades/Steps: {len(trade_log)}</p>
        <p>Final Return: {equity[-1]:.2f}</p>
    </body></html>
    """

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(report_html.encode())

if __name__ == "__main__":
    threading.Thread(target=run_backtest, daemon=True).start()
    HTTPServer(('', PORT), Handler).serve_forever()
