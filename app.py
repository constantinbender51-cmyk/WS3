import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

# Configuration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LIMIT = 100
PORT = 8080

# Global State
current_plot_data = None
exchange = ccxt.binance()

def get_data():
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
        return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    except:
        return pd.DataFrame()

def fit_ols(x, y):
    if len(x) < 2:
        return None, None
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def generate_plot(df):
    plt.figure(figsize=(12, 8))
    plt.style.use('dark_background')
    
    # Candles
    width = .6
    up = df[df.close >= df.open]
    down = df[df.close < df.open]
    
    plt.bar(up.index, up.close - up.open, width, bottom=up.open, color='green', alpha=0.3)
    plt.bar(up.index, up.high - np.maximum(up.close, up.open), 0.05, bottom=np.maximum(up.close, up.open), color='green', alpha=0.3)
    plt.bar(up.index, np.minimum(up.close, up.open) - up.low, 0.05, bottom=up.low, color='green', alpha=0.3)
    
    plt.bar(down.index, down.close - down.open, width, bottom=down.open, color='red', alpha=0.3)
    plt.bar(down.index, down.high - np.maximum(down.close, down.open), 0.05, bottom=np.maximum(down.close, down.open), color='red', alpha=0.3)
    plt.bar(down.index, np.minimum(down.close, down.open) - down.low, 0.05, bottom=down.low, color='red', alpha=0.3)
    
    # Multi-window OLS fitting
    windows = [10, 20, 50, 100]
    colors_up = ['#00FFFF', '#00CED1', '#4682B4', '#5F9EA0'] # Cyans
    colors_low = ['#FF00FF', '#DA70D6', '#BA55D3', '#9932CC'] # Magentas
    
    x_full = np.arange(len(df))
    
    for i, w in enumerate(windows):
        if len(df) < w:
            continue
            
        # Select trailing window
        window_df = df.suffix(w) if hasattr(df, 'suffix') else df.iloc[-w:]
        x_win = x_full[-w:]
        
        # Base Trend for the window
        m_mid, c_mid = fit_ols(x_win, window_df['close'].values)
        if m_mid is None: continue
        y_trend = m_mid * x_win + c_mid
        
        # Filter points above/below window trend
        upper_mask = window_df['high'].values > y_trend
        lower_mask = window_df['low'].values < y_trend
        
        # OLS Upper
        m_u, c_u = fit_ols(x_win[upper_mask], window_df['high'].values[upper_mask])
        if m_u is not None:
            plt.plot(x_win, m_u * x_win + c_u, color=colors_up[i], 
                     linewidth=1, label=f'Upper {w}', alpha=0.8)
            
        # OLS Lower
        m_l, c_l = fit_ols(x_win[lower_mask], window_df['low'].values[lower_mask])
        if m_l is not None:
            plt.plot(x_win, m_l * x_win + c_l, color=colors_low[i], 
                     linewidth=1, label=f'Lower {w}', alpha=0.8)

    plt.legend(loc='upper left', fontsize='small', ncol=2)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><body style="background:black;display:flex;justify-content:center"><img src="/chart.png"></body></html>')
        elif self.path == '/chart.png':
            if current_plot_data:
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.end_headers()
                self.wfile.write(current_plot_data.getvalue())
            else:
                self.send_error(404)

def run_server():
    server = HTTPServer(('', PORT), DashboardHandler)
    server.serve_forever()

def logic_loop():
    global current_plot_data
    while True:
        df = get_data()
        if not df.empty:
            current_plot_data = generate_plot(df)
            print(f"Plot updated: {df.iloc[-1]['close']}")
        time.sleep(10)

if __name__ == "__main__":
    t = threading.Thread(target=run_server)
    t.daemon = True
    t.start()
    logic_loop()
