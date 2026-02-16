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
    if len(x) == 0:
        return 0, 0
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def generate_plot(df):
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    
    # Candles
    width = .6
    up = df[df.close >= df.open]
    down = df[df.close < df.open]
    
    plt.bar(up.index, up.close - up.open, width, bottom=up.open, color='green')
    plt.bar(up.index, up.high - np.maximum(up.close, up.open), 0.05, bottom=np.maximum(up.close, up.open), color='green')
    plt.bar(up.index, np.minimum(up.close, up.open) - up.low, 0.05, bottom=up.low, color='green')
    
    plt.bar(down.index, down.close - down.open, width, bottom=down.open, color='red')
    plt.bar(down.index, down.high - np.maximum(down.close, down.open), 0.05, bottom=np.maximum(down.close, down.open), color='red')
    plt.bar(down.index, np.minimum(down.close, down.open) - down.low, 0.05, bottom=down.low, color='red')
    
    x_full = np.arange(len(df))
    
    # Base Trendline
    m_mid, c_mid = fit_ols(x_full, df['close'].values)
    y_trend = m_mid * x_full + c_mid
    
    # Filter points above/below base trend
    upper_mask = df['high'] > y_trend
    lower_mask = df['low'] < y_trend
    
    # Upper Boundary (OLS on points above trend)
    m_up, c_up = fit_ols(x_full[upper_mask], df['high'].values[upper_mask])
    
    # Lower Boundary (OLS on points below trend)
    m_low, c_low = fit_ols(x_full[lower_mask], df['low'].values[lower_mask])
    
    plt.plot(x_full, y_trend, color='yellow', linestyle='--', alpha=0.5, label='Mid Trend')
    plt.plot(x_full, m_up * x_full + c_up, color='cyan', linewidth=2, label='Upper Boundary (Filtered)')
    plt.plot(x_full, m_low * x_full + c_low, color='magenta', linewidth=2, label='Lower Boundary (Filtered)')

    plt.legend()
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
            self.wfile.write(b'<html><body style="background:black"><img src="/chart.png"></body></html>')
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
