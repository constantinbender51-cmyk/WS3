import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import random
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

def get_cost(m, c, x, target):
    line_y = m * x + c
    return np.sum(np.abs(line_y - target))

def optimize_slope(x, target_y):
    # Initialize Randomly around center
    cx = x[len(x)//2]
    cy = np.mean(target_y)
    m = random.uniform(-100, 100)
    
    # Hill Climbing Optimization for Slope
    step = 10.0
    for _ in range(200):
        c = cy - m * cx
        cost_curr = get_cost(m, c, x, target_y)
        
        # Test Up
        m_up = m + step
        c_up = cy - m_up * cx
        cost_up = get_cost(m_up, c_up, x, target_y)
        
        # Test Down
        m_dn = m - step
        c_dn = cy - m_dn * cx
        cost_dn = get_cost(m_dn, c_dn, x, target_y)
        
        if cost_up < cost_curr:
            m = m_up
        elif cost_dn < cost_curr:
            m = m_dn
        else:
            step *= 0.9 # Refine step
            
    return m, cy - m * cx

def generate_plot(df):
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    
    # Candles
    width = .6
    up = df[df.close >= df.open]
    down = df[df.close < df.open]
    plt.bar(up.index, up.close - up.open, width, bottom=up.open, color='green')
    plt.bar(up.index, up.high - up.close, 0.05, bottom=up.close, color='green')
    plt.bar(up.index, up.low - up.open, 0.05, bottom=up.open, color='green')
    plt.bar(down.index, down.close - down.open, width, bottom=down.open, color='red')
    plt.bar(down.index, down.high - down.open, 0.05, bottom=down.open, color='red')
    plt.bar(down.index, down.low - down.close, 0.05, bottom=down.close, color='red')
    
    x = np.arange(len(df))
    
    # Line 1: Resistance -> Minimizes distance to Low
    m1, c1 = optimize_slope(x, df['low'].values)
    plt.plot(x, m1 * x + c1, color='cyan', linewidth=2, label='Res (Target Low)')
    
    # Line 2: Support -> Minimizes distance to High
    m2, c2 = optimize_slope(x, df['high'].values)
    plt.plot(x, m2 * x + c2, color='magenta', linewidth=2, label='Sup (Target High)')

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
    HTTPServer(('', PORT), DashboardHandler).serve_forever()

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
