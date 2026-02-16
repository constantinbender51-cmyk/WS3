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
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return pd.DataFrame()

def generate_random_line_data(df):
    # Generate valid coordinates within chart limits
    x_min, x_max = 0, len(df)
    y_min, y_max = df['low'].min(), df['high'].max()
    
    # Random points
    x1, x2 = random.uniform(x_min, x_max), random.uniform(x_min, x_max)
    y1, y2 = random.uniform(y_min, y_max), random.uniform(y_min, y_max)
    
    # Slope (m) and Intercept (c)
    if x2 - x1 == 0: m = 0
    else: m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    
    # Generate Y values for the whole range
    x_vals = np.arange(len(df))
    y_vals = m * x_vals + c
    return x_vals, y_vals

def generate_plot(df):
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    
    # Plot Candles
    width = .6
    up = df[df.close >= df.open]
    down = df[df.close < df.open]
    plt.bar(up.index, up.close - up.open, width, bottom=up.open, color='green')
    plt.bar(up.index, up.high - up.close, 0.05, bottom=up.close, color='green')
    plt.bar(up.index, up.low - up.open, 0.05, bottom=up.open, color='green')
    plt.bar(down.index, down.close - down.open, width, bottom=down.open, color='red')
    plt.bar(down.index, down.high - down.open, 0.05, bottom=down.open, color='red')
    plt.bar(down.index, down.low - down.close, 0.05, bottom=down.close, color='red')
    
    # Generate and Plot 2 Random Lines
    for color in ['cyan', 'magenta']:
        x, y = generate_random_line_data(df)
        plt.plot(x, y, color=color, linewidth=2)

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
    print(f"Server on {PORT}")
    server.serve_forever()

def logic_loop():
    global current_plot_data
    while True:
        df = get_data()
        if not df.empty:
            current_plot_data = generate_plot(df)
            print("Plot updated.")
        time.sleep(10)

if __name__ == "__main__":
    t = threading.Thread(target=run_server)
    t.daemon = True
    t.start()
    logic_loop()
