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
LIMIT = 200
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
    plt.figure(figsize=(15, 9))
    plt.style.use('dark_background')
    
    x_full = np.arange(len(df))
    last_idx = x_full[-1]
    last_close = df['close'].iloc[-1]
    
    # 1. Render Lines First (Background)
    for w in range(10, 101):
        if len(df) < w:
            continue
            
        x_win = x_full[-w:]
        y_win_close = df['close'].values[-w:]
        y_win_high = df['high'].values[-w:]
        y_win_low = df['low'].values[-w:]
        
        m_mid, c_mid = fit_ols(x_win, y_win_close)
        if m_mid is None: continue
        y_trend = m_mid * x_win + c_mid
        
        upper_mask = y_win_high > y_trend
        lower_mask = y_win_low < y_trend
        
        m_u, c_u = fit_ols(x_win[upper_mask], y_win_high[upper_mask])
        m_l, c_l = fit_ols(x_win[lower_mask], y_win_low[lower_mask])
        
        # Determine if lines should be shown (only if crossed)
        show_lines = False
        if m_u is not None and m_l is not None:
            upper_val = m_u * last_idx + c_u
            lower_val = m_l * last_idx + c_l
            if last_close > upper_val or last_close < lower_val:
                show_lines = True
        
        if show_lines:
            plt.plot(x_win, y_trend, color='red', linewidth=0.5, zorder=1)
            plt.plot(x_win, m_u * x_win + c_u, color='red', linewidth=0.5, zorder=1)
            plt.plot(x_win, m_l * x_win + c_l, color='red', linewidth=0.5, zorder=1)

    # 2. Render Candles Second (Foreground)
    width = .6
    up = df[df.close >= df.open]
    down = df[df.close < df.open]
    
    plt.bar(up.index, up.close - up.open, width, bottom=up.open, color='green', zorder=2)
    plt.bar(up.index, up.high - np.maximum(up.close, up.open), 0.05, bottom=np.maximum(up.close, up.open), color='green', zorder=2)
    plt.bar(up.index, np.minimum(up.close, up.open) - up.low, 0.05, bottom=up.low, color='green', zorder=2)
    
    plt.bar(down.index, down.close - down.open, width, bottom=down.open, color='red', zorder=2)
    plt.bar(down.index, down.high - np.maximum(down.close, down.open), 0.05, bottom=np.maximum(down.close, down.open), color='red', zorder=2)
    plt.bar(down.index, np.minimum(down.close, down.open) - down.low, 0.05, bottom=down.low, color='red', zorder=2)

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
            html = """
            <html>
            <head>
                <script>
                    setInterval(function() {
                        document.getElementById('chart').src = '/chart.png?t=' + new Date().getTime();
                    }, 10000);
                </script>
            </head>
            <body style="background:black;margin:0;padding:0;overflow:hidden">
                <img id="chart" src="/chart.png" style="width:100%">
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        elif self.path.startswith('/chart.png'):
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
            print(f"Update: {df.iloc[-1]['close']}")
        time.sleep(10)

if __name__ == "__main__":
    t = threading.Thread(target=run_server)
    t.daemon = True
    t.start()
    logic_loop()
