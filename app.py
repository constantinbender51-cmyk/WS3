import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import sys
from datetime import datetime, timezone

def fetch_btc_data():
    """Fetch 1h BTC/USDT data from Binance for 2025-2026"""
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1h'
    
    # Set date range: 2025-01-01 to current date (2026-02-22)
    since = exchange.parse8601('2025-01-01T00:00:00Z')
    until = exchange.parse8601('2026-02-22T23:59:59Z')  # Current date per problem
    
    all_ohlcv = []
    start = since
    
    while start < until:
        ohlcv = exchange.fetch_ohlcv(
            symbol, 
            timeframe, 
            since=start,
            limit=1000
        )
        if not ohlcv:
            break
            
        all_ohlcv.extend(ohlcv)
        start = ohlcv[-1][0] + 3600000  # Move to next hour (ms)
        if start >= until:
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(
        all_ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    return df['close'].rename('price')

def find_crossings(prices, fit_line):
    """Find indices where price crosses the fit line"""
    diff = prices.values - fit_line
    sign = np.sign(diff)
    crossings = []
    
    for i in range(1, len(sign)):
        if sign[i] != sign[i-1] and not (sign[i] == 0 and sign[i-1] == 0):
            crossings.append(i)
    return crossings

def main():
    # 1. Fetch BTC data
    print("Fetching BTC data from Binance (2025-2026)...")
    prices = fetch_btc_data()
    
    # 2. Prepare data for analysis
    x = np.arange(len(prices))
    prices_arr = prices.values
    
    # 4. Fit global OLS line
    a_global, b_global = np.polyfit(x, prices_arr, 1)
    global_fit = a_global * x + b_global
    
    # 5. Find crossings
    crossings = find_crossings(prices, global_fit)
    
    # 6. Split prices at crossings
    segments = []
    start_idx = 0
    for idx in crossings:
        segments.append(prices.iloc[start_idx:idx])
        start_idx = idx
    segments.append(prices.iloc[start_idx:])
    
    # 7. Fit lines on individual segments
    segment_lines = []
    for seg in segments:
        if len(seg) < 2:  # Skip tiny segments
            continue
        x_seg = np.arange(len(seg))
        a, b = np.polyfit(x_seg, seg.values, 1)
        segment_lines.append((seg.index, a * x_seg + b))
    
    # 2 & 8. Plot everything
    plt.figure(figsize=(14, 7))
    plt.plot(prices.index, prices.values, 'b-', alpha=0.3, label='BTC Price')
    
    # Plot segment trend lines
    for dates, line in segment_lines:
        plt.plot(dates, line, 'r-', linewidth=1.5)
    
    plt.title('BTC/USDT 1h Price with Segment Trend Lines (2025-2026)')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('btc_plot.png', dpi=120)
    plt.close()
    
    print(f"Plot saved to btc_plot.png ({len(segments)} segments)")
    
    # 3. Serve plot via HTTP
    class PlotHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(
                    b'<html><body><h1>BTC Price Analysis</h1>'
                    b'<img src="/plot.png" style="max-width:100%"></body></html>'
                )
            elif self.path == '/plot.png':
                try:
                    with open('btc_plot.png', 'rb') as f:
                        self.send_response(200)
                        self.send_header('Content-type', 'image/png')
                        self.end_headers()
                        self.wfile.write(f.read())
                except:
                    self.send_error(404, 'Plot not found')
            else:
                self.send_error(404, 'Not Found')
    
    print("Starting HTTP server on http://0.0.0.0:8080")
    print("Press Ctrl+C to stop")
    try:
        httpd = HTTPServer(('0.0.0.0', 8080), PlotHandler)
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")

if __name__ == "__main__":
    main()