#!/usr/bin/env python3
"""
Simple BTC Trading Strategy Backtest
Run: python btc_strategy.py
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import http.server
import socketserver
import base64
from io import BytesIO
import json
import argparse

# ========== CONFIG ==========
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
DAYS = 30
K = 1.8
PORT = 8000

# ========== FETCH ==========
def fetch():
    """Fetch BTC data from Binance"""
    end = datetime.now()
    start = end - timedelta(days=DAYS)
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": int(start.timestamp() * 1000),
        "endTime": int(end.timestamp() * 1000),
        "limit": 1000
    }
    
    data = requests.get(url, params=params).json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    return df[['timestamp', 'close']]

# ========== FIT & OPTIMIZE ==========
def fit_and_optimize(prices):
    """For each point, find best window and get slope"""
    signals = []
    windows = []
    
    for i in range(10, len(prices)):
        # Try all windows from 10 to min(100, i)
        best_window = 10
        best_error = float('inf')
        
        for w in range(10, min(101, i+1)):
            # Get last w prices
            y = prices.iloc[i-w:i].values.reshape(-1, 1)
            x = np.arange(w).reshape(-1, 1)
            
            # Fit line
            model = LinearRegression()
            model.fit(x, y)
            pred = model.predict(x).flatten()
            
            # Calculate error
            error = np.sum(np.abs(y.flatten() - pred)) / (w ** K)
            
            if error < best_error:
                best_error = error
                best_window = w
        
        # Use best window to get slope
        y = prices.iloc[i-best_window:i].values.reshape(-1, 1)
        x = np.arange(best_window).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        slope = model.coef_[0][0]
        
        signals.append(1 if slope > 0 else -1)
        windows.append(best_window)
    
    return signals, windows

# ========== BACKTEST ==========
def backtest(prices, signals):
    """Run backtest"""
    equity = [1000]
    
    for i in range(min(len(signals), len(prices)-1)):
        ret = signals[i] * (prices.iloc[i+1] - prices.iloc[i]) / prices.iloc[i]
        equity.append(equity[-1] * (1 + ret))
    
    returns = [(equity[i] - equity[i-1])/equity[i-1] for i in range(1, len(equity))]
    return equity[1:], returns

# ========== STATS ==========
def stats(equity, returns, windows):
    """Calculate statistics"""
    total = (equity[-1] / 1000 - 1) * 100
    wins = sum(1 for r in returns if r > 0)
    
    return {
        'total_return': round(total, 2),
        'win_rate': round(wins/len(returns)*100, 1),
        'num_trades': len(returns),
        'avg_window': round(np.mean(windows), 1),
        'min_window': min(windows),
        'max_window': max(windows),
        'k': K
    }

# ========== PLOT ==========
def plot(prices, equity, signals, windows, stats):
    """Create plot"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Price
    ax1.plot(prices.index, prices.values, 'k-', alpha=0.5)
    for i, s in enumerate(signals):
        if i < len(prices)-1:
            color = 'g' if s == 1 else 'r'
            ax1.scatter(prices.index[i+1], prices.iloc[i+1], c=color, s=20, alpha=0.5)
    ax1.set_title('Price with Signals')
    ax1.grid(True, alpha=0.3)
    
    # Equity
    ax2.plot(prices.index[1:len(equity)+1], equity, 'b-')
    ax2.axhline(y=1000, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title(f'Equity (Return: {stats["total_return"]}%)')
    ax2.grid(True, alpha=0.3)
    
    # Windows
    ax3.hist(windows, bins=20, color='orange', edgecolor='black')
    ax3.axvline(x=stats['avg_window'], color='r', linestyle='--')
    ax3.set_title(f'Window Sizes (avg: {stats["avg_window"]})')
    ax3.grid(True, alpha=0.3)
    
    # Returns
    ax4.hist([(equity[i] - equity[i-1]) for i in range(1, len(equity))], 
             bins=20, color='purple', edgecolor='black')
    ax4.set_title('Returns Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return img

# ========== SERVE ==========
def serve(img, stats):
    """Simple HTTP server"""
    html = f"""
    <html>
    <head><title>BTC Strategy</title>
    <style>
        body {{ font-family: Arial; margin: 20px; background: #f0f2f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0; }}
        .stat {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 15px; border-radius: 5px; text-align: center; }}
        img {{ width: 100%; border-radius: 5px; }}
    </style>
    </head>
    <body>
    <div class="container">
        <h1>BTC Strategy</h1>
        <div class="stats">
            <div class="stat"><h3>Return</h3><p>{stats["total_return"]}%</p></div>
            <div class="stat"><h3>Win Rate</h3><p>{stats["win_rate"]}%</p></div>
            <div class="stat"><h3>Trades</h3><p>{stats["num_trades"]}</p></div>
            <div class="stat"><h3>Avg Window</h3><p>{stats["avg_window"]}</p></div>
        </div>
        <img src="data:image/png;base64,{img}">
        <p style="color: gray; text-align: center;">Generated: {datetime.now()}</p>
    </div>
    </body>
    </html>
    """
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Server at http://localhost:{PORT}")
        httpd.serve_forever()

# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true', help='Run HTTP server')
    parser.add_argument('--port', type=int, default=PORT)
    args = parser.parse_args()
    
    # Run strategy
    print("Fetching data...")
    df = fetch()
    prices = df['close']
    
    print("Fitting and optimizing...")
    signals, windows = fit_and_optimize(prices)
    
    print("Backtesting...")
    equity, returns = backtest(prices, signals)
    
    print("Calculating stats...")
    s = stats(equity, returns, windows)
    
    print("Creating plot...")
    img = plot(prices, equity, signals, windows, s)
    
    if args.serve:
        print(f"Starting server on port {args.port}...")
        serve(img, s)
    else:
        # Save files
        with open('report.html', 'w') as f:
            html = f"""
            <html><body>
            <h1>BTC Strategy Results</h1>
            <pre>{json.dumps(s, indent=2)}</pre>
            <img src="data:image/png;base64,{img}">
            </body></html>
            """
            f.write(html)
        print(f"Results: {s}")
        print("Report saved to report.html")

if __name__ == "__main__":
    main()