#!/usr/bin/env python3
"""
BTC Trading Strategy - Simple Web Server
Run: python btc_strategy.py
"""

import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import http.server
import socketserver
import base64
from io import BytesIO

# ========== CONFIG ==========
PORT = 8000
K = 1.8

# ========== FETCH DATA ==========
print("📥 Fetching BTC data...")
end = datetime.now()
start = end - timedelta(days=30)

url = "https://api.binance.com/api/v3/klines"
params = {
    "symbol": "BTCUSDT",
    "interval": "1h",
    "startTime": int(start.timestamp() * 1000),
    "endTime": int(end.timestamp() * 1000),
    "limit": 1000
}

data = requests.get(url, params=params).json()
prices = [float(x[4]) for x in data]
times = [datetime.fromtimestamp(x[0]/1000) for x in data]

print(f"✅ Got {len(prices)} candles")

# ========== RUN STRATEGY ==========
print("🧮 Computing signals...")
signals = []
windows = []

for i in range(10, len(prices)):
    # Find best window
    best_w = 10
    best_err = float('inf')
    
    for w in range(10, min(101, i+1)):
        y = np.array(prices[i-w:i]).reshape(-1, 1)
        x = np.arange(w).reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(x, y)
        pred = model.predict(x).flatten()
        
        err = np.sum(np.abs(y.flatten() - pred)) / (w ** K)
        
        if err < best_err:
            best_err = err
            best_w = w
    
    # Get slope with best window
    y = np.array(prices[i-best_w:i]).reshape(-1, 1)
    x = np.arange(best_w).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    
    signals.append(1 if model.coef_[0][0] > 0 else -1)
    windows.append(best_w)

print(f"✅ Got {len(signals)} signals")

# ========== CALCULATE RETURNS ==========
print("📊 Calculating returns...")
returns = []
for i in range(len(signals)):
    if i < len(prices) - 1:
        ret = signals[i] * (prices[i+1] - prices[i]) / prices[i] * 100
        returns.append(ret)

cumulative = []
total = 0
for r in returns:
    total += r
    cumulative.append(total)

total_return = sum(returns)
winning_trades = sum(1 for r in returns if r > 0)
win_rate = (winning_trades / len(returns) * 100) if returns else 0

# ========== CREATE PLOT ==========
print("🎨 Creating plot...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Price chart with signals
price_times = times[10:]  # Start from index 10
price_values = prices[10:]
ax1.plot(price_times, price_values, 'k-', alpha=0.5, linewidth=1)

# Add signals (signal for NEXT candle)
for i in range(min(len(signals), len(price_times)-1)):
    color = 'g' if signals[i] == 1 else 'r'
    ax1.scatter(price_times[i+1], price_values[i+1], c=color, s=10, alpha=0.5)

ax1.set_title('BTC Price with Signals')
ax1.set_ylabel('Price (USDT)')
ax1.grid(True, alpha=0.3)

# Cumulative returns
# Returns start from index 11 (first signal at 10, first return at 11)
return_times = times[11:11+len(cumulative)]
ax2.plot(return_times, cumulative, 'b-', linewidth=2)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_title(f'Cumulative Returns: {total_return:.1f}%')
ax2.set_ylabel('Return (%)')
ax2.grid(True, alpha=0.3)

# Windows
ax3.hist(windows, bins=20, color='orange', edgecolor='black')
ax3.axvline(x=np.mean(windows), color='r', linestyle='--', label=f'Avg: {np.mean(windows):.1f}')
ax3.set_title('Window Sizes')
ax3.set_xlabel('Window')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Returns distribution
ax4.hist(returns, bins=20, color='purple', edgecolor='black')
ax4.axvline(x=0, color='r', linestyle='--')
ax4.set_title('Trade Returns Distribution')
ax4.set_xlabel('Return (%)')
ax4.set_ylabel('Frequency')
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Convert to base64
buf = BytesIO()
plt.savefig(buf, format='png', dpi=100)
buf.seek(0)
img = base64.b64encode(buf.getvalue()).decode()
plt.close()

# ========== CREATE HTML ==========
html = f"""
<html>
<head>
    <title>BTC Strategy</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .stat {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .stat h3 {{ margin: 0; font-size: 14px; opacity: 0.9; }}
        .stat .value {{ font-size: 28px; font-weight: bold; margin: 10px 0 0; }}
        .positive {{ color: #4caf50; }}
        .negative {{ color: #f44336; }}
        img {{ width: 100%; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .footer {{ text-align: center; color: #666; margin-top: 20px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 BTC Trading Strategy</h1>
        
        <div class="stats">
            <div class="stat">
                <h3>Total Return</h3>
                <div class="value {'positive' if total_return > 0 else 'negative'}">{total_return:.1f}%</div>
            </div>
            <div class="stat">
                <h3>Win Rate</h3>
                <div class="value">{win_rate:.1f}%</div>
            </div>
            <div class="stat">
                <h3>Avg Window</h3>
                <div class="value">{np.mean(windows):.1f}</div>
            </div>
            <div class="stat">
                <h3>Total Trades</h3>
                <div class="value">{len(returns)}</div>
            </div>
        </div>
        
        <img src="data:image/png;base64,{img}">
        
        <div class="footer">
            k = {K} | Computed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data: {len(prices)} candles
        </div>
    </div>
</body>
</html>
"""

# ========== WEB SERVER ==========
class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

# ========== MAIN ==========
if __name__ == "__main__":
    print(f"✅ Computation complete!")
    print(f"🌐 Starting server on http://localhost:{PORT}")
    print("Press Ctrl+C to stop")
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()