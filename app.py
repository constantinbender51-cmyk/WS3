#!/usr/bin/env python3
"""
BTC Trading Strategy - Visualize Optimized Lines
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
print("🧮 Computing signals and optimized lines...")
signals = []
windows = []
slopes = []
intercepts = []
line_prices = []  # Store the fitted line prices for visualization

for i in range(10, len(prices)):
    # Find best window
    best_w = 10
    best_err = float('inf')
    best_model = None
    
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
            best_model = model
    
    # Store results with best window
    signals.append(1 if best_model.coef_[0][0] > 0 else -1)
    windows.append(best_w)
    slopes.append(best_model.coef_[0][0])
    intercepts.append(best_model.intercept_[0])
    
    # Generate line prices for the last point (for visualization)
    x_line = np.arange(best_w).reshape(-1, 1)
    y_line = best_model.predict(x_line).flatten()
    line_prices.append({
        'times': times[i-best_w:i],
        'prices': y_line,
        'slope': best_model.coef_[0][0],
        'window': best_w,
        'end_idx': i
    })

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
fig = plt.figure(figsize=(14, 10))

# Price chart with optimized lines
ax1 = plt.subplot(3, 2, 1)
price_times = times[10:]
price_values = prices[10:]
ax1.plot(price_times, price_values, 'k-', alpha=0.7, linewidth=1, label='BTC Price')

# Plot a few sample optimized lines (every 20th line to avoid clutter)
for idx, line in enumerate(line_prices[::20]):
    if idx < len(line_prices[::20]):
        color = 'g' if line['slope'] > 0 else 'r'
        alpha = 0.3
        ax1.plot(line['times'], line['prices'], color=color, alpha=alpha, linewidth=1)

ax1.set_title('BTC Price with Optimized Lines (Green=Up, Red=Down)')
ax1.set_ylabel('Price (USDT)')
ax1.grid(True, alpha=0.3)
ax1.legend(['Price', 'Optimized Lines'])

# Price with signals (traditional view)
ax2 = plt.subplot(3, 2, 2)
ax2.plot(price_times, price_values, 'k-', alpha=0.5, linewidth=1, label='Price')

# Add signals
for i in range(min(len(signals), len(price_times)-1)):
    color = 'g' if signals[i] == 1 else 'r'
    ax2.scatter(price_times[i+1], price_values[i+1], c=color, s=20, alpha=0.7, 
               marker='^' if signals[i]==1 else 'v')

ax2.set_title(f'Trading Signals (Return: {total_return:.1f}%)')
ax2.set_ylabel('Price (USDT)')
ax2.grid(True, alpha=0.3)

# Cumulative returns
ax3 = plt.subplot(3, 2, 3)
if len(cumulative) > 0:
    max_len = min(len(times[11:]), len(cumulative))
    return_times = times[11:11+max_len]
    returns_to_plot = cumulative[:max_len]
    ax3.plot(return_times, returns_to_plot, 'b-', linewidth=2)
    
ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax3.set_title(f'Cumulative Returns: {total_return:.1f}%')
ax3.set_ylabel('Return (%)')
ax3.grid(True, alpha=0.3)

# Window sizes over time
ax4 = plt.subplot(3, 2, 4)
window_times = times[10:10+len(windows)]
ax4.plot(window_times, windows, 'orange', linewidth=1)
ax4.set_title(f'Window Sizes Over Time (Avg: {np.mean(windows):.1f})')
ax4.set_ylabel('Window Size')
ax4.grid(True, alpha=0.3)

# Window distribution
ax5 = plt.subplot(3, 2, 5)
ax5.hist(windows, bins=20, color='orange', edgecolor='black', alpha=0.7)
ax5.axvline(x=np.mean(windows), color='r', linestyle='--', label=f'Avg: {np.mean(windows):.1f}')
ax5.set_title('Window Size Distribution')
ax5.set_xlabel('Window')
ax5.set_ylabel('Frequency')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Returns distribution
ax6 = plt.subplot(3, 2, 6)
ax6.hist(returns, bins=20, color='purple', edgecolor='black', alpha=0.7)
ax6.axvline(x=0, color='r', linestyle='--')
ax6.set_title(f'Trade Returns (Win Rate: {win_rate:.1f}%)')
ax6.set_xlabel('Return (%)')
ax6.set_ylabel('Frequency')
ax6.grid(True, alpha=0.3)

plt.tight_layout()

# Convert to base64
buf = BytesIO()
plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
buf.seek(0)
img = base64.b64encode(buf.getvalue()).decode()
plt.close()

# ========== CREATE HTML ==========
html = f"""
<html>
<head>
    <title>BTC Strategy - Optimized Lines</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .stat {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .stat h3 {{ margin: 0; font-size: 14px; opacity: 0.9; }}
        .stat .value {{ font-size: 28px; font-weight: bold; margin: 10px 0 0; }}
        .positive {{ color: #4caf50; }}
        .negative {{ color: #f44336; }}
        img {{ width: 100%; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .footer {{ text-align: center; color: #666; margin-top: 20px; font-size: 12px; }}
        .info {{ background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 BTC Trading Strategy - Optimized Lines</h1>
        
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
        
        <div class="info">
            <strong>How it works:</strong> For each point, we find the optimal window (10-100) that minimizes 
            error = sum(|actual - predicted|) / window^{K}. The fitted line determines the signal: 
            <span style="color:green">green = up (buy)</span>, <span style="color:red">red = down (sell)</span>.
            Top-left chart shows sample optimized lines overlaid on price.
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