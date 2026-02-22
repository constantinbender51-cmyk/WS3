#!/usr/bin/env python3
"""
BTC Trading Strategy - Hold Until Signal Changes
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
K = 1.6

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
raw_signals = []  # Signal at each hour (before position holding)
windows = []
line_prices = []

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
    
    raw_signals.append(1 if best_model.coef_[0][0] > 0 else -1)
    windows.append(best_w)
    
    # Store line for visualization
    x_line = np.arange(best_w).reshape(-1, 1)
    y_line = best_model.predict(x_line).flatten()
    line_prices.append({
        'times': times[i-best_w:i],
        'prices': y_line,
        'slope': best_model.coef_[0][0],
        'time': times[i]
    })

print(f"✅ Got {len(raw_signals)} raw signals")

# ========== APPLY POSITION HOLDING ==========
print("🔄 Applying position holding...")
positions = []  # Actual positions (hold until signal changes)
current_position = raw_signals[0]
position_changes = []  # Store when position changes

for i, signal in enumerate(raw_signals):
    if signal != current_position:
        # Position changes
        position_changes.append({
            'from': current_position,
            'to': signal,
            'time': times[10 + i],  # Adjust for starting index
            'index': 10 + i
        })
        current_position = signal
    positions.append(current_position)

print(f"✅ Position changes: {len(position_changes)}")

# ========== CALCULATE RETURNS ==========
print("📊 Calculating returns...")
returns = []
trade_times = []
position_start_idx = 10  # First signal at index 10
current_pos = positions[0]
entry_price = prices[10]

for i in range(1, len(positions)):
    current_idx = 10 + i
    if current_idx >= len(prices):
        break
        
    if positions[i] != positions[i-1] or i == len(positions)-1:
        # Position changed or end of data - close trade
        exit_price = prices[current_idx]
        
        if positions[i-1] == 1:  # Long
            ret = (exit_price - entry_price) / entry_price * 100
        else:  # Short
            ret = (entry_price - exit_price) / entry_price * 100
            
        returns.append(ret)
        trade_times.append(times[current_idx])
        
        # Start new position
        entry_price = prices[current_idx]
        current_pos = positions[i]

# Calculate cumulative returns
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
fig = plt.figure(figsize=(16, 10))

# Price with positions
ax1 = plt.subplot(3, 2, 1)
ax1.plot(times[10:], prices[10:], 'k-', alpha=0.7, linewidth=1, label='Price')

# Color price based on position
for i in range(len(positions)-1):
    start_idx = 10 + i
    end_idx = 10 + i + 1
    color = 'g' if positions[i] == 1 else 'r'
    ax1.axvspan(times[start_idx], times[end_idx], alpha=0.1, color=color)

# Mark position changes
for change in position_changes:
    marker = '^' if change['to'] == 1 else 'v'
    ax1.scatter(change['time'], prices[change['index']], 
               c='blue', s=50, marker=marker, zorder=5)

ax1.set_title('Price with Positions (Green=Long, Red=Short, ▲/▼=Signal Change)')
ax1.set_ylabel('Price (USDT)')
ax1.grid(True, alpha=0.3)

# Sample optimized lines
ax2 = plt.subplot(3, 2, 2)
ax2.plot(times[10:], prices[10:], 'k-', alpha=0.3, linewidth=1)
for idx, line in enumerate(line_prices[::10]):
    if idx < len(line_prices[::10]):
        color = 'g' if line['slope'] > 0 else 'r'
        ax2.plot(line['times'], line['prices'], color=color, alpha=0.2, linewidth=1)
ax2.set_title('Sample Optimized Lines')
ax2.set_ylabel('Price (USDT)')
ax2.grid(True, alpha=0.3)

# Position history
ax3 = plt.subplot(3, 2, 3)
pos_times = times[10:10+len(positions)]
ax3.plot(pos_times, positions, 'b-', linewidth=2)
ax3.set_ylim(-1.5, 1.5)
ax3.set_yticks([-1, 1])
ax3.set_yticklabels(['Short', 'Long'])
ax3.set_title('Position Over Time')
ax3.grid(True, alpha=0.3)

# Cumulative returns
ax4 = plt.subplot(3, 2, 4)
if len(cumulative) > 0:
    ax4.plot(trade_times, cumulative, 'b-', linewidth=2)
ax4.axhline(y=0, color='gray', linestyle='--')
ax4.set_title(f'Cumulative Returns: {total_return:.1f}%')
ax4.set_ylabel('Return (%)')
ax4.grid(True, alpha=0.3)

# Trade returns
ax5 = plt.subplot(3, 2, 5)
ax5.bar(range(len(returns)), returns, color=['g' if r>0 else 'r' for r in returns])
ax5.axhline(y=0, color='black', linewidth=1)
ax5.set_title(f'Trade Returns (Win Rate: {win_rate:.1f}%)')
ax5.set_xlabel('Trade #')
ax5.set_ylabel('Return (%)')
ax5.grid(True, alpha=0.3)

# Returns distribution
ax6 = plt.subplot(3, 2, 6)
ax6.hist(returns, bins=15, color='purple', edgecolor='black', alpha=0.7)
ax6.axvline(x=0, color='r', linestyle='--')
ax6.set_title('Returns Distribution')
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
    <title>BTC Strategy - Hold Until Signal Changes</title>
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
        .info {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 BTC Strategy - Hold Until Signal Changes</h1>
        
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
                <h3>Total Trades</h3>
                <div class="value">{len(returns)}</div>
            </div>
            <div class="stat">
                <h3>Avg Window</h3>
                <div class="value">{np.mean(windows):.1f}</div>
            </div>
        </div>
        
        <div class="info">
            <strong>Strategy:</strong> Enter position when signal changes. Hold until next signal change.<br>
            Green = Long, Red = Short, Blue ▲/▼ = Signal change points
        </div>
        
        <img src="data:image/png;base64,{img}">
        
        <div class="footer">
            k = {K} | Position changes: {len(position_changes)} | Computed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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