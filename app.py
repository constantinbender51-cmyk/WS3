import http.server
import socketserver
import json
import urllib.request
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Global variables
prices = None
timestamps = None
full_start_date = None
full_end_date = None

def fetch_data():
    """Fetch BTC 1h data for last 30 days from Binance"""
    global prices, timestamps, full_start_date, full_end_date
    
    try:
        full_end_date = datetime.now()
        full_start_date = full_end_date - timedelta(days=30)
        
        end_time = int(full_end_date.timestamp() * 1000)
        start_time = int(full_start_date.timestamp() * 1000)
        
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&startTime={start_time}&endTime={end_time}&limit=1000"
        
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        
        timestamps = np.array([int(item[0]) for item in data]).reshape(-1, 1)
        prices = np.array([float(item[4]) for item in data])
        
        print(f"✅ Fetched {len(prices)} hours of BTC data")
        return True
        
    except Exception as e:
        print(f"⚠️ Error: {e}")
        generate_sample_data()
        return False

def generate_sample_data():
    """Generate simple sample data"""
    global prices, timestamps, full_start_date, full_end_date
    
    np.random.seed(42)
    n_points = 720
    
    full_end_date = datetime.now()
    full_start_date = full_end_date - timedelta(days=30)
    base_time = full_start_date.timestamp() * 1000
    
    timestamps = np.array([base_time + i * 3600000 for i in range(n_points)]).reshape(-1, 1)
    
    # Sine wave with trend
    x = np.linspace(0, 4*np.pi, n_points)
    prices = 40000 + 2000 * np.sin(x) + 3000 * x/len(x) + np.random.normal(0, 200, n_points)
    
    print(f"⚠️ Generated {n_points} hours of sample data")

def find_best_line(data_prices, data_timestamps, start_idx):
    """Find the best line on the given data segment"""
    best_error = float('inf')
    best_line = None
    best_window = 0
    best_slope = 0
    best_end = 0
    
    n_points = len(data_prices)
    
    for window in range(10, min(101, n_points + 1)):
        X = data_timestamps[-window:]
        y = data_prices[-window:]
        
        X_mean = X.mean()
        X_std = X.std()
        if X_std == 0:
            continue
        X_norm = (X - X_mean) / X_std
        
        model = LinearRegression()
        model.fit(X_norm, y)
        
        y_pred = model.predict(X_norm)
        error = np.sum(np.abs(y - y_pred)) / (window * window)
        slope = model.coef_[0] / X_std * 3600000
        
        if error < best_error:
            best_error = error
            best_window = window
            best_line = (X, y_pred, model, (X_mean, X_std))
            best_slope = slope
            best_end = start_idx + n_points
    
    return {
        'start_idx': start_idx,
        'end_idx': best_end,
        'window': best_window,
        'slope': best_slope,
        'error': best_error,
        'line_data': best_line,
        'n_points': n_points
    }

def find_cascade():
    """Find cascade of lines and calculate trading returns"""
    print("\n" + "=" * 60)
    print("📊 Finding cascade of lines...")
    print("=" * 60)
    
    cascade = []
    trades = []  # (timestamp, return, slope_sign)
    cumulative_returns = [0]
    cumul_points = []
    
    current_prices = prices.copy()
    current_timestamps = timestamps.copy()
    current_start = 0
    iteration = 1
    
    while len(current_prices) >= 110:
        result = find_best_line(current_prices, current_timestamps, current_start)
        
        if result['window'] == 0:
            break
            
        cascade.append(result)
        
        # Calculate trading returns for this line
        line_start = result['end_idx'] - result['window']
        line_end = result['end_idx']
        
        # Trade from the end of this line until we run out of data
        # or until the next line starts (which will be at line_start)
        trade_end = line_end
        next_start = result['start_idx']  # This will be the start of next iteration
        
        print(f"\n📌 Line {iteration}:")
        print(f"   Period: {line_start}h to {line_end}h (window={result['window']}h)")
        print(f"   Slope: {result['slope']:+.2f} $/h")
        print(f"   Trading from {line_end}h to {next_start + result['window']}h")
        
        # Trade on each candle from line_end to next_start + window
        for t in range(line_end, min(next_start + result['window'], len(prices) - 1)):
            if t + 1 >= len(prices):
                break
                
            # Calculate next candle return
            ret = (prices[t + 1] - prices[t]) / prices[t] * 100
            
            # Apply based on slope direction
            if result['slope'] > 0:
                trade_ret = ret
                direction = 'LONG'
            else:
                trade_ret = -ret
                direction = 'SHORT'
            
            trades.append({
                'time': t,
                'return': trade_ret,
                'direction': direction,
                'line': iteration
            })
            
            # Update cumulative
            last_cumul = cumulative_returns[-1]
            new_cumul = last_cumul + trade_ret
            cumulative_returns.append(new_cumul)
            cumul_points.append(t)
        
        # Remove the window we just used
        new_end = line_start
        if new_end <= current_start:
            break
            
        current_prices = prices[current_start:new_end]
        current_timestamps = timestamps[current_start:new_end]
        iteration += 1
        
        if iteration > 20:
            break
    
    print(f"\n✅ Found {len(cascade)} cascade lines")
    print(f"   Generated {len(trades)} trades")
    
    if trades:
        wins = sum(1 for t in trades if t['return'] > 0)
        total_ret = cumulative_returns[-1]
        print(f"   Win rate: {wins/len(trades)*100:.1f}%")
        print(f"   Total return: {total_ret:+.2f}%")
    
    return cascade, trades, cumulative_returns[1:], cumul_points

def create_plot(cascade, trades, cumulative_returns, cumul_points):
    """Create plot with price, lines, and returns"""
    plt.figure(figsize=(14, 12))
    
    # Create 3 subplots
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    # Top plot: Price with cascade lines
    ax1 = plt.subplot(gs[0])
    
    # Plot price
    ax1.plot(range(len(prices)), prices, 'b-', alpha=0.5, label='BTC Price', linewidth=1)
    
    # Plot each cascade line
    colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    
    for i, line in enumerate(cascade):
        color = colors[i % len(colors)]
        line_start = line['end_idx'] - line['window']
        line_end = line['end_idx']
        
        # Get the line data
        X, y_pred, model, norm_params = line['line_data']
        
        # Plot the line
        x_range = range(line_start, line_end)
        ax1.plot(x_range, y_pred, color=color, linewidth=2.5, 
                label=f'Line {i+1}: win={line["window"]}h')
        
        # Mark the window
        ax1.scatter(x_range, prices[line_start:line_end], c=color, s=15, alpha=0.3)
        
        # Add line number
        ax1.text(line_start, prices[line_start] - 200, f'{i+1}', 
                fontsize=10, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
    
    ax1.set_title('BTC Price with Cascade Lines', fontsize=12)
    ax1.set_ylabel('Price (USDT)')
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper left', fontsize=7)
    
    # Middle plot: Individual trade returns
    ax2 = plt.subplot(gs[1])
    
    if trades:
        times = [t['time'] for t in trades]
        returns = [t['return'] for t in trades]
        colors2 = ['green' if r > 0 else 'red' for r in returns]
        
        ax2.bar(times, returns, width=0.8, color=colors2, alpha=0.6)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add line boundaries
        for i, line in enumerate(cascade):
            ax2.axvline(x=line['end_idx'], color=colors[i % len(colors)], 
                       linestyle='--', alpha=0.3, linewidth=1)
    
    ax2.set_ylabel('Return per Trade (%)')
    ax2.set_title('Individual Trade Returns (colors = line boundaries)')
    ax2.grid(True, alpha=0.2)
    
    # Bottom plot: Cumulative returns
    ax3 = plt.subplot(gs[2])
    
    if cumul_points and cumulative_returns:
        ax3.plot(cumul_points, cumulative_returns, 'b-', linewidth=2, label='Strategy')
        
        # Add buy & hold for comparison
        if cumul_points:
            start_price = prices[cumul_points[0]]
            buy_hold = [(prices[t] - start_price) / start_price * 100 for t in cumul_points]
            ax3.plot(cumul_points, buy_hold, 'gray', linewidth=1.5, alpha=0.7, label='Buy & Hold')
        
        # Add line boundaries
        for i, line in enumerate(cascade):
            if line['end_idx'] <= cumul_points[-1]:
                ax3.axvline(x=line['end_idx'], color=colors[i % len(colors)], 
                           linestyle='--', alpha=0.3, linewidth=1)
    
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Cumulative Return (%)')
    ax3.set_title('Cumulative Returns')
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc='upper left')
    
    # Add stats text
    if trades:
        wins = sum(1 for t in trades if t['return'] > 0)
        win_rate = wins/len(trades)*100
        total_ret = cumulative_returns[-1] if cumulative_returns else 0
        avg_ret = np.mean([t['return'] for t in trades])
        
        stats_text = f'Trades: {len(trades)} | Win Rate: {win_rate:.1f}% | Total: {total_ret:+.2f}% | Avg: {avg_ret:+.2f}%'
        ax3.text(0.02, 0.95, stats_text, transform=ax3.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')

class BTCRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Find cascade and trades
            cascade, trades, cumul_returns, cumul_points = find_cascade()
            
            # Create plot
            image_base64 = create_plot(cascade, trades, cumul_returns, cumul_points)
            
            # Calculate stats
            if trades:
                wins = sum(1 for t in trades if t['return'] > 0)
                win_rate = wins/len(trades)*100
                total_ret = cumul_returns[-1] if cumul_returns else 0
                avg_ret = np.mean([t['return'] for t in trades])
                max_win = max(t['return'] for t in trades)
                max_loss = min(t['return'] for t in trades)
            else:
                win_rate = total_ret = avg_ret = max_win = max_loss = 0
            
            # HTML
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>BTC Cascade Trading</title>
                <style>
                    body {{ margin: 20px; font-family: Arial; background: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
                    h1 {{ margin: 0 0 10px 0; font-size: 20px; }}
                    .stats {{ display: flex; gap: 15px; flex-wrap: wrap; margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 4px; }}
                    .stat {{ padding: 4px 12px; border-radius: 16px; font-size: 13px; }}
                    .positive {{ background: #d4edda; color: #155724; }}
                    .negative {{ background: #f8d7da; color: #721c24; }}
                    .neutral {{ background: #e2e3e5; color: #383d41; }}
                    img {{ width: 100%; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📈 BTC Cascade Trading Strategy</h1>
                    <div class="stats">
                        <span class="stat neutral">📊 {len(cascade)} lines</span>
                        <span class="stat neutral">🎯 {len(trades)} trades</span>
                        <span class="stat positive">📈 Win: {win_rate:.1f}%</span>
                        <span class="stat {"positive" if total_ret > 0 else "negative"}">💰 {total_ret:+.2f}%</span>
                        <span class="stat positive">🏆 {max_win:+.2f}%</span>
                        <span class="stat negative">📉 {max_loss:+.2f}%</span>
                        <span class="stat neutral">📏 {avg_ret:+.2f}%</span>
                    </div>
                    <img src="data:image/png;base64,{image_base64}">
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()

def main():
    print("=" * 60)
    print("🚀 BTC Cascade Trading Server")
    print("=" * 60)
    print("   Long when slope > 0, Short when slope < 0")
    
    # Fetch data on startup
    print("\n📡 Fetching BTC data...")
    fetch_data()
    
    # Start server
    PORT = 8080
    with socketserver.TCPServer(("", PORT), BTCRequestHandler) as httpd:
        print(f"\n🌐 http://localhost:{PORT}")
        print("   Press Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()