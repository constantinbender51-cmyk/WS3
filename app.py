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

# Global variables to store pre-calculated data
prices = None
timestamps = None
full_start_date = None
full_end_date = None
all_lines = []  # Store all lines from each beginning
trade_returns = []
cumulative_returns = []
line_endpoints = []

def fetch_data():
    """Fetch BTC 1h data for last 30 days from Binance"""
    global prices, timestamps, full_start_date, full_end_date
    
    try:
        # Calculate timestamps for last 30 days
        full_end_date = datetime.now()
        full_start_date = full_end_date - timedelta(days=30)
        
        end_time = int(full_end_date.timestamp() * 1000)
        start_time = int(full_start_date.timestamp() * 1000)
        
        # Binance API URL for klines/candlestick data
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&startTime={start_time}&endTime={end_time}&limit=1000"
        
        # Fetch data
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        
        # Extract timestamps and closing prices
        timestamps = np.array([int(item[0]) for item in data]).reshape(-1, 1)
        prices = np.array([float(item[4]) for item in data])  # Closing price
        
        print(f"✅ Fetched {len(prices)} hours of BTC data")
        print(f"   Period: {full_start_date.strftime('%Y-%m-%d')} to {full_end_date.strftime('%Y-%m-%d')}")
        return True
        
    except Exception as e:
        print(f"⚠️ Error fetching data: {e}")
        # Generate sample data for testing if API fails
        generate_sample_data()
        return False

def generate_sample_data():
    """Generate sample data for testing when API is unavailable"""
    global prices, timestamps, full_start_date, full_end_date
    
    np.random.seed(42)
    n_points = 720  # 30 days * 24 hours
    
    # Generate timestamps
    full_end_date = datetime.now()
    full_start_date = full_end_date - timedelta(days=30)
    base_time = full_start_date.timestamp() * 1000
    timestamps = np.array([base_time + i * 3600000 for i in range(n_points)]).reshape(-1, 1)
    
    # Generate synthetic BTC price with trend, cycles, and noise
    t = np.linspace(0, 4*np.pi, n_points)
    trend = np.linspace(40000, 45000, n_points)
    cycle = 2000 * np.sin(t)
    noise = np.random.normal(0, 300, n_points)
    prices = trend + cycle + noise
    
    print(f"⚠️ Generated {n_points} hours of sample data")
    print(f"   Period: {full_start_date.strftime('%Y-%m-%d')} to {full_end_date.strftime('%Y-%m-%d')}")

def calculate_line(start_idx: int, end_idx: int):
    """Calculate OLS line from start_idx to end_idx"""
    if end_idx - start_idx < 10:  # Minimum window size
        return None
    
    X = timestamps[start_idx:end_idx]
    y = prices[start_idx:end_idx]
    
    # Normalize timestamps
    X_mean = X.mean()
    X_std = X.std()
    X_normalized = (X - X_mean) / X_std
    
    # Fit OLS
    model = LinearRegression()
    model.fit(X_normalized, y)
    
    # Get slope (denormalize) - price change per hour
    slope = model.coef_[0] / X_std * 3600000
    
    # Calculate error normalized by window^2
    y_pred = model.predict(X_normalized)
    window_size = end_idx - start_idx
    error = np.sum(np.abs(y - y_pred)) / (window_size * window_size)  # Divide by window^2
    
    # Predict for all points
    all_X = timestamps[start_idx:end_idx].flatten()
    all_X_norm = (all_X - X_mean) / X_std
    all_y_pred = model.predict(all_X_norm.reshape(-1, 1))
    
    return {
        'start_idx': start_idx,
        'end_idx': end_idx,
        'slope': slope,
        'error': error,
        'model': model,
        'norm_params': (X_mean, X_std),
        'y_pred': all_y_pred,
        'window_size': window_size
    }

def analyze_all_lines():
    """Calculate lines from each possible beginning to each possible end"""
    global all_lines, trade_returns, cumulative_returns, line_endpoints
    
    print("\n" + "=" * 60)
    print("📊 CALCULATING lines from each beginning...")
    print("=" * 60)
    
    all_lines = []
    total_points = len(prices)
    
    # For each possible start point
    for start_idx in range(0, total_points - 10):  # Need at least 10 points
        # For each possible end point after start
        max_end = min(start_idx + 100, total_points)  # Max 100-hour window
        best_line = None
        best_error = float('inf')
        
        for end_idx in range(start_idx + 10, max_end + 1):
            line = calculate_line(start_idx, end_idx)
            if line and line['error'] < best_error:
                best_error = line['error']
                best_line = line
        
        if best_line:
            all_lines.append(best_line)
        
        if start_idx % 100 == 0:
            print(f"   Processed start {start_idx}/{total_points-10}...")
    
    print(f"\n✅ Calculated {len(all_lines)} optimal lines")
    
    # Now calculate trading returns using these lines
    print("\n📈 Calculating trading returns...")
    trade_returns = []
    line_endpoints = []
    
    min_start = 10  # Need at least 10 points before we can have a line
    
    for current_idx in range(min_start, total_points - 1):
        # Find the best line that ends at or before current_idx
        best_line_at_point = None
        best_error_at_point = float('inf')
        
        for line in all_lines:
            if line['end_idx'] <= current_idx:
                if line['error'] < best_error_at_point:
                    best_error_at_point = line['error']
                    best_line_at_point = line
        
        if best_line_at_point:
            # Use this line's slope for trading decision
            slope = best_line_at_point['slope']
            
            # Calculate return for next candle
            next_return = (prices[current_idx + 1] - prices[current_idx]) / prices[current_idx] * 100
            
            # Trade based on slope direction
            if slope > 0:
                trade_return = next_return
            else:
                trade_return = -next_return
            
            trade_returns.append(trade_return)
            line_endpoints.append(current_idx)
    
    # Calculate cumulative returns
    returns_array = np.array(trade_returns) / 100
    cumulative_returns = np.cumprod(1 + returns_array) - 1
    
    print(f"   Generated {len(trade_returns)} trades")
    print(f"   Return range: {min(trade_returns):+.2f}% to {max(trade_returns):+.2f}%")
    print(f"   Final cumulative return: {cumulative_returns[-1]*100:+.2f}%")

def create_plot():
    """Create plot showing lines and trading returns"""
    plt.figure(figsize=(14, 12))
    
    # Create 3x1 subplot grid
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    # Top plot: Price with all optimal lines
    ax_lines = plt.subplot(gs[0])
    
    # Plot price
    ax_lines.plot(range(len(prices)), prices, 'b-', alpha=0.7, label='BTC Price', linewidth=1.5)
    
    # Plot a sample of lines (show every 10th line to avoid overcrowding)
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_lines)))
    for i, line in enumerate(all_lines[::10]):  # Plot every 10th line
        start = line['start_idx']
        end = line['end_idx']
        x_range = range(start, end)
        color = 'green' if line['slope'] > 0 else 'red'
        alpha = 0.1 + (line['window_size'] / 100) * 0.2  # Longer windows more opaque
        ax_lines.plot(x_range, line['y_pred'], color=color, linewidth=1, alpha=alpha)
    
    ax_lines.set_title(f'BTC Price with Optimal Lines from Each Beginning (n={len(all_lines)})', fontsize=12)
    ax_lines.set_ylabel('Price (USDT)')
    ax_lines.grid(True, alpha=0.2)
    ax_lines.legend(['BTC Price'], loc='upper left')
    
    # Middle plot: Trade returns
    ax_returns = plt.subplot(gs[1])
    
    # Plot individual trade returns as bars
    x = line_endpoints
    colors = ['green' if r > 0 else 'red' for r in trade_returns]
    ax_returns.bar(x, trade_returns, width=0.8, color=colors, alpha=0.5, label='Trade Returns')
    
    # Add zero line
    ax_returns.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax_returns.set_ylabel('Return per Trade (%)')
    ax_returns.grid(True, alpha=0.2)
    ax_returns.set_title('Individual Trade Returns')
    
    # Bottom plot: Cumulative returns
    ax_cumul = plt.subplot(gs[2])
    
    ax_cumul.plot(line_endpoints, cumulative_returns * 100, 'b-', linewidth=2, label='Strategy')
    
    # Add buy and hold for comparison
    first_price = prices[min(line_endpoints)] if line_endpoints else prices[0]
    buy_hold = (prices[line_endpoints] - first_price) / first_price * 100 if line_endpoints else []
    if len(buy_hold) == len(line_endpoints):
        ax_cumul.plot(line_endpoints, buy_hold, 'gray', linewidth=1.5, alpha=0.7, label='Buy & Hold')
    
    ax_cumul.set_xlabel('Hour')
    ax_cumul.set_ylabel('Cumulative Return (%)')
    ax_cumul.grid(True, alpha=0.2)
    ax_cumul.legend(loc='upper left')
    
    # Add title with stats
    total_return = cumulative_returns[-1] * 100 if cumulative_returns.size > 0 else 0
    win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100 if trade_returns else 0
    ax_cumul.set_title(f'Cumulative Returns: {total_return:+.2f}% | Win Rate: {win_rate:.1f}%', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Encode as base64
    return base64.b64encode(buf.read()).decode('utf-8')

class BTCRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Create plot with pre-calculated data
            image_base64 = create_plot()
            
            # Calculate stats
            if cumulative_returns.size > 0:
                total_return = cumulative_returns[-1] * 100
            else:
                total_return = 0
                
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100 if trade_returns else 0
            avg_return = np.mean(trade_returns) if trade_returns else 0
            max_win = max(trade_returns) if trade_returns else 0
            max_loss = min(trade_returns) if trade_returns else 0
            
            # Calculate average window size
            avg_window = np.mean([line['window_size'] for line in all_lines]) if all_lines else 0
            
            # HTML page
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>BTC Multi-Beginning Lines Analysis</title>
                <style>
                    body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    h1 {{ margin: 0 0 10px 0; color: #333; font-size: 20px; }}
                    .plot {{ width: 100%; height: auto; }}
                    .stats {{ display: flex; gap: 15px; flex-wrap: wrap; margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 4px; }}
                    .stat {{ padding: 5px 12px; border-radius: 16px; font-size: 13px; }}
                    .stat.positive {{ background-color: #d4edda; color: #155724; }}
                    .stat.negative {{ background-color: #f8d7da; color: #721c24; }}
                    .stat.neutral {{ background-color: #e2e3e5; color: #383d41; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📈 BTC Multi-Beginning Lines Analysis (Error/Window²)</h1>
                    <div class="stats">
                        <span class="stat neutral">📊 {len(all_lines)} lines</span>
                        <span class="stat neutral">📏 Avg window: {avg_window:.1f}h</span>
                        <span class="stat neutral">🎯 {len(trade_returns)} trades</span>
                        <span class="stat positive">📈 Win rate: {win_rate:.1f}%</span>
                        <span class="stat {"positive" if total_return > 0 else "negative"}">💰 Total: {total_return:+.2f}%</span>
                        <span class="stat positive">🏆 Max win: {max_win:+.2f}%</span>
                        <span class="stat negative">📉 Max loss: {max_loss:+.2f}%</span>
                    </div>
                    <img class="plot" src="data:image/png;base64,{image_base64}" alt="BTC Multi-Beginning Analysis">
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()

def main():
    """Main function - fetches data and starts server"""
    print("=" * 60)
    print("🚀 BTC Multi-Beginning Lines Analysis Server")
    print("=" * 60)
    print("   Using error/Window² normalization")
    
    # Fetch data on startup
    print("\n📡 Fetching BTC data...")
    fetch_data()
    
    # Run analysis on startup
    analyze_all_lines()
    
    # Start server
    PORT = 8080
    handler = BTCRequestHandler
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"\n🌐 Server running at http://localhost:{PORT}")
        print("   Lines from each beginning, normalized by window²")
        print("   Green lines = positive slope, Red lines = negative slope")
        print("   Press Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped. Goodbye!")

if __name__ == "__main__":
    main()