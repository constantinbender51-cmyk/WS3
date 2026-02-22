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
cascade_lines = []  # Store cascade of lines
trade_returns = []
cumulative_returns = []
trade_points = []

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

def calculate_best_window(start_idx: int, max_end: int) -> dict:
    """Find the best window starting at start_idx that minimizes error/window²"""
    best_error = float('inf')
    best_line = None
    
    # Try windows from 10 to 100 hours
    for window_size in range(10, min(101, max_end - start_idx + 1)):
        end_idx = start_idx + window_size
        
        X = timestamps[start_idx:end_idx]
        y = prices[start_idx:end_idx]
        
        # Normalize timestamps
        X_mean = X.mean()
        X_std = X.std()
        X_normalized = (X - X_mean) / X_std
        
        # Fit OLS
        model = LinearRegression()
        model.fit(X_normalized, y)
        
        # Get slope (denormalize)
        slope = model.coef_[0] / X_std * 3600000
        
        # Calculate error normalized by window²
        y_pred = model.predict(X_normalized)
        error = np.sum(np.abs(y - y_pred)) / (window_size * window_size)
        
        if error < best_error:
            best_error = error
            best_line = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'window_size': window_size,
                'slope': slope,
                'error': error,
                'model': model,
                'norm_params': (X_mean, X_std)
            }
    
    return best_line

def build_cascade():
    """Build cascade of lines where each starts at previous window's beginning"""
    global cascade_lines, trade_returns, cumulative_returns, trade_points
    
    print("\n" + "=" * 60)
    print("📊 BUILDING cascade of lines...")
    print("=" * 60)
    
    cascade_lines = []
    trade_returns = []
    trade_points = []
    
    total_points = len(prices)
    
    # Start from the earliest possible point
    current_start = 0
    iteration = 1
    
    while current_start + 10 < total_points:
        # Find best window from current start
        best_line = calculate_best_window(current_start, total_points - 1)
        
        if not best_line:
            break
            
        cascade_lines.append(best_line)
        
        end_date = full_start_date + timedelta(hours=best_line['end_idx'])
        print(f"   Line {iteration}: Start={current_start}, End={best_line['end_idx']}, "
              f"Window={best_line['window_size']}h, Slope=${best_line['slope']:+.2f}/h")
        
        # Calculate trading returns using this line
        # Trade from the end of this window to the start of next window
        for trade_idx in range(best_line['end_idx'], min(best_line['end_idx'] + 10, total_points - 1)):
            if trade_idx + 1 < total_points:
                # Calculate next candle return
                next_return = (prices[trade_idx + 1] - prices[trade_idx]) / prices[trade_idx] * 100
                
                # Trade based on slope direction
                if best_line['slope'] > 0:
                    trade_return = next_return
                else:
                    trade_return = -next_return
                
                trade_returns.append(trade_return)
                trade_points.append(trade_idx)
        
        # Next line starts at this line's beginning
        current_start = best_line['start_idx']
        iteration += 1
        
        # Safety break to prevent infinite loop
        if iteration > 50:
            break
    
    # Calculate cumulative returns
    if trade_returns:
        returns_array = np.array(trade_returns) / 100
        cumulative_returns = np.cumprod(1 + returns_array) - 1
    
    print(f"\n✅ Built {len(cascade_lines)} cascade lines")
    print(f"   Generated {len(trade_returns)} trades")
    if trade_returns:
        print(f"   Return range: {min(trade_returns):+.2f}% to {max(trade_returns):+.2f}%")
        print(f"   Final cumulative return: {cumulative_returns[-1]*100:+.2f}%")

def create_plot():
    """Create plot showing cascade lines and trading returns"""
    plt.figure(figsize=(14, 12))
    
    # Create 3x1 subplot grid
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    # Top plot: Price with cascade lines
    ax_lines = plt.subplot(gs[0])
    
    # Plot price
    ax_lines.plot(range(len(prices)), prices, 'b-', alpha=0.7, label='BTC Price', linewidth=1.5)
    
    # Plot each cascade line
    colors = plt.cm.rainbow(np.linspace(0, 1, len(cascade_lines)))
    for i, line in enumerate(cascade_lines):
        start = line['start_idx']
        end = line['end_idx']
        
        # Generate predictions for this window
        X = timestamps[start:end].flatten()
        X_mean, X_std = line['norm_params']
        X_norm = (X - X_mean) / X_std
        y_pred = line['model'].predict(X_norm.reshape(-1, 1))
        
        # Plot the line
        color = 'green' if line['slope'] > 0 else 'red'
        ax_lines.plot(range(start, end), y_pred, 
                     color=color, linewidth=2.5, alpha=0.8)
        
        # Mark the start point
        ax_lines.plot(start, prices[start], 'o', color=color, markersize=6)
        
        # Add line number at start
        ax_lines.text(start, prices[start] - 200, f'{i+1}', 
                     fontsize=8, fontweight='bold', ha='center',
                     bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
    
    ax_lines.set_title(f'BTC Price with Cascade Lines (Each starts at previous beginning)', fontsize=12)
    ax_lines.set_ylabel('Price (USDT)')
    ax_lines.grid(True, alpha=0.2)
    ax_lines.legend(['BTC Price'], loc='upper left')
    
    # Middle plot: Trade returns
    ax_returns = plt.subplot(gs[1])
    
    if trade_points:
        # Plot individual trade returns as bars
        colors = ['green' if r > 0 else 'red' for r in trade_returns]
        ax_returns.bar(trade_points, trade_returns, width=0.8, color=colors, alpha=0.5, label='Trade Returns')
        
        # Add vertical lines at cascade line boundaries
        for line in cascade_lines:
            ax_returns.axvline(x=line['end_idx'], color='purple', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add zero line
    ax_returns.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax_returns.set_ylabel('Return per Trade (%)')
    ax_returns.grid(True, alpha=0.2)
    ax_returns.set_title('Individual Trade Returns (Purple dashes = new line starts)')
    
    # Bottom plot: Cumulative returns
    ax_cumul = plt.subplot(gs[2])
    
    if trade_points and cumulative_returns.size > 0:
        ax_cumul.plot(trade_points, cumulative_returns * 100, 'b-', linewidth=2, label='Strategy')
        
        # Add buy and hold for comparison (from first trade point)
        first_trade_idx = trade_points[0]
        first_price = prices[first_trade_idx]
        buy_hold = [(prices[idx] - first_price) / first_price * 100 for idx in trade_points]
        ax_cumul.plot(trade_points, buy_hold, 'gray', linewidth=1.5, alpha=0.7, label='Buy & Hold')
        
        # Add vertical lines at cascade line boundaries
        for line in cascade_lines:
            if line['end_idx'] <= trade_points[-1]:
                ax_cumul.axvline(x=line['end_idx'], color='purple', linestyle='--', alpha=0.3, linewidth=1)
    
    ax_cumul.set_xlabel('Hour')
    ax_cumul.set_ylabel('Cumulative Return (%)')
    ax_cumul.grid(True, alpha=0.2)
    ax_cumul.legend(loc='upper left')
    
    # Add title with stats
    if cumulative_returns.size > 0:
        total_return = cumulative_returns[-1] * 100
    else:
        total_return = 0
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
            avg_window = np.mean([line['window_size'] for line in cascade_lines]) if cascade_lines else 0
            
            # HTML page
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>BTC Cascade Lines Analysis</title>
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
                    <h1>📈 BTC Cascade Lines Analysis (Each line starts at previous beginning)</h1>
                    <div class="stats">
                        <span class="stat neutral">📊 {len(cascade_lines)} lines</span>
                        <span class="stat neutral">📏 Avg window: {avg_window:.1f}h</span>
                        <span class="stat neutral">🎯 {len(trade_returns)} trades</span>
                        <span class="stat positive">📈 Win rate: {win_rate:.1f}%</span>
                        <span class="stat {"positive" if total_return > 0 else "negative"}">💰 Total: {total_return:+.2f}%</span>
                        <span class="stat positive">🏆 Max win: {max_win:+.2f}%</span>
                        <span class="stat negative">📉 Max loss: {max_loss:+.2f}%</span>
                    </div>
                    <img class="plot" src="data:image/png;base64,{image_base64}" alt="BTC Cascade Analysis">
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
    print("🚀 BTC Cascade Lines Analysis Server")
    print("=" * 60)
    print("   Each line starts at previous window's beginning")
    print("   Error normalized by window²")
    
    # Fetch data on startup
    print("\n📡 Fetching BTC data...")
    fetch_data()
    
    # Build cascade on startup
    build_cascade()
    
    # Start server
    PORT = 8080
    handler = BTCRequestHandler
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"\n🌐 Server running at http://localhost:{PORT}")
        print("   Green lines = positive slope, Red lines = negative slope")
        print("   Numbers mark start of each line")
        print("   Purple dashes = new line starts")
        print("   Press Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped. Goodbye!")

if __name__ == "__main__":
    main()