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
slope_history = []
lookback_history = []
endpoint_indices = []
trade_returns = []  # Store returns from trading based on slope direction
cumulative_returns = []  # Store cumulative returns

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

def ols(prices_seg: np.ndarray, timestamps_seg: np.ndarray, lookback: int) -> Tuple[float, float]:
    """Perform OLS regression on the last 'lookback' data points
    Returns: slope, avg_error"""
    if lookback > len(prices_seg):
        lookback = len(prices_seg)
    
    # Get last 'lookback' data points
    X = timestamps_seg[-lookback:]
    y = prices_seg[-lookback:]
    
    # Normalize timestamps to avoid numerical issues
    X_mean = X.mean()
    X_std = X.std()
    X_normalized = (X - X_mean) / X_std
    
    # Fit OLS
    model = LinearRegression()
    model.fit(X_normalized, y)
    
    # Get slope (denormalize) - price change per hour
    slope = model.coef_[0] / X_std * 3600000
    
    # Calculate average error
    y_pred = model.predict(X_normalized)
    avg_error = np.sum(np.abs(y - y_pred)) / lookback
    
    return slope, avg_error

def find_optimal_lookback(prices_seg: np.ndarray, timestamps_seg: np.ndarray) -> Tuple[int, float]:
    """Find the optimal lookback window that minimizes average error
    Returns: lookback, slope"""
    best_avg_error = float('inf')
    best_lookback = 10
    best_slope = 0
    
    max_lookback = min(100, len(prices_seg))
    
    for lookback in range(10, max_lookback + 1):
        slope, avg_error = ols(prices_seg, timestamps_seg, lookback)
        
        if avg_error < best_avg_error:
            best_avg_error = avg_error
            best_lookback = lookback
            best_slope = slope
    
    return best_lookback, best_slope

def analyze_all_points():
    """Analyze every possible endpoint and calculate trading returns"""
    global slope_history, lookback_history, endpoint_indices, trade_returns, cumulative_returns, prices
    
    print("\n" + "=" * 60)
    print("📊 ANALYZING all endpoints and calculating returns...")
    print("=" * 60)
    
    slope_history = []
    lookback_history = []
    endpoint_indices = []
    trade_returns = []
    
    min_points = 110  # Need at least 100 lookback + 10 minimum
    total_points = len(prices)
    
    # We need to stop one short because we need next candle's return
    for endpoint in range(min_points, total_points):
        # Get data up to current endpoint
        test_prices = prices[:endpoint]
        test_timestamps = timestamps[:endpoint]
        
        # Find optimal lookback for this endpoint
        lookback, slope = find_optimal_lookback(test_prices, test_timestamps)
        
        slope_history.append(slope)
        lookback_history.append(lookback)
        endpoint_indices.append(endpoint)
        
        # Calculate return for next candle based on slope direction
        next_candle_return = (prices[endpoint] - prices[endpoint-1]) / prices[endpoint-1] * 100
        
        # If slope is positive (uptrend), we go long -> return is actual return
        # If slope is negative (downtrend), we go short -> return is negative of actual return
        if slope > 0:
            trade_return = next_candle_return
        else:
            trade_return = -next_candle_return
        
        trade_returns.append(trade_return)
        
        if endpoint % 100 == 0:
            print(f"   Analyzed {endpoint}/{total_points-1}...")
    
    # Calculate cumulative returns
    returns_array = np.array(trade_returns) / 100  # Convert to decimal
    cumulative_returns = np.cumprod(1 + returns_array) - 1
    
    # Calculate buy and hold returns for comparison (from first trade point)
    first_price = prices[min_points-1]
    buy_hold_returns = (prices[min_points:] - first_price) / first_price
    
    print(f"\n✅ Analyzed {len(slope_history)} endpoints")
    print(f"   Slope range: ${min(slope_history):+.2f}/h to ${max(slope_history):+.2f}/h")
    print(f"   Return range: {min(trade_returns):+.2f}% to {max(trade_returns):+.2f}%")
    print(f"   Final cumulative return: {cumulative_returns[-1]*100:+.2f}%")
    print(f"   Buy & Hold return: {buy_hold_returns[-1]*100:+.2f}%")

def create_plot():
    """Create plot with price and trading returns"""
    plt.figure(figsize=(14, 10))
    
    # Create 2x1 subplot grid
    gs = plt.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3)
    
    # Top plot: Price with slope-colored regions
    ax_price = plt.subplot(gs[0])
    
    # Plot price
    ax_price.plot(range(len(prices)), prices, 'b-', alpha=0.7, label='BTC Price', linewidth=1.5)
    
    # Color background based on slope sign
    for i, endpoint in enumerate(endpoint_indices):
        if slope_history[i] > 0:
            color = 'green'
        elif slope_history[i] < 0:
            color = 'red'
        else:
            color = 'gray'
        ax_price.axvspan(endpoint-1, endpoint, facecolor=color, alpha=0.2)
    
    # Add slope line above price (scaled)
    slope_scaled = np.array(slope_history) * 5
    slope_offset = np.max(prices) * 1.05
    
    for i in range(len(endpoint_indices)-1):
        if slope_history[i] > 0:
            color = 'green'
        else:
            color = 'red'
        ax_price.plot([endpoint_indices[i], endpoint_indices[i+1]], 
                     [slope_offset + slope_scaled[i], slope_offset + slope_scaled[i+1]], 
                     color=color, linewidth=2, alpha=0.8)
    
    ax_price.axhline(y=slope_offset, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax_price.text(5, slope_offset, 'Slope', fontsize=8, va='bottom')
    
    ax_price.set_title('BTC Price with Slope Direction (Green=Long, Red=Short)', fontsize=12)
    ax_price.set_ylabel('Price (USDT)')
    ax_price.grid(True, alpha=0.2)
    ax_price.legend(loc='upper left')
    
    # Bottom plot: Returns
    ax_returns = plt.subplot(gs[1])
    
    # Plot individual trade returns as bars
    x = endpoint_indices
    colors = []
    for r in trade_returns:
        if r > 0:
            colors.append('green')
        else:
            colors.append('red')
    
    ax_returns.bar(x, trade_returns, width=0.8, color=colors, alpha=0.5, label='Trade Returns')
    
    # Plot cumulative returns
    ax_cumul = ax_returns.twinx()
    ax_cumul.plot(x, cumulative_returns * 100, 'b-', linewidth=2, label='Cumulative Return')
    
    # Add zero line
    ax_returns.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax_returns.set_xlabel('Hour')
    ax_returns.set_ylabel('Return per Trade (%)', color='gray')
    ax_returns.tick_params(axis='y', labelcolor='gray')
    ax_returns.grid(True, alpha=0.2)
    
    ax_cumul.set_ylabel('Cumulative Return (%)', color='blue')
    ax_cumul.tick_params(axis='y', labelcolor='blue')
    
    # Add title with stats
    total_return = cumulative_returns[-1] * 100
    win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100
    ax_returns.set_title(f'Trading Returns: {total_return:+.2f}% cumulative | Win Rate: {win_rate:.1f}%', fontsize=12)
    
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
            if cumulative_returns.size > 0:  # Check if array has elements
                total_return = cumulative_returns[-1] * 100
            else:
                total_return = 0
                
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100 if trade_returns else 0
            avg_return = np.mean(trade_returns) if trade_returns else 0
            max_win = max(trade_returns) if trade_returns else 0
            max_loss = min(trade_returns) if trade_returns else 0
            
            # Minimal HTML page
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>BTC Slope-Based Trading</title>
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
                    <h1>📈 BTC Slope-Based Trading Strategy</h1>
                    <div class="stats">
                        <span class="stat neutral">📊 {len(trade_returns)} trades</span>
                        <span class="stat positive">📈 Win rate: {win_rate:.1f}%</span>
                        <span class="stat {"positive" if total_return > 0 else "negative"}">💰 Total: {total_return:+.2f}%</span>
                        <span class="stat positive">🏆 Max win: {max_win:+.2f}%</span>
                        <span class="stat negative">📉 Max loss: {max_loss:+.2f}%</span>
                        <span class="stat neutral">📏 Avg: {avg_return:+.2f}%</span>
                    </div>
                    <img class="plot" src="data:image/png;base64,{image_base64}" alt="BTC Trading Strategy">
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
    print("🚀 BTC Slope-Based Trading Strategy Server")
    print("=" * 60)
    
    # Fetch data on startup
    print("\n📡 Fetching BTC data...")
    fetch_data()
    
    # Run analysis on startup
    analyze_all_points()
    
    # Start server
    PORT = 8080
    handler = BTCRequestHandler
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"\n🌐 Server running at http://localhost:{PORT}")
        print("   Strategy: Long when slope > 0, Short when slope < 0")
        print("   Returns calculated on next candle")
        print("   Press Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped. Goodbye!")

if __name__ == "__main__":
    main()