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
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Global variables to store pre-calculated data
prices = None
timestamps = None
full_start_date = None
full_end_date = None
merged_lines = []  # Store merged OLS lines by sign
slope_history = []
lookback_history = []
endpoint_indices = []

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

def ols(prices_seg: np.ndarray, timestamps_seg: np.ndarray, lookback: int) -> Tuple[float, np.ndarray, np.ndarray, object]:
    """Perform OLS regression on the last 'lookback' data points"""
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
    
    # Predict for the window
    y_pred = model.predict(X_normalized)
    
    return slope, X.flatten(), y_pred, model, (X_mean, X_std)

def find_optimal_lookback(prices_seg: np.ndarray, timestamps_seg: np.ndarray) -> Tuple[int, float, np.ndarray, np.ndarray, object]:
    """Find the optimal lookback window that minimizes average error"""
    best_avg_error = float('inf')
    best_lookback = 10
    best_slope = 0
    best_X = None
    best_y_pred = None
    best_model = None
    best_norm_params = None
    
    max_lookback = min(100, len(prices_seg))
    
    for lookback in range(10, max_lookback + 1):
        slope, X, y_pred, model, norm_params = ols(prices_seg, timestamps_seg, lookback)
        
        # Calculate average error
        y = prices_seg[-lookback:]
        y_pred_local = model.predict((timestamps_seg[-lookback:] - norm_params[0]) / norm_params[1])
        avg_error = np.sum(np.abs(y - y_pred_local)) / lookback
        
        if avg_error < best_avg_error:
            best_avg_error = avg_error
            best_lookback = lookback
            best_slope = slope
            best_X = X
            best_y_pred = y_pred
            best_model = model
            best_norm_params = norm_params
    
    return best_lookback, best_slope, best_X, best_y_pred, best_model, best_norm_params

def analyze_all_points():
    """Analyze every possible endpoint and merge same-sign lines"""
    global slope_history, lookback_history, endpoint_indices, merged_lines, prices, timestamps
    
    print("\n" + "=" * 60)
    print("📊 ANALYZING all endpoints...")
    print("=" * 60)
    
    slope_history = []
    lookback_history = []
    endpoint_indices = []
    all_results = []  # Store all results with their endpoints
    
    min_points = 110  # Need at least 100 lookback + 10 minimum
    total_points = len(prices)
    
    for endpoint in range(min_points, total_points + 1):
        # Get data up to current endpoint
        test_prices = prices[:endpoint]
        test_timestamps = timestamps[:endpoint]
        
        # Find optimal lookback for this endpoint
        lookback, slope, X, y_pred, model, norm_params = find_optimal_lookback(test_prices, test_timestamps)
        
        slope_history.append(slope)
        lookback_history.append(lookback)
        endpoint_indices.append(endpoint)
        
        # Store full result
        all_results.append({
            'endpoint': endpoint,
            'lookback': lookback,
            'slope': slope,
            'X': X,
            'y_pred': y_pred,
            'model': model,
            'norm_params': norm_params
        })
        
        if endpoint % 100 == 0:
            print(f"   Analyzed {endpoint}/{total_points}...")
    
    # Merge consecutive same-sign results
    merged_lines = []
    current_group = []
    current_sign = None
    
    for result in all_results:
        sign = np.sign(result['slope'])
        
        if current_sign is None:
            # First result
            current_sign = sign
            current_group = [result]
        elif sign == current_sign:
            # Same sign, add to current group
            current_group.append(result)
        else:
            # Sign changed, merge the previous group
            if current_group:
                merged = merge_line_group(current_group)
                merged_lines.append(merged)
            
            # Start new group
            current_sign = sign
            current_group = [result]
    
    # Don't forget the last group
    if current_group:
        merged = merge_line_group(current_group)
        merged_lines.append(merged)
    
    print(f"\n✅ Analyzed {len(slope_history)} endpoints")
    print(f"   Merged into {len(merged_lines)} lines by sign")
    print(f"   Slope range: ${min(slope_history):+.2f}/h to ${max(slope_history):+.2f}/h")
    
    # Print merged line info
    for i, line in enumerate(merged_lines):
        sign_str = "🟢 UPTREND" if line['slope'] > 0 else "🔴 DOWNTREND"
        print(f"   Line {i+1}: {sign_str} | Period: {line['start_idx']}-{line['end_idx']} | Lookback: {line['avg_lookback']:.0f}h | Slope: ${line['slope']:+.2f}/h")

def merge_line_group(group: List[Dict]) -> Dict:
    """Merge a group of same-sign OLS results into a single representative line"""
    if not group:
        return None
    
    # Use the middle result as representative
    mid_idx = len(group) // 2
    representative = group[mid_idx]
    
    # Calculate average lookback and slope
    avg_lookback = np.mean([r['lookback'] for r in group])
    avg_slope = np.mean([r['slope'] for r in group])
    
    # Get the full range this group covers
    start_idx = group[0]['endpoint'] - group[0]['lookback']
    end_idx = group[-1]['endpoint']
    
    return {
        'start_idx': start_idx,
        'end_idx': end_idx,
        'lookback': representative['lookback'],
        'avg_lookback': avg_lookback,
        'slope': avg_slope,
        'model': representative['model'],
        'norm_params': representative['norm_params'],
        'representative_endpoint': representative['endpoint'],
        'num_merged': len(group)
    }

def create_plot():
    """Create plot with price, slope line, and merged OLS lines by sign"""
    plt.figure(figsize=(14, 8))
    
    # Plot price
    plt.plot(range(len(prices)), prices, 'b-', alpha=0.7, label='BTC Price', linewidth=1.5)
    
    # Plot merged OLS lines with different colors for each sign
    colors = {'uptrend': 'green', 'downtrend': 'red'}
    
    for i, line in enumerate(merged_lines):
        start_idx = line['start_idx']
        end_idx = line['end_idx']
        
        # Get the data for this merged line's representative window
        rep_endpoint = line['representative_endpoint']
        lookback = line['lookback']
        rep_start = rep_endpoint - lookback
        
        # Use the representative model to generate the line
        model = line['model']
        X_mean, X_std = line['norm_params']
        
        # Generate predictions for the entire merged range
        # We'll extend the representative line across the merged range
        X_range = timestamps[start_idx:end_idx].flatten()
        X_norm = (X_range - X_mean) / X_std
        y_pred = model.predict(X_norm.reshape(-1, 1))
        
        # Determine color based on slope sign
        color = colors['uptrend'] if line['slope'] > 0 else colors['downtrend']
        
        # Plot the merged line
        plt.plot(range(start_idx, end_idx), y_pred, 
                color=color, linewidth=2.5, alpha=0.8,
                label=f"{'Uptrend' if line['slope'] > 0 else 'Downtrend'} (n={line['num_merged']})" if i < 2 else "")
        
        # Add small markers at the endpoints
        plt.plot(start_idx, prices[start_idx] if start_idx < len(prices) else prices[-1], 
                'o', color=color, markersize=4, alpha=0.5)
        plt.plot(end_idx-1, prices[end_idx-1] if end_idx-1 < len(prices) else prices[-1], 
                's', color=color, markersize=4, alpha=0.5)
    
    # Create slope line (above 0 green, below 0 red)
    # Scale slopes to be visible but not overwhelm the price
    slope_scaled = np.array(slope_history) * 8  # Scale factor to make slope visible
    slope_offset = np.max(prices) * 1.1  # Position slope line above price
    
    # Plot slope line with color based on sign
    slope_line_x = endpoint_indices
    slope_line_y = slope_offset + slope_scaled
    
    for i in range(len(slope_line_x) - 1):
        color = 'green' if slope_history[i] > 0 else 'red'
        plt.plot([slope_line_x[i], slope_line_x[i+1]], 
                [slope_line_y[i], slope_line_y[i+1]], 
                color=color, linewidth=2, alpha=0.9)
    
    # Add zero line for slope reference
    plt.axhline(y=slope_offset, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Add slope value labels at trend change points
    last_sign = None
    for i in range(0, len(slope_line_x), max(1, len(slope_line_x)//15)):
        current_sign = np.sign(slope_history[i])
        if current_sign != last_sign:
            plt.text(slope_line_x[i], slope_line_y[i] + 80, 
                    f'{slope_history[i]:+.1f}', fontsize=8, ha='center',
                    color='green' if slope_history[i] > 0 else 'red',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))
            last_sign = current_sign
    
    # Minimal styling
    plt.title('BTC Price with Merged Trend Lines (Same Sign = Same Color)', fontsize=14)
    plt.xlabel('Hours from Start')
    plt.ylabel('Price (USDT)')
    plt.grid(True, alpha=0.2)
    plt.legend(loc='upper left', fontsize=8)
    
    # Add info about merged lines
    uptrend_count = sum(1 for line in merged_lines if line['slope'] > 0)
    downtrend_count = sum(1 for line in merged_lines if line['slope'] < 0)
    
    info_text = f'🟢 {uptrend_count} uptrend segments | 🔴 {downtrend_count} downtrend segments | Total: {len(merged_lines)} lines'
    plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
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
            
            # Minimal HTML page
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>BTC Merged Trend Lines</title>
                <style>
                    body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    h1 {{ margin: 0 0 10px 0; color: #333; font-size: 20px; }}
                    .plot {{ width: 100%; height: auto; }}
                    .info {{ margin: 10px 0; padding: 8px; background-color: #f8f9fa; border-radius: 4px; font-size: 12px; color: #666; display: flex; gap: 20px; }}
                    .stat {{ background-color: #e9ecef; padding: 2px 8px; border-radius: 12px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📈 BTC Merged Trend Lines</h1>
                    <div class="info">
                        <span class="stat">📊 {len(slope_history)} points</span>
                        <span class="stat">🟢 {sum(1 for s in slope_history if s > 0)} positive</span>
                        <span class="stat">🔴 {sum(1 for s in slope_history if s < 0)} negative</span>
                        <span class="stat">📏 {len(merged_lines)} merged lines</span>
                    </div>
                    <img class="plot" src="data:image/png;base64,{image_base64}" alt="BTC Merged Trend Lines">
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
    print("🚀 BTC Merged Trend Lines Server")
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
        print("   Page shows: price + slope line + merged trend lines by sign")
        print("   Press Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped. Goodbye!")

if __name__ == "__main__":
    main()