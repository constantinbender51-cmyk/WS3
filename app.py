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
K = 1.8  # Window exponent

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
        # Generate simple sample data
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
    
    # Simple sine wave with trend
    x = np.linspace(0, 4*np.pi, n_points)
    prices = 40000 + 2000 * np.sin(x) + 3000 * x/len(x) + np.random.normal(0, 200, n_points)
    
    print(f"⚠️ Generated {n_points} hours of sample data")

def find_best_line(data_prices, data_timestamps, start_idx):
    """Find the best line on the given data segment using error/window^K"""
    best_error = float('inf')
    best_line = None
    best_window = 0
    best_slope = 0
    best_end = 0
    
    n_points = len(data_prices)
    
    print(f"\n   Analyzing segment from {start_idx}h ({len(data_prices)} points):")
    
    # Try all window sizes from 10 to min(100, n_points)
    for window in range(10, min(101, n_points + 1)):
        # Get last 'window' points of this segment
        X = data_timestamps[-window:]
        y = data_prices[-window:]
        
        # Normalize
        X_mean = X.mean()
        X_std = X.std()
        if X_std == 0:
            continue
        X_norm = (X - X_mean) / X_std
        
        # Fit OLS
        model = LinearRegression()
        model.fit(X_norm, y)
        
        # Get predictions
        y_pred = model.predict(X_norm)
        
        # Calculate error / window^K
        error = np.sum(np.abs(y - y_pred)) / (window ** K)
        
        # Calculate slope (price change per hour)
        slope = model.coef_[0] / X_std * 3600000
        
        # Print every 10th window for debugging
        if window % 10 == 0 or window == 10:
            print(f"     win={window:3d}: error/win^{K}={error:10.2f}, slope={slope:8.2f}")
        
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
    """Find cascade of lines, each on the reduced dataset"""
    print("\n" + "=" * 60)
    print(f"📊 Finding cascade of lines (error/window^{K})...")
    print("=" * 60)
    
    cascade = []
    current_prices = prices.copy()
    current_timestamps = timestamps.copy()
    current_start = 0
    iteration = 1
    
    while len(current_prices) >= 110:  # Need at least 110 points (100 window + 10)
        # Find best line on current data
        result = find_best_line(current_prices, current_timestamps, current_start)
        
        if result['window'] == 0:
            break
            
        cascade.append(result)
        
        end_date = full_start_date + timedelta(hours=result['end_idx'])
        window_start = result['end_idx'] - result['window']
        window_start_date = full_start_date + timedelta(hours=window_start)
        
        print(f"\n📌 Line {iteration}:")
        print(f"   Window: {window_start}h to {result['end_idx']}h ({result['window']}h)")
        print(f"   Date: {window_start_date} to {end_date}")
        print(f"   Slope: {result['slope']:+.2f} $/h")
        print(f"   Error/win^{K}: {result['error']:.2f}")
        
        # Remove the window we just used
        # We keep everything before the start of the window
        new_end = window_start
        
        if new_end <= current_start:
            print(f"   Cannot go further (new end {new_end} <= current start {current_start})")
            break
            
        current_prices = prices[current_start:new_end]
        current_timestamps = timestamps[current_start:new_end]
        print(f"   Remaining data: {current_start}h to {new_end}h ({len(current_prices)} points)")
        
        iteration += 1
        
        if iteration > 20:  # Safety limit
            print("   Reached iteration limit")
            break
    
    print(f"\n✅ Found {len(cascade)} cascade lines")
    return cascade

def create_plot(cascade):
    """Create plot with all cascade lines"""
    plt.figure(figsize=(14, 8))
    
    # Plot price
    plt.plot(range(len(prices)), prices, 'b-', alpha=0.5, label='BTC Price', linewidth=1)
    
    # Plot each cascade line
    colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    
    for i, line in enumerate(cascade):
        color = colors[i % len(colors)]
        end_idx = line['end_idx']
        window = line['window']
        line_start = end_idx - window
        
        # Get the line data
        X, y_pred, model, norm_params = line['line_data']
        
        # Plot the line
        x_range = range(line_start, end_idx)
        plt.plot(x_range, y_pred, color=color, linewidth=2.5, 
                label=f'Line {i+1}: win={window}h, slope={line["slope"]:+.1f}')
        
        # Mark the window points
        plt.scatter(x_range, prices[line_start:end_idx], c=color, s=15, alpha=0.3)
        
        # Add line number
        plt.text(line_start, prices[line_start] - 200, f'{i+1}', 
                fontsize=10, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
        
        # Mark the boundary
        if i < len(cascade) - 1:
            plt.axvline(x=line_start, color=color, linestyle='--', alpha=0.3)
    
    plt.title(f'BTC Price with Cascade Lines (error/window^{K})', fontsize=14)
    plt.xlabel('Hours from Start')
    plt.ylabel('Price (USDT)')
    plt.grid(True, alpha=0.2)
    plt.legend(loc='upper left', fontsize=8)
    
    # Add K value info
    plt.text(0.98, 0.02, f'K = {K}', transform=plt.gca().transAxes, 
             fontsize=12, ha='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
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
            
            # Find cascade
            cascade = find_cascade()
            
            # Create plot
            image_base64 = create_plot(cascade)
            
            # Calculate some stats
            avg_window = np.mean([line['window'] for line in cascade]) if cascade else 0
            avg_slope = np.mean([line['slope'] for line in cascade]) if cascade else 0
            
            # Simple HTML
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>BTC Cascade Lines K={K}</title>
                <style>
                    body {{ margin: 20px; font-family: Arial; background: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
                    h1 {{ margin: 0 0 10px 0; font-size: 20px; }}
                    .stats {{ margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 4px; display: flex; gap: 20px; }}
                    .stat {{ padding: 5px 10px; background: #fff; border-radius: 4px; }}
                    img {{ width: 100%; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📈 BTC Cascade Lines (error/window^{K})</h1>
                    <div class="stats">
                        <span class="stat">📊 Lines: {len(cascade)}</span>
                        <span class="stat">📏 Avg window: {avg_window:.1f}h</span>
                        <span class="stat">📈 Avg slope: {avg_slope:+.1f} $/h</span>
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
    print(f"🚀 BTC Cascade Lines Server (K={K})")
    print("=" * 60)
    print(f"   Each line minimizes error/window^{K}")
    print("   Then removes that window and continues")
    
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