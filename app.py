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

def find_best_line(data_prices, data_timestamps, current_pos):
    """Find the best line starting from current_pos using error/window^K"""
    best_error = float('inf')
    best_line = None
    best_window = 0
    best_slope = 0
    best_end = 0
    
    max_end = min(current_pos + 100, len(prices))  # Can't go beyond data end
    
    print(f"\n   Analyzing from position {current_pos}h:")
    
    # Try all window sizes from 10 to 100 (or until data ends)
    for window in range(10, 101):
        end_pos = current_pos + window
        if end_pos > len(prices):
            break
            
        # Get window points
        X = timestamps[current_pos:end_pos]
        y = prices[current_pos:end_pos]
        
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
        if window % 10 == 0:
            print(f"     win={window:3d}: error/win^{K}={error:10.2f}, slope={slope:8.2f}")
        
        if error < best_error:
            best_error = error
            best_window = window
            best_line = (X, y_pred, model, (X_mean, X_std))
            best_slope = slope
            best_end = end_pos
    
    return {
        'start_idx': current_pos,
        'end_idx': best_end,
        'window': best_window,
        'slope': best_slope,
        'error': best_error,
        'line_data': best_line
    }

def find_cascade():
    """Find cascade of lines moving forward through time"""
    print("\n" + "=" * 60)
    print(f"📊 Finding cascade of lines (error/window^{K})...")
    print("=" * 60)
    
    cascade = []
    current_pos = 0
    iteration = 1
    
    while current_pos + 10 < len(prices):  # Need at least 10 points left
        # Find best line starting from current position
        result = find_best_line(prices, timestamps, current_pos)
        
        if result['window'] == 0:
            break
            
        cascade.append(result)
        
        end_date = full_start_date + timedelta(hours=result['end_idx'])
        start_date = full_start_date + timedelta(hours=current_pos)
        
        print(f"\n📌 Line {iteration}:")
        print(f"   Period: {current_pos}h to {result['end_idx']}h ({result['window']}h window)")
        print(f"   Date: {start_date} to {end_date}")
        print(f"   Slope: {result['slope']:+.2f} $/h")
        print(f"   Error/win^{K}: {result['error']:.2f}")
        
        # Move to the start of this window for the next line
        current_pos = result['start_idx']
        print(f"   Next start: {current_pos}h")
        
        iteration += 1
        
        if iteration > 20 or current_pos >= len(prices) - 10:  # Safety limits
            print("   Stopping: reached end or iteration limit")
            break
    
    print(f"\n✅ Found {len(cascade)} cascade lines")
    return cascade

def create_plot(cascade):
    """Create plot with all cascade lines"""
    plt.figure(figsize=(14, 8))
    
    # Plot price
    plt.plot(range(len(prices)), prices, 'b-', alpha=0.7, label='BTC Price', linewidth=1.5)
    
    # Plot each cascade line
    colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    
    for i, line in enumerate(cascade):
        color = colors[i % len(colors)]
        start_idx = line['start_idx']
        end_idx = line['end_idx']
        
        # Get the line data
        X, y_pred, model, norm_params = line['line_data']
        
        # Plot the line
        x_range = range(start_idx, end_idx)
        plt.plot(x_range, y_pred, color=color, linewidth=2.5, 
                label=f'Line {i+1}: win={line["window"]}h, slope={line["slope"]:+.1f}')
        
        # Add line number at start
        plt.text(start_idx, prices[start_idx] - 200, f'{i+1}', 
                fontsize=10, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
        
        # Mark the boundary between lines
        if i < len(cascade) - 1:
            plt.axvline(x=start_idx, color=color, linestyle='--', alpha=0.3)
    
    plt.title(f'BTC Price with Forward Cascade Lines (error/window^{K})', fontsize=14)
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
                <title>BTC Forward Cascade K={K}</title>
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
                    <h1>📈 BTC Forward Cascade (error/window^{K})</h1>
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
    print(f"🚀 BTC Forward Cascade Server (K={K})")
    print("=" * 60)
    print(f"   Moving forward through time")
    print(f"   Each line minimizes error/window^{K}")
    print(f"   Next line starts at beginning of current window")
    
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