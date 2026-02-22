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

def find_best_line():
    """Find the line (10-100 window) that minimizes error/window²"""
    print("\n📊 Finding best line...")
    
    best_error = float('inf')
    best_line = None
    best_window = 0
    best_slope = 0
    
    # Try all window sizes from 10 to 100
    for window in range(10, 101):
        if window > len(prices):
            break
            
        # Get last 'window' points
        X = timestamps[-window:]
        y = prices[-window:]
        
        # Normalize
        X_mean = X.mean()
        X_std = X.std()
        X_norm = (X - X_mean) / X_std
        
        # Fit OLS
        model = LinearRegression()
        model.fit(X_norm, y)
        
        # Get predictions
        y_pred = model.predict(X_norm)
        
        # Calculate error / window²
        error = np.sum(np.abs(y - y_pred)) / (window * window)
        
        # Calculate slope (price change per hour)
        slope = model.coef_[0] / X_std * 3600000
        
        print(f"   Window {window:3d}: error/win² = {error:10.2f}, slope = {slope:8.2f}")
        
        if error < best_error:
            best_error = error
            best_window = window
            best_line = (X, y_pred, model, (X_mean, X_std))
            best_slope = slope
    
    print(f"\n✅ Best window: {best_window} hours")
    print(f"   Best error/win²: {best_error:.2f}")
    print(f"   Slope: {best_slope:+.2f} $/h")
    
    return best_line, best_window, best_slope

def create_plot(line_data, window, slope):
    """Create simple plot with price and best line"""
    plt.figure(figsize=(12, 6))
    
    X, y_pred, model, norm_params = line_data
    
    # Plot price (all data)
    plt.plot(range(len(prices)), prices, 'b-', alpha=0.5, label='BTC Price', linewidth=1)
    
    # Plot the best line
    start_idx = len(prices) - window
    x_range = range(start_idx, len(prices))
    plt.plot(x_range, y_pred, 'r-', linewidth=2.5, label=f'Best Line (window={window}h)')
    
    # Mark the window points
    plt.scatter(x_range, prices[start_idx:], c='orange', s=20, alpha=0.5, label='Window Points')
    
    # Add slope info
    slope_text = f'Slope: {slope:+.2f} $/h'
    plt.text(0.02, 0.95, slope_text, transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Simple styling
    plt.title(f'BTC Price with Best OLS Line (minimizing error/window²)', fontsize=14)
    plt.xlabel('Hours from Start')
    plt.ylabel('Price (USDT)')
    plt.grid(True, alpha=0.2)
    plt.legend()
    
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
            
            # Find best line
            line_data, window, slope = find_best_line()
            
            # Create plot
            image_base64 = create_plot(line_data, window, slope)
            
            # Ultra simple HTML
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>BTC Best Line</title>
                <style>
                    body {{ margin: 20px; font-family: Arial; background: #f5f5f5; }}
                    .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
                    h1 {{ margin: 0 0 10px 0; font-size: 20px; }}
                    img {{ width: 100%; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📈 BTC Best Line (minimizing error/window²)</h1>
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
    print("=" * 50)
    print("🚀 BTC Best Line Server")
    print("=" * 50)
    
    # Fetch data on startup
    print("\n📡 Fetching BTC data...")
    fetch_data()
    
    # Start server
    PORT = 8080
    with socketserver.TCPServer(("", PORT), BTCRequestHandler) as httpd:
        print(f"\n🌐 http://localhost:{PORT}")
        print("   Finding line that minimizes error/window²")
        print("   Press Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()