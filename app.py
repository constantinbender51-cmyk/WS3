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
positions = []  # Store trading positions (1 = long, -1 = short, 0 = flat)
position_dates = []  # Store corresponding indices

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

def find_best_line(data_prices, data_timestamps, end_pos):
    """Find the best line ending at end_pos using error/window^K"""
    best_error = float('inf')
    best_line = None
    best_window = 0
    best_slope = 0
    best_start = 0
    
    min_start = max(0, end_pos - 100)  # Can't go more than 100 back
    
    print(f"\n   Analyzing window ending at {end_pos}h:")
    
    # Try all window sizes from 10 to 100 (or until start of data)
    for window in range(10, 101):
        start_pos = end_pos - window
        if start_pos < 0:
            break
            
        # Get window points
        X = timestamps[start_pos:end_pos]
        y = prices[start_pos:end_pos]
        
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
            best_start = start_pos
    
    return {
        'start_idx': best_start,
        'end_idx': end_pos,
        'window': best_window,
        'slope': best_slope,
        'error': best_error,
        'line_data': best_line
    }

def calculate_positions():
    """Calculate trading positions by moving backwards through time"""
    global positions, position_dates
    
    print("\n" + "=" * 60)
    print(f"📊 Calculating trading positions (error/window^{K})...")
    print("=" * 60)
    
    positions = []
    position_dates = []
    
    # Start from the end and move backwards
    current_end = len(prices)
    iteration = 1
    
    while current_end > 100:  # Need at least 100 points
        # Find best line ending at current_end
        result = find_best_line(prices, timestamps, current_end)
        
        if result['window'] == 0:
            break
        
        start_date = full_start_date + timedelta(hours=result['start_idx'])
        end_date = full_start_date + timedelta(hours=current_end)
        
        # Determine position based on slope
        position = 1 if result['slope'] > 0 else -1  # Long if slope positive, short if negative
        
        print(f"\n📌 Position {iteration}:")
        print(f"   Period: {result['start_idx']}h to {current_end}h ({result['window']}h window)")
        print(f"   Date: {start_date} to {end_date}")
        print(f"   Slope: {result['slope']:+.2f} $/h → {'LONG' if position == 1 else 'SHORT'}")
        print(f"   Error/win^{K}: {result['error']:.2f}")
        
        # Store position for every point in this window
        for idx in range(result['start_idx'], current_end):
            positions.append(position)
            position_dates.append(idx)
        
        # Move back to the start of this window
        current_end = result['start_idx']
        iteration += 1
        
        if iteration > 50:  # Safety limit
            print("   Reached iteration limit")
            break
    
    print(f"\n✅ Calculated {len(positions)} position points")
    print(f"   Long positions: {positions.count(1)}")
    print(f"   Short positions: {positions.count(-1)}")

def create_plot():
    """Create plot with price and trading positions"""
    plt.figure(figsize=(14, 10))
    
    # Create 2x1 subplot grid
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
    
    # Top plot: Price with position-colored background
    ax_price = plt.subplot(gs[0])
    
    # Plot price
    ax_price.plot(range(len(prices)), prices, 'b-', alpha=0.7, label='BTC Price', linewidth=1.5)
    
    # Color background based on position
    if positions and position_dates:
        # Group consecutive positions
        current_pos = positions[0]
        start_idx = position_dates[0]
        
        for i in range(1, len(position_dates)):
            if positions[i] != current_pos or i == len(position_dates) - 1:
                end_idx = position_dates[i]
                color = 'green' if current_pos == 1 else 'red'
                ax_price.axvspan(start_idx, end_idx, facecolor=color, alpha=0.2)
                current_pos = positions[i]
                start_idx = position_dates[i]
    
    ax_price.set_title('BTC Price with Trading Positions (Green=Long, Red=Short)', fontsize=12)
    ax_price.set_ylabel('Price (USDT)')
    ax_price.grid(True, alpha=0.2)
    ax_price.legend(loc='upper left')
    
    # Add K value info
    ax_price.text(0.98, 0.02, f'K = {K}', transform=ax_price.transAxes, 
                  fontsize=10, ha='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Bottom plot: Position over time
    ax_pos = plt.subplot(gs[1])
    
    if positions and position_dates:
        # Plot position as step function
        ax_pos.step(position_dates, positions, where='post', color='purple', linewidth=2)
        ax_pos.fill_between(position_dates, 0, positions, where=np.array(positions) > 0, 
                            color='green', alpha=0.3, step='post')
        ax_pos.fill_between(position_dates, 0, positions, where=np.array(positions) < 0, 
                            color='red', alpha=0.3, step='post')
    
    ax_pos.set_ylim(-1.5, 1.5)
    ax_pos.set_yticks([-1, 0, 1])
    ax_pos.set_yticklabels(['SHORT', 'FLAT', 'LONG'])
    ax_pos.set_xlabel('Hour')
    ax_pos.set_ylabel('Position')
    ax_pos.grid(True, alpha=0.2)
    ax_pos.set_title('Trading Position Over Time')
    
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
            
            # Calculate positions
            calculate_positions()
            
            # Create plot
            image_base64 = create_plot()
            
            # Calculate some stats
            long_pct = positions.count(1) / len(positions) * 100 if positions else 0
            short_pct = positions.count(-1) / len(positions) * 100 if positions else 0
            
            # Simple HTML
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>BTC Trading Positions K={K}</title>
                <style>
                    body {{ margin: 20px; font-family: Arial; background: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
                    h1 {{ margin: 0 0 10px 0; font-size: 20px; }}
                    .stats {{ margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 4px; display: flex; gap: 20px; }}
                    .stat {{ padding: 5px 10px; background: #fff; border-radius: 4px; }}
                    .long {{ color: green; font-weight: bold; }}
                    .short {{ color: red; font-weight: bold; }}
                    img {{ width: 100%; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📈 BTC Trading Positions (error/window^{K})</h1>
                    <div class="stats">
                        <span class="stat">📊 Points: {len(positions)}</span>
                        <span class="stat long">🟢 Long: {long_pct:.1f}%</span>
                        <span class="stat short">🔴 Short: {short_pct:.1f}%</span>
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
    print(f"🚀 BTC Trading Positions Server (K={K})")
    print("=" * 60)
    print(f"   Moving backwards through time")
    print(f"   Each line minimizes error/window^{K}")
    print(f"   Position = LONG if slope > 0, SHORT if slope < 0")
    
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