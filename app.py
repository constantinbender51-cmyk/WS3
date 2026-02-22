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
import urllib.parse
warnings.filterwarnings('ignore')

# Global variables
prices = None
timestamps = None
full_start_date = None
full_end_date = None
K = 1.8  # Default window exponent

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

def find_best_line(data_prices, data_timestamps, start_idx, k_value):
    """Find the best line on the given data segment using error/window^K"""
    best_error = float('inf')
    best_line = None
    best_window = 0
    best_slope = 0
    best_end = 0
    
    n_points = len(data_prices)
    
    print(f"\n   Analyzing segment from {start_idx}h ({len(data_prices)} points) with K={k_value}:")
    
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
        error = np.sum(np.abs(y - y_pred)) / (window ** k_value)
        
        # Calculate slope (price change per hour)
        slope = model.coef_[0] / X_std * 3600000
        
        # Print every 10th window for debugging
        if window % 10 == 0 or window == 10:
            print(f"     win={window:3d}: error/win^{k_value}={error:10.2f}, slope={slope:8.2f}")
        
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

def find_cascade(k_value):
    """Find cascade of lines, each on the reduced dataset"""
    print("\n" + "=" * 60)
    print(f"📊 Finding cascade of lines (error/window^{k_value})...")
    print("=" * 60)
    
    cascade = []
    current_prices = prices.copy()
    current_timestamps = timestamps.copy()
    current_start = 0
    iteration = 1
    
    while len(current_prices) >= 110:  # Need at least 110 points (100 window + 10)
        # Find best line on current data
        result = find_best_line(current_prices, current_timestamps, current_start, k_value)
        
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
        print(f"   Error/win^{k_value}: {result['error']:.2f}")
        
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

def calculate_returns(cascade):
    """Calculate total return from following cascade strategy"""
    if len(cascade) < 2:
        return 0, 0, 0
    
    total_return = 0
    positive_return = 0
    negative_return = 0
    
    for i in range(len(cascade) - 1):
        current_line = cascade[i]
        next_line = cascade[i + 1]
        
        # The current line's window start is the boundary
        boundary = current_line['end_idx'] - current_line['window']
        
        # Price at boundary
        start_price = prices[boundary]
        
        # Price at next line's start (which is the end of this region)
        end_price = prices[next_line['end_idx'] - next_line['window']]
        
        # Calculate return
        region_return = (end_price / start_price - 1) * 100
        
        # Add to total with sign based on slope
        if current_line['slope'] > 0:
            positive_return += region_return
            total_return += region_return
        else:
            negative_return += region_return
            total_return -= region_return  # Subtract negative returns (we want to avoid them)
    
    return total_return, positive_return, negative_return

def create_plot(cascade, k_value):
    """Create plot with all cascade lines and slope-based backgrounds"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot price
    ax.plot(range(len(prices)), prices, 'b-', alpha=0.5, label='BTC Price', linewidth=1)
    
    # Calculate returns
    total_return, pos_return, neg_return = calculate_returns(cascade)
    
    # Add colored backgrounds based on slope
    regions = []
    for i, line in enumerate(cascade):
        end_idx = line['end_idx']
        window = line['window']
        line_start = end_idx - window
        
        # Choose background color based on slope
        if line['slope'] > 0:
            color = 'green'
            alpha = 0.15
            regions.append({'type': 'positive', 'start': line_start, 'end': end_idx, 'slope': line['slope']})
        else:
            color = 'red'
            alpha = 0.15
            regions.append({'type': 'negative', 'start': line_start, 'end': end_idx, 'slope': line['slope']})
        
        # Add colored background for the window region
        ax.axvspan(line_start, end_idx, alpha=alpha, color=color, zorder=0)
    
    # Plot each cascade line (on top of backgrounds)
    colors = ['black', 'darkblue', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    
    for i, line in enumerate(cascade):
        color = colors[i % len(colors)]
        end_idx = line['end_idx']
        window = line['window']
        line_start = end_idx - window
        
        # Get the line data
        X, y_pred, model, norm_params = line['line_data']
        
        # Plot the line
        x_range = range(line_start, end_idx)
        ax.plot(x_range, y_pred, color=color, linewidth=2.5, 
                label=f'Line {i+1}: win={window}h, slope={line["slope"]:+.1f}')
        
        # Mark the window points
        ax.scatter(x_range, prices[line_start:end_idx], c=color, s=15, alpha=0.3)
        
        # Add line number with slope-based background
        slope_symbol = '↑' if line['slope'] > 0 else '↓'
        bg_color = 'lightgreen' if line['slope'] > 0 else 'lightcoral'
        ax.text(line_start, prices[line_start] - 200, f'{i+1}{slope_symbol}', 
                fontsize=10, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc=bg_color, alpha=0.7))
        
        # Mark the boundary
        if i < len(cascade) - 1:
            ax.axvline(x=line_start, color=color, linestyle='--', alpha=0.3)
    
    # Add return info box
    return_text = f"""Strategy Returns:
    Total: {total_return:+.2f}%
    From Green: +{pos_return:.2f}%
    From Red (avoided): +{abs(neg_return):.2f}%"""
    
    ax.text(0.02, 0.98, return_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Add slope legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.15, label='Positive slope (GO LONG)'),
                      Patch(facecolor='red', alpha=0.15, label='Negative slope (GO SHORT)')]
    
    # Add the first price line to legend
    price_line = ax.get_legend_handles_labels()[0][0]
    ax.legend(handles=legend_elements + [price_line], 
              loc='upper left', fontsize=8)
    
    ax.set_title(f'BTC Price with Cascade Lines (error/window^{k_value})', fontsize=14)
    ax.set_xlabel('Hours from Start')
    ax.set_ylabel('Price (USDT)')
    ax.grid(True, alpha=0.2)
    
    # Add K value info
    ax.text(0.98, 0.02, f'K = {k_value}', transform=ax.transAxes, 
             fontsize=12, ha='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add slope statistics
    positive_slopes = sum(1 for line in cascade if line['slope'] > 0)
    negative_slopes = sum(1 for line in cascade if line['slope'] < 0)
    ax.text(0.02, 0.85, f'Positive: {positive_slopes} | Negative: {negative_slopes}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')

class BTCRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global K
        
        if self.path.startswith('/?') or self.path == '/':
            # Parse query parameters
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            
            # Get K value from query, default to stored K
            try:
                k_value = float(query.get('k', [K])[0])
                K = k_value  # Update global K
            except (ValueError, TypeError):
                k_value = K
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Find cascade with current K
            cascade = find_cascade(k_value)
            
            # Calculate returns
            total_return, pos_return, neg_return = calculate_returns(cascade)
            
            # Create plot
            image_base64 = create_plot(cascade, k_value)
            
            # Calculate some stats
            avg_window = np.mean([line['window'] for line in cascade]) if cascade else 0
            avg_slope = np.mean([line['slope'] for line in cascade]) if cascade else 0
            positive_slopes = sum(1 for line in cascade if line['slope'] > 0)
            negative_slopes = sum(1 for line in cascade if line['slope'] < 0)
            
            # HTML with input form
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>BTC Cascade Lines</title>
                <style>
                    body {{ margin: 20px; font-family: Arial; background: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
                    h1 {{ margin: 0 0 10px 0; font-size: 20px; }}
                    .controls {{ margin: 20px 0; padding: 15px; background: #f0f0f0; border-radius: 4px; display: flex; gap: 20px; align-items: center; }}
                    .stats {{ margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 4px; display: flex; gap: 20px; flex-wrap: wrap; }}
                    .stat {{ padding: 5px 10px; background: #fff; border-radius: 4px; }}
                    .positive {{ background: #d4edda; color: #155724; padding: 5px 10px; border-radius: 4px; }}
                    .negative {{ background: #f8d7da; color: #721c24; padding: 5px 10px; border-radius: 4px; }}
                    .returns {{ background: #cce5ff; color: #004085; padding: 5px 10px; border-radius: 4px; }}
                    input[type=number] {{ padding: 8px; border: 1px solid #ddd; border-radius: 4px; width: 80px; }}
                    button {{ padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }}
                    button:hover {{ background: #0056b3; }}
                    img {{ width: 100%; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📈 BTC Cascade Lines</h1>
                    
                    <div class="controls">
                        <form method="get" style="display: flex; gap: 10px; align-items: center;">
                            <label for="k">K = </label>
                            <input type="number" id="k" name="k" step="0.1" min="0.1" max="5" value="{k_value}">
                            <button type="submit">Update</button>
                        </form>
                        <span style="color: #666; font-size: 14px;">Higher K = shorter windows</span>
                    </div>
                    
                    <div class="stats">
                        <span class="stat">📊 Lines: {len(cascade)}</span>
                        <span class="stat">📏 Avg window: {avg_window:.1f}h</span>
                        <span class="stat">📈 Avg slope: {avg_slope:+.1f} $/h</span>
                        <span class="stat positive">🟢 Positive: {positive_slopes}</span>
                        <span class="stat negative">🔴 Negative: {negative_slopes}</span>
                    </div>
                    
                    <div class="stats">
                        <span class="returns">💰 Total Strategy Return: <b>{total_return:+.2f}%</b></span>
                        <span class="positive">🟢 From Green periods: +{pos_return:.2f}%</span>
                        <span class="negative">🔴 Avoided Red periods: +{abs(neg_return):.2f}%</span>
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
    print("🚀 BTC Cascade Lines Server")
    print("=" * 60)
    print("   Each line minimizes error/window^K")
    print("   Then removes that window and continues")
    print("   Green = Long, Red = Short")
    
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