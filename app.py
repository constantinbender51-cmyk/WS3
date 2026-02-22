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
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import threading
import time
warnings.filterwarnings('ignore')

# Global variables
prices = None
timestamps = None
full_start_date = None
full_end_date = None
K = 1.8  # Default window exponent

# Cache for precomputed lines
cached_lines = {}
cache_lock = threading.Lock()
last_cache_update = None
CACHE_DURATION = 300  # Cache valid for 5 minutes

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

def find_best_line_at_position(end_idx, k_value, min_window=10, max_window=100):
    """
    Find the best line ending at end_idx by trying different window sizes
    Returns the line with minimum error/window^K
    """
    best_error = float('inf')
    best_line = None
    best_window = 0
    best_slope = 0
    best_start_idx = 0
    
    # Can't go before the start of data
    max_possible_window = min(max_window, end_idx + 1)
    
    if max_possible_window < min_window:
        return None
    
    # Try different window sizes
    for window in range(min_window, max_possible_window + 1):
        start_idx = end_idx - window + 1
        if start_idx < 0:
            continue
            
        # Get window data
        X = timestamps[start_idx:end_idx + 1]
        y = prices[start_idx:end_idx + 1]
        
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
        
        if error < best_error:
            best_error = error
            best_window = window
            best_line = (X, y_pred, model, (X_mean, X_std))
            best_slope = slope
            best_start_idx = start_idx
    
    if best_window > 0:
        return {
            'start_idx': best_start_idx,
            'end_idx': end_idx,
            'window': best_window,
            'slope': best_slope,
            'error': best_error,
            'line_data': best_line
        }
    
    return None

def precompute_lines_for_k(k_value):
    """Precompute lines for a given K value and cache them"""
    global cached_lines, last_cache_update
    
    cache_key = f"k_{k_value}"
    
    # Check if cache is still valid
    with cache_lock:
        if cache_key in cached_lines:
            cache_time, lines = cached_lines[cache_key]
            if time.time() - cache_time < CACHE_DURATION:
                print(f"📦 Using cached lines for K={k_value}")
                return lines
    
    print(f"🔄 Precomputing lines for K={k_value} (this may take a few seconds)...")
    
    total_points = len(prices)
    hourly_lines = []
    
    # Use a smaller step to reduce computation (every 5th hour for visualization)
    step = 5  # Only compute every 5th hour for the visualization lines
    positions = range(99, total_points, step)
    
    for current_pos in positions:
        line = find_best_line_at_position(current_pos, k_value)
        if line is not None:
            hourly_lines.append(line)
        
        if current_pos % 200 == 0:
            print(f"   Progress: {current_pos}/{total_points-1}")
    
    # Also compute the last 20 hours at full resolution for recent signals
    recent_positions = range(max(99, total_points-20), total_points)
    for current_pos in recent_positions:
        if current_pos % step != 0:  # Skip if already computed
            line = find_best_line_at_position(current_pos, k_value)
            if line is not None:
                hourly_lines.append(line)
    
    # Sort by index
    hourly_lines.sort(key=lambda x: x['end_idx'])
    
    # Cache the results
    with cache_lock:
        cached_lines[cache_key] = (time.time(), hourly_lines)
        last_cache_update = time.time()
    
    print(f"✅ Precomputed {len(hourly_lines)} lines for K={k_value}")
    return hourly_lines

def get_trading_signals(hourly_lines):
    """
    Generate trading signals based on slope changes
    This is much faster than recomputing lines
    """
    signals = []
    
    if len(hourly_lines) < 2:
        return signals
    
    # Create a dense signal array for all hours
    max_idx = len(prices) - 1
    signal_array = [None] * (max_idx + 1)
    
    # Fill in signals for hours where we have lines
    for line in hourly_lines:
        if line['end_idx'] <= max_idx:
            signal_array[line['end_idx']] = {
                'slope': line['slope'],
                'start_idx': line['start_idx'],
                'window': line['window']
            }
    
    # Interpolate missing signals (use last known signal)
    last_signal = None
    for i in range(len(signal_array)):
        if signal_array[i] is not None:
            last_signal = signal_array[i]
        elif last_signal is not None:
            # Copy the last signal to this hour
            signal_array[i] = {
                'slope': last_signal['slope'],
                'start_idx': last_signal['start_idx'],
                'window': last_signal['window'],
                'interpolated': True
            }
    
    return signal_array

def generate_trades_from_signals(signal_array):
    """
    Generate trades using the signal array
    Trade one hour after signal with that sign
    """
    trades = []
    
    if not signal_array:
        return trades
    
    # Start trading from hour 100 (when we have enough data)
    for i in range(100, len(signal_array) - 1):
        if signal_array[i-1] is None:  # Need previous hour's signal
            continue
            
        prev_signal = signal_array[i-1]
        
        # Entry at current hour
        entry_idx = i
        
        # Exit at next hour
        exit_idx = i + 1
        
        # Entry and exit prices
        entry_price = prices[entry_idx]
        exit_price = prices[exit_idx]
        
        # Determine trade type based on previous hour's signal
        if prev_signal['slope'] > 0:  # Long trade
            trade_return = (exit_price / entry_price - 1) * 100
            trade_type = "LONG"
            zone_color = 'green'
        else:  # Short trade
            trade_return = (entry_price / exit_price - 1) * 100
            trade_type = "SHORT"
            zone_color = 'red'
        
        # Only create a trade if the return is not zero (avoid flat periods)
        if abs(trade_return) > 0.001:
            # Convert indices to datetime
            entry_date = full_start_date + timedelta(hours=entry_idx)
            exit_date = full_start_date + timedelta(hours=exit_idx)
            
            trades.append({
                'id': len(trades) + 1,
                'type': trade_type,
                'zone_color': zone_color,
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'entry_date': entry_date.strftime('%Y-%m-%d %H:%M'),
                'exit_date': exit_date.strftime('%Y-%m-%d %H:%M'),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': trade_return,
                'signal_slope': prev_signal['slope'],
                'signal_hour': i-1
            })
    
    return trades

def calculate_returns(trades):
    """Calculate total return from trades"""
    total_return = 0
    long_return = 0
    short_return = 0
    
    for trade in trades:
        total_return += trade['return']
        if trade['type'] == 'LONG':
            long_return += trade['return']
        else:
            short_return += trade['return']
    
    return total_return, long_return, short_return

def create_plot(hourly_lines, trades, k_value):
    """Create plot with sampled lines and trades"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot price
    ax.plot(range(len(prices)), prices, 'b-', alpha=0.5, label='BTC Price', linewidth=1)
    
    # Add colored backgrounds based on trade zones
    current_zone_start = 100
    current_zone_type = None
    
    for i in range(100, len(prices) - 1):
        # Find trade that covers this hour
        for trade in trades:
            if trade['entry_idx'] <= i < trade['exit_idx']:
                if current_zone_type != trade['type']:
                    # Zone change, draw previous zone
                    if current_zone_type is not None:
                        color = 'green' if current_zone_type == 'LONG' else 'red'
                        ax.axvspan(current_zone_start, i, alpha=0.15, color=color, zorder=0)
                    current_zone_start = i
                    current_zone_type = trade['type']
                break
    
    # Draw last zone
    if current_zone_type is not None:
        color = 'green' if current_zone_type == 'LONG' else 'red'
        ax.axvspan(current_zone_start, len(prices)-1, alpha=0.15, color=color, zorder=0)
    
    # Plot sampled lines (every 20th to avoid clutter)
    for i, line in enumerate(hourly_lines):
        if i % 20 == 0:  # Plot every 20th line
            # Use color based on slope
            line_color = 'green' if line['slope'] > 0 else 'red'
            
            # Plot just the start and end points as a line
            ax.plot([line['start_idx'], line['end_idx']], 
                   [prices[line['start_idx']], prices[line['end_idx']]], 
                   color=line_color, linewidth=0.3, alpha=0.2)
    
    # Plot the most recent lines (last 10) more prominently
    recent_lines = hourly_lines[-10:] if len(hourly_lines) >= 10 else hourly_lines
    for line in recent_lines:
        line_color = 'green' if line['slope'] > 0 else 'red'
        ax.plot([line['start_idx'], line['end_idx']], 
               [prices[line['start_idx']], prices[line['end_idx']]], 
               color=line_color, linewidth=2, alpha=0.8, marker='o', markersize=3)
        
        # Add slope indicator at the end
        slope_symbol = '↑' if line['slope'] > 0 else '↓'
        ax.text(line['end_idx'], prices[line['end_idx']], f'{slope_symbol}', 
                fontsize=10, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7))
    
    # Add trade markers at entry/exit points (sample every 10th trade to avoid clutter)
    for i, trade in enumerate(trades):
        if i % 10 == 0 or i == len(trades) - 1:  # Show every 10th trade and last trade
            # Mark entry point
            ax.scatter(trade['entry_idx'], trade['entry_price'], 
                      c='black', marker='^', s=30, zorder=5, alpha=0.7)
            # Mark exit point
            ax.scatter(trade['exit_idx'], trade['exit_price'], 
                      c='black', marker='s', s=30, zorder=5, alpha=0.7)
    
    # Calculate returns
    total_return, long_return, short_return = calculate_returns(trades)
    
    # Add return info box
    return_text = f"""Strategy Returns:
    Total: {total_return:+.2f}%
    Long: +{long_return:.2f}%
    Short: +{short_return:.2f}%
    Trades: {len(trades)}"""
    
    ax.text(0.02, 0.98, return_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.15, label='LONG zone'),
        Patch(facecolor='red', alpha=0.15, label='SHORT zone'),
        plt.Line2D([0], [0], color='green', linewidth=2, label='Recent +Slope'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Recent -Slope'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=8, label='Entry'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=8, label='Exit')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    ax.set_title(f'BTC Hourly Lines Strategy (Sampled)', fontsize=14)
    ax.set_xlabel('Hours from Start')
    ax.set_ylabel('Price (USDT)')
    ax.grid(True, alpha=0.2)
    
    # Add info about sampling
    ax.text(0.98, 0.02, f'K={k_value} | Showing 1/{len(hourly_lines)//50} lines', 
            transform=ax.transAxes, fontsize=10, ha='right',
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
            
            # Get precomputed lines from cache
            start_time = time.time()
            hourly_lines = precompute_lines_for_k(k_value)
            
            # Generate signal array (fast operation)
            signal_array = get_trading_signals(hourly_lines)
            
            # Generate trades from signals
            trades = generate_trades_from_signals(signal_array)
            
            # Calculate returns
            total_return, long_return, short_return = calculate_returns(trades)
            
            # Create plot
            image_base64 = create_plot(hourly_lines, trades, k_value)
            
            compute_time = time.time() - start_time
            
            # Calculate some stats
            avg_window = np.mean([line['window'] for line in hourly_lines]) if hourly_lines else 0
            avg_slope = np.mean([line['slope'] for line in hourly_lines]) if hourly_lines else 0
            positive_slopes = sum(1 for line in hourly_lines if line['slope'] > 0)
            negative_slopes = sum(1 for line in hourly_lines if line['slope'] < 0)
            
            # Generate trades table HTML (show only last 20 trades for performance)
            trades_html = ""
            if trades:
                recent_trades = trades[-20:]  # Show only last 20 trades
                trades_html = """
                <div style="margin-top: 30px;">
                    <h3>📋 Recent Trades (Last 20)</h3>
                    <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                        <thead>
                            <tr style="background: #333; color: white;">
                                <th style="padding: 8px;">#</th>
                                <th style="padding: 8px;">Type</th>
                                <th style="padding: 8px;">Entry Date</th>
                                <th style="padding: 8px;">Exit Date</th>
                                <th style="padding: 8px;">Entry</th>
                                <th style="padding: 8px;">Exit</th>
                                <th style="padding: 8px;">Return</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                
                for trade in recent_trades:
                    row_color = "#d4edda" if trade['type'] == 'LONG' else "#f8d7da"
                    trades_html += f"""
                        <tr style="background: {row_color}; border-bottom: 1px solid #ddd;">
                            <td style="padding: 8px; text-align: center;">{trade['id']}</td>
                            <td style="padding: 8px; text-align: center; font-weight: bold;">{trade['type']}</td>
                            <td style="padding: 8px;">{trade['entry_date']}</td>
                            <td style="padding: 8px;">{trade['exit_date']}</td>
                            <td style="padding: 8px; text-align: right;">${trade['entry_price']:,.0f}</td>
                            <td style="padding: 8px; text-align: right;">${trade['exit_price']:,.0f}</td>
                            <td style="padding: 8px; text-align: right; font-weight: bold; color: {'green' if trade['return'] > 0 else 'red'};">{trade['return']:+.2f}%</td>
                        </tr>
                    """
                
                trades_html += """
                        </tbody>
                    </table>
                    <p style="font-size: 11px; color: #666; margin-top: 5px;">Showing last 20 of {} total trades</p>
                </div>
                """.format(len(trades))
            
            # HTML with input form
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>BTC Optimized Strategy</title>
                <style>
                    body {{ margin: 20px; font-family: Arial; background: #f5f5f5; }}
                    .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
                    h1 {{ margin: 0 0 10px 0; font-size: 24px; }}
                    .performance-note {{ color: #28a745; font-size: 12px; margin-left: 10px; }}
                    .controls {{ margin: 20px 0; padding: 15px; background: #f0f0f0; border-radius: 4px; display: flex; gap: 20px; align-items: center; }}
                    .stats {{ margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 4px; display: flex; gap: 20px; flex-wrap: wrap; }}
                    .stat {{ padding: 5px 10px; background: #fff; border-radius: 4px; }}
                    .positive {{ background: #d4edda; color: #155724; padding: 5px 10px; border-radius: 4px; }}
                    .negative {{ background: #f8d7da; color: #721c24; padding: 5px 10px; border-radius: 4px; }}
                    .returns {{ background: #cce5ff; color: #004085; padding: 5px 10px; border-radius: 4px; }}
                    input[type=number] {{ padding: 8px; border: 1px solid #ddd; border-radius: 4px; width: 80px; }}
                    button {{ padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }}
                    button:hover {{ background: #0056b3; }}
                    img {{ width: 100%; margin-top: 20px; border: 1px solid #ddd; }}
                    table {{ margin-top: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    th {{ background: #333; color: white; position: sticky; top: 0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📈 BTC Optimized Strategy 
                        <span class="performance-note">⚡ Loaded in {compute_time:.2f}s</span>
                    </h1>
                    <p style="color: #666; font-size: 14px;">
                        Precomputed lines (every 5th hour) + signal interpolation.
                        Green = LONG, Red = SHORT. ▲ Entry, ■ Exit
                    </p>
                    
                    <div class="controls">
                        <form method="get" style="display: flex; gap: 10px; align-items: center;">
                            <label for="k"><b>K Parameter:</b></label>
                            <input type="number" id="k" name="k" step="0.1" min="0.1" max="5" value="{k_value}">
                            <button type="submit">Update Strategy</button>
                        </form>
                    </div>
                    
                    <div class="stats">
                        <span class="stat">📊 Lines: {len(hourly_lines)}</span>
                        <span class="stat">📏 Avg window: {avg_window:.1f}h</span>
                        <span class="stat">📈 +Slopes: {positive_slopes}</span>
                        <span class="stat">📉 -Slopes: {negative_slopes}</span>
                    </div>
                    
                    <div class="stats">
                        <span class="returns">💰 Total Return: <b>{total_return:+.2f}%</b></span>
                        <span class="positive">🟢 Long: +{long_return:.2f}%</span>
                        <span class="negative">🔴 Short: +{short_return:.2f}%</span>
                        <span class="stat">📊 Trades: {len(trades)}</span>
                    </div>
                    
                    <img src="data:image/png;base64,{image_base64}">
                    
                    {trades_html}
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
    print("🚀 BTC Optimized Strategy Server")
    print("=" * 60)
    print("   Features:")
    print("   • Precomputes lines every 5th hour (cached)")
    print("   • Interpolates signals for all hours")
    print("   • Shows recent trades only")
    print("   • 5-minute cache for fast reloads")
    
    # Fetch data on startup
    print("\n📡 Fetching BTC data...")
    fetch_data()
    
    # Precompute for default K on startup
    print("\n🔄 Precomputing default K=1.8 lines...")
    precompute_lines_for_k(1.8)
    
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