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

def compute_lines_for_each_hour(k_value):
    """
    For each hour, compute the best line using data up to that hour
    """
    print("\n" + "=" * 60)
    print(f"📊 Computing lines for each hour (error/window^{k_value})...")
    print("=" * 60)
    
    hourly_lines = []
    total_points = len(prices)
    
    # Start from the first hour that has enough data (minimum 100 points)
    for current_pos in range(99, total_points):
        # Find best line ending at current_pos
        line = find_best_line_at_position(current_pos, k_value)
        
        if line is not None:
            hourly_lines.append(line)
            
            if current_pos % 100 == 0:
                print(f"   Processed hour {current_pos}/{total_points-1}")
    
    print(f"\n✅ Computed {len(hourly_lines)} lines (one for each hour from 99 to {total_points-1})")
    return hourly_lines

def generate_trades_from_hourly_lines(hourly_lines):
    """
    Generate trades based on previous hour's slope
    Trade one candle after the line with that sign
    """
    trades = []
    
    if len(hourly_lines) < 2:
        return trades
    
    # For each hour (starting from the second one), use previous hour's slope to trade
    for i in range(1, len(hourly_lines)):
        prev_line = hourly_lines[i-1]  # Line from previous hour
        current_line = hourly_lines[i]  # Line at current hour
        
        # Entry at current hour's start (using previous hour's slope)
        entry_idx = current_line['start_idx']
        
        # Exit at next hour's start (if available)
        if i < len(hourly_lines) - 1:
            next_line = hourly_lines[i + 1]
            exit_idx = next_line['start_idx']
        else:
            # Last trade exits at the end of data
            exit_idx = len(prices) - 1
        
        # Entry and exit prices
        entry_price = prices[entry_idx]
        exit_price = prices[exit_idx]
        
        # Determine trade type based on previous line's slope
        if prev_line['slope'] > 0:  # Long trade
            trade_return = (exit_price / entry_price - 1) * 100
            trade_type = "LONG"
            bg_color = "positive"
            zone_color = 'green'
        else:  # Short trade
            # For short: profit when price goes down
            trade_return = (entry_price / exit_price - 1) * 100
            trade_type = "SHORT"
            bg_color = "negative"
            zone_color = 'red'
        
        # Duration in hours
        duration = exit_idx - entry_idx
        
        # Convert indices to datetime
        entry_date = full_start_date + timedelta(hours=entry_idx)
        exit_date = full_start_date + timedelta(hours=exit_idx)
        
        # Calculate price change for verification
        price_change_pct = (exit_price / entry_price - 1) * 100
        
        trades.append({
            'id': i,
            'type': trade_type,
            'bg_color': bg_color,
            'zone_color': zone_color,
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'entry_date': entry_date.strftime('%Y-%m-%d %H:%M'),
            'exit_date': exit_date.strftime('%Y-%m-%d %H:%M'),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return': trade_return,
            'price_change': price_change_pct,
            'duration': duration,
            'signal_slope': prev_line['slope'],  # Slope that generated the signal
            'signal_hour': prev_line['end_idx']   # Hour when signal was generated
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
    """Create plot with all hourly lines and trades"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot price
    ax.plot(range(len(prices)), prices, 'b-', alpha=0.5, label='BTC Price', linewidth=1)
    
    # Add colored backgrounds based on trade zones
    for trade in trades:
        color = trade['zone_color']
        alpha = 0.15
        
        # Add colored background for the trade region
        ax.axvspan(trade['entry_idx'], trade['exit_idx'], alpha=alpha, color=color, zorder=0)
        
        # Add trade label
        y_pos = prices[trade['entry_idx']] - 300
        ax.text(trade['entry_idx'], y_pos, f"{trade['type']}", 
                fontsize=8, fontweight='bold', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    # Plot a subset of hourly lines (every 10th line to avoid clutter)
    for i, line in enumerate(hourly_lines):
        if i % 10 == 0:  # Plot every 10th line
            # Use color based on slope
            line_color = 'green' if line['slope'] > 0 else 'red'
            
            # Plot the line
            x_range = range(line['start_idx'], line['end_idx'] + 1)
            ax.plot(x_range, [prices[line['start_idx']]] * len(x_range), 
                   color=line_color, linewidth=0.5, alpha=0.3)
    
    # Plot the most recent line (last 5 hours) more prominently
    recent_lines = hourly_lines[-5:] if len(hourly_lines) >= 5 else hourly_lines
    for line in recent_lines:
        line_color = 'green' if line['slope'] > 0 else 'red'
        x_range = range(line['start_idx'], line['end_idx'] + 1)
        y_line = np.linspace(prices[line['start_idx']], prices[line['end_idx']], len(x_range))
        ax.plot(x_range, y_line, color=line_color, linewidth=2, alpha=0.8)
        
        # Add slope indicator
        slope_symbol = '↑' if line['slope'] > 0 else '↓'
        ax.text(line['end_idx'], prices[line['end_idx']], f'{slope_symbol}', 
                fontsize=12, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    # Add trade markers at entry/exit points
    for trade in trades:
        # Mark entry point
        ax.scatter(trade['entry_idx'], trade['entry_price'], 
                  c='black', marker='^', s=50, zorder=5, edgecolors='white', linewidth=1)
        # Mark exit point
        ax.scatter(trade['exit_idx'], trade['exit_price'], 
                  c='black', marker='s', s=50, zorder=5, edgecolors='white', linewidth=1)
    
    # Calculate returns
    total_return, long_return, short_return = calculate_returns(trades)
    
    # Add return info box
    return_text = f"""Strategy Returns:
    Total: {total_return:+.2f}%
    Long P&L: +{long_return:.2f}%
    Short P&L: +{short_return:.2f}%
    Trades: {len(trades)}"""
    
    ax.text(0.02, 0.98, return_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.15, label='LONG zone'),
        Patch(facecolor='red', alpha=0.15, label='SHORT zone'),
        plt.Line2D([0], [0], color='green', linewidth=2, alpha=0.8, label='Recent +Slope'),
        plt.Line2D([0], [0], color='red', linewidth=2, alpha=0.8, label='Recent -Slope'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=8, label='Entry'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=8, label='Exit')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    ax.set_title(f'BTC Hourly Lines & Trades (error/window^{k_value})', fontsize=14)
    ax.set_xlabel('Hours from Start')
    ax.set_ylabel('Price (USDT)')
    ax.grid(True, alpha=0.2)
    
    # Add K value info
    ax.text(0.98, 0.02, f'K = {k_value}', transform=ax.transAxes, 
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
            
            # Compute lines for each hour
            hourly_lines = compute_lines_for_each_hour(k_value)
            
            # Generate trades based on previous hour's slope
            trades = generate_trades_from_hourly_lines(hourly_lines)
            
            # Calculate returns
            total_return, long_return, short_return = calculate_returns(trades)
            
            # Create plot
            image_base64 = create_plot(hourly_lines, trades, k_value)
            
            # Calculate some stats
            avg_window = np.mean([line['window'] for line in hourly_lines]) if hourly_lines else 0
            avg_slope = np.mean([line['slope'] for line in hourly_lines]) if hourly_lines else 0
            positive_slopes = sum(1 for line in hourly_lines if line['slope'] > 0)
            negative_slopes = sum(1 for line in hourly_lines if line['slope'] < 0)
            
            # Generate trades table HTML
            trades_html = ""
            if trades:
                trades_html = """
                <div style="margin-top: 30px;">
                    <h3>📋 Trade History (Trading 1 hour after signal)</h3>
                    <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                        <thead>
                            <tr style="background: #333; color: white;">
                                <th style="padding: 8px;">#</th>
                                <th style="padding: 8px;">Type</th>
                                <th style="padding: 8px;">Signal Hour</th>
                                <th style="padding: 8px;">Entry Date</th>
                                <th style="padding: 8px;">Exit Date</th>
                                <th style="padding: 8px;">Duration</th>
                                <th style="padding: 8px;">Entry Price</th>
                                <th style="padding: 8px;">Exit Price</th>
                                <th style="padding: 8px;">Return</th>
                                <th style="padding: 8px;">Signal Slope</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                
                for trade in trades:
                    row_color = "#d4edda" if trade['type'] == 'LONG' else "#f8d7da"
                    signal_date = full_start_date + timedelta(hours=trade['signal_hour'])
                    trades_html += f"""
                        <tr style="background: {row_color}; border-bottom: 1px solid #ddd;">
                            <td style="padding: 8px; text-align: center;">{trade['id']}</td>
                            <td style="padding: 8px; text-align: center; font-weight: bold;">{trade['type']}</td>
                            <td style="padding: 8px;">{signal_date.strftime('%Y-%m-%d %H:%M')}</td>
                            <td style="padding: 8px;">{trade['entry_date']}</td>
                            <td style="padding: 8px;">{trade['exit_date']}</td>
                            <td style="padding: 8px; text-align: center;">{trade['duration']}h</td>
                            <td style="padding: 8px; text-align: right;">${trade['entry_price']:,.2f}</td>
                            <td style="padding: 8px; text-align: right;">${trade['exit_price']:,.2f}</td>
                            <td style="padding: 8px; text-align: right; font-weight: bold; color: {'green' if trade['return'] > 0 else 'red'};">{trade['return']:+.2f}%</td>
                            <td style="padding: 8px; text-align: right;">{trade['signal_slope']:+.1f}</td>
                        </tr>
                    """
                
                trades_html += """
                        </tbody>
                    </table>
                </div>
                """
            
            # HTML with input form and trades table
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>BTC Hourly Lines Strategy</title>
                <style>
                    body {{ margin: 20px; font-family: Arial; background: #f5f5f5; }}
                    .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
                    h1 {{ margin: 0 0 10px 0; font-size: 24px; }}
                    h3 {{ margin: 20px 0 10px 0; }}
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
                    tr:hover {{ opacity: 0.9; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📈 BTC Hourly Lines Strategy</h1>
                    <p style="color: #666; font-size: 14px;">
                        For each hour: compute best line (10-100h window). 
                        Trade next hour with previous hour's slope sign. 
                        Green = LONG, Red = SHORT. ▲ Entry, ■ Exit
                    </p>
                    
                    <div class="controls">
                        <form method="get" style="display: flex; gap: 10px; align-items: center;">
                            <label for="k"><b>K Parameter:</b></label>
                            <input type="number" id="k" name="k" step="0.1" min="0.1" max="5" value="{k_value}">
                            <button type="submit">Update Strategy</button>
                        </form>
                        <span style="color: #666; font-size: 14px;">Higher K = shorter windows</span>
                    </div>
                    
                    <div class="stats">
                        <span class="stat">📊 Hours analyzed: {len(hourly_lines)}</span>
                        <span class="stat">📏 Avg window: {avg_window:.1f}h</span>
                        <span class="stat">📈 Avg slope: {avg_slope:+.1f} $/h</span>
                        <span class="stat positive">🟢 Positive slopes: {positive_slopes}</span>
                        <span class="stat negative">🔴 Negative slopes: {negative_slopes}</span>
                    </div>
                    
                    <div class="stats">
                        <span class="returns">💰 Total Strategy Return: <b>{total_return:+.2f}%</b></span>
                        <span class="positive">🟢 Long P&L: +{long_return:.2f}%</span>
                        <span class="negative">🔴 Short P&L: +{short_return:.2f}%</span>
                        <span class="stat">📊 Total Trades: {len(trades)}</span>
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
    print("🚀 BTC Hourly Lines Strategy")
    print("=" * 60)
    print("   For each hour: compute best line (10-100h window)")
    print("   Set background based on slope sign")
    print("   Trade next hour with previous hour's slope sign")
    
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