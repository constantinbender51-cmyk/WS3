import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
import http.server
import socketserver
import threading
import webbrowser
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
import argparse

# ============================================================================
# CONFIGURABLE PARAMETERS - ADJUST THESE TO TUNE THE ALGORITHM
# ============================================================================

# TRADING PARAMETERS
SLOPE_THRESHOLD = 0.001  # Minimum slope to trigger trade (0.1% per hour)
                        # Higher = fewer trades, stronger trends only
                        # Lower = more trades, more noise

# ERROR MINIMIZATION PARAMETERS  
TIME_PENALTY_EXPONENT = 3  # Exponent for time in denominator: error / (time_range)^exp
                          # 0 = no penalty (neutral to timeframe)
                          # 1 = error / time_range (slightly favors shorter)
                          # 2 = error / time_range² (favors longer trends) - DEFAULT
                          # 3 = error / time_range³ (strongly favors longer)
                          # Negative = favors shorter trends

# LOOKBACK RANGE (in hours)
MIN_LOOKBACK = 4    # Minimum hours to consider for regression
MAX_LOOKBACK = 24   # Maximum hours to consider for regression

# POSITION SIZING
POSITION_SIZE = 1.0  # 1.0 = 100% of capital, 0.5 = 50%, etc.

# DATA PARAMETERS
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
HISTORY_DAYS = 30    # Days of historical data to fetch
LIMIT = HISTORY_DAYS * 24  # 30 days * 24h = 720 hours

# SERVER PARAMETERS
PORT = 8080
HOST = "0.0.0.0"

# ============================================================================
# END OF CONFIGURABLE PARAMETERS
# ============================================================================

# Global variable to store data (load once at startup)
cached_data = None
last_fetch_time = None
CACHE_DURATION = 60  # Cache data for 60 seconds to avoid excessive API calls

def fetch_binance_data(force_refresh=False):
    """Fetch Binance data and cache it"""
    global cached_data, last_fetch_time
    
    current_time = time.time()
    
    # Return cached data if it's still fresh
    if not force_refresh and cached_data is not None and last_fetch_time is not None:
        if current_time - last_fetch_time < CACHE_DURATION:
            print(f"📦 Using cached data from {datetime.fromtimestamp(last_fetch_time)}")
            return cached_data
    
    # Fetch new data
    print(f"🌐 Fetching fresh data from Binance...")
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        'symbol': SYMBOL,
        'interval': INTERVAL,
        'limit': LIMIT
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Process data
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert to numeric and datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['open'] = pd.to_numeric(df['open'])
        
        # Add time in hours for regression
        df['time_num'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600
        
        # Cache the data
        cached_data = df[['timestamp', 'open', 'high', 'low', 'close', 'time_num']]
        last_fetch_time = current_time
        
        print(f"✅ Fetched {len(df)} data points")
        print(f"📅 Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"💰 Current BTC: ${df['close'].iloc[-1]:,.2f}")
        
        return cached_data
    
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return cached_data  # Return cached data even if expired, rather than None

# Linear regression analysis and trading strategy
def analyze_and_trade(df):
    prices = df['close'].values
    times = df['time_num'].values
    timestamps = df['timestamp'].values
    
    positions = []
    pnl = []
    position_timestamps = []
    buy_signals = []
    sell_signals = []
    
    current_position = 0
    cumulative_pnl = 0
    entry_price = 0
    
    print(f"\n📊 Running analysis with parameters:")
    print(f"   Slope Threshold: {SLOPE_THRESHOLD*100:.3f}% per hour")
    print(f"   Time Penalty Exponent: {TIME_PENALTY_EXPONENT}")
    print(f"   Lookback Range: {MIN_LOOKBACK}-{MAX_LOOKBACK} hours")
    print(f"   Position Size: {POSITION_SIZE*100:.0f}%\n")
    
    for i in range(MAX_LOOKBACK, len(prices) - 1):
        current_time = timestamps[i]
        position_timestamps.append(current_time)
        
        best_error = float('inf')
        best_slope = 0
        best_lookback = 0
        
        for lookback in range(MIN_LOOKBACK, MAX_LOOKBACK + 1):
            start_idx = i - lookback
            if start_idx < 0:
                continue
            
            X = times[start_idx:i+1].reshape(-1, 1)
            y = prices[start_idx:i+1]
            
            model = LinearRegression()
            model.fit(X, y)
            
            slope = model.coef_[0]  # $ per hour
            
            # Calculate error with configurable time penalty
            predicted = model.predict(X)
            time_range_hours = times[i] - times[start_idx]
            
            if TIME_PENALTY_EXPONENT == 0:
                error = np.abs(predicted - y).mean()
            else:
                error = np.abs(predicted - y).mean() / (time_range_hours ** TIME_PENALTY_EXPONENT)
            
            if error < best_error:
                best_error = error
                best_slope = slope
                best_lookback = lookback
        
        # Trading logic - CORRECTED SHORT P&L
        if abs(best_slope) > SLOPE_THRESHOLD:
            if best_slope > 0:  # Bullish signal
                if current_position <= 0:  # Not long
                    # Close short if exists
                    if current_position == -1:
                        # SHORT P&L = (entry - exit) / entry
                        trade_pnl = (entry_price - prices[i]) / entry_price * 100
                        cumulative_pnl += trade_pnl * POSITION_SIZE
                        print(f"📈 Close SHORT at ${prices[i]:,.2f} P&L: {trade_pnl:.2f}%")
                    
                    # Open long
                    current_position = 1
                    entry_price = prices[i]
                    buy_signals.append((current_time, prices[i]))
                    print(f"🚀 OPEN LONG at ${prices[i]:,.2f} (lookback: {best_lookback}h, slope: {best_slope:.4f})")
                    
            else:  # Bearish signal
                if current_position >= 0:  # Not short
                    # Close long if exists
                    if current_position == 1:
                        # LONG P&L = (exit - entry) / entry
                        trade_pnl = (prices[i] - entry_price) / entry_price * 100
                        cumulative_pnl += trade_pnl * POSITION_SIZE
                        print(f"📉 Close LONG at ${prices[i]:,.2f} P&L: {trade_pnl:.2f}%")
                    
                    # Open short
                    current_position = -1
                    entry_price = prices[i]
                    sell_signals.append((current_time, prices[i]))
                    print(f"🔻 OPEN SHORT at ${prices[i]:,.2f} (lookback: {best_lookback}h, slope: {best_slope:.4f})")
        
        positions.append(current_position * POSITION_SIZE)
        pnl.append(cumulative_pnl)
    
    # Close any open position at the end
    final_price = prices[-1]
    if current_position == 1:
        trade_pnl = (final_price - entry_price) / entry_price * 100
        cumulative_pnl += trade_pnl * POSITION_SIZE
        print(f"\n🔚 Close final LONG at ${final_price:,.2f} P&L: {trade_pnl:.2f}%")
    elif current_position == -1:
        trade_pnl = (entry_price - final_price) / entry_price * 100
        cumulative_pnl += trade_pnl * POSITION_SIZE
        print(f"\n🔚 Close final SHORT at ${final_price:,.2f} P&L: {trade_pnl:.2f}%")
    
    print(f"\n📊 Final P&L: {cumulative_pnl:.2f}%")
    print(f"📈 Total Trades: {len(buy_signals) + len(sell_signals)}")
    print(f"   Buy Signals: {len(buy_signals)}")
    print(f"   Sell Signals: {len(sell_signals)}")
    
    return position_timestamps, positions, pnl, buy_signals, sell_signals
# Create plot using matplotlib
def create_plot(df, timestamps, positions, pnl, buy_signals, sell_signals):
    # Align data lengths
    min_len = min(len(timestamps), len(positions), len(pnl))
    plot_timestamps = timestamps[:min_len]
    plot_positions = positions[:min_len]
    plot_pnl = pnl[:min_len]
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(12, 10))
    
    # Price chart
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df['timestamp'], df['close'], 'b-', linewidth=1.5, label='BTC Price')
    
    # Add buy/sell signals
    if buy_signals:
        buy_dates, buy_prices = zip(*buy_signals)
        ax1.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy', zorder=5)
    
    if sell_signals:
        sell_dates, sell_prices = zip(*sell_signals)
        ax1.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell', zorder=5)
    
    ax1.set_title(f'BTC Price - {INTERVAL} Chart', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    
    # Position chart
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(plot_timestamps, plot_positions, 'o-', color='orange', linewidth=2, markersize=3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=POSITION_SIZE, color='green', linestyle='--', alpha=0.3)
    ax2.axhline(y=-POSITION_SIZE, color='red', linestyle='--', alpha=0.3)
    ax2.set_title(f'Position ({POSITION_SIZE*100:.0f}% max)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Position', fontsize=12)
    ax2.set_ylim(-POSITION_SIZE*1.5, POSITION_SIZE*1.5)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    
    # P&L chart
    ax3 = plt.subplot(3, 1, 3)
    ax3.fill_between(plot_timestamps, 0, plot_pnl, color='green', alpha=0.3)
    ax3.plot(plot_timestamps, plot_pnl, 'g-', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_title('Cumulative P&L', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('P&L (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    
    # Add parameter info as text
    params_text = f"Threshold: {SLOPE_THRESHOLD*100:.3f}%/h | Time^({TIME_PENALTY_EXPONENT}) | Lookback: {MIN_LOOKBACK}-{MAX_LOOKBACK}h"
    fig.suptitle(params_text, fontsize=10, y=0.98)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    return image_base64

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>BTC Trading Analysis</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=2.0, user-scalable=yes">
    <style>
        body {{
            margin: 0;
            padding: 10px;
            background-color: #111;
            color: white;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }}
        .container {{
            max-width: 100%;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 24px;
            text-align: center;
            margin: 10px 0;
            color: #ff9900;
        }}
        .params {{
            background-color: #222;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 15px;
            font-size: 14px;
            text-align: center;
            color: #aaa;
        }}
        .chart-container {{
            width: 100%;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            background-color: #222;
            border-radius: 10px;
            padding: 10px 0;
        }}
        .chart-container img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        .info-card {{
            background-color: #222;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #ff9900;
        }}
        .info-label {{
            font-size: 14px;
            color: #aaa;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .info-value {{
            font-size: 24px;
            font-weight: bold;
            color: #ff9900;
        }}
        .info-value.small {{
            font-size: 18px;
        }}
        .status {{
            background-color: #222;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            text-align: center;
            font-size: 18px;
        }}
        .refresh-btn {{
            background-color: #ff9900;
            color: #111;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            margin-top: 15px;
            width: 100%;
            cursor: pointer;
            -webkit-tap-highlight-color: transparent;
        }}
        .refresh-btn:active {{
            background-color: #ffaa22;
        }}
        .footer {{
            text-align: center;
            margin-top: 20px;
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 BTC {INTERVAL} Trading Analysis</h1>
        
        <div class="params">
            Threshold: {threshold}%/h | Time^({exponent}) | Lookback: {min_lb}-{max_lb}h
        </div>
        
        <div class="chart-container">
            <img src="data:image/png;base64,{chart_image}" alt="BTC Analysis Chart">
        </div>
        
        <div class="info-grid">
            <div class="info-card">
                <div class="info-label">Current Position</div>
                <div class="info-value" id="position">Loading...</div>
            </div>
            <div class="info-card">
                <div class="info-label">Total P&L</div>
                <div class="info-value" id="pnl">Loading...</div>
            </div>
            <div class="info-card">
                <div class="info-label">Signals</div>
                <div class="info-value small" id="signals">Loading...</div>
            </div>
        </div>
        
        <div class="status" id="last-update">
            Last Update: Loading...
        </div>
        
        <button class="refresh-btn" onclick="refreshData()">
            🔄 Refresh Data
        </button>
        
        <div class="footer">
            Data from Binance • Parameters at top of script
        </div>
    </div>
    
    <script>
        function refreshData() {{
            window.location.reload();
        }}
        
        function updateInfo() {{
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {{
                    document.getElementById('position').innerHTML = data.position;
                    document.getElementById('pnl').innerHTML = data.pnl.toFixed(2) + '%';
                    document.getElementById('signals').innerHTML = data.buy_signals + ' Buy / ' + data.sell_signals + ' Sell';
                    document.getElementById('last-update').innerHTML = 'Last Update: ' + data.last_update;
                }})
                .catch(error => {{
                    console.error('Error fetching data:', error);
                }});
        }}
        
        updateInfo();
        setInterval(updateInfo, 30000);
    </script>
</body>
</html>
'''

# Custom HTTP handler
class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Get data (from cache)
            df = fetch_binance_data()
            if df is not None:
                timestamps, positions, pnl, buy_signals, sell_signals = analyze_and_trade(df)
                chart_image = create_plot(df, timestamps, positions, pnl, buy_signals, sell_signals)
                
                # Send HTML with current parameters
                html = HTML_TEMPLATE.format(
                    chart_image=chart_image,
                    threshold=SLOPE_THRESHOLD*100,
                    exponent=TIME_PENALTY_EXPONENT,
                    min_lb=MIN_LOOKBACK,
                    max_lb=MAX_LOOKBACK,
                    INTERVAL=INTERVAL
                )
                self.wfile.write(html.encode())
            else:
                self.wfile.write(b"<html><body><h1>Error fetching data</h1></body></html>")
                
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            df = fetch_binance_data()
            if df is not None:
                timestamps, positions, pnl, buy_signals, sell_signals = analyze_and_trade(df)
                
                current_position = positions[-1] if positions else 0
                current_pnl = pnl[-1] if pnl else 0
                
                if current_position > 0.5:
                    pos_text = f'LONG 📈 ({current_position*100:.0f}%)'
                elif current_position < -0.5:
                    pos_text = f'SHORT 📉 ({abs(current_position)*100:.0f}%)'
                else:
                    pos_text = 'FLAT ⚪'
                
                status = {
                    'position': pos_text,
                    'pnl': current_pnl,
                    'buy_signals': len(buy_signals),
                    'sell_signals': len(sell_signals),
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.wfile.write(json.dumps(status).encode())
            else:
                self.wfile.write(json.dumps({'error': 'Failed to fetch data'}).encode())
        
        else:
            self.send_response(404)
            self.end_headers()

def start_server():
    with socketserver.TCPServer((HOST, PORT), CustomHandler) as httpd:
        print(f"\n{'='*50}")
        print(f"✅ Server running at http://{HOST}:{PORT}")
        print(f"{'='*50}")
        print("📱 Open on mobile:")
        
        try:
            import socket
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            print(f"   http://{local_ip}:{PORT}")
        except:
            print(f"   Check your network IP")
        
        print(f"\n🔄 Press Ctrl+C to stop")
        print(f"{'='*50}\n")
        httpd.serve_forever()

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 BTC TRADING ANALYSIS SERVER")
    print("=" * 60)
    
    # Load all data on startup
    print("\n📥 Loading initial data from Binance...")
    df = fetch_binance_data(force_refresh=True)
    
    if df is not None:
        print("\n✅ Initial data loaded successfully!")
        print(f"📊 Total data points: {len(df)}")
        print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Start server
        start_server()
    else:
        print("❌ Failed to fetch initial data. Check internet connection.")
        exit(1)