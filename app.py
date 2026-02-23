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
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64

# Configuration
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LIMIT = 720  # 30 days * 24h = 720 hours
PORT = 8080
HOST = "0.0.0.0"

# Fetch Binance data
def fetch_binance_data():
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
        
        return df[['timestamp', 'open', 'high', 'low', 'close']]
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Linear regression analysis and trading strategy
def analyze_and_trade(df):
    # Convert timestamp to numeric for regression
    df['time_num'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    
    prices = df['close'].values
    times = df['time_num'].values
    timestamps = df['timestamp'].values
    
    positions = []  # Track positions (1 for long, -1 for short, 0 for flat)
    pnl = []  # Track P&L
    position_timestamps = []
    buy_signals = []
    sell_signals = []
    
    current_position = 0
    cumulative_pnl = 0
    entry_price = 0
    
    threshold = 0.001  # Slope threshold (0.1%)
    
    # Iterate through the data
    for i in range(24, len(prices) - 1):
        current_time = timestamps[i]
        position_timestamps.append(current_time)
        
        # Find optimal lookback by minimizing the error function
        best_error = float('inf')
        best_slope = 0
        
        for lookback in range(4, 25):  # Test from 4 to 24 hours
            start_idx = i - lookback
            if start_idx < 0:
                continue
            
            # Perform linear regression
            X = times[start_idx:i+1].reshape(-1, 1)
            y = prices[start_idx:i+1]
            
            model = LinearRegression()
            model.fit(X, y)
            
            slope = model.coef_[0]
            
            # Calculate error function: |predicted - actual| / |time range|^2
            predicted = model.predict(X)
            error = np.abs(predicted - y).mean() / ((times[i] - times[start_idx]) ** 2)
            
            if error < best_error:
                best_error = error
                best_slope = slope
        
        # Trading logic
        if abs(best_slope) > threshold:
            price_change_pct = (prices[i] - prices[i-1]) / prices[i-1]
            
            if best_slope > 0:  # Bullish signal
                if current_position <= 0:  # Not long
                    # Close short if exists
                    if current_position == -1:
                        cumulative_pnl += (entry_price - prices[i]) / entry_price * 100
                    
                    # Open long
                    current_position = 1
                    entry_price = prices[i]
                    buy_signals.append((current_time, prices[i]))
                    
            else:  # Bearish signal
                if current_position >= 0:  # Not short
                    # Close long if exists
                    if current_position == 1:
                        cumulative_pnl += (prices[i] - entry_price) / entry_price * 100
                    
                    # Open short
                    current_position = -1
                    entry_price = prices[i]
                    sell_signals.append((current_time, prices[i]))
            
            # Add to P&L based on slope direction
            pnl_adjustment = np.sign(best_slope) * price_change_pct * 100
            cumulative_pnl += pnl_adjustment
        
        positions.append(current_position)
        pnl.append(cumulative_pnl)
    
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
    
    ax1.set_title('BTC Price - 1h Chart', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    
    # Position chart
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(plot_timestamps, plot_positions, 'o-', color='orange', linewidth=2, markersize=3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=1, color='green', linestyle='--', alpha=0.3)
    ax2.axhline(y=-1, color='red', linestyle='--', alpha=0.3)
    ax2.set_title('Position', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Position\n(1=Long, -1=Short, 0=Flat)', fontsize=12)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Short', 'Flat', 'Long'])
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    
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
    ax3.tick_params(axis='y', labelsize=10)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    return image_base64

# HTML template with viewport for mobile
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
        @media (max-width: 600px) {{
            h1 {{ font-size: 22px; }}
            .info-value {{ font-size: 20px; }}
            .info-value.small {{ font-size: 16px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 BTC 1H Trading Analysis</h1>
        
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
            Data from Binance • Updated every minute
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
        
        // Update info every 30 seconds
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
            
            # Get current data
            df = fetch_binance_data()
            if df is not None:
                timestamps, positions, pnl, buy_signals, sell_signals = analyze_and_trade(df)
                chart_image = create_plot(df, timestamps, positions, pnl, buy_signals, sell_signals)
                
                # Send HTML
                html = HTML_TEMPLATE.format(chart_image=chart_image)
                self.wfile.write(html.encode())
            else:
                self.wfile.write(b"<html><body><h1>Error fetching data</h1></body></html>")
                
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Get latest data
            df = fetch_binance_data()
            if df is not None:
                timestamps, positions, pnl, buy_signals, sell_signals = analyze_and_trade(df)
                
                # Get latest values
                current_position = positions[-1] if positions else 0
                current_pnl = pnl[-1] if pnl else 0
                
                # Map position to text
                if current_position == 1:
                    pos_text = 'LONG 📈'
                elif current_position == -1:
                    pos_text = 'SHORT 📉'
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
        print(f"✅ Server running at http://{HOST}:{PORT}")
        print("📱 Open this URL on your mobile device:")
        
        # Get local IP
        import socket
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
            print(f"   http://{local_ip}:{PORT}")
        except:
            print(f"   Check your network IP")
        
        print("\n🔄 Press Ctrl+C to stop the server")
        httpd.serve_forever()

# Main execution
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Starting BTC Trading Analysis Server")
    print("=" * 50)
    print(f"📊 Fetching {LIMIT} hours of BTC data from Binance...")
    
    # Test data fetch
    df = fetch_binance_data()
    if df is not None:
        print(f"✅ Successfully fetched {len(df)} data points")
        print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"💰 Current BTC Price: ${df['close'].iloc[-1]:,.2f}")
        
        # Start server
        print(f"\n🌐 Starting web server on port {PORT}...")
        start_server()
    else:
        print("❌ Failed to fetch data. Check your internet connection.")