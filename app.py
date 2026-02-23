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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import json

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
    
    positions = []  # Track positions (1 for long, -1 for short, 0 for flat)
    pnl = []  # Track P&L
    timestamps = []
    signals = []
    
    current_position = 0
    cumulative_pnl = 0
    entry_price = 0
    
    threshold = 0.001  # Slope threshold (0.1%)
    
    # Iterate through the data
    for i in range(24, len(prices) - 1):
        current_time = df['timestamp'].iloc[i]
        timestamps.append(current_time)
        
        # Find optimal lookback by minimizing the error function
        best_error = float('inf')
        best_slope = 0
        best_end_idx = i
        
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
                best_end_idx = i
        
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
                    signals.append(('BUY', current_time))
                    
            else:  # Bearish signal
                if current_position >= 0:  # Not short
                    # Close long if exists
                    if current_position == 1:
                        cumulative_pnl += (prices[i] - entry_price) / entry_price * 100
                    
                    # Open short
                    current_position = -1
                    entry_price = prices[i]
                    signals.append(('SELL', current_time))
            
            # Add to P&L based on slope direction
            pnl_adjustment = np.sign(best_slope) * price_change_pct * 100
            cumulative_pnl += pnl_adjustment
        
        positions.append(current_position)
        pnl.append(cumulative_pnl)
    
    return timestamps, positions, pnl, signals

# Create interactive plot
def create_plot(df, timestamps, positions, pnl, signals):
    # Align data lengths
    min_len = min(len(timestamps), len(positions), len(pnl))
    timestamps = timestamps[:min_len]
    positions = positions[:min_len]
    pnl = pnl[:min_len]
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('BTC Price', 'Position', 'Cumulative P&L (%)')
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['close'],
                  mode='lines',
                  name='BTC Price',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Add buy/sell signals
    buy_dates = [sig[1] for sig in signals if sig[0] == 'BUY']
    sell_dates = [sig[1] for sig in signals if sig[0] == 'SELL']
    
    buy_prices = df[df['timestamp'].isin(buy_dates)]['close'].values if buy_dates else []
    sell_prices = df[df['timestamp'].isin(sell_dates)]['close'].values if sell_dates else []
    
    fig.add_trace(
        go.Scatter(x=buy_dates, y=buy_prices,
                  mode='markers',
                  name='Buy',
                  marker=dict(color='green', size=10, symbol='triangle-up')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=sell_dates, y=sell_prices,
                  mode='markers',
                  name='Sell',
                  marker=dict(color='red', size=10, symbol='triangle-down')),
        row=1, col=1
    )
    
    # Position chart
    fig.add_trace(
        go.Scatter(x=timestamps, y=positions,
                  mode='lines+markers',
                  name='Position (1=Long, -1=Short, 0=Flat)',
                  line=dict(color='orange', width=2),
                  marker=dict(size=4)),
        row=2, col=1
    )
    
    # P&L chart
    fig.add_trace(
        go.Scatter(x=timestamps, y=pnl,
                  mode='lines',
                  name='Cumulative P&L',
                  line=dict(color='green', width=2),
                  fill='tozeroy'),
        row=3, col=1
    )
    
    # Update layout for mobile display
    fig.update_layout(
        title=f'{SYMBOL} - 1h Data Analysis',
        height=900,
        showlegend=True,
        hovermode='x unified',
        template='plotly_dark',
        font=dict(size=14)  # Larger font for mobile
    )
    
    # Make y-axis labels bigger
    fig.update_yaxes(title_font=dict(size=14), tickfont=dict(size=12))
    fig.update_xaxes(title_font=dict(size=14), tickfont=dict(size=12))
    
    return fig

# HTML template with viewport for mobile
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>BTC Trading Analysis</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=yes">
    <style>
        body {{
            margin: 0;
            padding: 10px;
            background-color: #111;
            color: white;
            font-family: Arial, sans-serif;
        }}
        .container {{
            max-width: 100%;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 24px;
            text-align: center;
            margin: 10px 0;
        }}
        .chart {{
            width: 100%;
            height: 900px;
        }}
        .info {{
            background-color: #222;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            font-size: 16px;
        }}
        .info-item {{
            margin: 8px 0;
        }}
        @media (max-width: 600px) {{
            h1 {{ font-size: 20px; }}
            .info {{ font-size: 14px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>BTC 1H Trading Analysis</h1>
        <div class="chart" id="chart"></div>
        <div class="info" id="info">
            <div class="info-item">Loading data...</div>
        </div>
    </div>
    
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script>
        const chartData = {chart_json};
        Plotly.newPlot('chart', chartData.data, chartData.layout, {{responsive: true}});
        
        // Update info
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {{
                const infoDiv = document.getElementById('info');
                infoDiv.innerHTML = `
                    <div class="info-item"><strong>Current Position:</strong> ${{data.position}}</div>
                    <div class="info-item"><strong>Total P&L:</strong> ${{data.pnl.toFixed(2)}}%</div>
                    <div class="info-item"><strong>Last Update:</strong> ${{data.last_update}}</div>
                `;
            }});
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
                timestamps, positions, pnl, signals = analyze_and_trade(df)
                fig = create_plot(df, timestamps, positions, pnl, signals)
                
                # Convert figure to JSON
                chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                
                # Send HTML
                html = HTML_TEMPLATE.format(chart_json=chart_json)
                self.wfile.write(html.encode())
                
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Get latest data
            df = fetch_binance_data()
            if df is not None:
                timestamps, positions, pnl, signals = analyze_and_trade(df)
                
                # Get latest values
                current_position = positions[-1] if positions else 0
                current_pnl = pnl[-1] if pnl else 0
                
                # Map position to text
                pos_text = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}.get(current_position, 'UNKNOWN')
                
                status = {
                    'position': pos_text,
                    'pnl': current_pnl,
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.wfile.write(json.dumps(status).encode())
        
        else:
            self.send_response(404)
            self.end_headers()

def start_server():
    with socketserver.TCPServer((HOST, PORT), CustomHandler) as httpd:
        print(f"Server running at http://{HOST}:{PORT}")
        print("Open this URL on your mobile device")
        httpd.serve_forever()

# Main execution
if __name__ == "__main__":
    print("Starting BTC Trading Analysis Server...")
    print(f"Fetching {LIMIT} hours of BTC data from Binance...")
    
    # Test data fetch
    df = fetch_binance_data()
    if df is not None:
        print(f"Successfully fetched {len(df)} data points")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Start server
        print(f"\nStarting web server on port {PORT}...")
        print("You can access the dashboard at:")
        print(f"http://localhost:{PORT} (local)")
        
        # Get local IP
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"http://{local_ip}:{PORT} (network - use this for mobile)")
        
        start_server()
    else:
        print("Failed to fetch data. Check your internet connection.")