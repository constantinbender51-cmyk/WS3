import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import http.server
import socketserver
import io
import base64
from datetime import datetime, timedelta
import requests
import argparse
from sklearn.linear_model import LinearRegression
import time
import threading
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BTCTrendAnalyzer:
    def __init__(self, symbol='BTCUSDT', interval='1h', days=30):
        self.symbol = symbol
        self.interval = interval
        self.days = days
        self.data = None
        self.optimized_lines = []
        self.horizontal_lines = []
        self.trades = []
        self.live_trades = []  # Store live trading results
        self.current_position = None
        self.current_signal = None
        self.last_update = None
        self.position_open_time = None
        self.position_open_price = None
        self.pnl_history = []
        self.total_pnl = 0
        
    def fetch_binance_data(self, limit=1000, start_time=None):
        """Fetch price data from Binance"""
        base_url = "https://api.binance.com/api/v3/klines"
        
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close'] = df['close'].astype(float)
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            
            return df[['timestamp', 'close']].copy()
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def fetch_current_price(self):
        """Fetch current BTC price"""
        try:
            url = "https://api.binance.com/api/v3/ticker/price"
            params = {'symbol': self.symbol}
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return float(data['price'])
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return None
    
    def ols_fit(self, prices):
        """Fit OLS line to price data"""
        x = np.arange(len(prices)).reshape(-1, 1)
        y = prices.values.reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(x, y)
        
        return model, model.predict(x)
    
    def calculate_error(self, prices, predicted_prices):
        """Calculate error term: sum(|line-price| / prices^2)"""
        errors = np.abs(predicted_prices.flatten() - prices.values) / (prices.values ** 2)
        return np.sum(errors)
    
    def find_optimal_window(self, prices, min_window=10, max_window=100):
        """Find optimal number of prices for OLS fit"""
        best_error = float('inf')
        best_window = min_window
        best_line = None
        best_slope = 0
        best_model = None
        
        for window in range(min_window, min(max_window, len(prices)) + 1):
            window_prices = prices[-window:]
            model, predicted = self.ols_fit(window_prices)
            error = self.calculate_error(window_prices, predicted)
            
            if error < best_error:
                best_error = error
                best_window = window
                best_line = predicted
                best_slope = model.coef_[0][0]
                best_model = model
        
        return best_window, best_line, best_slope, best_error, best_model
    
    def get_current_signal(self):
        """Get current trading signal based on latest data"""
        if self.data is None or len(self.data) < 100:
            return None
        
        prices = self.data['close'].copy()
        
        # Find optimal window for most recent data
        window_size, line_values, slope, error, model = self.find_optimal_window(
            prices, min_window=10, max_window=100
        )
        
        # Determine signal
        signal = 'long' if slope > 0 else 'short'
        
        # Get line values for the most recent points
        recent_prices = prices[-window_size:]
        x_recent = np.arange(len(recent_prices)).reshape(-1, 1)
        line_recent = model.predict(x_recent).flatten()
        
        return {
            'signal': signal,
            'slope': slope,
            'window_size': window_size,
            'error': error,
            'current_price': prices.iloc[-1],
            'line_price': line_recent[-1],
            'timestamp': self.data['timestamp'].iloc[-1],
            'model': model,
            'recent_prices': recent_prices,
            'line_values': line_recent
        }
    
    def update_live_data(self):
        """Fetch latest data and update analysis"""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updating live data...")
        
        # Fetch latest 200 candles
        new_data = self.fetch_binance_data(limit=200)
        
        if new_data is not None:
            self.data = new_data
            self.last_update = datetime.now()
            
            # Get current signal
            signal_info = self.get_current_signal()
            
            if signal_info:
                self.manage_position(signal_info)
                
            print(f"Update complete. Current price: ${signal_info['current_price']:,.2f}")
            print(f"Signal: {signal_info['signal'].upper()} (slope: {signal_info['slope']:.6f})")
            
            return True
        return False
    
    def manage_position(self, signal_info):
        """Manage trading positions based on signals"""
        current_signal = signal_info['signal']
        current_price = signal_info['current_price']
        current_time = signal_info['timestamp']
        
        # If no position, open one
        if self.current_position is None:
            self.open_position(current_signal, current_price, current_time)
            return
        
        # Check for signal change
        if current_signal != self.current_signal:
            self.close_position(current_price, current_time)
            self.open_position(current_signal, current_price, current_time)
    
    def open_position(self, signal, price, timestamp):
        """Open a new trading position"""
        self.current_position = signal
        self.current_signal = signal
        self.position_open_time = timestamp
        self.position_open_price = price
        
        print(f"\n📈 OPENING {signal.upper()} POSITION")
        print(f"   Time: {timestamp}")
        print(f"   Price: ${price:,.2f}")
    
    def close_position(self, close_price, close_time):
        """Close current position and calculate PnL"""
        if self.current_position is None:
            return
        
        # Calculate PnL
        if self.current_position == 'long':
            pnl = (close_price - self.position_open_price) / self.position_open_price * 100
        else:  # short
            pnl = (self.position_open_price - close_price) / self.position_open_price * 100
        
        self.total_pnl += pnl
        
        # Record trade
        trade = {
            'open_time': self.position_open_time,
            'close_time': close_time,
            'position': self.current_position,
            'open_price': self.position_open_price,
            'close_price': close_price,
            'pnl_percent': pnl,
            'cumulative_pnl': self.total_pnl
        }
        self.live_trades.append(trade)
        self.pnl_history.append(self.total_pnl)
        
        print(f"\n📉 CLOSING {self.current_position.upper()} POSITION")
        print(f"   Open Time: {self.position_open_time}")
        print(f"   Close Time: {close_time}")
        print(f"   Open Price: ${self.position_open_price:,.2f}")
        print(f"   Close Price: ${close_price:,.2f}")
        print(f"   PnL: {pnl:+.2f}%")
        print(f"   Total PnL: {self.total_pnl:+.2f}%")
        
        # Reset position
        self.current_position = None
    
    def run_live_trading(self, check_interval=3601):  # 1 hour + 1 second
        """Run live trading loop"""
        print("\n" + "="*60)
        print("STARTING LIVE TRADING SESSION")
        print("="*60)
        
        # Initial data fetch
        print("Fetching initial data...")
        self.update_live_data()
        
        # Main trading loop
        while True:
            try:
                # Wait for next interval
                next_check = datetime.now() + timedelta(seconds=check_interval)
                print(f"\nNext update at: {next_check.strftime('%Y-%m-%d %H:%M:%S')}")
                
                time.sleep(check_interval)
                
                # Update data and manage positions
                self.update_live_data()
                
                # Save trade history periodically
                self.save_trade_history()
                
            except KeyboardInterrupt:
                print("\n\nStopping live trading...")
                # Close any open position
                if self.current_position is not None:
                    current_price = self.fetch_current_price()
                    if current_price:
                        self.close_position(current_price, datetime.now())
                self.save_trade_history()
                break
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def save_trade_history(self):
        """Save trade history to file"""
        history_file = Path('trade_history.json')
        
        history = {
            'total_pnl': self.total_pnl,
            'trades': [
                {
                    'open_time': t['open_time'].isoformat() if isinstance(t['open_time'], datetime) else str(t['open_time']),
                    'close_time': t['close_time'].isoformat() if isinstance(t['close_time'], datetime) else str(t['close_time']),
                    'position': t['position'],
                    'open_price': float(t['open_price']),
                    'close_price': float(t['close_price']),
                    'pnl_percent': float(t['pnl_percent']),
                    'cumulative_pnl': float(t['cumulative_pnl'])
                }
                for t in self.live_trades
            ],
            'pnl_history': [float(p) for p in self.pnl_history],
            'last_update': datetime.now().isoformat()
        }
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nTrade history saved to {history_file}")
    
    def load_trade_history(self):
        """Load trade history from file"""
        history_file = Path('trade_history.json')
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            self.total_pnl = history.get('total_pnl', 0)
            self.pnl_history = history.get('pnl_history', [])
            
            # Convert string times back to datetime
            self.live_trades = []
            for t in history.get('trades', []):
                t['open_time'] = datetime.fromisoformat(t['open_time'])
                t['close_time'] = datetime.fromisoformat(t['close_time'])
                self.live_trades.append(t)
            
            print(f"Loaded {len(self.live_trades)} previous trades")
            print(f"Previous total PnL: {self.total_pnl:+.2f}%")

class TradingServer:
    def __init__(self, analyzer, port=8080, host='localhost'):
        self.analyzer = analyzer
        self.port = port
        self.host = host
        self.server = None
    
    def get_html_dashboard(self):
        """Generate HTML dashboard with live trading data"""
        if self.analyzer.data is None:
            return "<html><body><h1>No data available</h1></body></html>"
        
        # Create plot
        fig = self.create_live_plot()
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Get current signal
        signal_info = self.analyzer.get_current_signal()
        
        # Create HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BTC Live Trading Dashboard</title>
            <meta http-equiv="refresh" content="60">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 20px;
                    background-color: #f0f2f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .card {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .signal-box {{
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    font-size: 24px;
                    font-weight: bold;
                }}
                .long {{
                    background-color: #d4edda;
                    color: #155724;
                    border: 2px solid #28a745;
                }}
                .short {{
                    background-color: #f8d7da;
                    color: #721c24;
                    border: 2px solid #dc3545;
                }}
                .price {{
                    font-size: 36px;
                    font-weight: bold;
                    text-align: center;
                    margin: 10px 0;
                }}
                .pnl-positive {{
                    color: #28a745;
                    font-weight: bold;
                }}
                .pnl-negative {{
                    color: #dc3545;
                    font-weight: bold;
                }}
                .chart-container {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: white;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                th {{
                    background-color: #343a40;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                td {{
                    padding: 12px;
                    border-bottom: 1px solid #dee2e6;
                }}
                tr:hover {{
                    background-color: #f8f9fa;
                }}
                .last-updated {{
                    text-align: right;
                    color: #6c757d;
                    font-size: 14px;
                    margin-top: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 BTC Live Trading Dashboard</h1>
                    <p>Real-time trend analysis and automated trading</p>
                </div>
                
                <div class="dashboard-grid">
                    <div class="card">
                        <h3>Current Signal</h3>
                        <div class="signal-box {signal_info['signal']}">
                            {signal_info['signal'].upper()}
                        </div>
                        <div class="price">${signal_info['current_price']:,.2f}</div>
                        <p>Slope: {signal_info['slope']:.6f}</p>
                        <p>Window Size: {signal_info['window_size']}</p>
                    </div>
                    
                    <div class="card">
                        <h3>Position Status</h3>
                        <p>Current Position: <strong>{self.analyzer.current_position or 'None'}</strong></p>
                        <p>Open Price: ${self.analyzer.position_open_price:,.2f if self.analyzer.position_open_price else 'N/A'}</p>
                        <p>Open Time: {self.analyzer.position_open_time.strftime('%Y-%m-%d %H:%M') if self.analyzer.position_open_time else 'N/A'}</p>
                    </div>
                    
                    <div class="card">
                        <h3>Performance</h3>
                        <p>Total PnL: <span class="{'pnl-positive' if self.analyzer.total_pnl >= 0 else 'pnl-negative'}">
                            {self.analyzer.total_pnl:+.2f}%
                        </span></p>
                        <p>Trades Closed: {len(self.analyzer.live_trades)}</p>
                        <p>Last Update: {self.analyzer.last_update.strftime('%H:%M:%S') if self.analyzer.last_update else 'Never'}</p>
                    </div>
                </div>
                
                <div class="chart-container">
                    <img src="data:image/png;base64,{image_base64}" alt="BTC Chart" style="width: 100%;">
                </div>
                
                <h2>Trade History</h2>
                <table>
                    <tr>
                        <th>Open Time</th>
                        <th>Close Time</th>
                        <th>Position</th>
                        <th>Open Price</th>
                        <th>Close Price</th>
                        <th>PnL %</th>
                        <th>Cumulative PnL</th>
                    </tr>
        """
        
        # Add trade rows
        for trade in reversed(self.analyzer.live_trades[-20:]):  # Show last 20 trades
            pnl_class = 'pnl-positive' if trade['pnl_percent'] >= 0 else 'pnl-negative'
            html += f"""
                    <tr>
                        <td>{trade['open_time'].strftime('%Y-%m-%d %H:%M')}</td>
                        <td>{trade['close_time'].strftime('%Y-%m-%d %H:%M')}</td>
                        <td>{trade['position'].upper()}</td>
                        <td>${trade['open_price']:,.2f}</td>
                        <td>${trade['close_price']:,.2f}</td>
                        <td class="{pnl_class}">{trade['pnl_percent']:+.2f}%</td>
                        <td class="{pnl_class}">{trade['cumulative_pnl']:+.2f}%</td>
                    </tr>
            """
        
        html += f"""
                </table>
                
                <div class="last-updated">
                    Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    | Auto-refresh every 60 seconds
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def create_live_plot(self):
        """Create plot with live trading data"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price data
        ax1.plot(self.analyzer.data['timestamp'], self.analyzer.data['close'], 
                color='blue', alpha=0.3, linewidth=1, label='BTC Price')
        
        # Plot current optimal line
        signal_info = self.analyzer.get_current_signal()
        if signal_info:
            # Plot the most recent optimal window
            recent_times = self.analyzer.data['timestamp'].iloc[-signal_info['window_size']:]
            ax1.plot(recent_times, signal_info['line_values'], 
                    color='green' if signal_info['signal'] == 'long' else 'red',
                    linewidth=3, alpha=0.8, 
                    label=f"Current {signal_info['signal'].upper()} Signal")
        
        # Plot trade markers
        for trade in self.analyzer.live_trades:
            # Mark entry
            ax1.scatter(trade['open_time'], trade['open_price'], 
                       color='green' if trade['position'] == 'long' else 'red',
                       s=100, marker='^', zorder=5)
            # Mark exit
            ax1.scatter(trade['close_time'], trade['close_price'],
                       color='red' if trade['position'] == 'long' else 'green',
                       s=100, marker='v', zorder=5)
        
        # Plot current position if open
        if self.analyzer.current_position:
            ax1.axhline(y=self.analyzer.position_open_price, 
                       color='yellow', linestyle='--', alpha=0.5,
                       label=f"Open Position @ ${self.analyzer.position_open_price:,.0f}")
        
        ax1.set_title(f'{self.analyzer.symbol} Live Trading - {self.analyzer.interval} Interval', fontsize=16)
        ax1.set_ylabel('Price (USDT)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot cumulative PnL
        if self.analyzer.pnl_history:
            trades_x = list(range(len(self.analyzer.pnl_history)))
            ax2.plot(trades_x, self.analyzer.pnl_history, 
                    color='blue', linewidth=2, marker='o')
            ax2.fill_between(trades_x, 0, self.analyzer.pnl_history,
                            where=np.array(self.analyzer.pnl_history) >= 0,
                            color='green', alpha=0.3)
            ax2.fill_between(trades_x, 0, self.analyzer.pnl_history,
                            where=np.array(self.analyzer.pnl_history) < 0,
                            color='red', alpha=0.3)
        
        ax2.set_title('Cumulative PnL (%)', fontsize=16)
        ax2.set_xlabel('Trade Number', fontsize=12)
        ax2.set_ylabel('PnL (%)', fontsize=12)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def start_server(self):
        """Start the HTTP server"""
        handler = self.create_handler()
        
        try:
            with socketserver.TCPServer((self.host, self.port), handler) as httpd:
                print(f"\n📊 Dashboard running at http://{self.host}:{self.port}")
                print("   Refresh every 60 seconds")
                httpd.serve_forever()
        except Exception as e:
            print(f"Server error: {e}")
    
    def create_handler(self):
        """Create a custom request handler"""
        class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/index.html':
                    html = self.server.get_html_dashboard()
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(html.encode('utf-8'))
                else:
                    super().do_GET()
        
        return CustomHTTPRequestHandler

def main():
    parser = argparse.ArgumentParser(description='BTC Live Trading with Trend Analysis')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--interval', type=str, default='1h', help='Time interval')
    parser.add_argument('--days', type=int, default=30, help='Historical days for initial data')
    parser.add_argument('--port', type=int, default=8080, help='Dashboard port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host IP address')
    parser.add_argument('--live', action='store_true', help='Enable live trading')
    parser.add_argument('--check-interval', type=int, default=3601, 
                       help='Seconds between checks (default: 3601 = 1 hour + 1 sec)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("BTC LIVE TRADING SYSTEM")
    print("="*60)
    print(f"Symbol: {args.symbol}")
    print(f"Interval: {args.interval}")
    print(f"Live Trading: {'Enabled' if args.live else 'Disabled'}")
    
    # Create analyzer
    analyzer = BTCTrendAnalyzer(args.symbol, args.interval, args.days)
    
    # Load previous trade history
    analyzer.load_trade_history()
    
    # Initial data fetch
    print("\nFetching initial data...")
    analyzer.data = analyzer.fetch_binance_data(limit=1000)
    
    if analyzer.data is None:
        print("Error fetching data. Exiting.")
        return
    
    print(f"Loaded {len(analyzer.data)} candles")
    
    # Start live trading in a separate thread if enabled
    if args.live:
        trading_thread = threading.Thread(target=analyzer.run_live_trading, 
                                        args=(args.check_interval,),
                                        daemon=True)
        trading_thread.start()
        print("\n✅ Live trading thread started")
    
    # Start web server
    server = TradingServer(analyzer, args.port, args.host)
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        if args.live:
            analyzer.save_trade_history()

if __name__ == "__main__":
    main()