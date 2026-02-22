import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import http.server
import socketserver
import io
import base64
from datetime import datetime, timedelta
import requests
import argparse
from sklearn.linear_model import LinearRegression
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
        
    def fetch_binance_data(self):
        """Fetch price data from Binance"""
        base_url = "https://api.binance.com/api/v3/klines"
        
        # Calculate start time
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.days)
        
        # Convert interval to milliseconds
        interval_map = {
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000),
            'limit': 1000
        }
        
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
            
            self.data = df[['timestamp', 'close']].copy()
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Generate sample data for testing
            self.generate_sample_data()
            return False
    
    def generate_sample_data(self):
        """Generate sample data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=24*30, freq='1h')
        # Create a trending price with some noise
        trend = np.linspace(40000, 45000, len(dates))
        noise = np.random.normal(0, 1000, len(dates))
        prices = trend + noise
        
        self.data = pd.DataFrame({
            'timestamp': dates,
            'close': prices
        })
        print("Generated sample data for testing")
    
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
        
        for window in range(min_window, min(max_window, len(prices)) + 1):
            window_prices = prices[-window:]
            model, predicted = self.ols_fit(window_prices)
            error = self.calculate_error(window_prices, predicted)
            
            if error < best_error:
                best_error = error
                best_window = window
                best_line = predicted
                best_slope = model.coef_[0][0]
        
        return best_window, best_line, best_slope, best_error
    
    def analyze_trends(self):
        """Main analysis function"""
        if self.data is None or len(self.data) < 100:
            print("Insufficient data")
            return
        
        prices = self.data['close'].copy()
        timestamps = self.data['timestamp'].copy()
        
        # Start from the end and work backwards
        current_position = len(prices)
        min_window = 10
        
        while current_position > min_window:
            # Get prices up to current position
            current_prices = prices.iloc[:current_position]
            
            # Find optimal window
            window_size, line_values, slope, error = self.find_optimal_window(
                current_prices, 
                min_window=min_window, 
                max_window=min(100, current_position)
            )
            
            # Store the line
            start_idx = current_position - window_size
            end_idx = current_position
            
            line_data = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': timestamps.iloc[start_idx],
                'end_time': timestamps.iloc[end_idx-1],
                'values': line_values.flatten(),
                'slope': slope,
                'color': 'green' if slope > 0 else 'red',
                'window_size': window_size,
                'error': error
            }
            
            self.optimized_lines.append(line_data)
            
            # Calculate trade return
            start_price = prices.iloc[start_idx]
            end_price = prices.iloc[end_idx-1]
            
            if slope > 0:
                trade_return = (end_price - start_price) / start_price
            else:
                trade_return = -1 * ((end_price - start_price) / start_price)
            
            self.trades.append({
                'start_time': timestamps.iloc[start_idx],
                'end_time': timestamps.iloc[end_idx-1],
                'return': trade_return,
                'type': 'long' if slope > 0 else 'short'
            })
            
            # Move position back (remove prices and continue)
            current_position = start_idx
            
            # Check for color change and mark horizontal line
            if len(self.optimized_lines) >= 2:
                prev_line = self.optimized_lines[-2]
                curr_line = self.optimized_lines[-1]
                
                if prev_line['color'] != curr_line['color']:
                    # Mark horizontal line at the transition point
                    self.horizontal_lines.append({
                        'time': timestamps.iloc[start_idx],
                        'price': prices.iloc[start_idx]
                    })
    
    def plot_results(self):
        """Plot all results"""
        if self.data is None:
            print("No data to plot")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price data
        ax1.plot(self.data['timestamp'], self.data['close'], 
                color='blue', alpha=0.5, linewidth=1, label='BTC Price')
        
        # Plot optimized lines
        for line in self.optimized_lines:
            x_values = self.data['timestamp'].iloc[line['start_idx']:line['end_idx']]
            y_values = line['values']
            
            ax1.plot(x_values, y_values, 
                    color=line['color'], 
                    linewidth=2, 
                    alpha=0.7)
        
        # Plot horizontal lines
        for h_line in self.horizontal_lines:
            ax1.axhline(y=h_line['price'], 
                       xmin=0, xmax=1, 
                       color='black', 
                       linestyle='--', 
                       linewidth=1, 
                       alpha=0.5)
            ax1.axvline(x=h_line['time'], 
                       color='black', 
                       linestyle='--', 
                       linewidth=1, 
                       alpha=0.5)
        
        # Formatting
        ax1.set_title(f'{self.symbol} Price Analysis with OLS Trends', fontsize=16)
        ax1.set_ylabel('Price (USDT)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot trade returns
        trade_colors = ['green' if t['return'] > 0 else 'red' for t in self.trades]
        trade_returns = [t['return'] * 100 for t in self.trades]
        trade_times = [t['start_time'] for t in self.trades]
        
        bars = ax2.bar(range(len(trade_returns)), trade_returns, color=trade_colors, alpha=0.7)
        ax2.set_title('Trade Returns (%)', fontsize=16)
        ax2.set_xlabel('Trade Number', fontsize=12)
        ax2.set_ylabel('Return (%)', fontsize=12)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, ret) in enumerate(zip(bars, trade_returns)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ret:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        return fig
    
    def get_html_plot(self):
        """Convert plot to HTML image"""
        fig = self.plot_results()
        if fig is None:
            return "<html><body><h1>No data available</h1></body></html>"
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Create HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BTC Trend Analysis</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                img {{
                    width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .stats {{
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .positive {{
                    color: green;
                    font-weight: bold;
                }}
                .negative {{
                    color: red;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>BTC Trend Analysis - {self.interval} Interval</h1>
                <img src="data:image/png;base64,{image_base64}" alt="BTC Trend Analysis">
                
                <div class="stats">
                    <h2>Analysis Statistics</h2>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Data Points</td>
                            <td>{len(self.data)}</td>
                        </tr>
                        <tr>
                            <td>Optimized Lines</td>
                            <td>{len(self.optimized_lines)}</td>
                        </tr>
                        <tr>
                            <td>Trend Changes</td>
                            <td>{len(self.horizontal_lines)}</td>
                        </tr>
                        <tr>
                            <td>Trades Executed</td>
                            <td>{len(self.trades)}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="stats">
                    <h2>Trade Performance</h2>
                    <table>
                        <tr>
                            <th>Start Time</th>
                            <th>End Time</th>
                            <th>Type</th>
                            <th>Return</th>
                        </tr>
        """
        
        # Add trade rows
        for trade in self.trades[-10:]:  # Show last 10 trades
            return_class = 'positive' if trade['return'] > 0 else 'negative'
            html += f"""
                        <tr>
                            <td>{trade['start_time']}</td>
                            <td>{trade['end_time']}</td>
                            <td>{trade['type'].upper()}</td>
                            <td class="{return_class}">{trade['return']*100:.2f}%</td>
                        </tr>
            """
        
        html += """
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    analyzer = None
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            if self.analyzer:
                html = self.analyzer.get_html_plot()
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))
            else:
                self.send_response(500)
                self.end_headers()
        else:
            super().do_GET()

def main():
    parser = argparse.ArgumentParser(description='BTC Trend Analysis with OLS')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--interval', type=str, default='1h', help='Time interval')
    parser.add_argument('--days', type=int, default=30, help='Number of days')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    
    args = parser.parse_args()
    
    print(f"Starting BTC Trend Analysis for {args.symbol}")
    print(f"Interval: {args.interval}, Days: {args.days}")
    
    # Create analyzer
    analyzer = BTCTrendAnalyzer(args.symbol, args.interval, args.days)
    
    # Fetch data
    print("Fetching data from Binance...")
    success = analyzer.fetch_binance_data()
    
    if success:
        print("Data fetched successfully")
    else:
        print("Using sample data for demonstration")
    
    # Analyze trends
    print("Analyzing trends...")
    analyzer.analyze_trends()
    
    print(f"Found {len(analyzer.optimized_lines)} optimized lines")
    print(f"Found {len(analyzer.horizontal_lines)} trend changes")
    
    # Setup HTTP server
    CustomHTTPRequestHandler.analyzer = analyzer
    handler = CustomHTTPRequestHandler
    
    # Create server
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"\nServer running at http://localhost:{args.port}")
        print("Press Ctrl+C to stop")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    main()