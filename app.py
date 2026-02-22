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

class BTCOptimalLookback:
    def __init__(self):
        self.data = None
        self.timestamps = None
        self.prices = None
        
    def fetch(self) -> bool:
        """Fetch BTC 1h data for last 30 days from Binance"""
        try:
            # Calculate timestamps for last 30 days
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            
            # Binance API URL for klines/candlestick data
            url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&startTime={start_time}&endTime={end_time}&limit=1000"
            
            # Fetch data
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
            
            # Extract timestamps and closing prices
            self.timestamps = np.array([int(item[0]) for item in data]).reshape(-1, 1)
            self.prices = np.array([float(item[4]) for item in data])  # Closing price
            
            print(f"Fetched {len(self.prices)} hours of BTC data")
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Generate sample data for testing if API fails
            self.generate_sample_data()
            return False
    
    def generate_sample_data(self):
        """Generate sample data for testing when API is unavailable"""
        np.random.seed(42)
        n_points = 720  # 30 days * 24 hours
        
        # Generate timestamps
        base_time = datetime.now().timestamp() * 1000
        self.timestamps = np.array([base_time + i * 3600000 for i in range(n_points)]).reshape(-1, 1)
        
        # Generate synthetic BTC price with trend and noise
        trend = np.linspace(40000, 45000, n_points)
        noise = np.random.normal(0, 500, n_points)
        self.prices = trend + noise
        
        print(f"Generated {n_points} hours of sample data")
    
    def ols(self, lookback: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Perform OLS regression on the last 'lookback' data points
        Returns: X, y_pred, total_error, avg_error_per_candle"""
        if lookback > len(self.prices):
            lookback = len(self.prices)
        
        # Get last 'lookback' data points
        X = self.timestamps[-lookback:]
        y = self.prices[-lookback:]
        
        # Normalize timestamps to avoid numerical issues
        X_normalized = (X - X.mean()) / X.std()
        
        # Fit OLS
        model = LinearRegression()
        model.fit(X_normalized, y)
        
        # Predict
        y_pred = model.predict(X_normalized)
        
        # Calculate total error (sum of absolute differences)
        total_error = np.sum(np.abs(y - y_pred))
        
        # Calculate average error per candle
        avg_error_per_candle = total_error / lookback
        
        return X.flatten(), y_pred, total_error, avg_error_per_candle
    
    def iterate(self) -> Tuple[int, np.ndarray, np.ndarray, float, float]:
        """Iterate through lookback windows from 10 to 100 and find best average error"""
        best_avg_error = float('inf')
        best_lookback = 10
        best_X = None
        best_y_pred = None
        best_total_error = None
        
        errors = []  # Store (lookback, total_error, avg_error)
        
        print("Calculating OLS for lookback windows 10-100...")
        print("-" * 60)
        print(f"{'Lookback':^10} | {'Total Error':^15} | {'Avg Error/Candle':^18}")
        print("-" * 60)
        
        for lookback in range(10, 101):
            if lookback > len(self.prices):
                break
                
            X, y_pred, total_error, avg_error = self.ols(lookback)
            errors.append((lookback, total_error, avg_error))
            
            print(f"{lookback:^10} | ${total_error:>13,.2f} | ${avg_error:>16,.2f}")
            
            if avg_error < best_avg_error:
                best_avg_error = avg_error
                best_lookback = lookback
                best_X = X
                best_y_pred = y_pred
                best_total_error = total_error
        
        print("-" * 60)
        print(f"\n✅ Best lookback window: {best_lookback} hours")
        print(f"   Total Error: ${best_total_error:,.2f}")
        print(f"   Average Error per Candle: ${best_avg_error:,.2f}")
        
        # Print top 5 lookback windows by average error
        sorted_errors = sorted(errors, key=lambda x: x[2])[:5]
        print("\n🏆 Top 5 lookback windows (by average error per candle):")
        print(f"{'Rank':^6} | {'Lookback':^10} | {'Total Error':^15} | {'Avg Error/Candle':^18}")
        print("-" * 60)
        for rank, (lb, total_err, avg_err) in enumerate(sorted_errors, 1):
            print(f"{rank:^6} | {lb:^10} | ${total_err:>13,.2f} | ${avg_err:>16,.2f}")
        
        return best_lookback, best_X, best_y_pred, best_total_error, best_avg_error
    
    def plot(self, best_lookback: int, best_X: np.ndarray, best_y_pred: np.ndarray, 
             best_total_error: float, best_avg_error: float) -> str:
        """Create plot and return as base64 encoded string"""
        plt.figure(figsize=(14, 10))
        
        # Create subplot for price chart
        ax1 = plt.subplot(2, 1, 1)
        
        # Plot all prices
        ax1.plot(range(len(self.prices)), self.prices, 'b-', alpha=0.5, 
                label='BTC Price (All Data)', linewidth=1)
        
        # Highlight the best OLS segment
        if best_X is not None and best_y_pred is not None:
            # Find indices for the best lookback window
            start_idx = len(self.prices) - len(best_X)
            x_indices = range(start_idx, len(self.prices))
            
            ax1.plot(x_indices, best_y_pred, 'r-', linewidth=3, 
                    label=f'Best OLS Line (lookback={best_lookback})')
            
            # Highlight the data points used
            ax1.scatter(x_indices, self.prices[start_idx:], c='orange', s=30, 
                       alpha=0.7, label='Data Points in Best Window')
        
        ax1.set_title(f'BTC Price (1h) - Last 30 Days\nBest OLS Lookback: {best_lookback} hours')
        ax1.set_xlabel('Time (hours from now)')
        ax1.set_ylabel('Price (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add price statistics
        if len(self.prices) > 0:
            current_price = self.prices[-1]
            min_price = np.min(self.prices)
            max_price = np.max(self.prices)
            stats_text = f'Current: ${current_price:,.2f}\nMin: ${min_price:,.2f}\nMax: ${max_price:,.2f}'
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Create subplot for error analysis
        ax2 = plt.subplot(2, 1, 2)
        
        # Calculate errors for all lookback windows
        lookbacks = []
        avg_errors = []
        total_errors = []
        
        for lookback in range(10, min(101, len(self.prices) + 1)):
            _, _, total_error, avg_error = self.ols(lookback)
            lookbacks.append(lookback)
            avg_errors.append(avg_error)
            total_errors.append(total_error)
        
        # Plot average error
        ax2.plot(lookbacks, avg_errors, 'g-', linewidth=2, label='Avg Error per Candle')
        ax2.scatter([best_lookback], [best_avg_error], c='red', s=100, zorder=5, 
                   label=f'Best: {best_lookback} (${best_avg_error:,.2f})')
        
        ax2.set_xlabel('Lookback Window (hours)')
        ax2.set_ylabel('Average Error per Candle ($)')
        ax2.set_title('Error Analysis: Average Absolute Deviation per Candle')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Encode as base64
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return image_base64

class BTCRequestHandler(http.server.SimpleHTTPRequestHandler):
    btc_analyzer = None
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            if BTCRequestHandler.btc_analyzer and BTCRequestHandler.btc_analyzer.prices is not None:
                best_lookback, best_X, best_y_pred, best_total_error, best_avg_error = BTCRequestHandler.btc_analyzer.iterate()
                image_base64 = BTCRequestHandler.btc_analyzer.plot(best_lookback, best_X, best_y_pred, 
                                                                  best_total_error, best_avg_error)
                
                # Calculate some additional statistics
                prices = BTCRequestHandler.btc_analyzer.prices
                volatility = np.std(prices[-24:])  # 24h volatility
                
                # Create HTML page
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>BTC OLS Analysis - Average Error per Candle</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }}
                        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                        h1 {{ color: #333; }}
                        h2 {{ color: #666; }}
                        .info {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                        .stats {{ display: flex; justify-content: space-between; flex-wrap: wrap; }}
                        .stat-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; flex: 1; margin: 10px; min-width: 200px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                        .stat-box:nth-child(2) {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
                        .stat-box:nth-child(3) {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
                        .stat-box:nth-child(4) {{ background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }}
                        .stat-value {{ font-size: 28px; font-weight: bold; margin: 10px 0; }}
                        .stat-label {{ font-size: 14px; opacity: 0.9; }}
                        .stat-sub {{ font-size: 12px; margin-top: 5px; opacity: 0.8; }}
                        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 10px; margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                        .footer {{ margin-top: 30px; color: #666; font-size: 12px; text-align: center; padding-top: 20px; border-top: 1px solid #eee; }}
                        .error-metric {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>📊 BTC OLS Analysis - Average Error per Candle</h1>
                        <p>30 Days of 1-Hour Data | Optimizing for minimum average absolute deviation</p>
                        
                        <div class="error-metric">
                            <strong>🎯 Optimization Metric:</strong> Using <strong>Average Error per Candle</strong> (Total Error / Lookback Period) 
                            to fairly compare different lookback windows regardless of their length.
                        </div>
                        
                        <div class="info">
                            <h3>📈 Best Result:</h3>
                            <p><strong>Optimal Lookback Window:</strong> {best_lookback} hours</p>
                            <p><strong>Total Error (Sum |price-ols|):</strong> ${best_total_error:,.2f}</p>
                            <p><strong>Average Error per Candle:</strong> ${best_avg_error:,.2f}</p>
                            <p><strong>Total Data Points:</strong> {len(prices)} hours</p>
                        </div>
                        
                        <div class="stats">
                            <div class="stat-box">
                                <div class="stat-label">Current BTC Price</div>
                                <div class="stat-value">${prices[-1]:,.2f}</div>
                                <div class="stat-sub">as of now</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">24h Volatility (Std)</div>
                                <div class="stat-value">${volatility:,.2f}</div>
                                <div class="stat-sub">price fluctuation</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">30d Min</div>
                                <div class="stat-value">${np.min(prices):,.2f}</div>
                                <div class="stat-sub">lowest price</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">30d Max</div>
                                <div class="stat-value">${np.max(prices):,.2f}</div>
                                <div class="stat-sub">highest price</div>
                            </div>
                        </div>
                        
                        <img src="data:image/png;base64,{image_base64}" alt="BTC Price and Error Analysis">
                        
                        <div class="footer">
                            <p>Data from Binance API | Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                            <p>Analysis looks for lookback window that minimizes average absolute deviation per candle</p>
                        </div>
                    </div>
                </body>
                </html>
                """
                self.wfile.write(html.encode())
            else:
                self.wfile.write(b"<html><body><h1>Error: No data available</h1></body></html>")
        else:
            self.send_response(404)
            self.end_headers()

def main():
    """Main function to run the analysis and server"""
    print("=" * 70)
    print("BTC OLS Analysis - Finding optimal lookback window (avg error per candle)")
    print("=" * 70)
    
    # Create analyzer instance
    analyzer = BTCOptimalLookback()
    
    # Fetch data
    print("\n📡 Fetching BTC data from Binance...")
    success = analyzer.fetch()
    if not success:
        print("⚠️  Using generated sample data (Binance API unavailable)")
    
    # Store analyzer in handler class
    BTCRequestHandler.btc_analyzer = analyzer
    
    # Start server
    PORT = 8080
    handler = BTCRequestHandler
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"\n🚀 Server started at http://localhost:{PORT}")
        print("Press Ctrl+C to stop the server\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped. Goodbye!")

if __name__ == "__main__":
    main()