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
    
    def ols(self, lookback: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Perform OLS regression on the last 'lookback' data points"""
        if lookback > len(self.prices):
            lookback = len(self.prices)
        
        # Get last 'lookback' data points
        X = self.timestamps[-lookback:]
        y = self.prices[-lookback:]
        
        # Reshape X for sklearn
        X_reshaped = X.reshape(-1, 1)
        
        # Fit OLS
        model = LinearRegression()
        model.fit(X_reshaped, y)
        
        # Predict
        y_pred = model.predict(X_reshaped)
        
        # Calculate error (sum of absolute differences)
        error = np.sum(np.abs(y - y_pred))
        
        return X.flatten(), y_pred, error
    
    def iterate(self) -> Tuple[int, np.ndarray, np.ndarray, float]:
        """Iterate through lookback windows from 10 to 100 and find best error"""
        best_error = float('inf')
        best_lookback = 10
        best_X = None
        best_y_pred = None
        
        errors = []
        
        print("Calculating OLS for lookback windows 10-100...")
        
        for lookback in range(10, 101):
            if lookback > len(self.prices):
                break
                
            X, y_pred, error = self.ols(lookback)
            errors.append((lookback, error))
            
            if error < best_error:
                best_error = error
                best_lookback = lookback
                best_X = X
                best_y_pred = y_pred
        
        print(f"Best lookback window: {best_lookback} with error: {best_error:.2f}")
        
        # Print top 5 lookback windows
        sorted_errors = sorted(errors, key=lambda x: x[1])[:5]
        print("\nTop 5 lookback windows:")
        for lb, err in sorted_errors:
            print(f"  Lookback {lb}: error = {err:.2f}")
        
        return best_lookback, best_X, best_y_pred, best_error
    
    def plot(self, best_lookback: int, best_X: np.ndarray, best_y_pred: np.ndarray) -> str:
        """Create plot and return as base64 encoded string"""
        plt.figure(figsize=(14, 8))
        
        # Plot all prices
        plt.plot(range(len(self.prices)), self.prices, 'b-', alpha=0.5, label='BTC Price (All Data)', linewidth=1)
        
        # Highlight the best OLS segment
        if best_X is not None and best_y_pred is not None:
            # Find indices for the best lookback window
            start_idx = len(self.prices) - len(best_X)
            x_indices = range(start_idx, len(self.prices))
            
            plt.plot(x_indices, best_y_pred, 'r-', linewidth=3, 
                    label=f'Best OLS Line (lookback={best_lookback})')
            
            # Highlight the data points used
            plt.scatter(x_indices, self.prices[start_idx:], c='orange', s=30, 
                       alpha=0.7, label='Data Points in Best Window')
        
        plt.title(f'BTC Price (1h) - Last 30 Days\nBest OLS Lookback: {best_lookback} hours')
        plt.xlabel('Time (hours from now)')
        plt.ylabel('Price (USDT)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add price statistics
        if len(self.prices) > 0:
            current_price = self.prices[-1]
            min_price = np.min(self.prices)
            max_price = np.max(self.prices)
            stats_text = f'Current: ${current_price:.2f}\nMin: ${min_price:.2f}\nMax: ${max_price:.2f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
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
                best_lookback, best_X, best_y_pred, best_error = BTCRequestHandler.btc_analyzer.iterate()
                image_base64 = BTCRequestHandler.btc_analyzer.plot(best_lookback, best_X, best_y_pred)
                
                # Create HTML page
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>BTC OLS Analysis</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }}
                        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                        h1 {{ color: #333; }}
                        .info {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                        .stats {{ display: flex; justify-content: space-between; }}
                        .stat-box {{ background-color: #4CAF50; color: white; padding: 15px; border-radius: 5px; flex: 1; margin: 0 10px; text-align: center; }}
                        .stat-box:first-child {{ margin-left: 0; }}
                        .stat-box:last-child {{ margin-right: 0; }}
                        .stat-value {{ font-size: 24px; font-weight: bold; }}
                        .stat-label {{ font-size: 14px; margin-top: 5px; }}
                        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; margin-top: 20px; }}
                        .footer {{ margin-top: 20px; color: #666; font-size: 12px; text-align: center; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>📈 BTC OLS Analysis - 30 Days 1-Hour Data</h1>
                        <div class="info">
                            <h3>Analysis Results:</h3>
                            <p><strong>Best Lookback Window:</strong> {best_lookback} hours</p>
                            <p><strong>Best Error (Sum |price-ols|):</strong> ${best_error:.2f}</p>
                            <p><strong>Total Data Points:</strong> {len(BTCRequestHandler.btc_analyzer.prices)} hours</p>
                        </div>
                        
                        <div class="stats">
                            <div class="stat-box">
                                <div class="stat-value">${BTCRequestHandler.btc_analyzer.prices[-1]:.2f}</div>
                                <div class="stat-label">Current Price</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-value">${np.min(BTCRequestHandler.btc_analyzer.prices):.2f}</div>
                                <div class="stat-label">Min Price (30d)</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-value">${np.max(BTCRequestHandler.btc_analyzer.prices):.2f}</div>
                                <div class="stat-label">Max Price (30d)</div>
                            </div>
                        </div>
                        
                        <img src="data:image/png;base64,{image_base64}" alt="BTC Price Chart">
                        
                        <div class="footer">
                            <p>Data from Binance API | Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
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
    print("=" * 60)
    print("BTC OLS Analysis - Fetching data and finding optimal lookback window")
    print("=" * 60)
    
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