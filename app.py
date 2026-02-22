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
warnings.filterwarnings('ignore')

class BTCOptimalLookback:
    def __init__(self):
        self.data = None
        self.timestamps = None
        self.prices = None
        self.slope_history = []
        self.lookback_history = []
        self.error_history = []
        self.endpoint_indices = []  # Store the indices we analyzed
        self.full_start_date = None
        self.full_end_date = None
        
    def fetch(self) -> bool:
        """Fetch BTC 1h data for last 30 days from Binance"""
        try:
            # Calculate timestamps for last 30 days
            self.full_end_date = datetime.now()
            self.full_start_date = self.full_end_date - timedelta(days=30)
            
            end_time = int(self.full_end_date.timestamp() * 1000)
            start_time = int(self.full_start_date.timestamp() * 1000)
            
            # Binance API URL for klines/candlestick data
            url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&startTime={start_time}&endTime={end_time}&limit=1000"
            
            # Fetch data
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
            
            # Extract timestamps and closing prices
            self.timestamps = np.array([int(item[0]) for item in data]).reshape(-1, 1)
            self.prices = np.array([float(item[4]) for item in data])  # Closing price
            
            print(f"Fetched {len(self.prices)} hours of BTC data")
            print(f"Period: {self.full_start_date.strftime('%Y-%m-%d')} to {self.full_end_date.strftime('%Y-%m-%d')}")
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
        self.full_end_date = datetime.now()
        self.full_start_date = self.full_end_date - timedelta(days=30)
        base_time = self.full_start_date.timestamp() * 1000
        self.timestamps = np.array([base_time + i * 3600000 for i in range(n_points)]).reshape(-1, 1)
        
        # Generate synthetic BTC price with trend, cycles, and noise
        t = np.linspace(0, 4*np.pi, n_points)
        trend = np.linspace(40000, 45000, n_points)
        cycle = 2000 * np.sin(t)
        noise = np.random.normal(0, 300, n_points)
        self.prices = trend + cycle + noise
        
        print(f"Generated {n_points} hours of sample data")
        print(f"Period: {self.full_start_date.strftime('%Y-%m-%d')} to {self.full_end_date.strftime('%Y-%m-%d')}")
    
    def ols(self, prices: np.ndarray, timestamps: np.ndarray, lookback: int) -> Tuple[float, float, float]:
        """Perform OLS regression on the last 'lookback' data points
        Returns: slope, total_error, avg_error_per_candle"""
        if lookback > len(prices):
            lookback = len(prices)
        
        # Get last 'lookback' data points
        X = timestamps[-lookback:]
        y = prices[-lookback:]
        
        # Normalize timestamps to avoid numerical issues
        X_mean = X.mean()
        X_std = X.std()
        X_normalized = (X - X_mean) / X_std
        
        # Fit OLS
        model = LinearRegression()
        model.fit(X_normalized, y)
        
        # Get slope (denormalize)
        # Convert to price change per hour (timestamps are in milliseconds)
        slope = model.coef_[0] / X_std * 3600000  # Convert to price change per hour
        
        # Calculate total error (sum of absolute differences)
        total_error = np.sum(np.abs(y - model.predict(X_normalized)))
        
        # Calculate average error per candle
        avg_error_per_candle = total_error / lookback
        
        return slope, total_error, avg_error_per_candle
    
    def find_optimal_lookback(self, prices: np.ndarray, timestamps: np.ndarray) -> Tuple[int, float, float, float]:
        """Find the optimal lookback window that minimizes average error"""
        best_avg_error = float('inf')
        best_lookback = 10
        best_slope = 0
        best_total_error = 0
        
        max_lookback = min(100, len(prices))
        
        for lookback in range(10, max_lookback + 1):
            slope, total_error, avg_error = self.ols(prices, timestamps, lookback)
            
            if avg_error < best_avg_error:
                best_avg_error = avg_error
                best_lookback = lookback
                best_slope = slope
                best_total_error = total_error
        
        return best_lookback, best_slope, best_total_error, best_avg_error
    
    def analyze_all_points(self):
        """Analyze every possible endpoint from min_points to the end"""
        print("\n" + "=" * 70)
        print("📊 POINT-BY-POINT ANALYSIS - Calculating Optimal Slope for Each Hour")
        print("=" * 70)
        
        self.slope_history = []
        self.lookback_history = []
        self.error_history = []
        self.endpoint_indices = []
        
        min_points = 110  # Need at least 100 lookback + 10 minimum
        
        total_points = len(self.prices)
        
        # We analyze from min_points to total_points (inclusive)
        for endpoint in range(min_points, total_points + 1):
            # Get data up to current endpoint
            test_prices = self.prices[:endpoint]
            test_timestamps = self.timestamps[:endpoint]
            
            # Find optimal lookback for this endpoint
            lookback, slope, total_error, avg_error = self.find_optimal_lookback(test_prices, test_timestamps)
            
            self.slope_history.append(slope)
            self.lookback_history.append(lookback)
            self.error_history.append(avg_error)
            self.endpoint_indices.append(endpoint)  # Store the endpoint index
            
            if endpoint % 100 == 0 or endpoint == total_points:
                end_date = self.full_start_date + timedelta(hours=endpoint)
                print(f"   📍 {endpoint}: {end_date.strftime('%Y-%m-%d %H:%M')} - Lookback={lookback}, Slope=${slope:+.2f}/h, Error=${avg_error:.2f}")
        
        print(f"\n✅ Completed analysis of {len(self.slope_history)} endpoints")
        print(f"   Endpoint indices range: {self.endpoint_indices[0]} to {self.endpoint_indices[-1]}")
        
        # Calculate statistics
        positive_slopes = sum(1 for s in self.slope_history if s > 0)
        negative_slopes = sum(1 for s in self.slope_history if s < 0)
        
        print(f"\n📈 Slope Statistics:")
        print(f"   Positive trends: {positive_slopes} ({positive_slopes/len(self.slope_history)*100:.1f}%)")
        print(f"   Negative trends: {negative_slopes} ({negative_slopes/len(self.slope_history)*100:.1f}%)")
        print(f"   Avg slope: ${np.mean(self.slope_history):+.2f}/h")
        print(f"   Slope volatility: ${np.std(self.slope_history):.2f}/h")
    
    def plot(self) -> str:
        """Create plot showing price with slope heatmap and analysis"""
        plt.figure(figsize=(16, 12))
        
        # Create 3x1 subplot grid
        gs = plt.GridSpec(3, 1, height_ratios=[2.5, 1, 1], hspace=0.3)
        
        # Main price chart with slope coloring
        ax_main = plt.subplot(gs[0])
        
        # Plot price line
        ax_main.plot(range(len(self.prices)), self.prices, 'b-', alpha=0.5, 
                    label='BTC Price', linewidth=1)
        
        # Create a color-coded background based on slope
        max_abs_slope = max(abs(np.min(self.slope_history)), abs(np.max(self.slope_history)))
        
        # For each analyzed point, color its background
        for i, endpoint in enumerate(self.endpoint_indices):
            slope = self.slope_history[i]
            if slope > 0:
                # Green for positive slope, intensity based on magnitude
                intensity = min(1.0, slope / max_abs_slope)
                color = (0, intensity, 0, 0.3)  # RGBA with alpha
            else:
                # Red for negative slope, intensity based on magnitude
                intensity = min(1.0, abs(slope) / max_abs_slope)
                color = (intensity, 0, 0, 0.3)
            
            # Color the background of this point
            ax_main.axvspan(endpoint-0.5, endpoint+0.5, facecolor=color)
        
        # Create custom legend for slope
        from matplotlib.patches import Rectangle
        
        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor=(0, 1, 0, 0.3), label='Strong Positive'),
            Rectangle((0, 0), 1, 1, facecolor=(0, 0.3, 0, 0.3), label='Weak Positive'),
            Rectangle((0, 0), 1, 1, facecolor=(0.3, 0, 0, 0.3), label='Weak Negative'),
            Rectangle((0, 0), 1, 1, facecolor=(1, 0, 0, 0.3), label='Strong Negative')
        ]
        ax_main.legend(handles=legend_elements, loc='upper left', fontsize=8, title='Slope Strength')
        
        ax_main.set_title(f'BTC Price with Slope Heatmap\n'
                         f'Green = Positive Trend, Red = Negative Trend (Intensity = Slope Magnitude)', 
                         fontsize=14, fontweight='bold')
        ax_main.set_xlabel('Hours from Start')
        ax_main.set_ylabel('Price (USDT)')
        ax_main.grid(True, alpha=0.3)
        
        # Add date range info
        date_text = f'Period: {self.full_start_date.strftime("%Y-%m-%d")} to {self.full_end_date.strftime("%Y-%m-%d")}'
        ax_main.text(0.02, 0.98, date_text, transform=ax_main.transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Slope chart (middle)
        ax_slope = plt.subplot(gs[1])
        
        # Color-code the slope line itself
        for i in range(len(self.endpoint_indices)-1):
            if self.slope_history[i] > 0:
                ax_slope.plot([self.endpoint_indices[i], self.endpoint_indices[i+1]], 
                            [self.slope_history[i], self.slope_history[i+1]], 
                            'g-', linewidth=1, alpha=0.7)
            else:
                ax_slope.plot([self.endpoint_indices[i], self.endpoint_indices[i+1]], 
                            [self.slope_history[i], self.slope_history[i+1]], 
                            'r-', linewidth=1, alpha=0.7)
        
        # Add zero line
        ax_slope.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        
        ax_slope.fill_between(self.endpoint_indices, 0, self.slope_history, 
                              where=np.array(self.slope_history) > 0, 
                              color='green', alpha=0.3, interpolate=True)
        ax_slope.fill_between(self.endpoint_indices, 0, self.slope_history, 
                              where=np.array(self.slope_history) < 0, 
                              color='red', alpha=0.3, interpolate=True)
        
        ax_slope.set_title('Optimal Slope at Each Point ($/hour)', fontsize=12)
        ax_slope.set_xlabel('Hours from Start')
        ax_slope.set_ylabel('Slope ($/h)')
        ax_slope.grid(True, alpha=0.3)
        
        # Lookback and Error chart (bottom)
        ax_bottom = plt.subplot(gs[2])
        
        ax_bottom.plot(self.endpoint_indices, self.lookback_history, 'b-', linewidth=1.5, alpha=0.7, label='Optimal Lookback')
        ax_bottom.set_xlabel('Hours from Start')
        ax_bottom.set_ylabel('Lookback (hours)', color='b')
        ax_bottom.tick_params(axis='y', labelcolor='b')
        ax_bottom.grid(True, alpha=0.3)
        
        ax2 = ax_bottom.twinx()
        ax2.plot(self.endpoint_indices, self.error_history, 'orange', linewidth=1, alpha=0.7, label='Avg Error')
        ax2.set_ylabel('Avg Error ($)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        ax_bottom.set_title('Optimal Lookback Window and Average Error', fontsize=12)
        
        # Add legend
        lines1, labels1 = ax_bottom.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_bottom.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
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
                # Run point-by-point analysis
                BTCRequestHandler.btc_analyzer.analyze_all_points()
                image_base64 = BTCRequestHandler.btc_analyzer.plot()
                
                # Calculate some additional statistics
                prices = BTCRequestHandler.btc_analyzer.prices
                volatility = np.std(prices[-24:])  # 24h volatility
                analyzer = BTCRequestHandler.btc_analyzer
                
                # Get recent slope (last 24 hours)
                if len(analyzer.slope_history) >= 24:
                    recent_slopes = analyzer.slope_history[-24:]
                else:
                    recent_slopes = analyzer.slope_history
                    
                avg_recent_slope = np.mean(recent_slopes)
                
                if avg_recent_slope > 5:
                    trend_direction = "🟢 STRONG BULLISH"
                elif avg_recent_slope > 1:
                    trend_direction = "🟢 BULLISH"
                elif avg_recent_slope > -1:
                    trend_direction = "⚪ NEUTRAL"
                elif avg_recent_slope > -5:
                    trend_direction = "🔴 BEARISH"
                else:
                    trend_direction = "🔴 STRONG BEARISH"
                
                # Create HTML page
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>BTC Point-by-Point OLS Slope Analysis</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }}
                        .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
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
                        .methodology {{ background-color: #e8f4f8; border-left: 4px solid #2196F3; padding: 15px; margin: 20px 0; }}
                        .trend-indicator {{ font-size: 24px; text-align: center; padding: 20px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px; margin: 20px 0; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>📊 BTC Point-by-Point OLS Slope Analysis</h1>
                        <p>Calculating optimal trend for every hour</p>
                        
                        <div class="methodology">
                            <strong>🔬 Methodology:</strong> For every possible endpoint (starting from hour 110), 
                            we find the lookback window (10-100 hours) that minimizes average absolute error. 
                            The slope of that optimal line is then plotted on the price chart as a color-coded 
                            background (green = positive, red = negative, intensity = magnitude).
                        </div>
                        
                        <div class="trend-indicator">
                            <strong>Current 24h Trend:</strong> {trend_direction} (${avg_recent_slope:+.2f}/h)
                        </div>
                        
                        <div class="info">
                            <h3>📈 Analysis Summary:</h3>
                            <p><strong>Period:</strong> {analyzer.full_start_date.strftime('%Y-%m-%d')} to {analyzer.full_end_date.strftime('%Y-%m-%d')}</p>
                            <p><strong>Total Points Analyzed:</strong> {len(analyzer.slope_history)}</p>
                            <p><strong>Positive Slopes:</strong> {sum(1 for s in analyzer.slope_history if s > 0)} ({sum(1 for s in analyzer.slope_history if s > 0)/len(analyzer.slope_history)*100:.1f}%)</p>
                            <p><strong>Negative Slopes:</strong> {sum(1 for s in analyzer.slope_history if s < 0)} ({sum(1 for s in analyzer.slope_history if s < 0)/len(analyzer.slope_history)*100:.1f}%)</p>
                            <p><strong>Average Slope:</strong> ${np.mean(analyzer.slope_history):+.2f}/h</p>
                            <p><strong>Slope Volatility:</strong> ${np.std(analyzer.slope_history):.2f}/h</p>
                        </div>
                        
                        <div class="stats">
                            <div class="stat-box">
                                <div class="stat-label">Current BTC Price</div>
                                <div class="stat-value">${prices[-1]:,.2f}</div>
                                <div class="stat-sub">as of now</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">24h Volatility</div>
                                <div class="stat-value">${volatility:,.2f}</div>
                                <div class="stat-sub">price fluctuation</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">Current Slope</div>
                                <div class="stat-value">${analyzer.slope_history[-1]:+.2f}/h</div>
                                <div class="stat-sub">optimal trend</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">Current Lookback</div>
                                <div class="stat-value">{analyzer.lookback_history[-1]}h</div>
                                <div class="stat-sub">optimal window</div>
                            </div>
                        </div>
                        
                        <img src="data:image/png;base64,{image_base64}" alt="BTC Slope Analysis">
                        
                        <div class="footer">
                            <p>Data from Binance API | Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                            <p>Each point's background color shows the optimal trend slope at that time</p>
                            <p>Green = positive slope (uptrend), Red = negative slope (downtrend)</p>
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
    print("BTC Point-by-Point OLS Slope Analysis")
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