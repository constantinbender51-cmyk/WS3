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
        self.all_results = []
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
    
    def ols(self, prices: np.ndarray, timestamps: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray, float, float, object]:
        """Perform OLS regression on the last 'lookback' data points
        Returns: X, y_pred, total_error, avg_error_per_candle, model"""
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
        
        # Predict
        y_pred = model.predict(X_normalized)
        
        # Calculate total error (sum of absolute differences)
        total_error = np.sum(np.abs(y - y_pred))
        
        # Calculate average error per candle
        avg_error_per_candle = total_error / lookback
        
        return X.flatten(), y_pred, total_error, avg_error_per_candle, model, (X_mean, X_std)
    
    def run_analysis(self, prices: np.ndarray, timestamps: np.ndarray, end_point_idx: int) -> Dict:
        """Run OLS analysis on data up to a specific endpoint"""
        best_avg_error = float('inf')
        best_lookback = 10
        best_X = None
        best_y_pred = None
        best_total_error = None
        best_model = None
        best_normalization_params = None
        
        # Only consider lookbacks that fit within the available data
        max_lookback = min(100, len(prices))
        
        for lookback in range(10, max_lookback + 1):
            X, y_pred, total_error, avg_error, model, norm_params = self.ols(prices, timestamps, lookback)
            
            if avg_error < best_avg_error:
                best_avg_error = avg_error
                best_lookback = lookback
                best_X = X
                best_y_pred = y_pred
                best_total_error = total_error
                best_model = model
                best_normalization_params = norm_params
        
        return {
            'end_point_idx': end_point_idx,
            'end_date': self.full_start_date + timedelta(hours=end_point_idx),
            'data_length': len(prices),
            'best_lookback': best_lookback,
            'best_avg_error': best_avg_error,
            'best_total_error': best_total_error,
            'best_model': best_model,
            'best_normalization_params': best_normalization_params,
            'prices': prices,
            'timestamps': timestamps
        }
    
    def iterate(self):
        """Move endpoint 100 candles to the left and run analysis until data runs out"""
        print("\n" + "=" * 70)
        print("📊 PROGRESSIVE ANALYSIS - Moving Endpoint 100 Candles Left")
        print("=" * 70)
        
        self.all_results = []
        total_points = len(self.prices)
        
        # Start from the full dataset and move endpoint left by 100 each iteration
        current_end = total_points
        
        iteration = 1
        while current_end >= 200:  # Need at least 200 points for meaningful analysis (100+100)
            # Slice data from beginning to current endpoint
            test_prices = self.prices[:current_end]
            test_timestamps = self.timestamps[:current_end]
            
            end_date = self.full_start_date + timedelta(hours=current_end)
            
            print(f"\n📌 Iteration {iteration}: Data up to {end_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Data points: {len(test_prices)} hours")
            
            # Run analysis on this slice
            result = self.run_analysis(test_prices, test_timestamps, current_end)
            self.all_results.append(result)
            
            print(f"   ✅ Best lookback: {result['best_lookback']} hours")
            print(f"      Avg Error: ${result['best_avg_error']:,.2f}")
            
            # Move endpoint 100 candles to the left
            current_end -= 100
            iteration += 1
        
        print(f"\n✅ Completed {len(self.all_results)} progressive analyses")
        return self.all_results
    
    def plot(self) -> str:
        """Create plot showing all progressive OLS lines"""
        plt.figure(figsize=(16, 12))
        
        # Create 2x1 subplot grid
        gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
        
        # Main price chart with all OLS lines
        ax_main = plt.subplot(gs[0])
        
        # Plot all prices
        ax_main.plot(range(len(self.prices)), self.prices, 'b-', alpha=0.3, 
                    label='BTC Price (All Data)', linewidth=1)
        
        # Generate a colormap for the progressive lines
        cmap = plt.cm.viridis
        colors = [cmap(i / len(self.all_results)) for i in range(len(self.all_results))]
        
        # Plot OLS lines for each progressive analysis
        for idx, result in enumerate(reversed(self.all_results)):  # Reverse to show oldest first
            if result['best_model'] is not None:
                end_idx = result['end_point_idx']
                lookback = result['best_lookback']
                lookback_start = end_idx - lookback
                lookback_range = range(lookback_start, end_idx)
                
                # Get model and normalization params
                model = result['best_model']
                X_mean, X_std = result['best_normalization_params']
                
                # Generate predictions for the lookback window
                X_lookback = self.timestamps[lookback_start:end_idx].flatten()
                X_norm = (X_lookback - X_mean) / X_std
                y_pred = model.predict(X_norm.reshape(-1, 1))
                
                # Plot the OLS line
                alpha = 0.5 + (idx / len(self.all_results)) * 0.5  # Later lines more opaque
                linewidth = 1.5 + (idx / len(self.all_results)) * 1.5
                
                ax_main.plot(lookback_range, y_pred, 
                           color=colors[idx],
                           linewidth=linewidth,
                           alpha=alpha,
                           label=f'End: {result["end_date"].strftime("%m/%d")} (lookback={lookback})')
                
                # Mark the endpoint
                ax_main.axvline(x=end_idx, color=colors[idx], linestyle=':', alpha=0.3, linewidth=1)
        
        # Mark the starting point
        ax_main.axvline(x=len(self.prices) - 100, color='red', linestyle='--', alpha=0.5, 
                       label='First Analysis End', linewidth=1)
        
        ax_main.set_title(f'BTC Price with Progressive OLS Lines\n'
                         f'Moving Endpoint 100 Candles Left from {self.full_end_date.strftime("%Y-%m-%d")}', 
                         fontsize=14, fontweight='bold')
        ax_main.set_xlabel('Hours from Start')
        ax_main.set_ylabel('Price (USDT)')
        ax_main.legend(loc='upper left', fontsize=8, ncol=2)
        ax_main.grid(True, alpha=0.3)
        
        # Add date range info
        date_text = f'Period: {self.full_start_date.strftime("%Y-%m-%d")} to {self.full_end_date.strftime("%Y-%m-%d")}'
        ax_main.text(0.02, 0.98, date_text, transform=ax_main.transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Summary plot (bottom)
        ax_summary = plt.subplot(gs[1])
        
        iterations = range(1, len(self.all_results) + 1)
        lookbacks = [r['best_lookback'] for r in self.all_results]
        errors = [r['best_avg_error'] for r in self.all_results]
        
        # Create twin axes
        ax_summary.plot(iterations, lookbacks, 'b-o', linewidth=2, markersize=8, label='Best Lookback')
        ax_summary.set_xlabel('Iteration (Moving Left)')
        ax_summary.set_ylabel('Lookback Window (hours)', color='b')
        ax_summary.tick_params(axis='y', labelcolor='b')
        ax_summary.grid(True, alpha=0.3)
        
        ax2 = ax_summary.twinx()
        ax2.plot(iterations, errors, 'r-s', linewidth=2, markersize=8, label='Avg Error')
        ax2.set_ylabel('Avg Error ($)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add endpoint dates as x-tick labels
        dates = [r['end_date'].strftime('%m/%d') for r in self.all_results]
        ax_summary.set_xticks(iterations)
        ax_summary.set_xticklabels(dates, rotation=45, ha='right')
        
        ax_summary.set_title('Progression of Best Lookback and Error as Endpoint Moves Left', fontsize=12)
        
        # Add legend
        lines1, labels1 = ax_summary.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_summary.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        
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
                # Run progressive analysis
                results = BTCRequestHandler.btc_analyzer.iterate()
                image_base64 = BTCRequestHandler.btc_analyzer.plot()
                
                # Calculate some additional statistics
                prices = BTCRequestHandler.btc_analyzer.prices
                volatility = np.std(prices[-24:])  # 24h volatility
                analyzer = BTCRequestHandler.btc_analyzer
                
                # Create HTML page
html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>BTC Progressive OLS Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }}
        .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
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
        .results-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .results-table th {{ background-color: #2196F3; color: white; padding: 10px; text-align: left; }}
        .results-table td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        .results-table tr:hover {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 BTC Progressive OLS Analysis</h1>
        <p>Moving Endpoint 100 Candles Left from <strong>{analyzer.full_end_date.strftime('%Y-%m-%d')}</strong></p>
        
        <div class="methodology">
            <strong>🔬 Methodology:</strong> Starting from the full dataset, we progressively move the endpoint 
            100 candles to the left and run the OLS optimization algorithm at each step. Each colored line on 
            the price chart represents the best OLS fit for that endpoint. This shows how the optimal trend 
            line evolves as we remove recent data.
        </div>
        
        <div class="info">
            <h3>📈 Analysis Summary:</h3>
            <p><strong>Full Period:</strong> {analyzer.full_start_date.strftime('%Y-%m-%d')} to {analyzer.full_end_date.strftime('%Y-%m-%d')}</p>
            <p><strong>Total Iterations:</strong> {len(results)}</p>
            <p><strong>Step Size:</strong> 100 hours</p>
            <p><strong>Total Data Points:</strong> {len(prices)} hours</p>
        </div>
        
        <h3>📋 Progressive Results:</h3>
        <table class="results-table">
            <tr>
                <th>Iteration</th>
                <th>End Date</th>
                <th>Data Points</th>
                <th>Best Lookback</th>
                <th>Avg Error</th>
                <th>Change</th>
            </tr>
"""

# Add result rows
for i, result in enumerate(results, 1):
    change = ""
    change_color = "black"
    
    if i > 1:
        diff = result['best_lookback'] - results[i-2]['best_lookback']
        if diff > 0:
            change = f"+{diff}"
        elif diff < 0:
            change = f"{diff}"
        else:
            change = "0"
        
        # Determine color based on change magnitude
        if diff == 0:
            change_color = "green"
        elif abs(diff) < 20:
            change_color = "orange"
        else:
            change_color = "red"
    
    html += f"""
            <tr>
                <td>{i}</td>
                <td>{result['end_date'].strftime('%Y-%m-%d')}</td>
                <td>{result['data_length']}</td>
                <td>{result['best_lookback']} hours</td>
                <td>${result['best_avg_error']:,.2f}</td>
                <td style="color: {change_color};">{change}</td>
            </tr>
    """

html += f"""
        </table>
        
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
        
        <img src="data:image/png;base64,{image_base64}" alt="BTC Progressive OLS Analysis">
        
        <div class="footer">
            <p>Data from Binance API | Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Progressive analysis moves endpoint 100 candles left at each iteration</p>
            <p>Each colored line shows the optimal OLS fit for that endpoint</p>
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
    print("BTC Progressive OLS Analysis - Moving Endpoint 100 Candles Left")
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