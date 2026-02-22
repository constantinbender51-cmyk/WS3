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
    
    def run_analysis(self, prices: np.ndarray, timestamps: np.ndarray, end_point_idx: int, iteration_num: int) -> Dict:
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
        
        # Calculate the start of the optimal window
        optimal_window_start = end_point_idx - best_lookback
        
        return {
            'iteration': iteration_num,
            'end_point_idx': end_point_idx,
            'end_date': self.full_start_date + timedelta(hours=end_point_idx),
            'window_start_idx': optimal_window_start,
            'window_start_date': self.full_start_date + timedelta(hours=optimal_window_start),
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
        """Move endpoint to the start of the previous optimal window and run analysis"""
        print("\n" + "=" * 70)
        print("📊 PROGRESSIVE ANALYSIS - Moving Endpoint to Previous Window Start")
        print("=" * 70)
        
        self.all_results = []
        total_points = len(self.prices)
        
        # Start from the full dataset
        current_end = total_points
        iteration = 1
        
        while current_end >= 200:  # Need at least 200 points for meaningful analysis
            # Slice data from beginning to current endpoint
            test_prices = self.prices[:current_end]
            test_timestamps = self.timestamps[:current_end]
            
            end_date = self.full_start_date + timedelta(hours=current_end)
            
            print(f"\n📌 Iteration {iteration}: Data up to {end_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Data points: {len(test_prices)} hours")
            
            # Run analysis on this slice
            result = self.run_analysis(test_prices, test_timestamps, current_end, iteration)
            self.all_results.append(result)
            
            print(f"   ✅ Best lookback: {result['best_lookback']} hours")
            print(f"      Window: {result['window_start_date'].strftime('%Y-%m-%d %H:%M')} to {result['end_date'].strftime('%Y-%m-%d %H:%M')}")
            print(f"      Avg Error: ${result['best_avg_error']:,.2f}")
            
            # Move endpoint to the start of the current optimal window
            next_end = result['window_start_idx']
            print(f"   ➡️  Next endpoint will be: {self.full_start_date + timedelta(hours=next_end)}")
            
            if next_end < 200:  # Stop if we don't have enough data for next iteration
                break
                
            current_end = next_end
            iteration += 1
        
        print(f"\n✅ Completed {len(self.all_results)} progressive analyses")
        return self.all_results
    
    def plot(self) -> str:
        """Create plot showing all progressive OLS lines"""
        plt.figure(figsize=(16, 14))
        
        # Create 2x1 subplot grid
        gs = plt.GridSpec(3, 1, height_ratios=[2.5, 1, 0.8], hspace=0.3)
        
        # Main price chart with all OLS lines
        ax_main = plt.subplot(gs[0])
        
        # Plot all prices
        ax_main.plot(range(len(self.prices)), self.prices, 'b-', alpha=0.3, 
                    label='BTC Price (All Data)', linewidth=1)
        
        # Generate a colormap for the progressive lines
        cmap = plt.cm.viridis
        colors = [cmap(i / len(self.all_results)) for i in range(len(self.all_results))]
        
        # Plot OLS lines for each progressive analysis
        for idx, result in enumerate(self.all_results):
            if result['best_model'] is not None:
                end_idx = result['end_point_idx']
                start_idx = result['window_start_idx']
                lookback_range = range(start_idx, end_idx)
                
                # Get model and normalization params
                model = result['best_model']
                X_mean, X_std = result['best_normalization_params']
                
                # Generate predictions for the lookback window
                X_lookback = self.timestamps[start_idx:end_idx].flatten()
                X_norm = (X_lookback - X_mean) / X_std
                y_pred = model.predict(X_norm.reshape(-1, 1))
                
                # Plot the OLS line
                alpha = 0.6 + (idx / len(self.all_results)) * 0.4  # Later lines more opaque
                linewidth = 2 + (idx / len(self.all_results)) * 2
                
                ax_main.plot(lookback_range, y_pred, 
                           color=colors[idx],
                           linewidth=linewidth,
                           alpha=alpha,
                           label=f'Iter {result["iteration"]}: {result["window_start_date"].strftime("%m/%d")} to {result["end_date"].strftime("%m/%d")} (lb={result["best_lookback"]})')
                
                # Mark the window boundaries
                ax_main.axvline(x=start_idx, color=colors[idx], linestyle='--', alpha=0.3, linewidth=1)
                ax_main.axvline(x=end_idx, color=colors[idx], linestyle=':', alpha=0.3, linewidth=1)
                
                # Add iteration number at the top of the window
                ax_main.text(end_idx - 10, ax_main.get_ylim()[1] * 0.95, 
                            f'Iter {result["iteration"]}', fontsize=8, color=colors[idx],
                            ha='right', va='top')
        
        ax_main.set_title(f'BTC Price with Progressive OLS Lines\n'
                         f'Each Iteration Uses Window from Previous Optimal Start', 
                         fontsize=14, fontweight='bold')
        ax_main.set_xlabel('Hours from Start')
        ax_main.set_ylabel('Price (USDT)')
        ax_main.legend(loc='upper left', fontsize=7, ncol=2)
        ax_main.grid(True, alpha=0.3)
        
        # Add date range info
        date_text = f'Period: {self.full_start_date.strftime("%Y-%m-%d")} to {self.full_end_date.strftime("%Y-%m-%d")}'
        ax_main.text(0.02, 0.98, date_text, transform=ax_main.transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Window progression plot (middle)
        ax_windows = plt.subplot(gs[1])
        
        for i, result in enumerate(self.all_results):
            # Plot each window as a horizontal bar
            start = result['window_start_idx']
            end = result['end_point_idx']
            ax_windows.barh(i, end - start, left=start, height=0.5, 
                          color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add lookback label
            ax_windows.text(start + (end-start)/2, i, f'{result["best_lookback"]}h', 
                          ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax_windows.set_xlabel('Hours from Start')
        ax_windows.set_ylabel('Iteration')
        ax_windows.set_title('Optimal Windows for Each Iteration', fontsize=12)
        ax_windows.set_yticks(range(len(self.all_results)))
        ax_windows.set_yticklabels([f'Iter {r["iteration"]}' for r in self.all_results])
        ax_windows.grid(True, alpha=0.3, axis='x')
        
        # Summary plot (bottom)
        ax_summary = plt.subplot(gs[2])
        
        iterations = [r['iteration'] for r in self.all_results]
        lookbacks = [r['best_lookback'] for r in self.all_results]
        errors = [r['best_avg_error'] for r in self.all_results]
        
        # Create twin axes
        ax_summary.plot(iterations, lookbacks, 'b-o', linewidth=2, markersize=8, label='Best Lookback')
        ax_summary.set_xlabel('Iteration')
        ax_summary.set_ylabel('Lookback Window (hours)', color='b')
        ax_summary.tick_params(axis='y', labelcolor='b')
        ax_summary.grid(True, alpha=0.3)
        
        ax2 = ax_summary.twinx()
        ax2.plot(iterations, errors, 'r-s', linewidth=2, markersize=8, label='Avg Error')
        ax2.set_ylabel('Avg Error ($)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax_summary.set_title('Lookback and Error Progression', fontsize=12)
        
        # Add legend
        lines1, labels1 = ax_summary.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_summary.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
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
                    <title>BTC Progressive OLS Analysis - Moving to Window Start</title>
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
                        <p>Moving Endpoint to Start of Previous Optimal Window</p>
                        
                        <div class="methodology">
                            <strong>🔬 Methodology:</strong> Starting from the full dataset, we find the optimal lookback window.
                            Then we move the endpoint to the start of that window and repeat the analysis.
                            This shows how optimal trends cascade backward through time.
                        </div>
                        
                        <div class="info">
                            <h3>📈 Analysis Summary:</h3>
                            <p><strong>Full Period:</strong> {analyzer.full_start_date.strftime('%Y-%m-%d')} to {analyzer.full_end_date.strftime('%Y-%m-%d')}</p>
                            <p><strong>Total Iterations:</strong> {len(results)}</p>
                            <p><strong>Total Data Points:</strong> {len(prices)} hours</p>
                        </div>
                        
                        <h3>📋 Progressive Results:</h3>
                        <table class="results-table">
                            <tr>
                                <th>Iter</th>
                                <th>Window Period</th>
                                <th>Window Size</th>
                                <th>Lookback</th>
                                <th>Avg Error</th>
                                <th>Next Endpoint</th>
                            </tr>
                """
                
                # Add result rows
                for i, result in enumerate(results, 1):
                    window_period = f"{result['window_start_date'].strftime('%m/%d %H:%M')} to {result['end_date'].strftime('%m/%d %H:%M')}"
                    window_size = result['end_point_idx'] - result['window_start_idx']
                    
                    next_endpoint = ""
                    if i < len(results):
                        next_endpoint = results[i]['window_start_date'].strftime('%m/%d %H:%M')
                    else:
                        next_endpoint = "End"
                    
                    html += f"""
                            <tr>
                                <td>{result['iteration']}</td>
                                <td>{window_period}</td>
                                <td>{window_size} hours</td>
                                <td>{result['best_lookback']} hours</td>
                                <td>${result['best_avg_error']:,.2f}</td>
                                <td>{next_endpoint}</td>
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
                            <p>Each iteration's endpoint becomes the start of the previous optimal window</p>
                            <p>This reveals nested optimal trends within the price data</p>
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
    print("BTC Progressive OLS Analysis - Moving to Previous Window Start")
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