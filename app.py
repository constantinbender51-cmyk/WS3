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
import random
import warnings
warnings.filterwarnings('ignore')

class BTCOptimalLookback:
    def __init__(self):
        self.data = None
        self.timestamps = None
        self.prices = None
        self.scenario_results = []
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
        
        # Generate synthetic BTC price with trend and noise
        trend = np.linspace(40000, 45000, n_points)
        noise = np.random.normal(0, 500, n_points)
        self.prices = trend + noise
        
        print(f"Generated {n_points} hours of sample data")
        print(f"Period: {self.full_start_date.strftime('%Y-%m-%d')} to {self.full_end_date.strftime('%Y-%m-%d')}")
    
    def ols(self, prices: np.ndarray, timestamps: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Perform OLS regression on the last 'lookback' data points
        Returns: X, y_pred, total_error, avg_error_per_candle"""
        if lookback > len(prices):
            lookback = len(prices)
        
        # Get last 'lookback' data points
        X = timestamps[-lookback:]
        y = prices[-lookback:]
        
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
    
    def run_analysis(self, prices: np.ndarray, timestamps: np.ndarray, scenario_name: str, 
                     start_date: datetime, end_date: datetime) -> Dict:
        """Run OLS analysis on given data and return results"""
        best_avg_error = float('inf')
        best_lookback = 10
        best_X = None
        best_y_pred = None
        best_total_error = None
        
        results = {
            'name': scenario_name,
            'lookbacks': [],
            'avg_errors': [],
            'total_errors': [],
            'best_lookback': None,
            'best_avg_error': None,
            'best_total_error': None,
            'best_X': None,
            'best_y_pred': None,
            'data_length': len(prices),
            'start_date': start_date,
            'end_date': end_date
        }
        
        for lookback in range(10, min(101, len(prices) + 1)):
            X, y_pred, total_error, avg_error = self.ols(prices, timestamps, lookback)
            
            results['lookbacks'].append(lookback)
            results['avg_errors'].append(avg_error)
            results['total_errors'].append(total_error)
            
            if avg_error < best_avg_error:
                best_avg_error = avg_error
                best_lookback = lookback
                best_X = X
                best_y_pred = y_pred
                best_total_error = total_error
        
        results['best_lookback'] = best_lookback
        results['best_avg_error'] = best_avg_error
        results['best_total_error'] = best_total_error
        results['best_X'] = best_X
        results['best_y_pred'] = best_y_pred
        
        return results
    
    def iterate(self) -> Tuple[int, np.ndarray, np.ndarray, float, float]:
        """Iterate through lookback windows from 10 to 100 and find best average error"""
        print("\n" + "=" * 70)
        print("📊 MAIN ANALYSIS - Full Dataset (30 days)")
        print("=" * 70)
        
        # Run main analysis on full dataset
        main_results = self.run_analysis(
            self.prices, self.timestamps, 
            "Main Analysis (Full 30 Days)", 
            self.full_start_date, 
            self.full_end_date
        )
        
        print(f"\n✅ Best lookback window: {main_results['best_lookback']} hours")
        print(f"   Total Error: ${main_results['best_total_error']:,.2f}")
        print(f"   Average Error per Candle: ${main_results['best_avg_error']:,.2f}")
        
        # Generate 3 test scenarios with random end points (>100 hours from start)
        print("\n" + "=" * 70)
        print("🧪 TEST SCENARIOS - Fixed Start Date, Random End Points")
        print("=" * 70)
        
        self.scenario_results = [main_results]  # Store main results first
        
        # Calculate minimum end point (must have at least 100 hours of data)
        min_hours = 100
        max_hours = len(self.prices)
        
        # Generate 3 random end points between min_hours and max_hours
        random.seed(42)  # For reproducibility
        end_points = sorted([random.randint(min_hours, max_hours) for _ in range(3)], reverse=True)
        
        for i, hours_to_keep in enumerate(end_points, 1):
            # Keep data from start up to random end point
            test_prices = self.prices[:hours_to_keep]
            test_timestamps = self.timestamps[:hours_to_keep]
            
            # Calculate the actual end date
            test_end_date = self.full_start_date + timedelta(hours=hours_to_keep)
            
            scenario_name = f"Scenario {i}: Keep first {hours_to_keep} hours (cut off last {len(self.prices) - hours_to_keep} hours)"
            print(f"\n📌 {scenario_name}")
            print(f"   Period: {self.full_start_date.strftime('%Y-%m-%d')} to {test_end_date.strftime('%Y-%m-%d')}")
            print(f"   Data points: {len(test_prices)} hours")
            
            # Run analysis on test data
            test_results = self.run_analysis(
                test_prices, test_timestamps, 
                scenario_name,
                self.full_start_date,
                test_end_date
            )
            self.scenario_results.append(test_results)
            
            print(f"   ✅ Best lookback: {test_results['best_lookback']} hours")
            print(f"      Avg Error: ${test_results['best_avg_error']:,.2f}")
        
        return (main_results['best_lookback'], main_results['best_X'], 
                main_results['best_y_pred'], main_results['best_total_error'], 
                main_results['best_avg_error'])
    
    def plot(self, best_lookback: int, best_X: np.ndarray, best_y_pred: np.ndarray, 
             best_total_error: float, best_avg_error: float) -> str:
        """Create comprehensive plot with main analysis and 3 test scenarios"""
        plt.figure(figsize=(16, 14))
        
        # Create 2x2 subplot grid
        gs = plt.GridSpec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main price chart (top, spanning both columns)
        ax_main = plt.subplot(gs[0, :])
        
        # Plot all prices
        ax_main.plot(range(len(self.prices)), self.prices, 'b-', alpha=0.5, 
                    label='BTC Price (All Data)', linewidth=1)
        
        # Highlight the best OLS segment for main analysis
        if best_X is not None and best_y_pred is not None:
            start_idx = len(self.prices) - len(best_X)
            x_indices = range(start_idx, len(self.prices))
            
            ax_main.plot(x_indices, best_y_pred, 'r-', linewidth=3, 
                        label=f'Best OLS Line (lookback={best_lookback})')
            
            # Highlight the data points used
            ax_main.scatter(x_indices, self.prices[start_idx:], c='orange', s=30, 
                           alpha=0.7, label='Data Points in Best Window')
        
        # Mark the random end points for test scenarios
        colors = ['red', 'purple', 'brown']
        for i, result in enumerate(self.scenario_results[1:], 1):
            end_point = result['data_length']
            ax_main.axvline(x=end_point, color=colors[i-1], linestyle='--', alpha=0.7,
                           label=f'Scenario {i} End: {end_point}h')
        
        ax_main.set_title(f'BTC Price (1h) - Fixed Start: {self.full_start_date.strftime("%Y-%m-%d")}\n'
                         f'Main Analysis - Best Lookback: {best_lookback} hours | Test Scenarios with Random End Points', 
                         fontsize=14, fontweight='bold')
        ax_main.set_xlabel('Hours from Start')
        ax_main.set_ylabel('Price (USDT)')
        ax_main.legend(loc='upper left', fontsize=8)
        ax_main.grid(True, alpha=0.3)
        
        # Add date ranges
        date_text = f'Full Period: {self.full_start_date.strftime("%Y-%m-%d")} to {self.full_end_date.strftime("%Y-%m-%d")}'
        ax_main.text(0.02, 0.98, date_text, transform=ax_main.transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Error analysis subplots for each scenario
        titles = ["Main Analysis (Full 30 Days)", "Test Scenario 1", "Test Scenario 2", "Test Scenario 3"]
        axes = [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), 
                plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1])]
        
        for ax, result, title in zip(axes, self.scenario_results, titles):
            self.plot_error_comparison(ax, result, title)
        
        plt.suptitle('BTC OLS Analysis: Main Results vs 3 Test Scenarios (Fixed Start, Random End Points)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Encode as base64
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return image_base64
    
    def plot_error_comparison(self, ax, results: Dict, title: str):
        """Plot error comparison for a single scenario"""
        # Plot average error
        ax.plot(results['lookbacks'], results['avg_errors'], 'g-', linewidth=2, 
                label='Avg Error/Candle')
        ax.scatter([results['best_lookback']], [results['best_avg_error']], 
                  c='red', s=100, zorder=5, marker='*',
                  label=f"Best: {results['best_lookback']}h")
        
        # Add a horizontal line at the best error
        ax.axhline(y=results['best_avg_error'], color='red', linestyle='--', 
                   alpha=0.3, linewidth=1)
        
        # Format date range for title
        date_range = f"{results['start_date'].strftime('%m/%d')} - {results['end_date'].strftime('%m/%d')}"
        
        ax.set_xlabel('Lookback Window (hours)')
        ax.set_ylabel('Avg Error ($)')
        ax.set_title(f'{title}\n{date_range} | Best: {results["best_lookback"]}h', 
                    fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis to show dollars
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

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
                analyzer = BTCRequestHandler.btc_analyzer
                
                # Create HTML page
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>BTC OLS Analysis - Fixed Start, Random End Points</title>
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
                        .error-metric {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; }}
                        .scenario-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                        .scenario-table th {{ background-color: #4CAF50; color: white; padding: 12px; text-align: left; }}
                        .scenario-table td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                        .scenario-table tr:hover {{ background-color: #f5f5f5; }}
                        .main-row {{ background-color: #e3f2fd; font-weight: bold; }}
                        .date-range {{ font-family: monospace; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>📊 BTC OLS Analysis with 3 Test Scenarios</h1>
                        <p>Fixed Start Date: <strong>{analyzer.full_start_date.strftime('%Y-%m-%d')}</strong> | Random End Points (>100 hours from start)</p>
                        
                        <div class="error-metric">
                            <strong>🎯 Testing Methodology:</strong> Keep the same 30-day start date but randomly cut off recent data 
                            (must keep at least 100 hours). This tests how sensitive the optimal lookback is to having less recent data.
                        </div>
                        
                        <div class="info">
                            <h3>📈 Main Analysis Results (Full 30 Days):</h3>
                            <p><strong>Period:</strong> {analyzer.full_start_date.strftime('%Y-%m-%d')} to {analyzer.full_end_date.strftime('%Y-%m-%d')}</p>
                            <p><strong>Optimal Lookback Window:</strong> {best_lookback} hours</p>
                            <p><strong>Total Error (Sum |price-ols|):</strong> ${best_total_error:,.2f}</p>
                            <p><strong>Average Error per Candle:</strong> ${best_avg_error:,.2f}</p>
                            <p><strong>Total Data Points:</strong> {len(prices)} hours</p>
                        </div>
                        
                        <h3>🧪 Test Scenarios Results (Fixed Start, Random End Points):</h3>
                        <table class="scenario-table">
                            <tr>
                                <th>Scenario</th>
                                <th>Period</th>
                                <th>Data Points</th>
                                <th>Hours Cut</th>
                                <th>Best Lookback</th>
                                <th>Avg Error</th>
                                <th>Change</th>
                            </tr>
                """
                
                # Add scenario rows
                scenarios = analyzer.scenario_results
                main_result = scenarios[0]
                
                for i, result in enumerate(scenarios[1:], 1):
                    change = result['best_lookback'] - main_result['best_lookback']
                    change_str = f"{change:+d} hours" if change != 0 else "No change"
                    change_color = "green" if change == 0 else "orange" if abs(change) < 20 else "red"
                    
                    hours_cut = len(analyzer.prices) - result['data_length']
                    
                    html += f"""
                            <tr>
                                <td><strong>Scenario {i}</strong></td>
                                <td class="date-range">{result['start_date'].strftime('%m/%d')} - {result['end_date'].strftime('%m/%d')}</td>
                                <td>{result['data_length']}</td>
                                <td>{hours_cut} hrs</td>
                                <td>{result['best_lookback']} hours</td>
                                <td>${result['best_avg_error']:,.2f}</td>
                                <td style="color: {change_color};">{change_str}</td>
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
                        
                        <img src="data:image/png;base64,{image_base64}" alt="BTC Price and Error Analysis with Test Scenarios">
                        
                        <div class="footer">
                            <p>Data from Binance API | Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                            <p>Analysis looks for lookback window that minimizes average absolute deviation per candle</p>
                            <p>Test scenarios keep the same start date but randomly cut off recent data (minimum 100 hours retained)</p>
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
    print("BTC OLS Analysis - Fixed Start Date, Random End Points")
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