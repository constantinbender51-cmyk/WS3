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
        self.trend_change_points = []  # Store points where trend changes
        self.merged_slopes = []  # Store merged slope segments
        self.merged_indices = []  # Store indices for merged segments
        
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
    
    def merge_slopes_by_sign(self):
        """Merge consecutive slopes with the same sign"""
        self.merged_slopes = []
        self.merged_indices = []
        
        if not self.slope_history:
            return
        
        current_sign = np.sign(self.slope_history[0])
        current_sum = self.slope_history[0]
        current_count = 1
        current_indices = [self.endpoint_indices[0]]
        
        for i in range(1, len(self.slope_history)):
            new_sign = np.sign(self.slope_history[i])
            
            if new_sign == current_sign:
                # Same sign, merge
                current_sum += self.slope_history[i]
                current_count += 1
                current_indices.append(self.endpoint_indices[i])
            else:
                # Sign changed, store the merged segment
                avg_slope = current_sum / current_count
                self.merged_slopes.append(avg_slope)
                self.merged_indices.append(current_indices)
                
                # Start new segment
                current_sign = new_sign
                current_sum = self.slope_history[i]
                current_count = 1
                current_indices = [self.endpoint_indices[i]]
        
        # Store the last segment
        if current_count > 0:
            avg_slope = current_sum / current_count
            self.merged_slopes.append(avg_slope)
            self.merged_indices.append(current_indices)
        
        print(f"\n📊 Merged {len(self.slope_history)} points into {len(self.merged_slopes)} segments by sign")
        for i, (indices, slope) in enumerate(zip(self.merged_indices, self.merged_slopes)):
            sign_str = "🟢 POSITIVE" if slope > 0 else "🔴 NEGATIVE" if slope < 0 else "⚪ ZERO"
            print(f"   Segment {i+1}: {sign_str} ({len(indices)} points, avg slope=${slope:+.2f}/h)")
    
    def analyze_all_points(self):
        """Analyze every possible endpoint from min_points to the end"""
        print("\n" + "=" * 70)
        print("📊 POINT-BY-POINT ANALYSIS - Calculating Optimal Slope for Each Hour")
        print("=" * 70)
        
        self.slope_history = []
        self.lookback_history = []
        self.error_history = []
        self.endpoint_indices = []
        self.trend_change_points = []
        
        min_points = 110  # Need at least 100 lookback + 10 minimum
        
        total_points = len(self.prices)
        
        # We analyze from min_points to total_points (inclusive)
        previous_slope = None
        
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
            
            # Detect trend changes
            if previous_slope is not None:
                if (previous_slope > 0 and slope < 0) or (previous_slope < 0 and slope > 0):
                    self.trend_change_points.append(endpoint)
                    print(f"   🔄 Trend change at {endpoint}: {previous_slope:+.2f} → {slope:+.2f}")
            
            previous_slope = slope
            
            if endpoint % 100 == 0 or endpoint == total_points:
                end_date = self.full_start_date + timedelta(hours=endpoint)
                trend_icon = "🟢" if slope > 0 else "🔴" if slope < 0 else "⚪"
                print(f"   {trend_icon} {endpoint}: {end_date.strftime('%Y-%m-%d %H:%M')} - "
                      f"Lookback={lookback}, Slope=${slope:+.2f}/h, Error=${avg_error:.2f}")
        
        print(f"\n✅ Completed analysis of {len(self.slope_history)} endpoints")
        print(f"   Endpoint indices range: {self.endpoint_indices[0]} to {self.endpoint_indices[-1]}")
        print(f"   Trend changes detected: {len(self.trend_change_points)}")
        
        # Merge slopes by sign
        self.merge_slopes_by_sign()
        
        # Calculate statistics
        positive_slopes = sum(1 for s in self.slope_history if s > 0)
        negative_slopes = sum(1 for s in self.slope_history if s < 0)
        
        print(f"\n📈 Slope Statistics:")
        print(f"   Positive trends: {positive_slopes} ({positive_slopes/len(self.slope_history)*100:.1f}%)")
        print(f"   Negative trends: {negative_slopes} ({negative_slopes/len(self.slope_history)*100:.1f}%)")
        print(f"   Avg slope: ${np.mean(self.slope_history):+.2f}/h")
        print(f"   Slope volatility: ${np.std(self.slope_history):.2f}/h")
        
        # Lookback statistics
        print(f"\n📊 Lookback Statistics:")
        print(f"   Avg lookback: {np.mean(self.lookback_history):.1f} hours")
        print(f"   Median lookback: {np.median(self.lookback_history):.1f} hours")
        print(f"   Min lookback: {np.min(self.lookback_history)} hours")
        print(f"   Max lookback: {np.max(self.lookback_history)} hours")
    
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
        
        # Create a color-coded background based on merged slope segments
        colors = [(0, 1, 0, 0.2), (1, 0, 0, 0.2)]  # Green and red with low alpha
        
        for indices, slope in zip(self.merged_indices, self.merged_slopes):
            color = colors[0] if slope > 0 else colors[1] if slope < 0 else (0.5, 0.5, 0.5, 0.2)
            # Color the entire segment
            ax_main.axvspan(indices[0]-0.5, indices[-1]+0.5, facecolor=color)
        
        # Add horizontal lines at trend change points
        for i, change_point in enumerate(self.trend_change_points):
            # Get price at this point
            price_at_change = self.prices[change_point]
            # Draw horizontal line from this point to the end
            ax_main.hlines(y=price_at_change, xmin=change_point, xmax=len(self.prices)-1, 
                          colors='black', linestyles='--', linewidth=1, alpha=0.7)
            # Mark the point
            ax_main.plot(change_point, price_at_change, 'ko', markersize=6, alpha=0.9)
            
            # Add resistance/support label
            if i < len(self.trend_change_points) - 1:
                next_change = self.trend_change_points[i+1]
                label = f"R{i+1}" if self.slope_history[self.endpoint_indices.index(change_point)] < 0 else f"S{i+1}"
                ax_main.text(change_point + 5, price_at_change, label, 
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
        
        # Create custom legend
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor=(0, 1, 0, 0.3), label='Uptrend (Positive Slope)'),
            Rectangle((0, 0), 1, 1, facecolor=(1, 0, 0, 0.3), label='Downtrend (Negative Slope)'),
            Line2D([0], [0], color='black', linestyle='--', linewidth=1, alpha=0.7, label='Support/Resistance'),
            Line2D([0], [0], marker='o', color='black', markersize=6, linestyle='None', label='Trend Change Point')
        ]
        ax_main.legend(handles=legend_elements, loc='upper left', fontsize=8, title='Legend')
        
        ax_main.set_title(f'BTC Price with Merged Trend Segments\n'
                         f'Green = Uptrend, Red = Downtrend | Black lines mark support/resistance levels', 
                         fontsize=14, fontweight='bold')
        ax_main.set_xlabel('Hours from Start')
        ax_main.set_ylabel('Price (USDT)')
        ax_main.grid(True, alpha=0.3)
        
        # Add date range info
        date_text = f'Period: {self.full_start_date.strftime("%Y-%m-%d")} to {self.full_end_date.strftime("%Y-%m-%d")}'
        ax_main.text(0.02, 0.98, date_text, transform=ax_main.transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add trend change count
        change_text = f'Trend Changes: {len(self.trend_change_points)}'
        ax_main.text(0.98, 0.98, change_text, transform=ax_main.transAxes, 
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        # Slope chart with merged segments (second)
        ax_slope = plt.subplot(gs[1])
        
        # Plot merged segments
        for indices, slope in zip(self.merged_indices, self.merged_slopes):
            color = 'green' if slope > 0 else 'red' if slope < 0 else 'gray'
            ax_slope.plot([indices[0], indices[-1]], [slope, slope], 
                         color=color, linewidth=3, alpha=0.7)
            # Fill between
            ax_slope.fill_between([indices[0], indices[-1]], 0, slope, 
                                  color=color, alpha=0.2)
        
        # Add zero line
        ax_slope.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        
        # Mark trend changes on slope chart
        for change_point in self.trend_change_points:
            ax_slope.axvline(x=change_point, color='black', linestyle=':', alpha=0.5, linewidth=1)
        
        ax_slope.set_title(f'Merged Slope Segments ({len(self.merged_slopes)} segments)', fontsize=12)
        ax_slope.set_xlabel('Hours from Start')
        ax_slope.set_ylabel('Slope ($/h)')
        ax_slope.grid(True, alpha=0.3)
        ax_slope.set_ylim(-max(abs(np.array(self.merged_slopes)))*1.1, max(abs(np.array(self.merged_slopes)))*1.1)
        
        # Lookback chart (third)
        ax_lookback = plt.subplot(gs[2])
        
        ax_lookback.plot(self.endpoint_indices, self.lookback_history, 'b-', linewidth=1.5, alpha=0.7, label='Optimal Lookback')
        
        # Mark trend changes on lookback chart
        for change_point in self.trend_change_points:
            if change_point in self.endpoint_indices:
                idx = self.endpoint_indices.index(change_point)
                ax_lookback.plot(change_point, self.lookback_history[idx], 'ko', markersize=6)
                ax_lookback.axvline(x=change_point, color='black', linestyle=':', alpha=0.3, linewidth=1)
        
        ax_lookback.set_xlabel('Hours from Start')
        ax_lookback.set_ylabel('Lookback (hours)', color='b')
        ax_lookback.tick_params(axis='y', labelcolor='b')
        ax_lookback.grid(True, alpha=0.3)
        ax_lookback.set_title('Optimal Lookback Window at Each Point', fontsize=12)
        
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
                
                # Get current trend
                current_slope = analyzer.slope_history[-1] if analyzer.slope_history else 0
                
                if current_slope > 2:
                    trend_direction = "🟢 STRONG UPTREND"
                elif current_slope > 0.5:
                    trend_direction = "🟢 UPTREND"
                elif current_slope > -0.5:
                    trend_direction = "⚪ SIDEWAYS"
                elif current_slope > -2:
                    trend_direction = "🔴 DOWNTREND"
                else:
                    trend_direction = "🔴 STRONG DOWNTREND"
                
                # Create HTML page
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>BTC Trend Analysis with Support/Resistance Levels</title>
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
                        .segment-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                        .segment-table th {{ background-color: #2196F3; color: white; padding: 10px; text-align: left; }}
                        .segment-table td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                        .segment-table tr:hover {{ background-color: #f5f5f5; }}
                        .uptrend {{ background-color: #d4edda; }}
                        .downtrend {{ background-color: #f8d7da; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>📊 BTC Trend Analysis with Support/Resistance Levels</h1>
                        <p>Merged trend segments with horizontal lines at trend changes</p>
                        
                        <div class="methodology">
                            <strong>🔬 Methodology:</strong> For every possible endpoint (starting from hour 110), 
                            we find the lookback window (10-100 hours) that minimizes average error. 
                            Consecutive points with the same slope sign are merged into trend segments.
                            When the sign changes, a horizontal line is drawn at that price level (support in uptrend, resistance in downtrend).
                        </div>
                        
                        <div class="trend-indicator">
                            <strong>Current Trend:</strong> {trend_direction} (${current_slope:+.2f}/h)
                        </div>
                        
                        <div class="info">
                            <h3>📈 Analysis Summary:</h3>
                            <p><strong>Period:</strong> {analyzer.full_start_date.strftime('%Y-%m-%d')} to {analyzer.full_end_date.strftime('%Y-%m-%d')}</p>
                            <p><strong>Points Analyzed:</strong> {len(analyzer.slope_history)}</p>
                            <p><strong>Merged Segments:</strong> {len(analyzer.merged_slopes)}</p>
                            <p><strong>Trend Changes:</strong> {len(analyzer.trend_change_points)}</p>
                            <p><strong>Avg Lookback:</strong> {np.mean(analyzer.lookback_history):.1f} hours</p>
                        </div>
                        
                        <h3>📋 Trend Segments:</h3>
                        <table class="segment-table">
                            <tr>
                                <th>Segment</th>
                                <th>Period</th>
                                <th>Duration</th>
                                <th>Trend</th>
                                <th>Avg Slope</th>
                            </tr>
                """
                
                # Add segment rows
                for i, (indices, slope) in enumerate(zip(analyzer.merged_indices, analyzer.merged_slopes)):
                    start_date = analyzer.full_start_date + timedelta(hours=indices[0])
                    end_date = analyzer.full_start_date + timedelta(hours=indices[-1])
                    duration = len(indices)
                    trend_class = "uptrend" if slope > 0 else "downtrend" if slope < 0 else ""
                    trend_icon = "🟢" if slope > 0 else "🔴" if slope < 0 else "⚪"
                    
                    html += f"""
                            <tr class="{trend_class}">
                                <td>{i+1}</td>
                                <td>{start_date.strftime('%m/%d %H:%M')} - {end_date.strftime('%m/%d %H:%M')}</td>
                                <td>{duration} hours</td>
                                <td>{trend_icon} {'Uptrend' if slope > 0 else 'Downtrend' if slope < 0 else 'Sideways'}</td>
                                <td>${slope:+.2f}/h</td>
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
                                <div class="stat-label">24h Volatility</div>
                                <div class="stat-value">${volatility:,.2f}</div>
                                <div class="stat-sub">price fluctuation</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">Current Slope</div>
                                <div class="stat-value">${current_slope:+.2f}/h</div>
                                <div class="stat-sub">trend strength</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">Current Lookback</div>
                                <div class="stat-value">{analyzer.lookback_history[-1]}h</div>
                                <div class="stat-sub">optimal window</div>
                            </div>
                        </div>
                        
                        <img src="data:image/png;base64,{image_base64}" alt="BTC Trend Analysis">
                        
                        <div class="footer">
                            <p>Data from Binance API | Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                            <p>Green areas = Uptrend segments, Red areas = Downtrend segments</p>
                            <p>Black horizontal lines mark support/resistance levels where trend changed</p>
                            <p>R = Resistance (trend changed from up to down), S = Support (trend changed from down to up)</p>
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
    print("BTC Trend Analysis with Support/Resistance Levels")
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