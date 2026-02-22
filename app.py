import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import http.server
import socketserver
import threading
import json
from io import BytesIO
import base64

# Configuration
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
DAYS = 30
PORT = 8000

def fetch_binance_data(symbol="BTCUSDT", interval="1h", days=30):
    """Fetch historical data from Binance"""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Convert interval to Binance format
    interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "4h": "4h", "1d": "1d"
    }
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval_map.get(interval, "1h"),
        "startTime": int(start_time.timestamp() * 1000),
        "endTime": int(end_time.timestamp() * 1000),
        "limit": 1000
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df = df[['timestamp', 'close']]
    
    return df

def optimize_window_and_k(prices, window_range=range(10, 101), k_range=np.arange(1.0, 2.1, 0.1)):
    """Find optimal window size and k value that minimize error"""
    results = []
    
    for window in window_range:
        if len(prices) < window + 1:  # Need at least window+1 candles for validation
            continue
            
        for k in k_range:
            # Use all but last candle for fitting
            fit_prices = prices[:-1]
            errors = []
            
            # Iterate through windows
            for i in range(len(fit_prices) - window + 1):
                window_prices = fit_prices[i:i+window]
                x = np.arange(window).reshape(-1, 1)
                y = window_prices.values.reshape(-1, 1)
                
                # Fit OLS line
                model = LinearRegression()
                model.fit(x, y)
                
                # Calculate predictions
                predictions = model.predict(x).flatten()
                
                # Calculate error term: sum(|actual - predicted|) / window^k
                error = np.sum(np.abs(window_prices - predictions)) / (window ** k)
                errors.append(error)
            
            # Average error for this window and k
            avg_error = np.mean(errors)
            results.append({
                'window': window,
                'k': k,
                'error': avg_error
            })
    
    # Find best parameters
    best = min(results, key=lambda x: x['error'])
    return best['window'], best['k']

def calculate_slopes_with_optimal_params(prices, window, k):
    """Calculate slopes using optimal window size and k"""
    slopes = []
    
    for i in range(len(prices) - window):
        window_prices = prices[i:i+window]
        x = np.arange(window).reshape(-1, 1)
        y = window_prices.values.reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(x, y)
        slopes.append(model.coef_[0][0])
    
    return slopes

def generate_trading_signals(slopes):
    """Generate trading signals based on slope direction"""
    return [1 if slope > 0 else -1 for slope in slopes]

def backtest_strategy(prices, signals):
    """Backtest the strategy"""
    # Align signals with prices (signals are for next candle)
    if len(signals) < len(prices) - 1:
        signals = signals + [0]  # Pad if necessary
    
    returns = []
    equity_curve = [1000]  # Start with 1000 USD
    
    for i in range(1, min(len(prices), len(signals) + 1)):
        if i <= len(signals):
            signal = signals[i-1]
            price_change = (prices.iloc[i] - prices.iloc[i-1]) / prices.iloc[i-1]
            trade_return = signal * price_change
            returns.append(trade_return)
            
            new_equity = equity_curve[-1] * (1 + trade_return)
            equity_curve.append(new_equity)
    
    return returns, equity_curve[1:]  # Skip initial equity

def generate_plot(prices, equity_curve, signals):
    """Generate matplotlib plot and return as base64 string"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Price chart with signals
    ax1.plot(prices.index, prices.values, label='BTC Price', color='black', alpha=0.7)
    
    # Mark buy/sell signals
    for i, signal in enumerate(signals):
        if i < len(prices) - 1:
            if signal == 1:
                ax1.scatter(prices.index[i+1], prices.iloc[i+1], color='green', marker='^', s=50)
            else:
                ax1.scatter(prices.index[i+1], prices.iloc[i+1], color='red', marker='v', s=50)
    
    ax1.set_title('BTC Price with Trading Signals')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Equity curve
    ax2.plot(prices.index[1:len(equity_curve)+1], equity_curve, color='blue', linewidth=2)
    ax2.set_title('Equity Curve')
    ax2.set_ylabel('Portfolio Value (USD)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1000, color='gray', linestyle='--', alpha=0.5)
    
    # Returns histogram
    returns = [equity_curve[i] - equity_curve[i-1] for i in range(1, len(equity_curve))]
    ax3.hist(returns, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax3.set_title('Returns Distribution')
    ax3.set_xlabel('Return')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def create_html_report(image_base64, stats):
    """Create HTML report with plot and statistics"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BTC Trading Strategy Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
            .stat-card h3 {{ margin: 0; font-size: 14px; opacity: 0.9; }}
            .stat-card p {{ margin: 10px 0 0; font-size: 24px; font-weight: bold; }}
            .plot {{ text-align: center; margin: 20px 0; }}
            .plot img {{ max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            .positive {{ color: #4caf50; }}
            .negative {{ color: #f44336; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>BTC Trading Strategy Backtest Results</h1>
            
            <div class="stats">
                <div class="stat-card">
                    <h3>Total Return</h3>
                    <p class="{'positive' if stats['total_return'] > 0 else 'negative'}">{stats['total_return']:.2f}%</p>
                </div>
                <div class="stat-card">
                    <h3>Number of Trades</h3>
                    <p>{stats['num_trades']}</p>
                </div>
                <div class="stat-card">
                    <h3>Win Rate</h3>
                    <p>{stats['win_rate']:.1f}%</p>
                </div>
                <div class="stat-card">
                    <h3>Optimal Window</h3>
                    <p>{stats['optimal_window']}</p>
                </div>
                <div class="stat-card">
                    <h3>Optimal k</h3>
                    <p>{stats['optimal_k']:.1f}</p>
                </div>
                <div class="stat-card">
                    <h3>Max Drawdown</h3>
                    <p class="negative">{stats['max_drawdown']:.2f}%</p>
                </div>
            </div>
            
            <div class="plot">
                <h2>Strategy Performance</h2>
                <img src="data:image/png;base64,{image_base64}" alt="Strategy Plot">
            </div>
            
            <p style="text-align: center; color: #666; margin-top: 20px;">
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </div>
    </body>
    </html>
    """
    return html

def calculate_statistics(equity_curve, returns):
    """Calculate performance statistics"""
    if not equity_curve or not returns:
        return {
            'total_return': 0,
            'num_trades': 0,
            'win_rate': 0,
            'optimal_window': 0,
            'optimal_k': 0,
            'max_drawdown': 0
        }
    
    total_return = (equity_curve[-1] / 1000 - 1) * 100
    
    winning_trades = sum(1 for r in returns if r > 0)
    num_trades = len(returns)
    win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
    
    # Calculate max drawdown
    peak = equity_curve[0]
    max_drawdown = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)
    
    return {
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown
    }

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Fetch data and run strategy
            df = fetch_binance_data(SYMBOL, INTERVAL, DAYS)
            prices = df['close']
            
            # Find optimal parameters
            optimal_window, optimal_k = optimize_window_and_k(prices)
            
            # Calculate slopes and generate signals
            slopes = calculate_slopes_with_optimal_params(prices, optimal_window, optimal_k)
            signals = generate_trading_signals(slopes)
            
            # Backtest
            returns, equity_curve = backtest_strategy(prices, signals)
            
            # Calculate statistics
            stats = calculate_statistics(equity_curve, returns)
            stats['optimal_window'] = optimal_window
            stats['optimal_k'] = optimal_k
            
            # Generate plot
            image_base64 = generate_plot(prices, equity_curve, signals)
            
            # Create HTML report
            html = create_html_report(image_base64, stats)
            self.wfile.write(html.encode())
        else:
            super().do_GET()

def run_server():
    """Run HTTP server"""
    handler = CustomHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Server running at http://localhost:{PORT}")
        print("Press Ctrl+C to stop")
        httpd.serve_forever()

if __name__ == "__main__":
    # Run server in a separate thread
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    print(f"Open your browser and go to http://localhost:{PORT}")
    print("Press Enter to stop the server...")
    input()