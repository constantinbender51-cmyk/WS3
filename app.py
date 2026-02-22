#!/usr/bin/env python3
"""
BTC Trading Strategy Backtest - With HTTP Server Option
Run with: python btc_strategy.py --serve
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os
import argparse
from typing import Tuple, Dict, List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import http.server
import socketserver
import threading
import base64
from io import BytesIO
import urllib.parse
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
DAYS = 30
K_FIXED = 1.8  # Fixed k value as specified
PORT = 8000

# Global variable to cache results for HTTP server
cached_results = None
cached_html = None

def fetch_binance_data(symbol: str = "BTCUSDT", interval: str = "1h", days: int = 30) -> pd.DataFrame:
    """Fetch historical data from Binance"""
    try:
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
        
        logger.info(f"Fetching data for {symbol}...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = df['close'].astype(float)
        df = df[['timestamp', 'close']]
        
        logger.info(f"Fetched {len(df)} candles")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data from Binance: {e}")
        raise

def find_optimal_window_for_line(prices: pd.Series, window_range: range = range(10, 101), k: float = 1.8) -> Tuple[int, float]:
    """
    Find the optimal window size for a single line that minimizes the error term
    Returns: (optimal_window, min_error)
    """
    best_window = 10
    min_error = float('inf')
    
    # Ensure we have enough data points
    max_window = min(max(window_range), len(prices))
    if max_window < min(window_range):
        return min(window_range), float('inf')
    
    for window in window_range:
        if window > len(prices):
            continue
            
        # Use the last 'window' data points
        recent_prices = prices.iloc[-window:].values
        
        x = np.arange(window).reshape(-1, 1)
        y = recent_prices.reshape(-1, 1)
        
        try:
            # Fit OLS line
            model = LinearRegression()
            model.fit(x, y)
            
            # Calculate predictions
            predictions = model.predict(x).flatten()
            
            # Calculate error term: sum(|actual - predicted|) / window^k
            error = np.sum(np.abs(recent_prices - predictions)) / (window ** k)
            
            if error < min_error:
                min_error = error
                best_window = window
        except Exception as e:
            logger.debug(f"Error fitting line with window {window}: {e}")
            continue
    
    return best_window, min_error

def process_single_window(args: Tuple) -> Tuple[int, float, int]:
    """
    Process a single window to find optimal line and slope
    Returns: (index, slope, optimal_window)
    """
    i, all_prices, min_window, max_window, k = args
    
    # Get prices up to current point
    current_prices = all_prices.iloc[:i + min_window]
    
    if len(current_prices) < min_window:
        return i, 0, min_window
    
    # Create window range for this specific point
    max_possible_window = min(max_window, len(current_prices))
    window_range = range(min_window, max_possible_window + 1)
    
    # Find optimal window for this line
    optimal_window, _ = find_optimal_window_for_line(current_prices, window_range, k)
    
    # Use the optimal window to fit the line on the most recent data
    recent_prices = current_prices.iloc[-optimal_window:].values
    x = np.arange(optimal_window).reshape(-1, 1)
    y = recent_prices.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_[0][0]
    
    return i, slope, optimal_window

def calculate_signals_with_optimal_windows(prices: pd.Series, 
                                          min_window: int = 10,
                                          max_window: int = 100,
                                          k: float = 1.8,
                                          parallel: bool = True) -> Tuple[List[int], List[int], List[float]]:
    """
    For each line, find optimal window size and calculate slope
    Returns: (signals, optimal_windows, slopes)
    """
    signals = []
    optimal_windows = []
    slopes = []
    
    total_windows = len(prices) - min_window
    logger.info(f"Processing {total_windows} windows...")
    
    if parallel and total_windows > 50:  # Use parallel processing for larger datasets
        # Prepare arguments for parallel processing
        args_list = []
        for i in range(total_windows):
            args_list.append((i, prices, min_window, max_window, k))
        
        # Process in parallel
        results = [None] * len(args_list)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_single_window, args) for args in args_list]
            
            for future in as_completed(futures):
                i, slope, optimal_window = future.result()
                results[i] = (slope, optimal_window)
        
        # Sort results by index
        for i, result in enumerate(results):
            if result is not None:
                slope, optimal_window = result
                slopes.append(slope)
                optimal_windows.append(optimal_window)
                signals.append(1 if slope > 0 else -1)
    
    else:  # Sequential processing
        for i in range(total_windows):
            # Get prices up to current point
            current_prices = prices.iloc[:i + min_window]
            
            if len(current_prices) < min_window:
                continue
            
            # Determine max possible window for this point
            max_possible_window = min(max_window, len(current_prices))
            window_range = range(min_window, max_possible_window + 1)
            
            # Find optimal window for this line
            optimal_window, _ = find_optimal_window_for_line(current_prices, window_range, k)
            
            # Use the optimal window to fit the line on the most recent data
            recent_prices = current_prices.iloc[-optimal_window:].values
            x = np.arange(optimal_window).reshape(-1, 1)
            y = recent_prices.reshape(-1, 1)
            
            model = LinearRegression()
            model.fit(x, y)
            slope = model.coef_[0][0]
            
            slopes.append(slope)
            optimal_windows.append(optimal_window)
            signals.append(1 if slope > 0 else -1)
    
    logger.info(f"Generated {len(signals)} signals")
    if optimal_windows:
        logger.info(f"Optimal windows range: {min(optimal_windows)}-{max(optimal_windows)}")
    
    return signals, optimal_windows, slopes

def backtest_strategy(prices: pd.Series, signals: List[int]) -> Tuple[List[float], List[float]]:
    """Backtest the strategy"""
    returns = []
    equity_curve = [1000.0]  # Start with 1000 USD
    
    # Align signals with prices (signal for next candle)
    min_length = min(len(prices) - 1, len(signals))
    
    for i in range(min_length):
        signal = signals[i]
        price_change = (prices.iloc[i + 1] - prices.iloc[i]) / prices.iloc[i]
        trade_return = signal * price_change
        returns.append(trade_return)
        
        new_equity = equity_curve[-1] * (1 + trade_return)
        equity_curve.append(new_equity)
    
    return returns, equity_curve[1:]

def calculate_statistics(equity_curve: List[float], returns: List[float], 
                        optimal_windows: List[int], k_fixed: float) -> Dict:
    """Calculate performance statistics"""
    if not equity_curve or not returns or not optimal_windows:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'num_trades': 0,
            'profit_factor': 0.0,
            'avg_optimal_window': 0,
            'min_optimal_window': 0,
            'max_optimal_window': 0,
            'k_fixed': k_fixed
        }
    
    total_return = (equity_curve[-1] / 1000 - 1) * 100
    
    # Annualized return (assuming 365*24 trading hours for crypto)
    hours = len(returns)
    annual_return = ((1 + total_return/100) ** (8760/hours) - 1) * 100 if hours > 0 else 0
    
    # Sharpe ratio (assuming 0% risk-free rate)
    returns_array = np.array(returns)
    sharpe_ratio = np.sqrt(8760) * returns_array.mean() / returns_array.std() if returns_array.std() > 0 else 0
    
    # Win rate
    winning_trades = sum(1 for r in returns if r > 0)
    num_trades = len(returns)
    win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
    
    # Profit factor
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max drawdown
    peak = equity_curve[0]
    max_drawdown = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)
    
    return {
        'total_return': round(total_return, 2),
        'annual_return': round(annual_return, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'max_drawdown': round(max_drawdown, 2),
        'win_rate': round(win_rate, 1),
        'num_trades': num_trades,
        'profit_factor': round(profit_factor, 2),
        'avg_optimal_window': round(np.mean(optimal_windows), 1),
        'min_optimal_window': min(optimal_windows),
        'max_optimal_window': max(optimal_windows),
        'k_fixed': k_fixed
    }

def generate_plot_base64(prices: pd.Series, equity_curve: List[float], 
                        signals: List[int], optimal_windows: List[int], 
                        stats: Dict) -> str:
    """Generate plot and return as base64 string"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Price chart with signals
        ax1.plot(prices.index, prices.values, label='BTC Price', color='black', alpha=0.7, linewidth=1)
        
        # Mark buy/sell signals
        buy_dates, buy_prices = [], []
        sell_dates, sell_prices = [], []
        
        for i, signal in enumerate(signals):
            if i < len(prices) - 1:
                if signal == 1:
                    buy_dates.append(prices.index[i+1])
                    buy_prices.append(prices.iloc[i+1])
                else:
                    sell_dates.append(prices.index[i+1])
                    sell_prices.append(prices.iloc[i+1])
        
        if buy_dates:
            ax1.scatter(buy_dates, buy_prices, color='green', marker='^', s=50, label='Buy Signal', zorder=5, alpha=0.7)
        if sell_dates:
            ax1.scatter(sell_dates, sell_prices, color='red', marker='v', s=50, label='Sell Signal', zorder=5, alpha=0.7)
        
        ax1.set_title(f'BTC Price with Trading Signals (k={stats["k_fixed"]})')
        ax1.set_ylabel('Price (USDT)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # Equity curve
        equity_dates = prices.index[1:len(equity_curve)+1]
        ax2.plot(equity_dates, equity_curve, color='blue', linewidth=2, label='Portfolio Value')
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Portfolio Value (USD)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax2.legend(loc='upper left')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # Optimal windows distribution
        if optimal_windows:
            ax3.hist(optimal_windows, bins=20, color='orange', alpha=0.7, edgecolor='black')
            ax3.set_title('Optimal Window Size Distribution')
            ax3.set_xlabel('Window Size')
            ax3.set_ylabel('Frequency')
            ax3.axvline(x=stats['avg_optimal_window'], color='red', linestyle='--', label=f"Avg: {stats['avg_optimal_window']}")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Returns histogram
        returns = [equity_curve[i] - equity_curve[i-1] for i in range(1, len(equity_curve))]
        ax4.hist(returns, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax4.set_title('Returns Distribution')
        ax4.set_xlabel('Return (USD)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64
        
    except Exception as e:
        logger.error(f"Error generating plot: {e}")
        return ""

def generate_html_report(prices: pd.Series, equity_curve: List[float], 
                        signals: List[int], optimal_windows: List[int], 
                        stats: Dict, image_base64: str) -> str:
    """Generate HTML report"""
    
    # Calculate some additional metrics for display
    total_candles = len(prices)
    trading_days = len(prices) * (1 if INTERVAL.endswith('h') else 24) / 24
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BTC Trading Strategy Report</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                backdrop-filter: blur(10px);
            }}
            
            h1 {{
                color: #333;
                margin-bottom: 20px;
                font-size: 2.5em;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }}
            
            h2 {{
                color: #555;
                margin: 20px 0 15px 0;
                font-size: 1.5em;
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                transition: transform 0.3s ease;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            }}
            
            .stat-card:hover {{
                transform: translateY(-5px);
            }}
            
            .stat-card h3 {{
                font-size: 16px;
                font-weight: 400;
                margin-bottom: 10px;
                opacity: 0.9;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .stat-card p {{
                font-size: 32px;
                font-weight: 700;
                margin: 0;
            }}
            
            .stat-card.positive p {{ color: #4ade80; }}
            .stat-card.negative p {{ color: #f87171; }}
            
            .plot-container {{
                background: white;
                border-radius: 15px;
                padding: 20px;
                margin: 30px 0;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            }}
            
            .plot-container img {{
                width: 100%;
                height: auto;
                border-radius: 10px;
            }}
            
            .info-panel {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }}
            
            .info-card {{
                background: #f8f9fa;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            }}
            
            .info-card h3 {{
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.2em;
            }}
            
            .info-item {{
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #e9ecef;
            }}
            
            .info-item:last-child {{
                border-bottom: none;
            }}
            
            .info-label {{
                color: #6c757d;
                font-weight: 500;
            }}
            
            .info-value {{
                color: #333;
                font-weight: 600;
            }}
            
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #e9ecef;
                color: #6c757d;
            }}
            
            .badge {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                margin-left: 10px;
            }}
            
            .badge.success {{
                background: #4ade80;
                color: white;
            }}
            
            .badge.warning {{
                background: #fbbf24;
                color: white;
            }}
            
            .refresh-btn {{
                background: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 10px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: background 0.3s ease;
                margin-bottom: 20px;
            }}
            
            .refresh-btn:hover {{
                background: #5a67d8;
            }}
            
            @media (max-width: 768px) {{
                .container {{
                    padding: 15px;
                }}
                
                h1 {{
                    font-size: 2em;
                }}
                
                .stat-card p {{
                    font-size: 24px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h1>🚀 BTC Trading Strategy Report</h1>
                <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Data</button>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Return</h3>
                    <p class="{'positive' if stats['total_return'] > 0 else 'negative'}">{stats['total_return']}%</p>
                </div>
                
                <div class="stat-card">
                    <h3>Sharpe Ratio</h3>
                    <p>{stats['sharpe_ratio']}</p>
                </div>
                
                <div class="stat-card">
                    <h3>Win Rate</h3>
                    <p>{stats['win_rate']}%</p>
                </div>
                
                <div class="stat-card">
                    <h3>Max Drawdown</h3>
                    <p class="negative">{stats['max_drawdown']}%</p>
                </div>
                
                <div class="stat-card">
                    <h3>Profit Factor</h3>
                    <p>{stats['profit_factor']}</p>
                </div>
                
                <div class="stat-card">
                    <h3>Total Trades</h3>
                    <p>{stats['num_trades']}</p>
                </div>
            </div>
            
            <div class="plot-container">
                <img src="data:image/png;base64,{image_base64}" alt="Strategy Performance">
            </div>
            
            <div class="info-panel">
                <div class="info-card">
                    <h3>📊 Strategy Parameters</h3>
                    <div class="info-item">
                        <span class="info-label">Fixed k value:</span>
                        <span class="info-value">{stats['k_fixed']}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Window range:</span>
                        <span class="info-value">10-100</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Symbol:</span>
                        <span class="info-value">{SYMBOL}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Interval:</span>
                        <span class="info-value">{INTERVAL}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Period:</span>
                        <span class="info-value">{DAYS} days</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Data points:</span>
                        <span class="info-value">{total_candles}</span>
                    </div>
                </div>
                
                <div class="info-card">
                    <h3>📈 Window Optimization</h3>
                    <div class="info-item">
                        <span class="info-label">Average window:</span>
                        <span class="info-value">{stats['avg_optimal_window']}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Min window:</span>
                        <span class="info-value">{stats['min_optimal_window']}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Max window:</span>
                        <span class="info-value">{stats['max_optimal_window']}</span>
                    </div>
                </div>
                
                <div class="info-card">
                    <h3>💰 Performance Details</h3>
                    <div class="info-item">
                        <span class="info-label">Annual Return:</span>
                        <span class="info-value">{stats['annual_return']}%</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Final Equity:</span>
                        <span class="info-value">${equity_curve[-1]:.2f}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Best Trade:</span>
                        <span class="info-value positive">+{max([r for r in returns if r > 0] or [0])*100:.2f}%</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Worst Trade:</span>
                        <span class="info-value negative">{min([r for r in returns if r < 0] or [0])*100:.2f}%</span>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
                <p style="font-size: 12px; margin-top: 10px;">Data source: Binance API</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def run_strategy(k_value: float = K_FIXED, min_window: int = 10, max_window: int = 100) -> Dict:
    """Run the complete strategy and return results"""
    global cached_results, cached_html
    
    # Fetch data
    df = fetch_binance_data(SYMBOL, INTERVAL, DAYS)
    prices = df['close']
    
    logger.info(f"Loaded {len(prices)} price points")
    
    # Calculate signals with optimal windows for each line
    logger.info(f"Calculating signals with fixed k={k_value}...")
    signals, optimal_windows, slopes = calculate_signals_with_optimal_windows(
        prices, 
        min_window=min_window,
        max_window=max_window,
        k=k_value,
        parallel=True
    )
    
    # Backtest
    logger.info("Running backtest...")
    returns, equity_curve = backtest_strategy(prices, signals)
    
    # Calculate statistics
    stats = calculate_statistics(equity_curve, returns, optimal_windows, k_value)
    
    # Generate plot
    image_base64 = generate_plot_base64(prices, equity_curve, signals, optimal_windows, stats)
    
    # Generate HTML
    html = generate_html_report(prices, equity_curve, signals, optimal_windows, stats, image_base64)
    
    # Cache results
    cached_results = {
        'prices': prices,
        'equity_curve': equity_curve,
        'signals': signals,
        'optimal_windows': optimal_windows,
        'stats': stats,
        'slopes': slopes,
        'image_base64': image_base64
    }
    cached_html = html
    
    return cached_results

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global cached_html
        
        parsed_path = urllib.parse.urlparse(self.path)
        
        if parsed_path.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            if cached_html is None:
                # Run strategy if not cached
                run_strategy()
            
            self.wfile.write(cached_html.encode())
            
        elif parsed_path.path == '/api/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if cached_results is None:
                run_strategy()
            
            # Convert numpy types to Python types for JSON
            stats_json = json.dumps(cached_results['stats'], default=lambda x: float(x) if isinstance(x, np.floating) else x)
            self.wfile.write(stats_json.encode())
            
        elif parsed_path.path == '/api/refresh':
            # Force refresh
            run_strategy()
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
            
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<h1>404 Not Found</h1>')
    
    def log_message(self, format, *args):
        # Suppress HTTP server logs
        pass

def run_http_server(port: int = PORT):
    """Run HTTP server"""
    handler = CustomHTTPRequestHandler
    
    # Run strategy once at startup
    logger.info("Pre-calculating strategy results...")
    run_strategy()
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        logger.info(f"🌐 HTTP Server running at http://localhost:{port}")
        logger.info("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped")

def save_results_to_file(stats: Dict, signals: List[int], optimal_windows: List[int], 
                        slopes: List[float], image_base64: str, output_dir: str = 'results'):
    """Save results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete results
    results = {
        'statistics': stats,
        'signals': signals,
        'optimal_windows': optimal_windows,
        'slopes': [float(s) for s in slopes],
        'generated_at': datetime.now().isoformat(),
        'parameters': {
            'symbol': SYMBOL,
            'interval': INTERVAL,
            'days': DAYS,
            'k_fixed': stats['k_fixed'],
            'window_range': '10-100'
        }
    }
    
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save HTML report
    html_file = os.path.join(output_dir, 'report.html')
    
    # We need to recreate the HTML with the image embedded
    # For file saving, we can use the base64 image directly
    html_content = generate_html_report(
        pd.Series(dtype=float),  # Dummy, not used in HTML generation
        [],  # Dummy
        [],  # Dummy
        [],  # Dummy
        stats,
        image_base64
    )
    
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    # Save plot as PNG
    if image_base64:
        plot_file = os.path.join(output_dir, 'strategy_plot.png')
        with open(plot_file, 'wb') as f:
            f.write(base64.b64decode(image_base64))
    
    # Save summary text
    summary_file = os.path.join(output_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BTC TRADING STRATEGY BACKTEST RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Symbol: {SYMBOL}\n")
        f.write(f"Interval: {INTERVAL}\n")
        f.write(f"Period: {DAYS} days\n\n")
        f.write("STRATEGY PARAMETERS:\n")
        f.write(f"  k (fixed): {stats['k_fixed']}\n")
        f.write(f"  Window range: 10-100\n\n")
        f.write("OPTIMIZATION RESULTS:\n")
        f.write(f"  Average Optimal Window: {stats['avg_optimal_window']}\n")
        f.write(f"  Min Optimal Window: {stats['min_optimal_window']}\n")
        f.write(f"  Max Optimal Window: {stats['max_optimal_window']}\n\n")
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  Total Return: {stats['total_return']}%\n")
        f.write(f"  Annual Return: {stats['annual_return']}%\n")
        f.write(f"  Sharpe Ratio: {stats['sharpe_ratio']}\n")
        f.write(f"  Max Drawdown: {stats['max_drawdown']}%\n")
        f.write(f"  Win Rate: {stats['win_rate']}%\n")
        f.write(f"  Profit Factor: {stats['profit_factor']}\n")
        f.write(f"  Number of Trades: {stats['num_trades']}\n")
    
    logger.info(f"Results saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='BTC Trading Strategy Backtest')
    parser.add_argument('--symbol', type=str, default=SYMBOL, help='Trading pair symbol')
    parser.add_argument('--interval', type=str, default=INTERVAL, help='Candle interval')
    parser.add_argument('--days', type=int, default=DAYS, help='Days of historical data')
    parser.add_argument('--k', type=float, default=K_FIXED, help='Fixed k value (default: 1.8)')
    parser.add_argument('--min-window', type=int, default=10, help='Minimum window size')
    parser.add_argument('--max-window', type=int, default=100, help='Maximum window size')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--no-plot', action='store_true', help='Skip generating plot')
    parser.add_argument('--sequential', action='store_true', help='Use sequential processing (no parallel)')
    parser.add_argument('--serve', action='store_true', help='Run HTTP server instead of CLI')
    parser.add_argument('--port', type=int, default=PORT, help='HTTP server port (default: 8000)')
    
    args = parser.parse_args()
    
    # Update global config
    global SYMBOL, INTERVAL, DAYS, K_FIXED, PORT
    SYMBOL = args.symbol
    INTERVAL = args.interval
    DAYS = args.days
    K_FIXED = args.k
    PORT = args.port
    
    if args.serve:
        # Run HTTP server
        run_http_server(args.port)
    else:
        # Run CLI version
        try:
            # Run strategy
            results = run_strategy(args.k, args.min_window, args.max_window)
            
            # Save results
            if not args.no_plot:
                save_results_to_file(
                    results['stats'], 
                    results['signals'], 
                    results['optimal_windows'], 
                    results['slopes'],
                    results['image_base64'],
                    args.output_dir
                )
            
            # Print summary to console
            stats = results['stats']
            print("\n" + "="*60)
            print("BACKTEST SUMMARY")
            print("="*60)
            print(f"k (fixed): {stats['k_fixed']}")
            print(f"Average Optimal Window: {stats['avg_optimal_window']}")
            print(f"Min/Max Window: {stats['min_optimal_window']}/{stats['max_optimal_window']}")
            print("-" * 40)
            print(f"Total Return: {stats['total_return']}%")
            print(f"Annual Return: {stats['annual_return']}%")
            print(f"Sharpe Ratio: {stats['sharpe_ratio']}")
            print(f"Max Drawdown: {stats['max_drawdown']}%")
            print(f"Win Rate: {stats['win_rate']}%")
            print(f"Profit Factor: {stats['profit_factor']}")
            print(f"Number of Trades: {stats['num_trades']}")
            print("="*60)
            
            if not args.no_plot:
                print(f"\n📊 Report saved to: {args.output_dir}/report.html")
                print(f"📈 Plot saved to: {args.output_dir}/strategy_plot.png")
            
        except Exception as e:
            logger.error(f"Error in main execution: {e}")
            raise

if __name__ == "__main__":
    main()