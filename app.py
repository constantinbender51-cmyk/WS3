#!/usr/bin/env python3
"""
BTC Trading Strategy Backtest - Optimized per line with fixed k
Run with: python btc_strategy.py
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import argparse
from typing import Tuple, Dict, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def find_optimal_window_for_line(prices: pd.Series, window_range: range = range(10, 101), k: float = K_FIXED) -> Tuple[int, float]:
    """
    Find the optimal window size for a single line that minimizes the error term
    Returns: (optimal_window, min_error)
    """
    best_window = 10
    min_error = float('inf')
    
    for window in window_range:
        if len(prices) < window:
            continue
            
        x = np.arange(window).reshape(-1, 1)
        y = prices.values.reshape(-1, 1)
        
        # Fit OLS line
        model = LinearRegression()
        model.fit(x, y)
        
        # Calculate predictions
        predictions = model.predict(x).flatten()
        
        # Calculate error term: sum(|actual - predicted|) / window^k
        error = np.sum(np.abs(prices.values - predictions)) / (window ** k)
        
        if error < min_error:
            min_error = error
            best_window = window
    
    return best_window, min_error

def process_single_window(args: Tuple) -> Tuple[int, float, float]:
    """
    Process a single window to find optimal line and slope
    Returns: (index, slope, optimal_window)
    """
    i, window_prices, window_range, k = args
    
    # Find optimal window size for this specific line
    optimal_window, min_error = find_optimal_window_for_line(window_prices, window_range, k)
    
    # Fit line with optimal window
    x = np.arange(optimal_window).reshape(-1, 1)
    y = window_prices.values.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_[0][0]
    
    return i, slope, optimal_window

def calculate_signals_with_optimal_windows(prices: pd.Series, 
                                          window_range: range = range(10, 101),
                                          k: float = K_FIXED,
                                          parallel: bool = True) -> Tuple[List[int], List[int], List[float]]:
    """
    For each line, find optimal window size and calculate slope
    Returns: (signals, optimal_windows, slopes)
    """
    signals = []
    optimal_windows = []
    slopes = []
    
    total_windows = len(prices) - min(window_range)
    logger.info(f"Processing {total_windows} windows...")
    
    if parallel and total_windows > 50:  # Use parallel processing for larger datasets
        # Prepare arguments for parallel processing
        args_list = []
        for i in range(len(prices) - min(window_range)):
            # Use all available prices up to the current point for optimization
            window_prices = prices.iloc[:i + min(window_range)]
            if len(window_prices) >= min(window_range):
                args_list.append((i, window_prices, window_range, k))
        
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
        for i in range(len(prices) - min(window_range)):
            # Use all available prices up to the current point for optimization
            window_prices = prices.iloc[:i + min(window_range)]
            
            if len(window_prices) >= min(window_range):
                # Find optimal window for this line
                optimal_window, _ = find_optimal_window_for_line(window_prices, window_range, k)
                
                # Fit line with optimal window
                x = np.arange(optimal_window).reshape(-1, 1)
                y = window_prices.values.reshape(-1, 1)
                
                model = LinearRegression()
                model.fit(x, y)
                slope = model.coef_[0][0]
                
                slopes.append(slope)
                optimal_windows.append(optimal_window)
                signals.append(1 if slope > 0 else -1)
    
    logger.info(f"Generated {len(signals)} signals")
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
                        optimal_windows: List[int]) -> Dict:
    """Calculate performance statistics"""
    if not equity_curve or not returns:
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
            'k_fixed': K_FIXED
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
        'k_fixed': K_FIXED
    }

def generate_plot(prices: pd.Series, equity_curve: List[float], 
                 signals: List[int], optimal_windows: List[int], 
                 stats: Dict, output_file: str = 'strategy_plot.png'):
    """Generate and save matplotlib plot"""
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
        
        ax1.set_title(f'BTC Price with Trading Signals (k={K_FIXED})')
        ax1.set_ylabel('Price (USDT)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Equity curve
        equity_dates = prices.index[1:len(equity_curve)+1]
        ax2.plot(equity_dates, equity_curve, color='blue', linewidth=2, label='Portfolio Value')
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Portfolio Value (USD)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax2.legend(loc='upper left')
        
        # Optimal windows distribution
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
        
        # Add statistics text
        stats_text = (
            f"Total Return: {stats['total_return']}%\n"
            f"Sharpe: {stats['sharpe_ratio']}\n"
            f"Win Rate: {stats['win_rate']}%\n"
            f"k fixed: {stats['k_fixed']}"
        )
        ax2.text(0.02, 0.85, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating plot: {e}")

def save_results(stats: Dict, signals: List[int], optimal_windows: List[int], 
                slopes: List[float], output_dir: str = 'results'):
    """Save results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete results
    results = {
        'statistics': stats,
        'signals': signals,
        'optimal_windows': optimal_windows,
        'slopes': [float(s) for s in slopes],  # Convert numpy floats to Python floats
        'generated_at': datetime.now().isoformat(),
        'parameters': {
            'symbol': SYMBOL,
            'interval': INTERVAL,
            'days': DAYS,
            'k_fixed': K_FIXED,
            'window_range': '10-100'
        }
    }
    
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
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
        f.write(f"  k (fixed): {K_FIXED}\n")
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
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--no-plot', action='store_true', help='Skip generating plot')
    parser.add_argument('--sequential', action='store_true', help='Use sequential processing (no parallel)')
    
    args = parser.parse_args()
    
    try:
        # Update global K_FIXED if provided
        global K_FIXED
        K_FIXED = args.k
        
        # Fetch data
        df = fetch_binance_data(args.symbol, args.interval, args.days)
        prices = df['close']
        
        # Calculate signals with optimal windows for each line
        logger.info(f"Calculating signals with fixed k={K_FIXED}...")
        signals, optimal_windows, slopes = calculate_signals_with_optimal_windows(
            prices, 
            window_range=range(10, 101),
            k=K_FIXED,
            parallel=not args.sequential
        )
        
        # Backtest
        logger.info("Running backtest...")
        returns, equity_curve = backtest_strategy(prices, signals)
        
        # Calculate statistics
        stats = calculate_statistics(equity_curve, returns, optimal_windows)
        
        # Generate plot (if not disabled)
        if not args.no_plot:
            plot_file = os.path.join(args.output_dir, 'strategy_plot.png')
            generate_plot(prices, equity_curve, signals, optimal_windows, stats, plot_file)
        
        # Save results
        save_results(stats, signals, optimal_windows, slopes, args.output_dir)
        
        # Print summary to console
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        print(f"k (fixed): {K_FIXED}")
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
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()