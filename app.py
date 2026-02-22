#!/usr/bin/env python3
"""
BTC Trading Strategy Backtest - Headless Cloud Version
Run with: python btc_strategy.py
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import json
import os
import argparse
from typing import Tuple, Dict, List
import logging

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
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from Binance: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

def optimize_window_and_k(prices: pd.Series, 
                         window_range: range = range(10, 101), 
                         k_range: np.ndarray = np.arange(1.0, 2.1, 0.1)) -> Tuple[int, float]:
    """Find optimal window size and k value that minimize error"""
    results = []
    
    for window in window_range:
        if len(prices) < window + 1:
            continue
            
        for k in k_range:
            fit_prices = prices[:-1]
            errors = []
            
            for i in range(len(fit_prices) - window + 1):
                window_prices = fit_prices[i:i+window]
                x = np.arange(window).reshape(-1, 1)
                y = window_prices.values.reshape(-1, 1)
                
                model = LinearRegression()
                model.fit(x, y)
                
                predictions = model.predict(x).flatten()
                error = np.sum(np.abs(window_prices - predictions)) / (window ** k)
                errors.append(error)
            
            if errors:
                avg_error = np.mean(errors)
                results.append({
                    'window': window,
                    'k': k,
                    'error': avg_error
                })
    
    if not results:
        return 20, 1.5  # Default values if optimization fails
    
    best = min(results, key=lambda x: x['error'])
    logger.info(f"Optimal window: {best['window']}, k: {best['k']:.2f}, error: {best['error']:.4f}")
    return best['window'], best['k']

def calculate_slopes_with_optimal_params(prices: pd.Series, window: int, k: float) -> List[float]:
    """Calculate slopes using optimal window size"""
    slopes = []
    
    for i in range(len(prices) - window):
        window_prices = prices[i:i+window]
        x = np.arange(window).reshape(-1, 1)
        y = window_prices.values.reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(x, y)
        slopes.append(model.coef_[0][0])
    
    return slopes

def backtest_strategy(prices: pd.Series, signals: List[int]) -> Tuple[List[float], List[float]]:
    """Backtest the strategy"""
    returns = []
    equity_curve = [1000.0]  # Start with 1000 USD
    
    for i in range(1, min(len(prices), len(signals) + 1)):
        if i <= len(signals):
            signal = signals[i-1]
            price_change = (prices.iloc[i] - prices.iloc[i-1]) / prices.iloc[i-1]
            trade_return = signal * price_change
            returns.append(trade_return)
            
            new_equity = equity_curve[-1] * (1 + trade_return)
            equity_curve.append(new_equity)
    
    return returns, equity_curve[1:]

def calculate_statistics(equity_curve: List[float], returns: List[float], 
                        optimal_window: int, optimal_k: float) -> Dict:
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
            'optimal_window': optimal_window,
            'optimal_k': round(optimal_k, 2)
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
        'optimal_window': optimal_window,
        'optimal_k': round(optimal_k, 2)
    }

def generate_plot(prices: pd.Series, equity_curve: List[float], 
                 signals: List[int], stats: Dict, output_file: str = 'strategy_plot.png'):
    """Generate and save matplotlib plot"""
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
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
            ax1.scatter(buy_dates, buy_prices, color='green', marker='^', s=50, label='Buy Signal', zorder=5)
        if sell_dates:
            ax1.scatter(sell_dates, sell_prices, color='red', marker='v', s=50, label='Sell Signal', zorder=5)
        
        ax1.set_title('BTC Price with Trading Signals')
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
        
        # Add statistics text
        stats_text = f"Total Return: {stats['total_return']}% | Sharpe: {stats['sharpe_ratio']} | Win Rate: {stats['win_rate']}%"
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Returns histogram
        returns = [equity_curve[i] - equity_curve[i-1] for i in range(1, len(equity_curve))]
        ax3.hist(returns, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax3.set_title('Returns Distribution')
        ax3.set_xlabel('Return (USD)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating plot: {e}")

def save_results(stats: Dict, output_dir: str = 'results'):
    """Save results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save statistics
    stats_file = os.path.join(output_dir, 'statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save summary text
    summary_file = os.path.join(output_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("BTC TRADING STRATEGY BACKTEST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Symbol: {SYMBOL}\n")
        f.write(f"Interval: {INTERVAL}\n")
        f.write(f"Period: {DAYS} days\n\n")
        f.write("OPTIMIZATION RESULTS:\n")
        f.write(f"  Optimal Window: {stats['optimal_window']}\n")
        f.write(f"  Optimal k: {stats['optimal_k']}\n\n")
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
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--no-plot', action='store_true', help='Skip generating plot')
    
    args = parser.parse_args()
    
    try:
        # Fetch data
        df = fetch_binance_data(args.symbol, args.interval, args.days)
        prices = df['close']
        
        # Find optimal parameters
        logger.info("Optimizing window size and k value...")
        optimal_window, optimal_k = optimize_window_and_k(prices)
        
        # Calculate slopes and generate signals
        logger.info("Calculating slopes and generating signals...")
        slopes = calculate_slopes_with_optimal_params(prices, optimal_window, optimal_k)
        signals = [1 if slope > 0 else -1 for slope in slopes]
        
        # Backtest
        logger.info("Running backtest...")
        returns, equity_curve = backtest_strategy(prices, signals)
        
        # Calculate statistics
        stats = calculate_statistics(equity_curve, returns, optimal_window, optimal_k)
        
        # Generate plot (if not disabled)
        if not args.no_plot:
            plot_file = os.path.join(args.output_dir, 'strategy_plot.png')
            generate_plot(prices, equity_curve, signals, stats, plot_file)
        
        # Save results
        save_results(stats, args.output_dir)
        
        # Print summary to console
        print("\n" + "="*50)
        print("BACKTEST SUMMARY")
        print("="*50)
        print(f"Optimal Window: {stats['optimal_window']}")
        print(f"Optimal k: {stats['optimal_k']}")
        print(f"Total Return: {stats['total_return']}%")
        print(f"Annual Return: {stats['annual_return']}%")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']}")
        print(f"Max Drawdown: {stats['max_drawdown']}%")
        print(f"Win Rate: {stats['win_rate']}%")
        print(f"Profit Factor: {stats['profit_factor']}")
        print(f"Number of Trades: {stats['num_trades']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()