import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import http.server
import socketserver
import io
import base64
from datetime import datetime, timedelta
import requests
import argparse
from sklearn.linear_model import LinearRegression
import time
import threading
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BTCTrendAnalyzer:
    def __init__(self, symbol='BTCUSDT', interval='1h', days=30):
        self.symbol = symbol
        self.interval = interval
        self.days = days
        self.data = None
        self.optimized_lines = []
        self.horizontal_lines = []
        self.trades = []
        self.live_trades = []  # Store live trading results
        self.current_position = None
        self.current_signal = None
        self.last_update = None
        self.position_open_time = None
        self.position_open_price = None
        self.pnl_history = []
        self.total_pnl = 0
        
    def fetch_binance_data(self):
        """Fetch price data from Binance"""
        base_url = "https://api.binance.com/api/v3/klines"
        
        # Calculate start time
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.days)
        
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000),
            'limit': 1000
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close'] = df['close'].astype(float)
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            
            self.data = df[['timestamp', 'close']].copy()
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Generate sample data for testing
            self.generate_sample_data()
            return False
    
    def generate_sample_data(self):
        """Generate sample data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=24*30, freq='1h')
        # Create a trending price with some noise
        trend = np.linspace(40000, 45000, len(dates))
        noise = np.random.normal(0, 1000, len(dates))
        prices = trend + noise
        
        self.data = pd.DataFrame({
            'timestamp': dates,
            'close': prices
        })
        print("Generated sample data for testing")
    
    def fetch_latest_candles(self, limit=100):
        """Fetch only the most recent candles for live trading"""
        base_url = "https://api.binance.com/api/v3/klines"
        
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'limit': limit
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close'] = df['close'].astype(float)
            
            return df[['timestamp', 'close']].copy()
            
        except Exception as e:
            print(f"Error fetching latest candles: {e}")
            return None
    
    def ols_fit(self, prices):
        """Fit OLS line to price data"""
        x = np.arange(len(prices)).reshape(-1, 1)
        y = prices.values.reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(x, y)
        
        return model, model.predict(x)
    
    def calculate_error(self, prices, predicted_prices):
        """Calculate error term: sum(|line-price| / prices^2)"""
        errors = np.abs(predicted_prices.flatten() - prices.values) / (prices.values ** 2)
        return np.sum(errors)
    
    def find_optimal_window(self, prices, min_window=10, max_window=100):
        """Find optimal number of prices for OLS fit"""
        best_error = float('inf')
        best_window = min_window
        best_line = None
        best_slope = 0
        best_model = None
        
        for window in range(min_window, min(max_window, len(prices)) + 1):
            window_prices = prices[-window:]
            model, predicted = self.ols_fit(window_prices)
            error = self.calculate_error(window_prices, predicted)
            
            if error < best_error:
                best_error = error
                best_window = window
                best_line = predicted
                best_slope = model.coef_[0][0]
                best_model = model
        
        return best_window, best_line, best_slope, best_error, best_model
    
    def analyze_trends(self):
        """Main analysis function - YOUR ORIGINAL FUNCTION"""
        if self.data is None or len(self.data) < 100:
            print("Insufficient data")
            return
        
        prices = self.data['close'].copy()
        timestamps = self.data['timestamp'].copy()
        
        # Start from the end and work backwards
        current_position = len(prices)
        min_window = 10
        
        while current_position > min_window:
            # Get prices up to current position
            current_prices = prices.iloc[:current_position]
            
            # Find optimal window
            window_size, line_values, slope, error, model = self.find_optimal_window(
                current_prices, 
                min_window=min_window, 
                max_window=min(100, current_position)
            )
            
            # Store the line
            start_idx = current_position - window_size
            end_idx = current_position
            
            line_data = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': timestamps.iloc[start_idx],
                'end_time': timestamps.iloc[end_idx-1],
                'values': line_values.flatten(),
                'slope': slope,
                'color': 'green' if slope > 0 else 'red',
                'window_size': window_size,
                'error': error
            }
            
            self.optimized_lines.append(line_data)
            
            # Calculate trade return
            start_price = prices.iloc[start_idx]
            end_price = prices.iloc[end_idx-1]
            
            if slope > 0:
                trade_return = (end_price - start_price) / start_price
            else:
                trade_return = -1 * ((end_price - start_price) / start_price)
            
            self.trades.append({
                'start_time': timestamps.iloc[start_idx],
                'end_time': timestamps.iloc[end_idx-1],
                'return': trade_return,
                'type': 'long' if slope > 0 else 'short'
            })
            
            # Move position back (remove prices and continue)
            current_position = start_idx
            
            # Check for color change and mark horizontal line
            if len(self.optimized_lines) >= 2:
                prev_line = self.optimized_lines[-2]
                curr_line = self.optimized_lines[-1]
                
                if prev_line['color'] != curr_line['color']:
                    # Mark horizontal line at the transition point
                    self.horizontal_lines.append({
                        'time': timestamps.iloc[start_idx],
                        'price': prices.iloc[start_idx]
                    })
    
    def get_current_signal(self):
        """Get current trading signal based on latest data"""
        if self.data is None or len(self.data) < 100:
            return None
        
        prices = self.data['close'].copy()
        
        # Find optimal window for most recent data
        window_size, line_values, slope, error, model = self.find_optimal_window(
            prices, min_window=10, max_window=100
        )
        
        # Determine signal
        signal = 'long' if slope > 0 else 'short'
        
        # Get line values for the most recent points
        recent_prices = prices[-window_size:]
        x_recent = np.arange(len(recent_prices)).reshape(-1, 1)
        line_recent = model.predict(x_recent).flatten()
        
        return {
            'signal': signal,
            'slope': slope,
            'window_size': window_size,
            'error': error,
            'current_price': prices.iloc[-1],
            'line_price': line_recent[-1],
            'timestamp': self.data['timestamp'].iloc[-1],
            'model': model,
            'recent_prices': recent_prices,
            'line_values': line_recent
        }
    
    def update_live_data(self):
        """Fetch latest data and update analysis"""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updating live data...")
        
        # Fetch latest 200 candles
        new_data = self.fetch_latest_candles(limit=200)
        
        if new_data is not None:
            self.data = new_data
            self.last_update = datetime.now()
            
            # Re-run analysis on new data
            self.optimized_lines = []
            self.horizontal_lines = []
            self.trades = []
            self.analyze_trends()
            
            # Get current signal
            signal_info = self.get_current_signal()
            
            if signal_info:
                self.manage_position(signal_info)
                
            print(f"Update complete. Current price: ${signal_info['current_price']:,.2f}")
            print(f"Signal: {signal_info['signal'].upper()} (slope: {signal_info['slope']:.6f})")
            
            return True
        return False
    
    def manage_position(self, signal_info):
        """Manage trading positions based on signals"""
        current_signal = signal_info['signal']
        current_price = signal_info['current_price']
        current_time = signal_info['timestamp']
        
        # If no position, open one
        if self.current_position is None:
            self.open_position(current_signal, current_price, current_time)
            return
        
        # Check for signal change
        if current_signal != self.current_signal:
            self.close_position(current_price, current_time)
            self.open_position(current_signal, current_price, current_time)
    
    def open_position(self, signal, price, timestamp):
        """Open a new trading position"""
        self.current_position = signal
        self.current_signal = signal
        self.position_open_time = timestamp
        self.position_open_price = price
        
        print(f"\n📈 OPENING {signal.upper()} POSITION")
        print(f"   Time: {timestamp}")
        print(f"   Price: ${price:,.2f}")
    
    def close_position(self, close_price, close_time):
        """Close current position and calculate PnL"""
        if self.current_position is None:
            return
        
        # Calculate PnL
        if self.current_position == 'long':
            pnl = (close_price - self.position_open_price) / self.position_open_price * 100
        else:  # short
            pnl = (self.position_open_price - close_price) / self.position_open_price * 100
        
        self.total_pnl += pnl
        
        # Record trade
        trade = {
            'open_time': self.position_open_time,
            'close_time': close_time,
            'position': self.current_position,
            'open_price': self.position_open_price,
            'close_price': close_price,
            'pnl_percent': pnl,
            'cumulative_pnl': self.total_pnl
        }
        self.live_trades.append(trade)
        self.pnl_history.append(self.total_pnl)
        
        print(f"\n📉 CLOSING {self.current_position.upper()} POSITION")
        print(f"   Open Time: {self.position_open_time}")
        print(f"   Close Time: {close_time}")
        print(f"   Open Price: ${self.position_open_price:,.2f}")
        print(f"   Close Price: ${close_price:,.2f}")
        print(f"   PnL: {pnl:+.2f}%")
        print(f"   Total PnL: {self.total_pnl:+.2f}%")
        
        # Reset position
        self.current_position = None
    
    def run_live_trading(self, check_interval=3601):  # 1 hour + 1 second
        """Run live trading loop"""
        print("\n" + "="*60)
        print("STARTING LIVE TRADING SESSION")
        print("="*60)
        
        # Initial data fetch and analysis
        print("Fetching initial data...")
        self.update_live_data()
        
        # Main trading loop
        while True:
            try:
                # Wait for next interval
                next_check = datetime.now() + timedelta(seconds=check_interval)
                print(f"\nNext update at: {next_check.strftime('%Y-%m-%d %H:%M:%S')}")
                
                time.sleep(check_interval)
                
                # Update data and manage positions
                self.update_live_data()
                
                # Save trade history periodically
                self.save_trade_history()
                
            except KeyboardInterrupt:
                print("\n\nStopping live trading...")
                # Close any open position
                if self.current_position is not None:
                    current_price = self.fetch_current_price()
                    if current_price:
                        self.close_position(current_price, datetime.now())
                self.save_trade_history()
                break
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def fetch_current_price(self):
        """Fetch current BTC price"""
        try:
            url = "https://api.binance.com/api/v3/ticker/price"
            params = {'symbol': self.symbol}
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return float(data['price'])
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return None
    
    def save_trade_history(self):
        """Save trade history to file"""
        history_file = Path('trade_history.json')
        
        history = {
            'total_pnl': self.total_pnl,
            'trades': [
                {
                    'open_time': t['open_time'].isoformat() if isinstance(t['open_time'], datetime) else str(t['open_time']),
                    'close_time': t['close_time'].isoformat() if isinstance(t['close_time'], datetime) else str(t['close_time']),
                    'position': t['position'],
                    'open_price': float(t['open_price']),
                    'close_price': float(t['close_price']),
                    'pnl_percent': float(t['pnl_percent']),
                    'cumulative_pnl': float(t['cumulative_pnl'])
                }
                for t in self.live_trades
            ],
            'pnl_history': [float(p) for p in self.pnl_history],
            'last_update': datetime.now().isoformat()
        }
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nTrade history saved to {history_file}")
    
    def load_trade_history(self):
        """Load trade history from file"""
        history_file = Path('trade_history.json')
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            self.total_pnl = history.get('total_pnl', 0)
            self.pnl_history = history.get('pnl_history', [])
            
            # Convert string times back to datetime
            self.live_trades = []
            for t in history.get('trades', []):
                t['open_time'] = datetime.fromisoformat(t['open_time'])
                t['close_time'] = datetime.fromisoformat(t['close_time'])
                self.live_trades.append(t)
            
            print(f"Loaded {len(self.live_trades)} previous trades")
            print(f"Previous total PnL: {self.total_pnl:+.2f}%")
    
    def plot_results(self):
        """Plot all results - FIXED VERSION with bounds checking"""
        if self.data is None or len(self.data) == 0:
            print("No data to plot")
            return None
        
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                            gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot price data
            ax1.plot(self.data['timestamp'], self.data['close'], 
                    color='blue', alpha=0.5, linewidth=1, label='BTC Price')
            
            # Plot optimized lines - with bounds checking
            valid_lines = 0
            for line in self.optimized_lines:
                try:
                    # Make sure indices are within current data bounds
                    start_idx = max(0, min(line['start_idx'], len(self.data) - 1))
                    end_idx = max(start_idx + 1, min(line['end_idx'], len(self.data)))
                    
                    if start_idx < end_idx and start_idx < len(self.data):
                        x_values = self.data['timestamp'].iloc[start_idx:end_idx]
                        y_values = line['values']
                        
                        # Trim y_values to match x_values length if needed
                        if len(x_values) < len(y_values):
                            y_values = y_values[:len(x_values)]
                        
                        if len(x_values) > 0 and len(y_values) > 0 and len(x_values) == len(y_values):
                            ax1.plot(x_values, y_values, 
                                    color=line['color'], 
                                    linewidth=2, 
                                    alpha=0.7)
                            valid_lines += 1
                except Exception as e:
                    print(f"Error plotting line: {e}")
                    continue
            
            print(f"Plotted {valid_lines} optimized lines")
            
            # Plot horizontal lines
            for h_line in self.horizontal_lines:
                try:
                    ax1.axhline(y=h_line['price'], 
                               xmin=0, xmax=1, 
                               color='black', 
                               linestyle='--', 
                               linewidth=1, 
                               alpha=0.5)
                    ax1.axvline(x=h_line['time'], 
                               color='black', 
                               linestyle='--', 
                               linewidth=1, 
                               alpha=0.5)
                except Exception as e:
                    print(f"Error plotting horizontal line: {e}")
                    continue
            
            # Formatting
            ax1.set_title(f'{self.symbol} Price Analysis with OLS Trends', fontsize=16)
            ax1.set_ylabel('Price (USDT)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Format x-axis dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot trade returns
            if self.trades:
                trade_colors = ['green' if t['return'] > 0 else 'red' for t in self.trades]
                trade_returns = [t['return'] * 100 for t in self.trades]
                
                bars = ax2.bar(range(len(trade_returns)), trade_returns, color=trade_colors, alpha=0.7)
                
                # Add value labels on bars
                for i, (bar, ret) in enumerate(zip(bars, trade_returns)):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{ret:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
            
            ax2.set_title('Trade Returns (%)', fontsize=16)
            ax2.set_xlabel('Trade Number', fontsize=12)
            ax2.set_ylabel('Return (%)', fontsize=12)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error in plot_results: {e}")
            return None
    
    def create_live_plot(self):
        """Create plot with live trading data"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                            gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot price data
            ax1.plot(self.data['timestamp'], self.data['close'], 
                    color='blue', alpha=0.3, linewidth=1, label='BTC Price')
            
            # Plot optimized lines from original analysis - with bounds checking
            for line in self.optimized_lines:
                try:
                    start_idx = max(0, min(line['start_idx'], len(self.data) - 1))
                    end_idx = max(start_idx + 1, min(line['end_idx'], len(self.data)))
                    
                    if start_idx < end_idx:
                        x_values = self.data['timestamp'].iloc[start_idx:end_idx]
                        y_values = line['values']
                        
                        if len(x_values) < len(y_values):
                            y_values = y_values[:len(x_values)]
                        
                        if len(x_values) > 0 and len(y_values) > 0 and len(x_values) == len(y_values):
                            ax1.plot(x_values, y_values, 
                                    color=line['color'], 
                                    linewidth=1.5, 
                                    alpha=0.5)
                except:
                    continue
            
            # Plot current optimal line
            signal_info = self.get_current_signal()
            if signal_info:
                # Plot the most recent optimal window
                recent_times = self.data['timestamp'].iloc[-signal_info['window_size']:]
                recent_line_values = signal_info['line_values']
                
                if len(recent_times) == len(recent_line_values):
                    ax1.plot(recent_times, recent_line_values, 
                            color='green' if signal_info['signal'] == 'long' else 'red',
                            linewidth=3, alpha=0.8, 
                            label=f"Current {signal_info['signal'].upper()} Signal")
            
            # Plot trade markers from live trading
            for trade in self.live_trades:
                try:
                    # Mark entry
                    ax1.scatter(trade['open_time'], trade['open_price'], 
                               color='green' if trade['position'] == 'long' else 'red',
                               s=100, marker='^', zorder=5)
                    # Mark exit
                    ax1.scatter(trade['close_time'], trade['close_price'],
                               color='red' if trade['position'] == 'long' else 'green',
                               s=100, marker='v', zorder=5)
                except:
                    continue
            
            # Plot current position if open
            if self.current_position and self.position_open_price:
                ax1.axhline(y=self.position_open_price, 
                           color='yellow', linestyle='--', alpha=0.5,
                           label=f"Open Position @ ${self.position_open_price:,.0f}")
            
            ax1.set_title(f'{self.symbol} Live Trading - {self.interval} Interval', fontsize=16)
            ax1.set_ylabel('Price (USDT)', fontsize=12)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Format x-axis dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot cumulative PnL
            if self.pnl_history:
                trades_x = list(range(len(self.pnl_history)))
                ax2.plot(trades_x, self.pnl_history, 
                        color='blue', linewidth=2, marker='o')
                ax2.fill_between(trades_x, 0, self.pnl_history,
                                where=np.array(self.pnl_history) >= 0,
                                color='green', alpha=0.3)
                ax2.fill_between(trades_x, 0, self.pnl_history,
                                where=np.array(self.pnl_history) < 0,
                                color='red', alpha=0.3)
            
            ax2.set_title('Cumulative PnL (%)', fontsize=16)
            ax2.set_xlabel('Trade Number', fontsize=12)
            ax2.set_ylabel('PnL (%)', fontsize=12)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error in create_live_plot: {e}")
            return None
    
    def get_html_plot(self):
        """Convert plot to HTML image - FIXED VERSION with error handling"""
        try:
            fig = self.plot_results()
            if fig is None:
                return "<html><body><h1>No data available or error generating plot</h1></body></html>"
        except Exception as e:
            print(f"Error generating plot: {e}")
            return f"<html><body><h1>Error generating plot</h1><p>{str(e)}</p></body></html>"
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Create HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BTC Trend Analysis</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                img {{
                    width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .stats {{
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .positive {{
                    color: green;
                    font-weight: bold;
                }}
                .negative {{
                    color: red;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>BTC Trend Analysis - {self.interval} Interval</h1>
                <img src="data:image/png;base64,{image_base64}" alt="BTC Trend Analysis">
                
                <div class="stats">
                    <h2>Analysis Statistics</h2>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Data Points</td>
                            <td>{len(self.data)}</td>
                        </tr>
                        <tr>
                            <td>Optimized Lines</td>
                            <td>{len(self.optimized_lines)}</td>
                        </tr>
                        <tr>
                            <td>Trend Changes</td>
                            <td>{len(self.horizontal_lines)}</td>
                        </tr>
                        <tr>
                            <td>Trades Executed</td>
                            <td>{len(self.trades)}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="stats">
                    <h2>Trade Performance</h2>
                    <table>
                        <tr>
                            <th>Start Time</th>
                            <th>End Time</th>
                            <th>Type</th>
                            <th>Return</th>
                        </tr>
        """
        
        # Add trade rows
        for trade in self.trades[-10:]:  # Show last 10 trades
            return_class = 'positive' if trade['return'] > 0 else 'negative'
            html += f"""
                        <tr>
                            <td>{trade['start_time']}</td>
                            <td>{trade['end_time']}</td>
                            <td>{trade['type'].upper()}</td>
                            <td class="{return_class}">{trade['return']*100:.2f}%</td>
                        </tr>
            """
        
        html += """
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    analyzer = None
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            if self.analyzer:
                try:
                    html = self.analyzer.get_html_plot()
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(html.encode('utf-8'))
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    error_html = f"<html><body><h1>Server Error</h1><p>{str(e)}</p></body></html>"
                    self.wfile.write(error_html.encode('utf-8'))
            else:
                self.send_response(500)
                self.end_headers()
        else:
            super().do_GET()

def main():
    parser = argparse.ArgumentParser(description='BTC Trend Analysis with OLS')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--interval', type=str, default='1h', help='Time interval')
    parser.add_argument('--days', type=int, default=30, help='Number of days')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--host', type=str, default='localhost', help='Host IP address to bind to')
    parser.add_argument('--live', action='store_true', help='Enable live trading')
    parser.add_argument('--check-interval', type=int, default=3601, 
                       help='Seconds between checks (default: 3601 = 1 hour + 1 sec)')
    
    args = parser.parse_args()
    
    print(f"Starting BTC Trend Analysis for {args.symbol}")
    print(f"Interval: {args.interval}, Days: {args.days}")
    print(f"Server will run at http://{args.host}:{args.port}")
    print(f"Live Trading: {'Enabled' if args.live else 'Disabled'}")
    
    # Create analyzer
    analyzer = BTCTrendAnalyzer(args.symbol, args.interval, args.days)
    
    # Load previous trade history if live trading
    if args.live:
        analyzer.load_trade_history()
    
    # Fetch data
    print("Fetching data from Binance...")
    success = analyzer.fetch_binance_data()
    
    if success:
        print("Data fetched successfully")
    else:
        print("Using sample data for demonstration")
    
    # Analyze trends (YOUR ORIGINAL ANALYSIS)
    print("Analyzing trends...")
    analyzer.analyze_trends()
    
    print(f"Found {len(analyzer.optimized_lines)} optimized lines")
    print(f"Found {len(analyzer.horizontal_lines)} trend changes")
    
    # Start live trading in a separate thread if enabled
    if args.live:
        trading_thread = threading.Thread(target=analyzer.run_live_trading, 
                                        args=(args.check_interval,),
                                        daemon=True)
        trading_thread.start()
        print("\n✅ Live trading thread started")
    
    # Setup HTTP server
    CustomHTTPRequestHandler.analyzer = analyzer
    handler = CustomHTTPRequestHandler
    
    # Create server with user-specified host and port
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"\nServer running at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            if args.live:
                analyzer.save_trade_history()
            httpd.shutdown()

if __name__ == "__main__":
    main()