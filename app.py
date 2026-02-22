import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
import traceback
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
        self.live_trades = []
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
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close'] = df['close'].astype(float)
            
            self.data = df[['timestamp', 'close']].copy()
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            self.generate_sample_data()
            return False
    
    def generate_sample_data(self):
        """Generate sample data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=24*30, freq='1h')
        trend = np.linspace(40000, 45000, len(dates))
        noise = np.random.normal(0, 1000, len(dates))
        prices = trend + noise
        
        self.data = pd.DataFrame({
            'timestamp': dates,
            'close': prices
        })
        print("Generated sample data for testing")
    
    def fetch_closed_candles(self, limit=100):
        """
        Fetch only closed candles (skip the current open candle)
        Returns candles ending at the previous full hour
        """
        base_url = "https://api.binance.com/api/v3/klines"
        
        # Get current time and align to hour
        now = datetime.now()
        
        # For 1h interval, we want candles that end at the previous hour mark
        # If current time is 14:35, we want candles up to 14:00
        
        # Calculate end time for the last COMPLETED candle
        if self.interval == '1h':
            # Round down to the nearest hour for the last completed candle
            if now.minute < 1 or now.second < 1:
                # If we're at the very start of a new candle, go back 1 more hour
                last_completed_hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            else:
                last_completed_hour = now.replace(minute=0, second=0, microsecond=0)
            
            end_time = last_completed_hour
            start_time = end_time - timedelta(hours=limit)
        
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000),
            'limit': limit
        }
        
        try:
            print(f"Fetching {limit} closed candles ending at {end_time.strftime('%Y-%m-%d %H:00')}")
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close'] = df['close'].astype(float)
            
            print(f"Got {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            
            return df[['timestamp', 'close']].copy()
            
        except Exception as e:
            print(f"Error fetching closed candles: {e}")
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
        """Main analysis function"""
        if self.data is None or len(self.data) < 100:
            print("Insufficient data")
            return
        
        prices = self.data['close'].copy()
        timestamps = self.data['timestamp'].copy()
        
        current_position = len(prices)
        min_window = 10
        
        while current_position > min_window:
            current_prices = prices.iloc[:current_position]
            
            window_size, line_values, slope, error, model = self.find_optimal_window(
                current_prices, 
                min_window=min_window, 
                max_window=min(100, current_position)
            )
            
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
            
            current_position = start_idx
            
            if len(self.optimized_lines) >= 2:
                prev_line = self.optimized_lines[-2]
                curr_line = self.optimized_lines[-1]
                
                if prev_line['color'] != curr_line['color']:
                    self.horizontal_lines.append({
                        'time': timestamps.iloc[start_idx],
                        'price': prices.iloc[start_idx]
                    })
    
    def get_current_signal(self):
        """Get current trading signal based on latest closed candles"""
        if self.data is None or len(self.data) < 100:
            return None
        
        prices = self.data['close'].copy()
        
        window_size, line_values, slope, error, model = self.find_optimal_window(
            prices, min_window=10, max_window=100
        )
        
        signal = 'long' if slope > 0 else 'short'
        
        recent_prices = prices[-window_size:]
        x_recent = np.arange(len(recent_prices)).reshape(-1, 1)
        line_recent = model.predict(x_recent).flatten()
        
        return {
            'signal': signal,
            'slope': slope,
            'window_size': window_size,
            'error': error,
            'current_price': prices.iloc[-1],  # This is now the last CLOSED candle
            'line_price': line_recent[-1],
            'timestamp': self.data['timestamp'].iloc[-1],
            'model': model,
            'recent_prices': recent_prices,
            'line_values': line_recent
        }
    
    def update_live_data(self):
        """Fetch latest closed candles and update analysis"""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching closed candles...")
        
        # Fetch 100 closed candles (ending at last completed hour)
        new_data = self.fetch_closed_candles(limit=100)
        
        if new_data is not None and len(new_data) > 0:
            self.data = new_data
            self.last_update = datetime.now()
            
            # Re-run analysis on new data
            self.optimized_lines = []
            self.horizontal_lines = []
            self.trades = []
            self.analyze_trends()
            
            signal_info = self.get_current_signal()
            
            if signal_info:
                self.manage_position(signal_info)
                
            print(f"✅ Update complete. Last closed candle: {signal_info['timestamp'].strftime('%Y-%m-%d %H:00')} @ ${signal_info['current_price']:,.2f}")
            print(f"📊 Signal: {signal_info['signal'].upper()} (slope: {signal_info['slope']:.6f})")
            
            return True
        return False
    
    def manage_position(self, signal_info):
        """Manage trading positions based on signals"""
        current_signal = signal_info['signal']
        current_price = signal_info['current_price']
        current_time = signal_info['timestamp']
        
        if self.current_position is None:
            self.open_position(current_signal, current_price, current_time)
            return
        
        if current_signal != self.current_signal:
            self.close_position(current_price, current_time)
            self.open_position(current_signal, current_price, current_time)
    
    def open_position(self, signal, price, timestamp):
        """Open a new trading position"""
        self.current_position = signal
        self.current_signal = signal
        self.position_open_time = timestamp
        self.position_open_price = price
        
        print(f"\n📈 OPENING {signal.upper()} @ ${price:,.2f} (Candle: {timestamp.strftime('%Y-%m-%d %H:00')})")
    
    def close_position(self, close_price, close_time):
        """Close current position and calculate PnL"""
        if self.current_position is None:
            return
        
        if self.current_position == 'long':
            pnl = (close_price - self.position_open_price) / self.position_open_price * 100
        else:
            pnl = (self.position_open_price - close_price) / self.position_open_price * 100
        
        self.total_pnl += pnl
        
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
        
        print(f"\n📉 CLOSING {self.current_position.upper()} @ ${close_price:,.2f}")
        print(f"   PnL: {pnl:+.2f}% | Total: {self.total_pnl:+.2f}%")
        
        self.current_position = None
    
    def run_live_trading(self):
        """Run live trading loop synced to hour + 1 second"""
        print("\n" + "="*60)
        print("STARTING LIVE TRADING - SYNCED TO HOUR + 1 SECOND")
        print("="*60)
        
        # Initial data fetch
        self.update_live_data()
        
        while True:
            try:
                # Calculate next run time (top of next hour + 1 second)
                now = datetime.now()
                
                # Next hour at 1 second past the hour
                if now.minute == 0 and now.second >= 1:
                    # We're in the first minute, schedule for next hour
                    next_run = (now + timedelta(hours=1)).replace(minute=0, second=1, microsecond=0)
                else:
                    # Schedule for next hour at 1 second past
                    next_run = (now + timedelta(hours=1)).replace(minute=0, second=1, microsecond=0)
                
                sleep_seconds = (next_run - now).total_seconds()
                
                print(f"\n⏰ Next update at: {next_run.strftime('%Y-%m-%d %H:%M:%S')} (in {sleep_seconds:.0f} seconds)")
                
                time.sleep(sleep_seconds)
                
                # Update data and manage positions
                self.update_live_data()
                
                # Save trade history
                self.save_trade_history()
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping live trading...")
                if self.current_position is not None:
                    current_price = self.fetch_current_price()
                    if current_price:
                        self.close_position(current_price, datetime.now())
                self.save_trade_history()
                break
            except Exception as e:
                print(f"❌ Error in trading loop: {e}")
                time.sleep(60)
    
    def fetch_current_price(self):
        """Fetch current BTC price (for emergency closes only)"""
        try:
            url = "https://api.binance.com/api/v3/ticker/price"
            params = {'symbol': self.symbol}
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return float(data['price'])
        except Exception as e:
            print(f"Error fetching price: {e}")
            return None
    
    def save_trade_history(self):
        """Save trade history to file"""
        history_file = Path('trade_history.json')
        
        history = {
            'total_pnl': self.total_pnl,
            'trades': [
                {
                    'open_time': t['open_time'].isoformat(),
                    'close_time': t['close_time'].isoformat(),
                    'position': t['position'],
                    'open_price': float(t['open_price']),
                    'close_price': float(t['close_price']),
                    'pnl_percent': float(t['pnl_percent']),
                    'cumulative_pnl': float(t['cumulative_pnl'])
                }
                for t in self.live_trades
            ],
            'last_update': datetime.now().isoformat()
        }
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_trade_history(self):
        """Load trade history from file"""
        history_file = Path('trade_history.json')
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            self.total_pnl = history.get('total_pnl', 0)
            
            for t in history.get('trades', []):
                t['open_time'] = datetime.fromisoformat(t['open_time'])
                t['close_time'] = datetime.fromisoformat(t['close_time'])
                self.live_trades.append(t)
            
            print(f"📚 Loaded {len(self.live_trades)} previous trades")
            print(f"💰 Previous total PnL: {self.total_pnl:+.2f}%")
    
    def plot_results(self):
        """Plot results with bounds checking"""
        if self.data is None or len(self.data) == 0:
            return None
        
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Price line
            ax1.plot(self.data['timestamp'], self.data['close'], color='blue', alpha=0.5, linewidth=1, label='Price')
            
            # Trend lines
            for line in self.optimized_lines:
                try:
                    start_idx = max(0, min(line['start_idx'], len(self.data)-1))
                    end_idx = max(start_idx+1, min(line['end_idx'], len(self.data)))
                    
                    if start_idx < end_idx:
                        x_vals = self.data['timestamp'].iloc[start_idx:end_idx]
                        y_vals = line['values'][:len(x_vals)]
                        
                        if len(x_vals) == len(y_vals):
                            ax1.plot(x_vals, y_vals, color=line['color'], linewidth=2, alpha=0.7)
                except:
                    continue
            
            # Current signal
            signal = self.get_current_signal()
            if signal:
                recent_times = self.data['timestamp'].iloc[-signal['window_size']:]
                recent_vals = signal['line_values'][:len(recent_times)]
                if len(recent_times) == len(recent_vals):
                    ax1.plot(recent_times, recent_vals, 
                            color='lime' if signal['signal']=='long' else 'red',
                            linewidth=3, label=f'Signal: {signal["signal"].upper()}')
            
            # Mark last closed candle
            last_time = self.data['timestamp'].iloc[-1]
            last_price = self.data['close'].iloc[-1]
            ax1.scatter(last_time, last_price, color='black', s=100, zorder=5, marker='o')
            
            ax1.set_title(f'{self.symbol} - {self.interval} (Last Closed: {last_time.strftime("%H:00")})', fontsize=14)
            ax1.set_ylabel('Price (USDT)')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # PnL chart
            if self.pnl_history:
                ax2.plot(range(len(self.pnl_history)), self.pnl_history, 'b-', linewidth=2)
                ax2.fill_between(range(len(self.pnl_history)), 0, self.pnl_history,
                                where=np.array(self.pnl_history) >= 0, color='green', alpha=0.3)
                ax2.fill_between(range(len(self.pnl_history)), 0, self.pnl_history,
                                where=np.array(self.pnl_history) < 0, color='red', alpha=0.3)
            
            ax2.set_ylabel('PnL %')
            ax2.axhline(y=0, color='black', linewidth=0.5)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Plot error: {e}")
            return None
    
    def get_html_dashboard(self):
        """Generate minimal HTML dashboard"""
        try:
            fig = self.plot_results()
            if fig is None:
                return "<html><body><h3>No data</h3></body></html>"
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            signal = self.get_current_signal()
            
            # Get last closed candle time
            last_candle_time = self.data['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:00') if self.data is not None else 'N/A'
            
            # Simple stats
            total_trades = len(self.live_trades)
            wins = sum(1 for t in self.live_trades if t['pnl_percent'] > 0) if self.live_trades else 0
            win_rate = (wins/total_trades*100) if total_trades > 0 else 0
            
            # Next update time
            now = datetime.now()
            next_update = (now + timedelta(hours=1)).replace(minute=0, second=1, microsecond=0)
            next_update_str = next_update.strftime('%H:%M:%S')
            
            # Format values safely
            signal_class = 'neutral'
            signal_text = 'WAITING'
            current_price_display = '---'
            signal_slope = '---'
            current_position_display = 'NONE'
            position_price_display = '---'
            position_time_display = ''
            
            if signal:
                signal_class = signal['signal']
                signal_text = signal['signal'].upper()
                current_price_display = f"${signal['current_price']:,.2f}"
                signal_slope = f"{signal['slope']:.6f}"
            
            if self.current_position:
                current_position_display = self.current_position.upper()
                if self.position_open_price:
                    position_price_display = f"${self.position_open_price:,.2f}"
                if self.position_open_time:
                    position_time_display = self.position_open_time.strftime('%m-%d %H:00')
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>BTC Trader</title>
                <meta http-equiv="refresh" content="30">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                           margin: 0; padding: 20px; background: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .row {{ display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 20px; }}
                    .card {{ background: white; padding: 15px 20px; border-radius: 8px; 
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; min-width: 200px; }}
                    .signal {{ font-size: 24px; font-weight: bold; padding: 10px; border-radius: 6px; 
                              text-align: center; margin: 10px 0; }}
                    .long {{ background: #d4edda; color: #155724; }}
                    .short {{ background: #f8d7da; color: #721c24; }}
                    .neutral {{ background: #e2e3e5; color: #383d41; }}
                    .price {{ font-size: 28px; font-weight: bold; text-align: center; margin: 5px 0; }}
                    .green {{ color: #28a745; }} .red {{ color: #dc3545; }}
                    .chart {{ background: white; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                    .chart img {{ width: 100%; height: auto; }}
                    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; }}
                    th {{ background: #333; color: white; padding: 10px; text-align: left; }}
                    td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                    .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
                    .badge-long {{ background: #d4edda; color: #155724; }}
                    .badge-short {{ background: #f8d7da; color: #721c24; }}
                    .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 20px; }}
                    .info {{ background: #e7f3ff; padding: 10px; border-radius: 6px; margin-bottom: 15px; 
                            font-size: 14px; border-left: 4px solid #0066cc; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>BTC Live Trading • {self.interval}</h2>
                    
                    <div class="info">
                        ⏰ Last closed candle: {last_candle_time} | Next update: {next_update_str} (hour +1s)
                    </div>
                    
                    <div class="row">
                        <div class="card">
                            <div style="font-size: 14px; color: #666;">CURRENT SIGNAL</div>
                            <div class="signal {signal_class}">
                                {signal_text}
                            </div>
                            <div class="price">{current_price_display}</div>
                            <div style="text-align: center; font-size: 12px; color: #666;">
                                Last closed candle
                            </div>
                        </div>
                        
                        <div class="card">
                            <div style="font-size: 14px; color: #666;">POSITION</div>
                            <div style="font-size: 20px; margin: 10px 0;">
                                {current_position_display}
                            </div>
                            <div>Open: {position_price_display}</div>
                            <div style="font-size: 12px; color: #666;">
                                {position_time_display}
                            </div>
                        </div>
                        
                        <div class="card">
                            <div style="font-size: 14px; color: #666;">TOTAL P&L</div>
                            <div style="font-size: 28px; font-weight: bold;" 
                                 class="{'green' if self.total_pnl >= 0 else 'red'}">
                                {self.total_pnl:+.2f}%
                            </div>
                            <div>Trades: {total_trades} | Win: {win_rate:.0f}%</div>
                        </div>
                    </div>
                    
                    <div class="chart">
                        <img src="data:image/png;base64,{img_b64}">
                    </div>
                    
                    <h3>Trade History</h3>
                    <table>
                        <tr>
                            <th>Close Time</th>
                            <th>Type</th>
                            <th>PnL</th>
                            <th>Total</th>
                        </tr>
            """
            
            for trade in reversed(self.live_trades[-20:]):
                pnl_class = 'green' if trade['pnl_percent'] >= 0 else 'red'
                pos_class = 'badge-long' if trade['position'] == 'long' else 'badge-short'
                html += f"""
                        <tr>
                            <td>{trade['close_time'].strftime('%m-%d %H:00')}</td>
                            <td><span class="badge {pos_class}">{trade['position'].upper()}</span></td>
                            <td class="{pnl_class}">{trade['pnl_percent']:+.2f}%</td>
                            <td class="{pnl_class}">{trade['cumulative_pnl']:+.2f}%</td>
                        </tr>
                """
            
            if not self.live_trades:
                html += "<tr><td colspan='4' style='text-align:center;padding:20px;'>No trades yet</td></tr>"
            
            html += f"""
                    </table>
                    
                    <div class="footer">
                        Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                        Using closed candles only | Next sync: {next_update_str}
                    </div>
                </div>
            </body>
            </html>
            """
            
            return html
        except Exception as e:
            return f"<html><body><h3>Error: {str(e)}</h3><pre>{traceback.format_exc()}</pre></body></html>"

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    analyzer = None
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            if self.analyzer:
                try:
                    html = self.analyzer.get_html_dashboard()
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(html.encode('utf-8'))
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(f"<html><body><h3>Error</h3><pre>{traceback.format_exc()}</pre></body></html>".encode('utf-8'))
            else:
                self.send_response(500)
                self.end_headers()
        else:
            super().do_GET()

def main():
    parser = argparse.ArgumentParser(description='BTC Live Trading')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--interval', type=str, default='1h', help='Time interval')
    parser.add_argument('--days', type=int, default=30, help='Days of history')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--host', type=str, default='localhost', help='Host IP')
    parser.add_argument('--live', action='store_true', help='Enable live trading')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"🚀 BTC Trader - {args.symbol} {args.interval}")
    print(f"{'='*60}")
    print(f"📊 Dashboard: http://{args.host}:{args.port}")
    print(f"🎯 Live Trading: {'ON' if args.live else 'OFF'}")
    if args.live:
        print(f"⏰ Sync: Hour + 1 second (using closed candles only)")
    
    analyzer = BTCTrendAnalyzer(args.symbol, args.interval, args.days)
    
    if args.live:
        analyzer.load_trade_history()
    
    print("\n📡 Fetching initial data...")
    success = analyzer.fetch_binance_data()
    print(f"   {'✅ Success' if success else '⚠️ Using sample data'}")
    
    print("📈 Analyzing trends...")
    analyzer.analyze_trends()
    print(f"   Found {len(analyzer.optimized_lines)} trend lines")
    
    if args.live:
        thread = threading.Thread(target=analyzer.run_live_trading, daemon=True)
        thread.start()
        print("✅ Live trading thread started")
    
    CustomHTTPRequestHandler.analyzer = analyzer
    
    with socketserver.TCPServer((args.host, args.port), CustomHTTPRequestHandler) as httpd:
        print(f"\n📊 Dashboard active: http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            if args.live:
                analyzer.save_trade_history()
                print("💾 Trade history saved")

if __name__ == "__main__":
    main()