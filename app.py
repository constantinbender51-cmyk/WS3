import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import ccxt
import matplotlib.pyplot as plt
import http.server
import socketserver
import os
import threading
import webbrowser

class CryptoBreakoutStrategy:
    def __init__(self, symbols: List[str] = None, fee: float = 0.0004):
        """
        Initialize strategy with major crypto symbols and trading fee.
        Default symbols represent ~95% of crypto market cap.
        """
        self.symbols = symbols or [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
            'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT', 'DOT/USDT'
        ]
        self.fee = fee
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
        self.sequence_lengths = range(10, 101)

    def _to_pandas_freq(self, tf: str) -> str:
        """Convert ccxt timeframe to Pandas frequency."""
        # Pandas 2.2+ deprecates 'M' (MonthEnd) for 'ME'
        # Also map 'm' to 'min' to avoid ambiguity
        mapping = {
            'm': 'min',
            'h': 'h',
            'd': 'D',
            'w': 'W',
            'M': 'ME'  # Specific compliance for Pandas 2.2+
        }
        unit = tf[-1]
        value = tf[:-1]
        if unit in mapping:
            return f"{value}{mapping[unit]}"
        return tf

    def fetch_ohlc(self, symbol: str, timeframe: str, days: int = 180) -> pd.DataFrame:
        """Fetch OHLCV data with error handling and rate limiting."""
        since = self.exchange.milliseconds() - days * 24 * 60 * 60 * 1000
        all_ohlcv = []
        
        while since < self.exchange.milliseconds():
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
            except Exception as e:
                print(f"Error fetching {symbol} {timeframe}: {e}")
                break
                
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
        return df

    def fit_channel(self, highs: np.ndarray, lows: np.ndarray) -> Tuple[float, float, float]:
        """Fit horizontal channel boundaries."""
        top_line = np.max(highs)
        bottom_line = np.min(lows)
        
        # Distance metric
        candle_ranges = highs - lows
        channel_height = top_line - bottom_line + 1e-10
        
        distances = ((top_line - highs) + (lows - bottom_line)) / channel_height
        avg_distance = np.mean(distances)
        
        return top_line, bottom_line, avg_distance

    def detect_breakout(self, df: pd.DataFrame, seq_len: int) -> List[Dict]:
        """Scan entire series for breakouts using sequence of length seq_len."""
        signals = []
        if len(df) < seq_len + 1:
            return signals

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Iterate through the DataFrame
        # Optimization: Avoid iterating by index if possible, but window logic requires slices
        for i in range(seq_len, len(df)):
            seq_highs = highs[i-seq_len:i-1]
            seq_lows = lows[i-seq_len:i-1]
            
            top_line, bottom_line, avg_dist = self.fit_channel(seq_highs, seq_lows)
            
            current_high = highs[i-1]
            current_low = lows[i-1]
            current_close = closes[i-1]
            
            breakout = None
            direction = 0
            entry_price = 0
            stop_loss = 0
            
            if current_high > top_line:
                breakout = 'long'
                direction = 1
                entry_price = max(current_close, top_line)
                stop_loss = bottom_line
            elif current_low < bottom_line:
                breakout = 'short'
                direction = -1
                entry_price = min(current_close, bottom_line)
                stop_loss = top_line
            
            if breakout:
                signals.append({
                    'timestamp': df.index[i-1],
                    'symbol': df.name,
                    'timeframe': getattr(df, 'timeframe', 'unknown'),
                    'sequence_length': seq_len,
                    'top_line': top_line,
                    'bottom_line': bottom_line,
                    'avg_distance': avg_dist,
                    'direction': direction,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'breakout_type': breakout
                })
        
        return signals

    def apply_trailing_stop(self, trade: Dict, future_prices: pd.Series) -> Tuple[float, int]:
        """Simulate trade with trailing stop."""
        direction = trade['direction']
        entry_price = trade['entry_price']
        initial_stop = trade['stop_loss']
        
        max_profit_price = entry_price
        trailing_stop = initial_stop
        exit_idx = -1
        exit_price = None
        
        for idx, price in enumerate(future_prices):
            if direction == 1:
                if price > max_profit_price:
                    max_profit_price = price
                    trailing_stop = entry_price + 0.9 * (max_profit_price - entry_price)
                
                if price <= trailing_stop:
                    exit_price = price
                    exit_idx = idx
                    break
            else:
                if price < max_profit_price:
                    max_profit_price = price
                    trailing_stop = entry_price - 0.9 * (entry_price - max_profit_price)
                
                if price >= trailing_stop:
                    exit_price = price
                    exit_idx = idx
                    break
        
        if exit_price is None:
            exit_price = future_prices.iloc[-1]
            exit_idx = len(future_prices) - 1
        
        if direction == 1:
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price
        
        net_return = gross_return - (2 * self.fee)
        return net_return, exit_idx

    def resolve_conflicts(self, signals_at_timestamp: List[Dict]) -> Dict:
        """Select channel with lowest average distance."""
        return min(signals_at_timestamp, key=lambda s: s['avg_distance'])

    def backtest_symbol_timeframe(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Dict]:
        """Run full backtest for single symbol/timeframe."""
        df.name = symbol
        df.timeframe = timeframe
        
        all_signals = []
        for seq_len in self.sequence_lengths:
            signals = self.detect_breakout(df, seq_len)
            all_signals.extend(signals)
        
        signals_by_time = {}
        for signal in all_signals:
            ts = signal['timestamp']
            if ts not in signals_by_time:
                signals_by_time[ts] = []
            signals_by_time[ts].append(signal)
        
        trades = []
        position_active = False
        position_end_idx = -1
        
        for ts in sorted(signals_by_time.keys()):
            ts_idx = df.index.get_loc(ts)
            
            # Check if previous position is still active
            if position_active:
                if ts_idx <= position_end_idx:
                    continue
                else:
                    position_active = False
            
            best_signal = self.resolve_conflicts(signals_by_time[ts])
            future_prices = df['close'].iloc[ts_idx+1:]
            
            if len(future_prices) < 2:
                continue
                
            net_return, exit_offset = self.apply_trailing_stop(best_signal, future_prices)
            
            trades.append({
                'entry_time': ts,
                'exit_time': df.index[ts_idx + 1 + exit_offset] if exit_offset < len(future_prices) else df.index[-1],
                'symbol': symbol,
                'timeframe': timeframe,
                'sequence_length': best_signal['sequence_length'],
                'direction': best_signal['direction'],
                'entry_price': best_signal['entry_price'],
                'exit_price': future_prices.iloc[exit_offset] if exit_offset < len(future_prices) else future_prices.iloc[-1],
                'net_return': net_return,
                'hold_bars': exit_offset + 1
            })
            
            position_active = True
            position_end_idx = ts_idx + 1 + exit_offset
        
        return trades

    def run_full_backtest(self, days: int = 180) -> pd.DataFrame:
        """Execute complete backtest."""
        all_trades = []
        
        for symbol in self.symbols:
            print(f"Fetching data for {symbol}...")
            try:
                # Fetch 1m data to allow accurate resampling
                df_1m = self.fetch_ohlc(symbol, '1m', days)
                
                if df_1m.empty:
                    continue

                for tf in self.timeframes:
                    pandas_freq = self._to_pandas_freq(tf)
                    
                    df_tf = df_1m.resample(pandas_freq).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    
                    if len(df_tf) < 20: # Skip if insufficient data
                        continue

                    print(f"  Backtesting {symbol} on {tf} ({len(df_tf)} candles)...")
                    trades = self.backtest_symbol_timeframe(df_tf, symbol, tf)
                    all_trades.extend(trades)
                    
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    def analyze_results(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        if trades_df.empty:
            return {'error': 'No trades generated'}
        
        trades_df['cum_return'] = (1 + trades_df['net_return']).cumprod()
        total_return = trades_df['cum_return'].iloc[-1] - 1
        
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['net_return'] > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        returns = trades_df['net_return'].values
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        running_max = np.maximum.accumulate(trades_df['cum_return'])
        drawdowns = (trades_df['cum_return'] - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        gross_profit = winning_trades['net_return'].sum()
        gross_loss = trades_df[trades_df['net_return'] < 0]['net_return'].sum()
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': np.mean(returns),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
        }

def generate_report_files(trades: pd.DataFrame, metrics: Dict, filename='report.html'):
    """Generate HTML report with embedded plots and CSV data."""
    
    # 1. Generate Plots
    plt.style.use('bmh')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Crypto Breakout Strategy Performance', fontsize=16)
    
    # Equity Curve
    trades['equity'] = (1 + trades['net_return']).cumprod()
    axes[0, 0].plot(trades['equity'])
    axes[0, 0].set_title('Equity Curve')
    axes[0, 0].set_ylabel('Portfolio Value (Start=1.0)')
    
    # Drawdown
    running_max = np.maximum.accumulate(trades['equity'])
    drawdown = (trades['equity'] - running_max) / running_max
    axes[0, 1].fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
    axes[0, 1].set_title('Drawdown')
    
    # Returns Distribution
    axes[1, 0].hist(trades['net_return'], bins=50, color='skyblue', edgecolor='black')
    axes[1, 0].set_title('Net Returns Distribution')
    
    # Timeframe Performance
    tf_perf = trades.groupby('timeframe')['net_return'].mean()
    tf_perf.plot(kind='bar', ax=axes[1, 1], color='orange')
    axes[1, 1].set_title('Avg Return by Timeframe')
    
    plt.tight_layout()
    plt.savefig('performance_charts.png')
    plt.close()
    
    # 2. Write CSV
    trades.to_csv('breakout_trades.csv', index=False)
    
    # 3. Create HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Report</title>
        <style>
            body {{ font-family: monospace; margin: 20px; background: #f0f0f0; }}
            .container {{ max_width: 1200px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #333; }}
            .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px; }}
            .metric-box {{ background: #e8e8e8; padding: 15px; border-radius: 5px; }}
            .metric-val {{ font-size: 1.2em; font-weight: bold; color: #2c3e50; }}
            img {{ max_width: 100%; height: auto; border: 1px solid #ddd; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.8em; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f9fa; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Strategy Performance Report</h1>
            <div class="metrics">
                <div class="metric-box">Total Return<br><span class="metric-val">{metrics.get('total_return', 0):.2%}</span></div>
                <div class="metric-box">Win Rate<br><span class="metric-val">{metrics.get('win_rate', 0):.2%}</span></div>
                <div class="metric-box">Sharpe Ratio<br><span class="metric-val">{metrics.get('sharpe_ratio', 0):.2f}</span></div>
                <div class="metric-box">Max Drawdown<br><span class="metric-val">{metrics.get('max_drawdown', 0):.2%}</span></div>
            </div>
            
            <h2>Performance Charts</h2>
            <img src="performance_charts.png" alt="Performance Charts">
            
            <h2>Trade Data</h2>
            <p><a href="breakout_trades.csv">Download CSV</a></p>
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w') as f:
        f.write(html_content)
    
    return filename

def run_server(port=8080):
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"\nServing analysis at http://localhost:{port}")
        print("Press Ctrl+C to stop server")
        httpd.serve_forever()

if __name__ == "__main__":
    # Backtest
    print("Initializing Strategy...")
    strategy = CryptoBreakoutStrategy()
    
    # Limit scope for demonstration if needed, or run full
    # strategy.symbols = ['BTC/USDT', 'ETH/USDT'] # Uncomment for faster testing
    
    print("Running Backtest (this may take time)...")
    trades = strategy.run_full_backtest(days=180)
    
    if not trades.empty:
        # Analyze
        results = strategy.analyze_results(trades)
        
        # Report
        print("\nGenerating Report...")
        generate_report_files(trades, results)
        
        # Server
        server_thread = threading.Thread(target=run_server, args=(8080,))
        server_thread.daemon = True
        server_thread.start()
        
        # Open browser
        webbrowser.open('http://localhost:8080/report.html')
        
        # Keep main thread alive
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down.")
    else:
        print("No trades generated. Check data connection or reduce constraints.")
