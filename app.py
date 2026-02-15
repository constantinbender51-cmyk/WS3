import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import ccxt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import http.server
import socketserver
import threading
import webbrowser
import os

# --- Visual Style Setup ---
plt.style.use('dark_background')

class CryptoBreakoutStrategy:
    def __init__(self, symbols: List[str] = None, fee: float = 0.0004):
        """
        Initialize strategy with major crypto symbols and trading fee.
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
        self.visual_contexts = []  # Store data for visualization

    def _to_pandas_freq(self, tf: str) -> str:
        mapping = {'m': 'min', 'h': 'h', 'd': 'D', 'w': 'W', 'M': 'ME'}
        unit = tf[-1]
        value = tf[:-1]
        if unit in mapping:
            return f"{value}{mapping[unit]}"
        return tf

    def fetch_ohlc(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        since = self.exchange.milliseconds() - days * 24 * 60 * 60 * 1000
        all_ohlcv = []
        
        while since < self.exchange.milliseconds():
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                if not ohlcv: break
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
        top_line = np.max(highs)
        bottom_line = np.min(lows)
        candle_ranges = highs - lows
        channel_height = top_line - bottom_line + 1e-10
        distances = ((top_line - highs) + (lows - bottom_line)) / channel_height
        avg_distance = np.mean(distances)
        return top_line, bottom_line, avg_distance

    def detect_breakout(self, df: pd.DataFrame, seq_len: int) -> List[Dict]:
        signals = []
        if len(df) < seq_len + 1: return signals

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
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
                    'breakout_type': breakout,
                    'index_loc': i-1 
                })
        return signals

    def apply_trailing_stop(self, trade: Dict, future_prices: pd.Series) -> Tuple[float, int, float, float]:
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
        return net_return, exit_idx, exit_price, trailing_stop

    def resolve_conflicts(self, signals_at_timestamp: List[Dict]) -> Dict:
        return min(signals_at_timestamp, key=lambda s: s['avg_distance'])

    def backtest_symbol_timeframe(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Dict]:
        df.name = symbol
        df.timeframe = timeframe
        
        all_signals = []
        for seq_len in self.sequence_lengths:
            signals = self.detect_breakout(df, seq_len)
            all_signals.extend(signals)
        
        signals_by_time = {}
        for signal in all_signals:
            ts = signal['timestamp']
            if ts not in signals_by_time: signals_by_time[ts] = []
            signals_by_time[ts].append(signal)
        
        trades = []
        position_active = False
        position_end_idx = -1
        
        sorted_timestamps = sorted(signals_by_time.keys())
        
        for ts in sorted_timestamps:
            ts_idx = df.index.get_loc(ts)
            
            if position_active:
                if ts_idx <= position_end_idx: continue
                else: position_active = False
            
            best_signal = self.resolve_conflicts(signals_by_time[ts])
            future_prices = df['close'].iloc[ts_idx+1:]
            
            if len(future_prices) < 2: continue
                
            net_return, exit_offset, exit_price, final_stop = self.apply_trailing_stop(best_signal, future_prices)
            
            # --- Visualization Capture Logic ---
            if len(self.visual_contexts) < 3:
                # Capture context: Lookback + Hold Period + Buffer
                lookback = best_signal['sequence_length'] + 10
                hold_period = exit_offset + 5
                
                start_loc = max(0, ts_idx - lookback)
                end_loc = min(len(df), ts_idx + hold_period)
                
                context_df = df.iloc[start_loc:end_loc].copy()
                
                self.visual_contexts.append({
                    'id': len(self.visual_contexts) + 1,
                    'df': context_df,
                    'signal': best_signal,
                    'exit_price': exit_price,
                    'exit_time': df.index[ts_idx + 1 + exit_offset] if exit_offset < len(future_prices) else df.index[-1],
                    'net_return': net_return
                })
            # -----------------------------------

            trades.append({
                'entry_time': ts,
                'exit_time': df.index[ts_idx + 1 + exit_offset] if exit_offset < len(future_prices) else df.index[-1],
                'symbol': symbol,
                'timeframe': timeframe,
                'sequence_length': best_signal['sequence_length'],
                'direction': best_signal['direction'],
                'entry_price': best_signal['entry_price'],
                'exit_price': exit_price,
                'net_return': net_return,
            })
            
            position_active = True
            position_end_idx = ts_idx + 1 + exit_offset
        
        return trades

    def run_full_backtest(self, days: int = 90) -> pd.DataFrame:
        all_trades = []
        for symbol in self.symbols:
            print(f"Fetching data for {symbol}...")
            try:
                df_1m = self.fetch_ohlc(symbol, '1m', days)
                if df_1m.empty: continue

                for tf in self.timeframes:
                    pandas_freq = self._to_pandas_freq(tf)
                    df_tf = df_1m.resample(pandas_freq).agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                    }).dropna()
                    
                    if len(df_tf) < 20: continue

                    print(f"  Backtesting {symbol} on {tf} ({len(df_tf)} candles)...")
                    trades = self.backtest_symbol_timeframe(df_tf, symbol, tf)
                    all_trades.extend(trades)
                    
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    def analyze_results(self, trades_df: pd.DataFrame) -> Dict:
        if trades_df.empty: return {'error': 'No trades'}
        trades_df['cum_return'] = (1 + trades_df['net_return']).cumprod()
        total_return = trades_df['cum_return'].iloc[-1] - 1
        winning_trades = trades_df[trades_df['net_return'] > 0]
        
        return {
            'total_return': total_return,
            'total_trades': len(trades_df),
            'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            'avg_return': trades_df['net_return'].mean(),
            'max_drawdown': ((trades_df['cum_return'] - trades_df['cum_return'].cummax()) / trades_df['cum_return'].cummax()).min()
        }

def plot_single_trade(context: Dict):
    """Generate a candlestick chart with channel and trade markers."""
    df = context['df']
    sig = context['signal']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Candlesticks (Manual implementation to avoid mplfinance dependency)
    width = 0.6
    width2 = 0.1
    up = df[df.close >= df.open]
    down = df[df.close < df.open]
    
    # Up candles
    ax.bar(up.index, up.close - up.open, width, bottom=up.open, color='green', alpha=0.6)
    ax.bar(up.index, up.high - up.close, width2, bottom=up.close, color='green', alpha=0.6)
    ax.bar(up.index, up.low - up.open, width2, bottom=up.open, color='green', alpha=0.6)
    
    # Down candles
    ax.bar(down.index, down.close - down.open, width, bottom=down.open, color='red', alpha=0.6)
    ax.bar(down.index, down.high - down.open, width2, bottom=down.open, color='red', alpha=0.6)
    ax.bar(down.index, down.low - down.close, width2, bottom=down.close, color='red', alpha=0.6)
    
    # Plot Channel (Only during the setup phase)
    # Channel start time = Entry Time - Sequence Length * Timeframe (approx)
    # We use index location for simplicity
    
    entry_time = sig['timestamp']
    entry_idx = df.index.get_loc(entry_time)
    seq_start_idx = max(0, entry_idx - sig['sequence_length'])
    
    # Channel X-axis range
    channel_dates = df.index[seq_start_idx : entry_idx]
    
    ax.plot(channel_dates, [sig['top_line']] * len(channel_dates), color='cyan', linestyle='--', linewidth=1.5, label='Resistance')
    ax.plot(channel_dates, [sig['bottom_line']] * len(channel_dates), color='magenta', linestyle='--', linewidth=1.5, label='Support')
    
    # Plot Entry/Exit markers
    ax.scatter([entry_time], [sig['entry_price']], color='yellow', marker='^' if sig['direction']==1 else 'v', s=200, label='Entry', zorder=5)
    ax.scatter([context['exit_time']], [context['exit_price']], color='white', marker='x', s=200, label='Exit', zorder=5)
    
    title = f"Trade #{context['id']} | {sig['symbol']} {sig['timeframe']} | {sig['breakout_type'].upper()} | Return: {context['net_return']:.2%}"
    ax.set_title(title, fontsize=14, color='white')
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    # Format Date Axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()
    
    filename = f"trade_{context['id']}.png"
    plt.savefig(filename, bbox_inches='tight', facecolor='black')
    plt.close()
    return filename

def generate_report_files(trades: pd.DataFrame, metrics: Dict, contexts: List[Dict], filename='report.html'):
    # Generate Performance Charts (Aggregate)
    plt.style.use('bmh') # Switch style for aggregate charts
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    trades['equity'] = (1 + trades['net_return']).cumprod()
    axes[0].plot(trades['equity'])
    axes[0].set_title('Equity Curve')
    
    trades['net_return'].hist(bins=30, ax=axes[1])
    axes[1].set_title('Return Distribution')
    
    plt.tight_layout()
    plt.savefig('performance_charts.png')
    plt.close()
    
    # Generate Individual Trade Plots
    plt.style.use('dark_background') # Switch back
    trade_imgs = []
    for ctx in contexts:
        img_file = plot_single_trade(ctx)
        trade_imgs.append(img_file)
    
    trades.to_csv('breakout_trades.csv', index=False)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Breakout Analysis</title>
        <style>
            body {{ font-family: monospace; margin: 20px; background: #1a1a1a; color: #e0e0e0; }}
            .container {{ max_width: 1200px; margin: 0 auto; padding: 20px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px; }}
            .metric-box {{ background: #333; padding: 15px; border-radius: 5px; }}
            .val {{ font-size: 1.5em; font-weight: bold; color: #4CAF50; }}
            .trade-viz {{ margin-bottom: 40px; border: 1px solid #444; padding: 10px; }}
            img {{ max_width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Strategy Report (First 3 Assets / 90 Days)</h1>
            <div class="metrics">
                <div class="metric-box">Total Return<br><span class="val">{metrics.get('total_return', 0):.2%}</span></div>
                <div class="metric-box">Win Rate<br><span class="val">{metrics.get('win_rate', 0):.2%}</span></div>
                <div class="metric-box">Trades<br><span class="val">{metrics.get('total_trades', 0)}</span></div>
                <div class="metric-box">Max DD<br><span class="val">{metrics.get('max_drawdown', 0):.2%}</span></div>
            </div>
            
            <h2>Aggregate Performance</h2>
            <img src="performance_charts.png">
            
            <h2>First 3 Breakout Visualizations</h2>
            {''.join([f'<div class="trade-viz"><h3>Breakout #{i+1}</h3><img src="{img}"></div>' for i, img in enumerate(trade_imgs)])}
            
            <p><a href="breakout_trades.csv" style="color: #4CAF50;">Download Raw Data</a></p>
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w') as f:
        f.write(html_content)

def run_server(port=8080):
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"\nServing at http://localhost:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    strategy = CryptoBreakoutStrategy()
    
    # 1. Restrict Asset List (First 3)
    strategy.symbols = strategy.symbols[:3]
    print(f"Active Assets: {strategy.symbols}")
    
    # 2. Run Backtest (90 Days)
    print("Running 90-day backtest...")
    trades = strategy.run_full_backtest(days=90)
    
    if not trades.empty:
        results = strategy.analyze_results(trades)
        
        # 3. Generate Report with Visuals
        print(f"\nGenerating visual report for {len(strategy.visual_contexts)} captured contexts...")
        generate_report_files(trades, results, strategy.visual_contexts)
        
        server_thread = threading.Thread(target=run_server, args=(8080,))
        server_thread.daemon = True
        server_thread.start()
        
        webbrowser.open('http://localhost:8080/report.html')
        
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nTerminated.")
    else:
        print("No trades found in this period.")
