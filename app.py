import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import ccxt  # Requires: pip install ccxt

class CryptoBreakoutStrategy:
    def __init__(self, symbols: List[str] = None, fee: float = 0.0004):
        """
        Initialize strategy with major crypto symbols and trading fee
        Default symbols represent ~95% of crypto market cap
        """
        self.symbols = symbols or [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
            'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT', 'DOT/USDT'
        ]
        self.fee = fee  # 0.04% per trade side (entry + exit = 0.08% total)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
        self.sequence_lengths = range(10, 101)  # 10 to 100 candles

    def fetch_ohlc(self, symbol: str, timeframe: str, days: int = 180) -> pd.DataFrame:
        """Fetch OHLCV data with error handling and rate limiting"""
        since = self.exchange.milliseconds() - days * 24 * 60 * 60 * 1000
        all_ohlcv = []
        
        while since < self.exchange.milliseconds():
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1  # Next timestamp
            except Exception as e:
                print(f"Error fetching {symbol} {timeframe}: {e}")
                break
                
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def fit_channel(self, highs: np.ndarray, lows: np.ndarray) -> Tuple[float, float, float]:
        """
        Fit horizontal channel boundaries that:
        1. Touch at least one high/low point
        2. Never cross any highs/lows in the sequence (except last candle)
        3. Return top_line, bottom_line, and average distance metric
        
        Uses minimal envelope approach satisfying all constraints
        """
        # Horizontal lines satisfy constraints by construction
        top_line = np.max(highs)
        bottom_line = np.min(lows)
        
        # Verify constraint satisfaction (should always pass with this method)
        assert np.all(highs <= top_line + 1e-10), "Top line crosses highs"
        assert np.all(lows >= bottom_line - 1e-10), "Bottom line crosses lows"
        
        # Calculate average distance metric for channel tightness
        # Distance = normalized vertical distance from candles to channel boundaries
        candle_ranges = highs - lows
        channel_height = top_line - bottom_line + 1e-10  # Avoid div by zero
        
        # For each candle: distance to top + distance to bottom, normalized by channel height
        distances = ((top_line - highs) + (lows - bottom_line)) / channel_height
        avg_distance = np.mean(distances)
        
        return top_line, bottom_line, avg_distance

    def detect_breakout(self, df: pd.DataFrame, seq_len: int) -> List[Dict]:
        """
        Scan entire series for breakouts using sequence of length seq_len
        Returns list of breakout signals with metadata
        """
        signals = []
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Slide window across series (exclude incomplete windows)
        for i in range(seq_len, len(df)):
            # Sequence excluding last candle (for channel fitting)
            seq_highs = highs[i-seq_len:i-1]
            seq_lows = lows[i-seq_len:i-1]
            
            # Fit channel to historical data (excluding breakout candle)
            top_line, bottom_line, avg_dist = self.fit_channel(seq_highs, seq_lows)
            
            # Check breakout conditions on last candle
            current_high = highs[i-1]
            current_low = lows[i-1]
            current_close = closes[i-1]
            
            breakout = None
            direction = 0
            
            # Long breakout: price breaks above resistance
            if current_high > top_line:
                breakout = 'long'
                direction = 1
                entry_price = max(current_close, top_line)  # Conservative entry
                
            # Short breakout: price breaks below support
            elif current_low < bottom_line:
                breakout = 'short'
                direction = -1
                entry_price = min(current_close, bottom_line)  # Conservative entry
            
            if breakout:
                signals.append({
                    'timestamp': df.index[i-1],
                    'symbol': df.name if hasattr(df, 'name') else 'unknown',
                    'timeframe': getattr(df, 'timeframe', 'unknown'),
                    'sequence_length': seq_len,
                    'top_line': top_line,
                    'bottom_line': bottom_line,
                    'avg_distance': avg_dist,
                    'direction': direction,
                    'entry_price': entry_price,
                    'stop_loss': bottom_line if direction == 1 else top_line,
                    'breakout_type': breakout
                })
        
        return signals

    def apply_trailing_stop(self, trade: Dict, future_prices: pd.Series) -> Tuple[float, int]:
        """
        Simulate trade with trailing stop that locks in 90% of max profit
        Stop triggers when price gives back 10% of max achieved profit
        
        Returns: (net_return, exit_index)
        """
        direction = trade['direction']
        entry_price = trade['entry_price']
        initial_stop = trade['stop_loss']
        
        # Initialize tracking variables
        max_profit_price = entry_price if direction == 1 else entry_price
        trailing_stop = initial_stop
        exit_idx = -1
        exit_price = None
        
        # Simulate price evolution
        for idx, price in enumerate(future_prices):
            # Update max profit reference
            if direction == 1:  # Long
                if price > max_profit_price:
                    max_profit_price = price
                    # Trail stop at 90% of max profit level
                    trailing_stop = entry_price + 0.9 * (max_profit_price - entry_price)
            else:  # Short
                if price < max_profit_price:
                    max_profit_price = price
                    trailing_stop = entry_price - 0.9 * (entry_price - max_profit_price)
            
            # Check stop trigger (use low for longs, high for shorts would be more accurate)
            # Simplified: use close price for stop detection
            if (direction == 1 and price <= trailing_stop) or \
               (direction == -1 and price >= trailing_stop):
                exit_price = price
                exit_idx = idx
                break
        
        # Handle no exit (hold to end of data)
        if exit_price is None:
            exit_price = future_prices.iloc[-1]
            exit_idx = len(future_prices) - 1
        
        # Calculate gross return
        if direction == 1:
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price
        
        # Apply fees (entry + exit = 2 * fee)
        net_return = gross_return - (2 * self.fee)
        
        return net_return, exit_idx

    def resolve_conflicts(self, signals_at_timestamp: List[Dict]) -> Dict:
        """
        When multiple sequence lengths trigger at same timestamp:
        Select channel with LOWEST average distance (tightest fit)
        """
        return min(signals_at_timestamp, key=lambda s: s['avg_distance'])

    def backtest_symbol_timeframe(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Dict]:
        """Run full backtest for single symbol/timeframe combination"""
        # Attach metadata for later reference
        df.name = symbol
        df.timeframe = timeframe
        
        # Collect all potential signals across sequence lengths
        all_signals = []
        for seq_len in self.sequence_lengths:
            signals = self.detect_breakout(df, seq_len)
            all_signals.extend(signals)
        
        # Group signals by timestamp to resolve conflicts
        signals_by_time = {}
        for signal in all_signals:
            ts = signal['timestamp']
            if ts not in signals_by_time:
                signals_by_time[ts] = []
            signals_by_time[ts].append(signal)
        
        # Resolve conflicts and prepare trades
        trades = []
        position_active = False
        position_end_idx = -1
        
        # Sort timestamps chronologically
        for ts in sorted(signals_by_time.keys()):
            # Skip if we're still in an active position
            if position_active and ts <= position_end_idx:
                continue
                
            # Resolve multiple signals at same timestamp
            best_signal = self.resolve_conflicts(signals_by_time[ts])
            
            # Find future price series for trailing stop simulation
            ts_idx = df.index.get_loc(ts)
            future_prices = df['close'].iloc[ts_idx+1:]
            
            if len(future_prices) < 2:  # Need at least 2 candles for trailing stop logic
                continue
                
            # Simulate trade
            net_return, exit_offset = self.apply_trailing_stop(best_signal, future_prices)
            
            # Record trade
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
            
            # Block new entries until current position exits
            position_active = True
            position_end_idx = ts_idx + 1 + exit_offset
        
        return trades

    def run_full_backtest(self, days: int = 180) -> pd.DataFrame:
        """Execute complete backtest across all symbols and timeframes"""
        all_trades = []
        
        for symbol in self.symbols:
            print(f"\nFetching data for {symbol}...")
            try:
                # Fetch base 1m data and resample
                df_1m = self.fetch_ohlc(symbol, '1m', days)
                
                for tf in self.timeframes:
                    # Resample to target timeframe
                    df_tf = df_1m.resample(tf).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    
                    print(f"  Backtesting {symbol} on {tf} ({len(df_tf)} candles)...")
                    trades = self.backtest_symbol_timeframe(df_tf, symbol, tf)
                    all_trades.extend(trades)
                    
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    def analyze_results(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        if trades_df.empty:
            return {'error': 'No trades generated'}
        
        # Basic metrics
        total_return = (1 + trades_df['net_return']).prod() - 1
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['net_return'] > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Risk metrics
        returns = trades_df['net_return'].values
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0  # Annualized
        
        # Drawdown calculation
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['net_return'].sum()
        gross_loss = trades_df[trades_df['net_return'] < 0]['net_return'].sum()
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'best_trade': returns.max(),
            'worst_trade': returns.min(),
            'timeframe_distribution': trades_df['timeframe'].value_counts().to_dict(),
            'symbol_distribution': trades_df['symbol'].value_counts().to_dict()
        }


# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    # Initialize strategy
    strategy = CryptoBreakoutStrategy()
    
    # Run backtest (will take 10-30 minutes depending on API rate limits)
    print("Starting 6-month backtest across 10 symbols and 6 timeframes...")
    trades = strategy.run_full_backtest(days=180)
    
    # Save raw trades
    trades.to_csv('breakout_trades.csv', index=False)
    print(f"\nBacktest complete! Generated {len(trades)} trades")
    
    # Analyze results
    results = strategy.analyze_results(trades)
    print("\n=== PERFORMANCE METRICS ===")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:8.2%}" if 'return' in key or 'drawdown' in key else f"{key:20s}: {value:8.4f}")
        else:
            print(f"{key:20s}: {value}")
    
    # Example trade preview
    if not trades.empty:
        print("\n=== SAMPLE TRADES ===")
        print(trades[['entry_time', 'symbol', 'timeframe', 'direction', 'net_return']].head(10).to_string(index=False))