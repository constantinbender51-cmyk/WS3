import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
DAYS_BACK = 30
STOP_LOSS_PCT = 0.02  # 2% Stop Loss
STARTING_BALANCE = 10000

def fetch_binance_data(symbol, timeframe, days):
    """Fetches historical data from Binance using CCXT."""
    print(f"Fetching {days} days of {timeframe} data for {symbol}...")
    exchange = ccxt.binance()
    
    # Calculate timestamp for 30 days ago
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    
    # Binance limits to 1000 candles per request. 30 days * 24 hours = 720 candles.
    # So a single request is sufficient.
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def run_backtest(df):
    print("Running Backtest...")
    
    # Pre-calculate indicators
    # 5h avg volume (shifted by 1 so the current spiking bar isn't in its own average)
    df['vol_5h_avg'] = df['volume'].shift(1).rolling(window=5).mean()
    
    # Color: 1 for Green (Close > Open), -1 for Red (Close <= Open)
    df['color'] = np.where(df['close'] > df['open'], 1, -1)
    
    # State tracking variables
    state = 2             # Start in State 2 (No position)
    position = 0          # 1 for Long, -1 for Short, 0 for Flat
    entry_price = 0.0
    consecutive_sl = 0    # Tracks consecutive stop-loss hits
    
    balance = STARTING_BALANCE
    equity_curve = [balance] * 6  # Pad the first 6 hours used for moving averages
    
    # Loop through the dataframe (starting from index 6 to ensure we have the 5h MA)
    for i in range(6, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # --- 1. CHECK STATE TRANSITIONS ---
        # Trigger State 1: Previous volume was > 3x its previous 5h average
        if prev['volume'] > (3 * prev['vol_5h_avg']):
            state = 1
            consecutive_sl = 0 # Reset SL counter when entering State 1 newly
            
        # Trigger State 2: 3 consecutive Stop Losses hit
        if consecutive_sl >= 3:
            state = 2
            
            
        # --- 2. EXECUTE LOGIC BASED ON STATE ---
        if state == 2:
            # If we are holding a position when transitioning to State 2, close it
            if position != 0:
                pnl_pct = (current['open'] - entry_price) / entry_price if position == 1 else (entry_price - current['open']) / entry_price
                balance *= (1 + pnl_pct)
                position = 0
            
            equity_curve.append(balance)
            continue
            
            
        if state == 1:
            target_position = 1 if prev['color'] == 1 else -1
            
            # Flip or Open Position at the open of the current candle
            if position != target_position:
                # Close existing position if there is one
                if position != 0:
                    pnl_pct = (current['open'] - entry_price) / entry_price if position == 1 else (entry_price - current['open']) / entry_price
                    balance *= (1 + pnl_pct)
                
                # Open new position
                position = target_position
                entry_price = current['open']
                
            # Calculate exact Stop Loss Price
            sl_price = entry_price * (1 - STOP_LOSS_PCT) if position == 1 else entry_price * (1 + STOP_LOSS_PCT)
            
            # Check if Stop Loss is hit during this hour's candle
            sl_hit = False
            if position == 1 and current['low'] <= sl_price:
                sl_hit = True
                pnl_pct = -STOP_LOSS_PCT
            elif position == -1 and current['high'] >= sl_price:
                sl_hit = True
                pnl_pct = -STOP_LOSS_PCT
                
            if sl_hit:
                balance *= (1 + pnl_pct)  # Deduct SL loss
                position = 0              # Flatten position
                consecutive_sl += 1       # Increment consecutive SL counter
            else:
                consecutive_sl = 0        # Reset SL counter if we survived the hour
                
            # Track Equity (Unrealized if still holding, Realized if stopped out)
            if position != 0:
                unrealized_pnl = (current['close'] - entry_price) / entry_price if position == 1 else (entry_price - current['close']) / entry_price
                equity_curve.append(balance * (1 + unrealized_pnl))
            else:
                equity_curve.append(balance)
                
    # Add equity curve to dataframe
    df['equity'] = equity_curve + [balance] * (len(df) - len(equity_curve))
    return df, balance

def plot_results(df):
    plt.figure(figsize=(14, 7))
    
    # Plot Price
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df['timestamp'], df['close'], label='BTC Price', color='black')
    ax1.set_title('BTC/USDT 1h Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Equity Curve
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(df['timestamp'], df['equity'], label='Strategy Equity', color='blue')
    ax2.set_title('Backtest Equity Curve')
    ax2.set_ylabel('Balance (USDT)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Fetch Data
    df = fetch_binance_data(SYMBOL, TIMEFRAME, DAYS_BACK)
    
    # 2. Run Backtest
    results_df, final_balance = run_backtest(df)
    
    # 3. Print Stats
    roi = ((final_balance - STARTING_BALANCE) / STARTING_BALANCE) * 100
    print("-" * 30)
    print(f"Starting Balance: ${STARTING_BALANCE:.2f}")
    print(f"Final Balance:    ${final_balance:.2f}")
    print(f"Net ROI:          {roi:.2f}%")
    print("-" * 30)
    
    # 4. Plot
    plot_results(results_df)