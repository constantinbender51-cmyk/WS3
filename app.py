import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server
import matplotlib.pyplot as plt
import io
from flask import Flask, send_file
from datetime import datetime, timedelta

# 1. Fetch Data
def fetch_btc_data():
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1d'
    since = exchange.parse8601('2018-01-01T00:00:00Z')
    
    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1
        if len(candles) < 1000:
            break
            
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    return df

# 2. Strategy Logic
def apply_strategy(df):
    # Anchor date: July 1, 2018 (First July 1 in dataset)
    anchor_date = pd.Timestamp('2018-07-01')
    
    # Calculate daily returns
    df['returns'] = df['close'].pct_change()
    
    # Define signal function
    def get_signal(date):
        if date < anchor_date:
            return 0 # No position before cycle start
        
        # Calculate years since anchor
        # We use days/365.25 to approximate position in 4-year cycle
        days_since = (date - anchor_date).days
        years_passed = days_since / 365.25
        cycle_position = years_passed % 4
        
        # 0.0 - 1.0: Short (1st year)
        # 1.0 - 4.0: Long (Next 3 years)
        if 0 <= cycle_position < 1:
            return -1
        else:
            return 1

    df['signal'] = df.index.to_series().apply(get_signal)
    
    # Calculate Strategy Returns
    df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
    
    # Cumulative Returns
    df['cum_bnh'] = (1 + df['returns']).cumprod()
    df['cum_strat'] = (1 + df['strategy_returns']).cumprod()
    
    return df

# 3. Server
app = Flask(__name__)

@app.route('/')
def serve_plot():
    # Fetch and Calculate
    df = fetch_btc_data()
    df = apply_strategy(df)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Buy & Hold
    ax.plot(df.index, df['cum_bnh'], label='Buy & Hold (BTC)', alpha=0.5, color='gray')
    
    # Plot Strategy
    ax.plot(df.index, df['cum_strat'], label='4Y Cycle (1S/3L)', color='blue')
    
    # Visual markers for cycle resets (July 1sts)
    cycle_years = range(2018, df.index.year.max() + 1, 4)
    for y in cycle_years:
        d = pd.Timestamp(f'{y}-07-01')
        if d >= df.index.min() and d <= df.index.max():
            ax.axvline(d, color='red', linestyle='--', alpha=0.3, label='Cycle Start (Short)' if y == 2018 else "")

    ax.set_title('BTC 4-Year Cycle Strategy (Start July 1: 1y Short, 3y Long)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    # Save to buffer
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    print("Serving plot on port 8080...")
    app.run(host='0.0.0.0', port=8080)
