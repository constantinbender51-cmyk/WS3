import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from flask import Flask, send_file

# 1. Fetch Data
def fetch_btc_data():
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1d'
    since = exchange.parse8601('2017-01-01T00:00:00Z') # Extended history for moving averages
    
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

# 2. Strategy & Analysis Logic
def apply_analysis(df):
    # --- Strategy Anchors ---
    anchor_date = pd.Timestamp('2017-07-01')
    
    # --- Basic Returns ---
    df['returns'] = df['close'].pct_change()
    
    # --- Moving Averages ---
    df['sma730'] = df['close'].rolling(window=730).mean()
    df['sma1460'] = df['close'].rolling(window=1460).mean()
    
    # Offset 1460 SMA by -1460 days (shift backwards)
    # Note: This moves future values to the past. Recent 1460 days will be NaN.
    df['sma1460_shifted'] = df['sma1460'].shift(-1460)

    # --- Linear Regression (Log-Linear) ---
    # Convert dates to ordinal for regression
    df_reg = df.dropna(subset=['close']).copy()
    df_reg['ordinal'] = df_reg.index.map(pd.Timestamp.toordinal)
    
    # Fit log(price) = mx + c
    slope, intercept = np.polyfit(df_reg['ordinal'], np.log(df_reg['close']), 1)
    
    # Calculate Regression Line Values for original df
    df['ordinal'] = df.index.map(pd.Timestamp.toordinal)
    df['log_lin_reg'] = np.exp(slope * df['ordinal'] + intercept)
    
    # --- Deduction (730 SMA - Regression Line) ---
    df['deduced_metric'] = df['sma730'] - df['log_lin_reg']

    # --- Cycle Signal ---
    def get_signal(date):
        if date < anchor_date:
            return 0 
        
        days_since = (date - anchor_date).days
        years_passed = days_since / 365.25
        cycle_position = years_passed % 4
        
        if 0 <= cycle_position < 1:
            return -1
        else:
            return 1

    df['signal'] = df.index.to_series().apply(get_signal)
    df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
    
    # --- Cumulative Results ---
    df['cum_bnh'] = (1 + df['returns']).cumprod()
    df['cum_strat'] = (1 + df['strategy_returns']).cumprod()
    
    return df

# 3. Server
app = Flask(__name__)

@app.route('/')
def serve_plot():
    df = fetch_btc_data()
    df = apply_analysis(df)
    
    # Create 2 subplots sharing x-axis
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Top Plot: Strategy & Price ---
    ax2 = ax1.twinx() # Secondary y-axis for Price/SMAs
    
    # Left Axis: Cumulative Returns
    ax1.plot(df.index, df['cum_bnh'], label='Buy & Hold (Left)', alpha=0.3, color='gray', linestyle=':')
    ax1.plot(df.index, df['cum_strat'], label='Strat (Left)', color='blue', linewidth=2)
    ax1.set_ylabel('Cum. Return')
    ax1.set_yscale('log')
    
    # Right Axis: Price, SMAs, Regression
    ax2.plot(df.index, df['close'], label='Price', color='black', alpha=0.1)
    ax2.plot(df.index, df['sma730'], label='SMA 730', color='orange')
    ax2.plot(df.index, df['sma1460_shifted'], label='SMA 1460 (Shift -1460)', color='purple', linestyle='--')
    ax2.plot(df.index, df['log_lin_reg'], label='Log-Lin Reg', color='green', linestyle='-.')
    ax2.set_ylabel('Price (USDT)')
    ax2.set_yscale('log')

    # Cycle Markers (Top Plot)
    cycle_years = range(2017, df.index.year.max() + 1, 4)
    for y in cycle_years:
        d = pd.Timestamp(f'{y}-07-01')
        if d >= df.index.min() and d <= df.index.max():
            ax1.axvline(d, color='red', linestyle='--', alpha=0.3)
    
    ax1.set_title('BTC Strategy | 730 SMA | 1460 SMA Offset | Regression')
    
    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='small')

    # --- Bottom Plot: Deduction ---
    ax3.plot(df.index, df['deduced_metric'], label='730 SMA - Regression', color='brown')
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_ylabel('Diff (USDT)')
    ax3.set_title('Deduction: SMA 730 - Log Linear Regression')
    ax3.legend(loc='upper left', fontsize='small')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    print("Serving plot on port 8080...")
    app.run(host='0.0.0.0', port=8080)
