import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from flask import Flask, send_file
import io
import numpy as np

app = Flask(__name__)

def fetch_data():
    exchange = ccxt.binance({'enableRateLimit': True})
    symbol = 'ETH/USDT'
    timeframe = '1h'
    limit = 1000
    
    now = exchange.milliseconds()
    since = now - (5 * 365 * 24 * 60 * 60 * 1000)
    
    all_ohlcv = []
    
    print(f"Fetching 5 years of {timeframe} data for {symbol}...")
    
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 3600000
            if len(all_ohlcv) % 10000 == 0:
                print(f"Fetched {len(all_ohlcv)} candles...")
        except Exception as e:
            print(f"Error: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

@app.route('/')
def plot_png():
    global df
    
    # 1. Calculate 365-Day SMA (365 days * 24 hours)
    # 5 years is ~43,800 hours. 365 days is 8,760 hours.
    sma_window = 365 * 24
    df['sma'] = df['close'].rolling(window=sma_window).mean()
    
    # 2. Simultaneous Long and Short Logic
    # Assumption: Hold 1 unit Long and 1 unit Short from the start of the plotting period.
    # Start analysis from when SMA is available
    plot_data = df.dropna(subset=['sma']).copy()
    plot_data = plot_data.reset_index(drop=True)
    
    initial_price = plot_data['close'].iloc[0]
    
    # Long PnL: Price - Initial
    plot_data['long_pnl'] = plot_data['close'] - initial_price
    
    # Short PnL: Initial - Price
    plot_data['short_pnl'] = initial_price - plot_data['close']
    
    # Net Equity (PnL)
    plot_data['net_pnl'] = plot_data['long_pnl'] + plot_data['short_pnl']
    
    # Plot Setup
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), dpi=100, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # --- Top Panel: Price vs 365-Day SMA ---
    x = matplotlib.dates.date2num(plot_data['timestamp'])
    y = plot_data['close'].values
    sma = plot_data['sma'].values
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Color logic: Green if Price > SMA, Red if Price <= SMA
    colors = ['green' if p > s else 'red' for p, s in zip(y[:-1], sma[:-1])]
    
    lc = LineCollection(segments, colors=colors, linewidth=1)
    ax1.add_collection(lc)
    ax1.plot(plot_data['timestamp'], sma, color='white', linewidth=1.5, label='365d SMA')
    
    ax1.set_title('ETH/USDT 1H - Price vs 365 Day SMA')
    ax1.set_ylabel('Price (USDT)')
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper left')
    
    # --- Bottom Panel: Equity Curve ---
    # Plotting Long, Short, and Net to demonstrate the hedge
    ax2.plot(plot_data['timestamp'], plot_data['long_pnl'], color='green', alpha=0.3, linewidth=1, label='Long Only PnL')
    ax2.plot(plot_data['timestamp'], plot_data['short_pnl'], color='red', alpha=0.3, linewidth=1, label='Short Only PnL')
    ax2.plot(plot_data['timestamp'], plot_data['net_pnl'], color='cyan', linewidth=2, label='Net Equity (Simultaneous)')
    
    ax2.set_title('Simultaneous Long + Short Equity (PnL)')
    ax2.set_ylabel('PnL (USDT)')
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper left')
    
    # Styling
    for ax in [ax1, ax2]:
        ax.set_facecolor('black')
        ax.autoscale_view()
    
    fig.patch.set_facecolor('white')
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    df = fetch_data()
    print("Serving plot on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)
