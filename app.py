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
            # Reduced print frequency for density
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
    
    # 1. Calculate 200-period SMA (200 Hours)
    df['sma'] = df['close'].rolling(window=200).mean()
    
    # 2. Strategy Logic: Stop and Reverse
    # Signal: 1 (Long) if Close > SMA, -1 (Short) if Close < SMA
    # "Open a long and short simultaneously" -> Interpreted as switching bias instant execution (Reversal)
    df['signal'] = np.where(df['close'] > df['sma'], 1, -1)
    
    # Calculate Returns
    # Shift signal by 1 because we trade at the Open of the NEXT candle based on Close of CURRENT
    df['market_return'] = df['close'].pct_change()
    df['strategy_return'] = df['market_return'] * df['signal'].shift(1)
    
    # 3. Equity Plot
    # Fill NaN from SMA window with 0 return
    df['strategy_return'] = df['strategy_return'].fillna(0)
    df['equity'] = (1 + df['strategy_return']).cumprod()
    
    # Filter for plotting
    plot_data = df.dropna(subset=['sma']).copy()
    
    # Plot Setup
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), dpi=100, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # --- Top Panel: Price vs SMA ---
    x = matplotlib.dates.date2num(plot_data['timestamp'])
    y = plot_data['close'].values
    sma = plot_data['sma'].values
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Color logic: Green if Price > SMA, Red if Price < SMA
    colors = ['green' if p > s else 'red' for p, s in zip(y[:-1], sma[:-1])]
    
    lc = LineCollection(segments, colors=colors, linewidth=1)
    ax1.add_collection(lc)
    ax1.plot(plot_data['timestamp'], sma, color='white', linewidth=1.5, label='200 SMA')
    
    ax1.set_title('ETH/USDT 1H - Price vs 200 SMA')
    ax1.set_ylabel('Price (USDT)')
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper left')
    
    # --- Bottom Panel: Equity Curve ---
    ax2.plot(plot_data['timestamp'], plot_data['equity'], color='cyan', linewidth=1.5)
    ax2.fill_between(plot_data['timestamp'], plot_data['equity'], 1, alpha=0.1, color='cyan')
    
    ax2.set_title('Strategy Equity (Reversal: Long > SMA, Short < SMA)')
    ax2.set_ylabel('Normalized Equity (Start=1.0)')
    ax2.grid(True, alpha=0.2)
    ax2.axhline(1, color='gray', linestyle='--', alpha=0.5)
    
    # Styling
    for ax in [ax1, ax2]:
        ax.set_facecolor('black')
    fig.patch.set_facecolor('white')
    
    # Format Date Axis
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
