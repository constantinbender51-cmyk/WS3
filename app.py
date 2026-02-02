import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
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
            print(f"Fetched up to {pd.to_datetime(last_timestamp, unit='ms')}")
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

@app.route('/')
def plot_png():
    global df
    
    # Calculate 365-day SMA (24 hours * 365 days)
    sma_window = 365 * 24
    df['sma'] = df['close'].rolling(window=sma_window).mean()
    
    # Filter out NaN values from SMA calculation for clean plotting
    plot_data = df.dropna(subset=['sma']).copy()
    
    fig, ax = plt.subplots(figsize=(15, 8), dpi=100)
    
    # Data conversion for matplotlib
    x = matplotlib.dates.date2num(plot_data['timestamp'])
    y = plot_data['close'].values
    sma = plot_data['sma'].values
    
    # Create segments for LineCollection
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Determine colors based on Price vs SMA relationship
    # We compare the start of the segment to the SMA
    # Logic: Green if Price > SMA, Red if Price <= SMA
    # Using the first point of the segment for color determination
    # Note: This is an approximation at the exact crossover point but sufficient for high density
    colors = ['green' if p > s else 'red' for p, s in zip(y[:-1], sma[:-1])]
    
    lc = LineCollection(segments, colors=colors, linewidth=1)
    ax.add_collection(lc)
    
    # Plot SMA
    ax.plot(plot_data['timestamp'], sma, color='white', linewidth=1, label='365d SMA')
    
    ax.autoscale_view()
    ax.set_title('ETH/USDT - 5 Year 1H Close Price vs 365d SMA')
    ax.set_ylabel('Price (USDT)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('white')
    
    # Format Date Axis
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    df = fetch_data()
    print("Serving plot on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)
