import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    # Re-fetch or cache data here. For static serve, we use global df or fetch once. 
    # Assuming global df for persistence as per previous logic flow.
    global df
    
    fig, ax = plt.subplots(figsize=(15, 8), dpi=100)
    
    # Vectorized candlestick construction using standard matplotlib
    up = df[df.close >= df.open]
    down = df[df.close < df.open]
    
    # Width of 1h candle in matplotlib date units (1 day = 1.0). 
    # 1h = 1/24. 0.03 is roughly 0.7 of an hour for spacing.
    width = 0.03 
    
    # Plot Up candles
    ax.bar(up.timestamp, up.close - up.open, width, bottom=up.open, color='green', edgecolor='green')
    ax.vlines(up.timestamp, up.low, up.high, color='green', linewidth=1)
    
    # Plot Down candles
    ax.bar(down.timestamp, down.close - down.open, width, bottom=down.open, color='red', edgecolor='red')
    ax.vlines(down.timestamp, down.low, down.high, color='red', linewidth=1)
    
    ax.set_title('ETH/USDT - 5 Year 1H OHLC')
    ax.set_ylabel('Price (USDT)')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('white')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    df = fetch_data()
    print("Serving plot on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)
