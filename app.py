import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from flask import Flask, send_file, request
import io
import numpy as np

app = Flask(__name__)

# Global dataframe storage
df = None

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
            
    data = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    return data

def get_pnl_series(prices, highs, lows, entry_price, is_long, tp_pct, sl_pct):
    """
    Calculates PnL series for a single leg handling TP/SL exits.
    Returns: Series of PnL values over time.
    """
    n = len(prices)
    if n == 0:
        return np.zeros(0)

    tp_mult = (1 + tp_pct/100) if is_long else (1 - tp_pct/100)
    sl_mult = (1 - sl_pct/100) if is_long else (1 + sl_pct/100)
    
    tp_price = entry_price * tp_mult
    sl_price = entry_price * sl_mult
    
    # Identify exit points
    if is_long:
        # Long: High hits TP, Low hits SL
        hit_tp = highs >= tp_price
        hit_sl = lows <= sl_price
    else:
        # Short: Low hits TP, High hits SL
        hit_tp = lows <= tp_price
        hit_sl = highs >= sl_price
        
    # Find first index of exit (if any)
    first_tp_idx = np.argmax(hit_tp) if hit_tp.any() else n
    first_sl_idx = np.argmax(hit_sl) if hit_sl.any() else n
    
    exit_idx = min(first_tp_idx, first_sl_idx)
    
    # Calculate PnL
    current_pnl = (prices - entry_price) if is_long else (entry_price - prices)
    
    # Create a writable copy of the values array
    pnl_values = current_pnl.values.copy()
    
    if exit_idx < n:
        if first_sl_idx <= first_tp_idx:
            final_pnl = (sl_price - entry_price) if is_long else (entry_price - sl_price)
        else:
            final_pnl = (tp_price - entry_price) if is_long else (entry_price - tp_price)
            
        # Overwrite PnL after exit on the copy
        pnl_values[exit_idx:] = final_pnl
        
    return pnl_values

@app.route('/')
def index():
    sl = request.args.get('sl', '20')
    tp = request.args.get('tp', '50')
    
    html = f"""
    <html>
        <body style="font-family: monospace; background: #222; color: #ddd; text-align: center;">
            <h2>ETH/USDT Simultaneous Long + Short Strategy</h2>
            <form action="/" method="get" style="margin-bottom: 20px; padding: 10px; background: #333; display: inline-block; border-radius: 5px;">
                <label>Stop Loss (%): <input type="number" step="0.1" name="sl" value="{sl}" style="width: 60px;"></label>
                <label style="margin-left: 20px;">Take Profit (%): <input type="number" step="0.1" name="tp" value="{tp}" style="width: 60px;"></label>
                <input type="submit" value="Update" style="margin-left: 20px;">
            </form>
            <br>
            <img src="/plot.png?sl={sl}&tp={tp}" style="border: 1px solid #555;">
        </body>
    </html>
    """
    return html

@app.route('/plot.png')
def plot_png():
    global df
    
    try:
        sl_pct = float(request.args.get('sl', 20))
        tp_pct = float(request.args.get('tp', 50))
    except ValueError:
        sl_pct = 20
        tp_pct = 50

    # 1. 365-Day SMA
    sma_window = 365 * 24
    if 'sma' not in df.columns:
        df['sma'] = df['close'].rolling(window=sma_window).mean()
    
    # Prepare data starting from valid SMA
    plot_data = df.dropna(subset=['sma']).copy()
    plot_data = plot_data.reset_index(drop=True)
    
    if plot_data.empty:
        return "Not enough data", 400

    initial_price = plot_data['close'].iloc[0]
    prices = plot_data['close']
    highs = plot_data['high']
    lows = plot_data['low']
    
    # 2. Calculate PnL for legs
    plot_data['long_pnl'] = get_pnl_series(prices, highs, lows, initial_price, True, tp_pct, sl_pct)
    plot_data['short_pnl'] = get_pnl_series(prices, highs, lows, initial_price, False, tp_pct, sl_pct)
    plot_data['net_pnl'] = plot_data['long_pnl'] + plot_data['short_pnl']
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), dpi=100, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # Panel 1: Price & SMA
    x = matplotlib.dates.date2num(plot_data['timestamp'])
    y = plot_data['close'].values
    sma = plot_data['sma'].values
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    colors = ['green' if p > s else 'red' for p, s in zip(y[:-1], sma[:-1])]
    lc = LineCollection(segments, colors=colors, linewidth=1)
    ax1.add_collection(lc)
    ax1.plot(plot_data['timestamp'], sma, color='white', linewidth=1.5, label='365d SMA')
    
    # Mark entry point
    ax1.axhline(initial_price, color='gray', linestyle='--', alpha=0.5, label='Entry Price')
    
    ax1.set_title(f'ETH/USDT Price vs SMA (Entry: {initial_price:.2f})')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    
    # Panel 2: PnL
    ax2.plot(plot_data['timestamp'], plot_data['long_pnl'], color='green', alpha=0.3, label='Long PnL')
    ax2.plot(plot_data['timestamp'], plot_data['short_pnl'], color='red', alpha=0.3, label='Short PnL')
    ax2.plot(plot_data['timestamp'], plot_data['net_pnl'], color='cyan', linewidth=2, label='Net Equity')
    
    ax2.set_title(f'Equity (SL: {sl_pct}%, TP: {tp_pct}%)')
    ax2.set_ylabel('PnL (USDT)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.2)
    
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
    print("Serving on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)
