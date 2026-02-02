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

def calculate_continuous_equity(data, tp_pct, sl_pct):
    """
    Simulates continuous trading for Long and Short legs independently.
    If a leg hits TP/SL, it closes and re-opens on the next candle's Open.
    """
    n = len(data)
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    
    # Storage for equity curves
    long_equity = np.zeros(n)
    short_equity = np.zeros(n)
    
    # State variables
    # Long
    long_active = False
    long_entry = 0.0
    long_realized = 0.0
    
    # Short
    short_active = False
    short_entry = 0.0
    short_realized = 0.0
    
    # Multipliers
    long_tp_mult = 1 + tp_pct / 100.0
    long_sl_mult = 1 - sl_pct / 100.0
    short_tp_mult = 1 - tp_pct / 100.0
    short_sl_mult = 1 + sl_pct / 100.0
    
    for i in range(n):
        current_open = opens[i]
        current_high = highs[i]
        current_low = lows[i]
        current_close = closes[i]
        
        # --- LONG LEG ---
        if not long_active:
            long_entry = current_open
            long_active = True
            
        # Check TP/SL
        l_tp_price = long_entry * long_tp_mult
        l_sl_price = long_entry * long_sl_mult
        
        # Determine if hit (Assuming SL hit first if both in range for safety)
        if current_low <= l_sl_price:
            long_realized += (l_sl_price - long_entry)
            long_active = False # Will re-open next iteration (i+1)
            long_val = long_realized
        elif current_high >= l_tp_price:
            long_realized += (l_tp_price - long_entry)
            long_active = False
            long_val = long_realized
        else:
            # Mark to Market
            long_val = long_realized + (current_close - long_entry)
            
        long_equity[i] = long_val
        
        # --- SHORT LEG ---
        if not short_active:
            short_entry = current_open
            short_active = True
            
        s_tp_price = short_entry * short_tp_mult
        s_sl_price = short_entry * short_sl_mult
        
        if current_high >= s_sl_price:
            short_realized += (short_entry - s_sl_price)
            short_active = False
            short_val = short_realized
        elif current_low <= s_tp_price:
            short_realized += (short_entry - s_tp_price)
            short_active = False
            short_val = short_realized
        else:
            short_val = short_realized + (short_entry - current_close)
            
        short_equity[i] = short_val
        
    return long_equity, short_equity

@app.route('/')
def index():
    sl = request.args.get('sl', '20')
    tp = request.args.get('tp', '50')
    
    html = f"""
    <html>
        <body style="font-family: monospace; background: #222; color: #ddd; text-align: center;">
            <h2>ETH/USDT Continuous Hedge Strategy</h2>
            <p>Logic: Always in Long + Short. If TP/SL hit, re-open next hour.</p>
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
    
    # 2. Calculate Continuous PnL
    long_eq, short_eq = calculate_continuous_equity(plot_data, tp_pct, sl_pct)
    
    plot_data['long_pnl'] = long_eq
    plot_data['short_pnl'] = short_eq
    plot_data['net_pnl'] = long_eq + short_eq
    
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
    
    ax1.set_title(f'ETH/USDT Price vs SMA')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    
    # Panel 2: PnL
    ax2.plot(plot_data['timestamp'], plot_data['long_pnl'], color='green', alpha=0.3, label='Long Leg PnL')
    ax2.plot(plot_data['timestamp'], plot_data['short_pnl'], color='red', alpha=0.3, label='Short Leg PnL')
    ax2.plot(plot_data['timestamp'], plot_data['net_pnl'], color='cyan', linewidth=2, label='Net Equity')
    
    ax2.set_title(f'Continuous Hedge Equity (SL: {sl_pct}%, TP: {tp_pct}%)')
    ax2.set_ylabel('Cumulative PnL (USDT)')
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
