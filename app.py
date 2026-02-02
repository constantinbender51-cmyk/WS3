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

def calculate_strategy(data, tp_pct, sl_pct):
    """
    Simulates continuous trading. Returns equity curves and trade log.
    """
    n = len(data)
    timestamps = data['timestamp'].values
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    
    long_equity = np.zeros(n)
    short_equity = np.zeros(n)
    trades = []
    
    # State
    l_active = False
    l_entry_price = 0.0
    l_entry_idx = 0
    l_realized = 0.0
    
    s_active = False
    s_entry_price = 0.0
    s_entry_idx = 0
    s_realized = 0.0
    
    l_tp_mult = 1 + tp_pct / 100.0
    l_sl_mult = 1 - sl_pct / 100.0
    s_tp_mult = 1 - tp_pct / 100.0
    s_sl_mult = 1 + sl_pct / 100.0
    
    for i in range(n):
        ts = timestamps[i]
        op = opens[i]
        hi = highs[i]
        lo = lows[i]
        cl = closes[i]
        
        # --- LONG ---
        if not l_active:
            l_entry_price = op
            l_entry_idx = i
            l_active = True
            
        l_tp = l_entry_price * l_tp_mult
        l_sl = l_entry_price * l_sl_mult
        
        l_pnl = 0.0
        l_status = ""
        l_exit_price = 0.0
        
        if lo <= l_sl:
            l_pnl = l_sl - l_entry_price
            l_realized += l_pnl
            l_active = False
            l_status = "SL"
            l_exit_price = l_sl
        elif hi >= l_tp:
            l_pnl = l_tp - l_entry_price
            l_realized += l_pnl
            l_active = False
            l_status = "TP"
            l_exit_price = l_tp
        
        if not l_active:
            trades.append({
                'entry_time': pd.to_datetime(timestamps[l_entry_idx]),
                'exit_time': pd.to_datetime(ts),
                'type': 'LONG',
                'entry_price': l_entry_price,
                'exit_price': l_exit_price,
                'pnl': l_pnl,
                'status': l_status
            })
            long_equity[i] = l_realized
        else:
            long_equity[i] = l_realized + (cl - l_entry_price)

        # --- SHORT ---
        if not s_active:
            s_entry_price = op
            s_entry_idx = i
            s_active = True
            
        s_tp = s_entry_price * s_tp_mult
        s_sl = s_entry_price * s_sl_mult
        
        s_pnl = 0.0
        s_status = ""
        s_exit_price = 0.0
        
        if hi >= s_sl:
            s_pnl = s_entry_price - s_sl
            s_realized += s_pnl
            s_active = False
            s_status = "SL"
            s_exit_price = s_sl
        elif lo <= s_tp:
            s_pnl = s_entry_price - s_tp
            s_realized += s_pnl
            s_active = False
            s_status = "TP"
            s_exit_price = s_tp
            
        if not s_active:
            trades.append({
                'entry_time': pd.to_datetime(timestamps[s_entry_idx]),
                'exit_time': pd.to_datetime(ts),
                'type': 'SHORT',
                'entry_price': s_entry_price,
                'exit_price': s_exit_price,
                'pnl': s_pnl,
                'status': s_status
            })
            short_equity[i] = s_realized
        else:
            short_equity[i] = s_realized + (s_entry_price - cl)
            
    return long_equity, short_equity, trades

@app.route('/')
def index():
    global df
    try:
        sl = float(request.args.get('sl', 20))
        tp = float(request.args.get('tp', 50))
    except ValueError:
        sl = 20.0
        tp = 50.0

    # Ensure data availability
    sma_window = 365 * 24
    if 'sma' not in df.columns:
        df['sma'] = df['close'].rolling(window=sma_window).mean()
        
    calc_data = df.dropna(subset=['sma']).reset_index(drop=True)
    
    # Calculate strategy to get trades
    _, _, trades = calculate_strategy(calc_data, tp, sl)
    
    # Generate Table Rows (Reverse order: newest first)
    rows = ""
    for t in reversed(trades):
        color = "#00ff00" if t['pnl'] > 0 else "#ff0000"
        rows += f"""
        <tr>
            <td>{t['entry_time']}</td>
            <td>{t['exit_time']}</td>
            <td style="color: {'cyan' if t['type']=='LONG' else 'orange'}">{t['type']}</td>
            <td>{t['entry_price']:.2f}</td>
            <td>{t['exit_price']:.2f}</td>
            <td>{t['status']}</td>
            <td style="color: {color}">{t['pnl']:.2f}</td>
        </tr>
        """
    
    html = f"""
    <html>
        <head>
            <style>
                body {{ font-family: monospace; background: #222; color: #ddd; margin: 0; padding: 20px; }}
                .container {{ max_width: 1200px; margin: 0 auto; text-align: center; }}
                input {{ background: #333; color: white; border: 1px solid #555; padding: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 12px; }}
                th {{ background: #444; padding: 10px; text-align: left; position: sticky; top: 0; }}
                td {{ border-bottom: 1px solid #333; padding: 5px; text-align: left; }}
                .scroll-table {{ max_height: 500px; overflow-y: auto; border: 1px solid #444; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>ETH/USDT Continuous Hedge Strategy</h2>
                <form action="/" method="get">
                    <label>Stop Loss (%): <input type="number" step="0.1" name="sl" value="{sl}"></label>
                    <label style="margin-left: 20px;">Take Profit (%): <input type="number" step="0.1" name="tp" value="{tp}"></label>
                    <input type="submit" value="Update" style="margin-left: 20px; cursor: pointer;">
                </form>
                <br>
                <img src="/plot.png?sl={sl}&tp={tp}" style="border: 1px solid #555; max-width: 100%;">
                
                <h3>Trade History ({len(trades)} Trades)</h3>
                <div class="scroll-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Entry Time</th>
                                <th>Exit Time</th>
                                <th>Type</th>
                                <th>Entry Price</th>
                                <th>Exit Price</th>
                                <th>Status</th>
                                <th>PnL</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows}
                        </tbody>
                    </table>
                </div>
            </div>
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

    sma_window = 365 * 24
    if 'sma' not in df.columns:
        df['sma'] = df['close'].rolling(window=sma_window).mean()
    
    plot_data = df.dropna(subset=['sma']).copy()
    plot_data = plot_data.reset_index(drop=True)
    
    if plot_data.empty:
        return "Not enough data", 400
    
    # Calculate Equity
    long_eq, short_eq, _ = calculate_strategy(plot_data, tp_pct, sl_pct)
    
    plot_data['long_pnl'] = long_eq
    plot_data['short_pnl'] = short_eq
    plot_data['net_pnl'] = long_eq + short_eq
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), dpi=100, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # Panel 1
    x = matplotlib.dates.date2num(plot_data['timestamp'])
    y = plot_data['close'].values
    sma = plot_data['sma'].values
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = ['green' if p > s else 'red' for p, s in zip(y[:-1], sma[:-1])]
    
    lc = LineCollection(segments, colors=colors, linewidth=1)
    ax1.add_collection(lc)
    ax1.plot(plot_data['timestamp'], sma, color='white', linewidth=1.5, label='365d SMA')
    ax1.set_title('ETH/USDT Price vs SMA')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    
    # Panel 2
    ax2.plot(plot_data['timestamp'], plot_data['long_pnl'], color='green', alpha=0.3, label='Long Leg PnL')
    ax2.plot(plot_data['timestamp'], plot_data['short_pnl'], color='red', alpha=0.3, label='Short Leg PnL')
    ax2.plot(plot_data['timestamp'], plot_data['net_pnl'], color='cyan', linewidth=2, label='Net Equity')
    ax2.set_title(f'Equity Curve (SL: {sl_pct}%, TP: {tp_pct}%)')
    ax2.set_ylabel('PnL (USDT)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.2)
    
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
