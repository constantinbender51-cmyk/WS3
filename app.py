import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from flask import Flask, send_file, request, jsonify
import io
import numpy as np
import time

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: Numba not installed. Optimization will be slow.")
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

app = Flask(__name__)

# Global storage
df = None
market_data = {}

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

@njit(fastmath=True)
def optimize_single_pnl(opens, highs, lows, closes):
    """
    Optimizes for Total Net PnL (Profit) using a single SL/TP pair.
    """
    n = len(opens)
    fee_rate = 0.0002
    
    best_pnl = -np.inf
    best_tp = 5.0
    best_sl = 2.0
    
    # Grid Search 0.2% - 10%
    for tp_int in range(2, 101, 2): 
        for sl_int in range(2, 101, 2):
            tp_pct = tp_int / 10.0
            sl_pct = sl_int / 10.0
            
            l_tp_mult = 1.0 + tp_pct / 100.0
            l_sl_mult = 1.0 - sl_pct / 100.0
            s_tp_mult = 1.0 - tp_pct / 100.0
            s_sl_mult = 1.0 + sl_pct / 100.0
            
            total_pnl = 0.0
            
            l_closed_prev = True
            s_closed_prev = True
            
            for i in range(n):
                op = opens[i]
                hi = highs[i]
                lo = lows[i]
                cl = closes[i]
                
                # LONG
                l_entry_fee = op * fee_rate if l_closed_prev else 0.0
                l_sl_price = op * l_sl_mult
                l_tp_price = op * l_tp_mult
                
                l_exit_fee = 0.0
                l_trade_pnl = 0.0
                
                if lo <= l_sl_price:
                    l_trade_pnl = l_sl_price - op
                    l_exit_fee = l_sl_price * fee_rate
                    l_closed_prev = True 
                elif hi >= l_tp_price:
                    l_trade_pnl = l_tp_price - op
                    l_exit_fee = l_tp_price * fee_rate
                    l_closed_prev = True
                else:
                    l_trade_pnl = cl - op
                    l_exit_fee = 0.0
                    l_closed_prev = False
                
                l_net = l_trade_pnl - l_entry_fee - l_exit_fee
                
                # SHORT
                s_entry_fee = op * fee_rate if s_closed_prev else 0.0
                s_sl_price = op * s_sl_mult
                s_tp_price = op * s_tp_mult
                
                s_exit_fee = 0.0
                s_trade_pnl = 0.0
                
                if hi >= s_sl_price:
                    s_trade_pnl = op - s_sl_price
                    s_exit_fee = s_sl_price * fee_rate
                    s_closed_prev = True
                elif lo <= s_tp_price:
                    s_trade_pnl = op - s_tp_price
                    s_exit_fee = s_tp_price * fee_rate
                    s_closed_prev = True
                else:
                    s_trade_pnl = op - cl
                    s_exit_fee = 0.0
                    s_closed_prev = False
                
                s_net = s_trade_pnl - s_entry_fee - s_exit_fee
                
                total_pnl += (l_net + s_net)
            
            if total_pnl > best_pnl:
                best_pnl = total_pnl
                best_tp = tp_pct
                best_sl = sl_pct
                        
    return best_tp, best_sl, best_pnl

def calculate_strategy_single(data, tp_pct, sl_pct):
    n = len(data)
    timestamps = data['timestamp'].values
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    
    net_equity = np.zeros(n)
    trades = []
    current_cum_pnl = 0.0
    fee_rate = 0.0002
    
    l_closed_prev = True
    s_closed_prev = True
    
    tp_mult_l = 1 + tp_pct / 100.0
    sl_mult_l = 1 - sl_pct / 100.0
    tp_mult_s = 1 - tp_pct / 100.0
    sl_mult_s = 1 + sl_pct / 100.0
    
    for i in range(n):
        ts = timestamps[i]
        op = opens[i]
        hi = highs[i]
        lo = lows[i]
        cl = closes[i]
        
        # LONG
        l_entry_fee = op * fee_rate if l_closed_prev else 0.0
        l_sl_price = op * sl_mult_l
        l_tp_price = op * tp_mult_l
        
        l_exit = 0.0
        l_status = ""
        l_exit_fee = 0.0
        
        if lo <= l_sl_price:
            l_exit = l_sl_price
            l_status = "SL"
            l_exit_fee = l_sl_price * fee_rate
            l_closed_prev = True
        elif hi >= l_tp_price:
            l_exit = l_tp_price
            l_status = "TP"
            l_exit_fee = l_tp_price * fee_rate
            l_closed_prev = True
        else:
            l_exit = cl
            l_status = "TIME"
            l_exit_fee = 0.0
            l_closed_prev = False
            
        l_pnl = (l_exit - op) - l_entry_fee - l_exit_fee
        
        # SHORT
        s_entry_fee = op * fee_rate if s_closed_prev else 0.0
        s_sl_price = op * sl_mult_s
        s_tp_price = op * tp_mult_s
        
        s_exit = 0.0
        s_status = ""
        s_exit_fee = 0.0
        
        if hi >= s_sl_price:
            s_exit = s_sl_price
            s_status = "SL"
            s_exit_fee = s_sl_price * fee_rate
            s_closed_prev = True
        elif lo <= s_tp_price:
            s_exit = s_tp_price
            s_status = "TP"
            s_exit_fee = s_tp_price * fee_rate
            s_closed_prev = True
        else:
            s_exit = cl
            s_status = "TIME"
            s_exit_fee = 0.0
            s_closed_prev = False
            
        s_pnl = (op - s_exit) - s_entry_fee - s_exit_fee
        
        net_pnl = l_pnl + s_pnl
        current_cum_pnl += net_pnl
        net_equity[i] = current_cum_pnl
        
        if l_status != "TIME" or s_status != "TIME":
             trades.append({
                'entry_time': pd.to_datetime(ts),
                'l_status': l_status,
                's_status': s_status,
                'pnl': net_pnl
            })
            
    return net_equity, trades

@app.route('/')
def index():
    global df
    try:
        sl = float(request.args.get('sl', 2.0))
        tp = float(request.args.get('tp', 5.0))
    except:
        sl, tp = 2.0, 5.0
        
    if df is None:
        return "Data not loaded", 500

    calc_data = df.dropna(subset=['close']).reset_index(drop=True)
    net_equity, trades = calculate_strategy_single(calc_data, tp, sl)
    
    rows = ""
    for t in reversed(trades[-50:]):
        color = "#00ff00" if t['pnl'] > 0 else "#ff0000"
        rows += f"""
        <tr>
            <td>{t['entry_time']}</td>
            <td>L:{t['l_status']} / S:{t['s_status']}</td>
            <td style="color: {color}">{t['pnl']:.2f}</td>
        </tr>
        """
    
    html = f"""
    <html>
        <head>
            <style>
                body {{ font-family: monospace; background: #222; color: #ddd; margin: 0; padding: 20px; }}
                .container {{ max_width: 1200px; margin: 0 auto; text-align: center; }}
                input, button {{ background: #333; color: white; border: 1px solid #555; padding: 5px; width: 80px; }}
                button {{ cursor: pointer; width: auto; background: #554400; }}
                .params-box {{ background: #333; padding: 15px; border-radius: 5px; display: inline-block; text-align: left; }}
                .params-row {{ margin-bottom: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 12px; }}
                th {{ background: #444; padding: 10px; text-align: left; }}
                td {{ border-bottom: 1px solid #333; padding: 5px; text-align: left; }}
                .scroll-table {{ max_height: 500px; overflow-y: auto; border: 1px solid #444; }}
                #optimize-status {{ margin-top: 10px; color: yellow; }}
            </style>
            <script>
                function runOptimization() {{
                    document.getElementById('optimize-status').innerText = "Optimizing Single SL/TP (Train PnL)... Please wait.";
                    fetch('/optimize')
                        .then(response => response.json())
                        .then(data => {{
                            document.getElementById('optimize-status').innerText = "Done! SL: " + data.sl + "% | TP: " + data.tp + "% (PnL: " + data.pnl.toFixed(2) + ")";
                            document.querySelector('input[name="tp"]').value = data.tp;
                            document.querySelector('input[name="sl"]').value = data.sl;
                        }})
                        .catch(err => {{
                            document.getElementById('optimize-status').innerText = "Error: " + err;
                        }});
                }}
            </script>
        </head>
        <body>
            <div class="container">
                <h2>ETH/USDT Single Strategy Optimization</h2>
                <p>Grid Search: Single TP/SL (0.2-10%). Train < 2026. Metric: PnL.</p>
                
                <form action="/" method="get">
                    <div class="params-box">
                        <div class="params-row">
                            <strong>Global Params:</strong>
                            SL % <input type="number" step="0.1" name="sl" value="{sl}">
                            TP % <input type="number" step="0.1" name="tp" value="{tp}">
                        </div>
                        <div style="text-align: center; margin-top: 10px;">
                            <input type="submit" value="Update Plots" style="width: auto; cursor: pointer;">
                        </div>
                    </div>
                </form>
                
                <button onclick="runOptimization()">Run Optimization</button>
                <div id="optimize-status"></div>
                
                <br>
                <img src="/plot.png?sl={sl}&tp={tp}" style="border: 1px solid #555; max-width: 100%; margin-top: 20px;">
                
                <h3>Recent Events</h3>
                <div class="scroll-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Entry Time</th>
                                <th>Status</th>
                                <th>Net PnL</th>
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

@app.route('/optimize')
def optimize():
    global market_data, df
    
    cutoff_ts = pd.Timestamp('2026-01-01')
    
    if 'opens' not in market_data:
        calc_data = df.dropna(subset=['close']).reset_index(drop=True)
        market_data['timestamps'] = calc_data['timestamp'].values
        market_data['opens'] = calc_data['open'].values.astype(np.float64)
        market_data['highs'] = calc_data['high'].values.astype(np.float64)
        market_data['lows'] = calc_data['low'].values.astype(np.float64)
        market_data['closes'] = calc_data['close'].values.astype(np.float64)

    train_mask = market_data['timestamps'] < np.datetime64(cutoff_ts)
    
    opens = market_data['opens'][train_mask]
    highs = market_data['highs'][train_mask]
    lows = market_data['lows'][train_mask]
    closes = market_data['closes'][train_mask]
    
    t0 = time.time()
    
    best_tp, best_sl, best_pnl = optimize_single_pnl(opens, highs, lows, closes)
    
    print(f"Optimization done in {time.time() - t0:.2f}s")
    
    return jsonify({
        'tp': round(best_tp, 1),
        'sl': round(best_sl, 1),
        'pnl': best_pnl
    })

@app.route('/plot.png')
def plot_png():
    global df
    try:
        sl = float(request.args.get('sl', 2.0))
        tp = float(request.args.get('tp', 5.0))
    except:
        sl, tp = 2.0, 5.0

    plot_data = df.dropna(subset=['close']).reset_index(drop=True).copy()
    
    net_equity, _ = calculate_strategy_single(plot_data, tp, sl)
    plot_data['net_equity'] = net_equity
    
    cutoff_ts = pd.Timestamp('2026-01-01')
    unseen_data = plot_data[plot_data['timestamp'] >= cutoff_ts].copy()
    if not unseen_data.empty:
        unseen_data['net_equity'] -= unseen_data['net_equity'].iloc[0]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), dpi=100, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    x = matplotlib.dates.date2num(plot_data['timestamp'])
    y = plot_data['close'].values
    
    # Simple price plot, no color segments needed since no SMA
    ax1.plot(plot_data['timestamp'], y, color='white', linewidth=1)
    ax1.set_title('ETH/USDT Price')
    ax1.axvline(cutoff_ts, color='yellow', linestyle='--')
    ax1.set_xlim(plot_data['timestamp'].min(), plot_data['timestamp'].max())
    ax1.grid(True, alpha=0.2)
    
    ax2.plot(plot_data['timestamp'], plot_data['net_equity'], color='cyan')
    ax2.set_title('Full History PnL')
    ax2.axvline(cutoff_ts, color='yellow', linestyle='--')
    ax2.set_xlim(plot_data['timestamp'].min(), plot_data['timestamp'].max())
    ax2.grid(True, alpha=0.2)
    
    if not unseen_data.empty:
        ax3.plot(unseen_data['timestamp'], unseen_data['net_equity'], color='magenta')
        ax3.set_title('Unseen Data PnL (2026+)')
        ax3.axhline(0, color='white', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No Unseen Data", color='white')
    ax3.grid(True, alpha=0.2)
    
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('black')
    fig.patch.set_facecolor('white')
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
    
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
