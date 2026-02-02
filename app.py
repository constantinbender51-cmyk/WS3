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
    data['sma'] = data['close'].rolling(window=365*24).mean()
    
    return data

@njit(fastmath=True)
def optimize_regime_sharpe(opens, highs, lows, closes, smas, target_above_sma):
    n = len(opens)
    fee_rate = 0.0002
    
    best_sharpe = -np.inf
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
            
            sum_returns = 0.0
            sum_returns_sq = 0.0
            count = 0
            
            l_closed_prev = True
            s_closed_prev = True
            
            for i in range(n):
                op = opens[i]
                sma = smas[i]
                
                # Check Regime
                is_above = op > sma
                if is_above != target_above_sma:
                    l_closed_prev = True
                    s_closed_prev = True
                    continue
                
                hi = highs[i]
                lo = lows[i]
                cl = closes[i]
                
                # --- LONG LEG ---
                l_entry_fee = op * fee_rate if l_closed_prev else 0.0
                l_sl_price = op * l_sl_mult
                l_tp_price = op * l_tp_mult
                
                l_pnl = 0.0
                l_exit_fee = 0.0
                
                if lo <= l_sl_price:
                    l_pnl = l_sl_price - op
                    l_exit_fee = l_sl_price * fee_rate
                    l_closed_prev = True 
                elif hi >= l_tp_price:
                    l_pnl = l_tp_price - op
                    l_exit_fee = l_tp_price * fee_rate
                    l_closed_prev = True
                else:
                    l_pnl = cl - op
                    l_exit_fee = 0.0
                    l_closed_prev = False
                
                l_net = l_pnl - l_entry_fee - l_exit_fee
                
                # --- SHORT LEG ---
                s_entry_fee = op * fee_rate if s_closed_prev else 0.0
                s_sl_price = op * s_sl_mult
                s_tp_price = op * s_tp_mult
                
                s_pnl = 0.0
                s_exit_fee = 0.0
                
                if hi >= s_sl_price:
                    s_pnl = op - s_sl_price
                    s_exit_fee = s_sl_price * fee_rate
                    s_closed_prev = True
                elif lo <= s_tp_price:
                    s_pnl = op - s_tp_price
                    s_exit_fee = s_tp_price * fee_rate
                    s_closed_prev = True
                else:
                    s_pnl = op - cl
                    s_exit_fee = 0.0
                    s_closed_prev = False
                
                s_net = s_pnl - s_entry_fee - s_exit_fee
                net_pnl = l_net + s_net
                
                sum_returns += net_pnl
                sum_returns_sq += (net_pnl * net_pnl)
                count += 1
            
            if count > 50:
                mean = sum_returns / count
                var = (sum_returns_sq / count) - (mean * mean)
                if var > 1e-9:
                    std = np.sqrt(var)
                    sharpe = (mean / std) * 93.59
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_tp = tp_pct
                        best_sl = sl_pct
                        
    return best_tp, best_sl, best_sharpe

def calculate_strategy_hourly_smart_fees(data, tp_above, sl_above, tp_below, sl_below):
    n = len(data)
    timestamps = data['timestamp'].values
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    smas = data['sma'].values
    
    net_equity = np.zeros(n)
    trades = []
    current_cum_pnl = 0.0
    fee_rate = 0.0002
    
    l_closed_prev = True
    s_closed_prev = True
    
    tp_mult_above_l = 1 + tp_above / 100.0
    sl_mult_above_l = 1 - sl_above / 100.0
    tp_mult_above_s = 1 - tp_above / 100.0
    sl_mult_above_s = 1 + sl_above / 100.0
    
    tp_mult_below_l = 1 + tp_below / 100.0
    sl_mult_below_l = 1 - sl_below / 100.0
    tp_mult_below_s = 1 - tp_below / 100.0
    sl_mult_below_s = 1 + sl_below / 100.0
    
    for i in range(n):
        ts = timestamps[i]
        op = opens[i]
        hi = highs[i]
        lo = lows[i]
        cl = closes[i]
        sma = smas[i]
        
        is_above = op > sma
        
        if is_above:
            l_tp_m, l_sl_m = tp_mult_above_l, sl_mult_above_l
            s_tp_m, s_sl_m = tp_mult_above_s, sl_mult_above_s
        else:
            l_tp_m, l_sl_m = tp_mult_below_l, sl_mult_below_l
            s_tp_m, s_sl_m = tp_mult_below_s, sl_mult_below_s
            
        # --- LONG ---
        l_entry_fee = op * fee_rate if l_closed_prev else 0.0
        l_sl_price = op * l_sl_m
        l_tp_price = op * l_tp_m
        
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
        
        # --- SHORT ---
        s_entry_fee = op * fee_rate if s_closed_prev else 0.0
        s_sl_price = op * s_sl_m
        s_tp_price = op * s_tp_m
        
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
                'regime': 'Above' if is_above else 'Below',
                'l_status': l_status,
                's_status': s_status,
                'pnl': net_pnl
            })
            
    return net_equity, trades

@app.route('/')
def index():
    global df
    try:
        sl_above = float(request.args.get('sl_above', 2.0))
        tp_above = float(request.args.get('tp_above', 5.0))
        sl_below = float(request.args.get('sl_below', 2.0))
        tp_below = float(request.args.get('tp_below', 5.0))
    except:
        sl_above, tp_above = 2.0, 5.0
        sl_below, tp_below = 2.0, 5.0
        
    if df is None:
        return "Data not loaded", 500

    calc_data = df.dropna(subset=['sma']).reset_index(drop=True)
    net_equity, trades = calculate_strategy_hourly_smart_fees(calc_data, tp_above, sl_above, tp_below, sl_below)
    
    rows = ""
    for t in reversed(trades[-50:]):
        color = "#00ff00" if t['pnl'] > 0 else "#ff0000"
        rows += f"""
        <tr>
            <td>{t['entry_time']}</td>
            <td>{t['regime']}</td>
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
                    document.getElementById('optimize-status').innerText = "Optimizing Training Data (<2026)...";
                    fetch('/optimize_split')
                        .then(response => response.json())
                        .then(data => {{
                            document.getElementById('optimize-status').innerText = "Done! Above: " + data.tp_above + "/" + data.sl_above + " | Below: " + data.tp_below + "/" + data.sl_below;
                            document.querySelector('input[name="tp_above"]').value = data.tp_above;
                            document.querySelector('input[name="sl_above"]').value = data.sl_above;
                            document.querySelector('input[name="tp_below"]').value = data.tp_below;
                            document.querySelector('input[name="sl_below"]').value = data.sl_below;
                        }})
                        .catch(err => {{
                            document.getElementById('optimize-status').innerText = "Error: " + err;
                        }});
                }}
            </script>
        </head>
        <body>
            <div class="container">
                <h2>ETH/USDT Walk-Forward Analysis</h2>
                <p>Optimization Train: 2021-2025. Unseen Test: 2026-Present.</p>
                
                <form action="/" method="get">
                    <div class="params-box">
                        <div class="params-row">
                            <strong>Above SMA (>365):</strong>
                            SL % <input type="number" step="0.1" name="sl_above" value="{sl_above}">
                            TP % <input type="number" step="0.1" name="tp_above" value="{tp_above}">
                        </div>
                        <div class="params-row">
                            <strong>Below SMA (<365):</strong>
                            SL % <input type="number" step="0.1" name="sl_below" value="{sl_below}">
                            TP % <input type="number" step="0.1" name="tp_below" value="{tp_below}">
                        </div>
                        <div style="text-align: center; margin-top: 10px;">
                            <input type="submit" value="Update Plots" style="width: auto; cursor: pointer;">
                        </div>
                    </div>
                </form>
                
                <button onclick="runOptimization()">Run Optimization (Train Set Only)</button>
                <div id="optimize-status"></div>
                
                <br>
                <img src="/plot.png?sl_above={sl_above}&tp_above={tp_above}&sl_below={sl_below}&tp_below={tp_below}" style="border: 1px solid #555; max-width: 100%; margin-top: 20px;">
                
                <h3>Triggered Trade Events</h3>
                <div class="scroll-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Entry Time</th>
                                <th>Regime</th>
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

@app.route('/optimize_split')
def optimize_split():
    global market_data, df
    
    # CUTOFF DATE for Training
    cutoff_ts = pd.Timestamp('2026-01-01')
    
    if 'opens' not in market_data:
        calc_data = df.dropna(subset=['sma']).reset_index(drop=True)
        # Store full arrays
        market_data['timestamps'] = calc_data['timestamp'].values
        market_data['opens'] = calc_data['open'].values.astype(np.float64)
        market_data['highs'] = calc_data['high'].values.astype(np.float64)
        market_data['lows'] = calc_data['low'].values.astype(np.float64)
        market_data['closes'] = calc_data['close'].values.astype(np.float64)
        market_data['smas'] = calc_data['sma'].values.astype(np.float64)
    
    # Create Boolean Mask for Training Data
    train_mask = market_data['timestamps'] < np.datetime64(cutoff_ts)
    
    # Slice arrays for optimizer
    opens = market_data['opens'][train_mask]
    highs = market_data['highs'][train_mask]
    lows = market_data['lows'][train_mask]
    closes = market_data['closes'][train_mask]
    smas = market_data['smas'][train_mask]
    
    t0 = time.time()
    tp_a, sl_a, sharpe_a = optimize_regime_sharpe(opens, highs, lows, closes, smas, True)
    tp_b, sl_b, sharpe_b = optimize_regime_sharpe(opens, highs, lows, closes, smas, False)
    
    print(f"Training Set Optimization done in {time.time() - t0:.2f}s")
    
    return jsonify({
        'tp_above': round(tp_a, 1),
        'sl_above': round(sl_a, 1),
        'sharpe_above': sharpe_a,
        'tp_below': round(tp_b, 1),
        'sl_below': round(sl_b, 1),
        'sharpe_below': sharpe_b
    })

@app.route('/plot.png')
def plot_png():
    global df
    try:
        sl_a = float(request.args.get('sl_above', 2.0))
        tp_a = float(request.args.get('tp_above', 5.0))
        sl_b = float(request.args.get('sl_below', 2.0))
        tp_b = float(request.args.get('tp_below', 5.0))
    except:
        sl_a, tp_a, sl_b, tp_b = 2.0, 5.0, 2.0, 5.0

    plot_data = df.dropna(subset=['sma']).copy()
    plot_data = plot_data.reset_index(drop=True)
    
    if plot_data.empty:
        return "Not enough data", 400
    
    net_equity, _ = calculate_strategy_hourly_smart_fees(plot_data, tp_a, sl_a, tp_b, sl_b)
    plot_data['net_equity'] = net_equity
    
    # Prepare Unseen Data Plot (2026+)
    cutoff_ts = pd.Timestamp('2026-01-01')
    unseen_data = plot_data[plot_data['timestamp'] >= cutoff_ts].copy()
    
    # Rebase Unseen Equity to 0
    if not unseen_data.empty:
        base_val = unseen_data['net_equity'].iloc[0]
        unseen_data['net_equity'] = unseen_data['net_equity'] - base_val
    
    # Plotting (3 Subplots)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), dpi=100, gridspec_kw={'height_ratios': [2, 1, 1]}, sharex=False)
    
    # 1. Price
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
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    # Cutoff Line
    ax1.axvline(cutoff_ts, color='yellow', linestyle='--', linewidth=2, label='Training Cutoff')
    ax1.set_xlim(plot_data['timestamp'].min(), plot_data['timestamp'].max())
    
    # 2. Full Equity
    ax2.plot(plot_data['timestamp'], plot_data['net_equity'], color='cyan', linewidth=1.5, label='Full History Equity')
    ax2.set_title('Full History PnL (Training + Unseen)')
    ax2.set_ylabel('PnL (USDT)')
    ax2.axvline(cutoff_ts, color='yellow', linestyle='--', linewidth=2)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(plot_data['timestamp'].min(), plot_data['timestamp'].max())
    
    # 3. Unseen Data Zoom
    if not unseen_data.empty:
        ax3.plot(unseen_data['timestamp'], unseen_data['net_equity'], color='magenta', linewidth=1.5, label='Unseen Data PnL')
        ax3.set_title('Unseen Data PnL (2026-Present) - Rebased to 0')
        ax3.set_ylabel('PnL (USDT)')
        ax3.grid(True, alpha=0.2)
        ax3.axhline(0, color='white', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No Unseen Data Available (>2026)", ha='center', va='center', color='white')
    
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
