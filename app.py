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

# Check for Numba
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
    
    # Pre-calculate 365 SMA
    data['sma'] = data['close'].rolling(window=365*24).mean()
    
    return data

@njit(fastmath=True)
def optimize_regime_sharpe(opens, highs, lows, closes, smas, target_above_sma):
    """
    Optimizes TP/SL for a specific regime (Above or Below SMA).
    Returns: Best TP, Best SL, Best Sharpe for that regime.
    """
    n = len(opens)
    fee_rate = 0.0002
    
    best_sharpe = -np.inf
    best_tp = 20.0
    best_sl = 20.0
    
    # Grid Search Range
    # Using slightly coarser steps or limited range if speed is an issue, 
    # but 0.1-10.0 is requested.
    
    # We loop manually through the grid
    # 0.1 to 10.0 = 100 steps. 100x100 = 10,000 iters.
    
    for tp_int in range(1, 101):
        for sl_int in range(1, 101):
            tp_pct = tp_int / 10.0
            sl_pct = sl_int / 10.0
            
            l_tp_mult = 1.0 + tp_pct / 100.0
            l_sl_mult = 1.0 - sl_pct / 100.0
            s_tp_mult = 1.0 - tp_pct / 100.0
            s_sl_mult = 1.0 + sl_pct / 100.0
            
            # Simulation State
            l_active = False
            l_entry = 0.0
            
            s_active = False
            s_entry = 0.0
            
            # Return Stats
            sum_returns = 0.0
            sum_returns_sq = 0.0
            count = 0
            
            # Running Equity for Delta Calc
            l_cash = 0.0
            s_cash = 0.0
            prev_equity = 0.0
            
            for i in range(n):
                op = opens[i]
                hi = highs[i]
                lo = lows[i]
                cl = closes[i]
                sma = smas[i]
                
                # Determine Regime of the current moment (using Open vs SMA or strict logic)
                # We only count stats for the regime we are optimizing.
                # However, we must SIMULATE the whole path to keep equity consistent, 
                # but only accumulate Sharpe stats if the trade *started* in the regime?
                # Or if the current candle is in the regime?
                # "Optimize Above and Below Separately" implies finding parameters that perform best
                # when the condition is met.
                
                # Logic: We simulate the strategy using these params ONLY when the condition is met.
                # If condition not met, we assume flat (0 return) for the optimization metric of this specific regime.
                
                is_above = op > sma
                
                # Check if this candle belongs to the target regime
                in_regime = (is_above == target_above_sma)
                
                # Trade Logic (Simplified for speed: Always in market if in regime?)
                # The strategy is "Continuous Hedge".
                # If we are optimizing "Above", we assume we are using these params when Above.
                # We ignore what happens when Below (assume 0 returns for Sharpe calc of this component).
                
                # --- LONG ---
                if not l_active:
                    l_entry = op
                    l_cash -= (l_entry * fee_rate)
                    l_active = True
                    
                l_tp = l_entry * l_tp_mult
                l_sl = l_entry * l_sl_mult
                
                l_float = 0.0
                if lo <= l_sl:
                    l_cash += (l_sl - l_entry) - (l_sl * fee_rate)
                    l_active = False
                elif hi >= l_tp:
                    l_cash += (l_tp - l_entry) - (l_tp * fee_rate)
                    l_active = False
                else:
                    l_float = cl - l_entry
                    
                # --- SHORT ---
                if not s_active:
                    s_entry = op
                    s_cash -= (s_entry * fee_rate)
                    s_active = True
                
                s_tp = s_entry * s_tp_mult
                s_sl = s_entry * s_sl_mult
                
                s_float = 0.0
                if hi >= s_sl:
                    s_cash += (s_entry - s_sl) - (s_sl * fee_rate)
                    s_active = False
                elif lo <= s_tp:
                    s_cash += (s_entry - s_tp) - (s_tp * fee_rate)
                    s_active = False
                else:
                    s_float = s_entry - cl
                    
                # --- Stats Accumulation ---
                current_equity = l_cash + l_float + s_cash + s_float
                
                if i == 0:
                    delta = current_equity
                else:
                    delta = current_equity - prev_equity
                
                prev_equity = current_equity
                
                # Only accumulate Sharpe stats if we are in the target regime
                if in_regime:
                    sum_returns += delta
                    sum_returns_sq += (delta * delta)
                    count += 1
            
            # Calculate Sharpe for this regime
            if count > 100: # Min samples
                mean = sum_returns / count
                var = (sum_returns_sq / count) - (mean * mean)
                if var > 0:
                    std = np.sqrt(var)
                    # Annualize roughly
                    sharpe = (mean / std) * 93.59
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_tp = tp_pct
                        best_sl = sl_pct
                        
    return best_tp, best_sl, best_sharpe

def calculate_strategy_split(data, tp_above, sl_above, tp_below, sl_below):
    n = len(data)
    timestamps = data['timestamp'].values
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    smas = data['sma'].values
    
    long_equity = np.zeros(n)
    short_equity = np.zeros(n)
    trades = []
    
    fee_rate = 0.0002
    
    l_active = False
    l_entry_price = 0.0
    l_entry_idx = 0
    l_realized = 0.0
    # Track which params were used for the active trade
    l_current_tp_mult = 0.0
    l_current_sl_mult = 0.0
    
    s_active = False
    s_entry_price = 0.0
    s_entry_idx = 0
    s_realized = 0.0
    s_current_tp_mult = 0.0
    s_current_sl_mult = 0.0
    
    # Pre-calc multipliers
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
        
        # Determine params for NEW trades
        if is_above:
            cur_l_tp_m = tp_mult_above_l
            cur_l_sl_m = sl_mult_above_l
            cur_s_tp_m = tp_mult_above_s
            cur_s_sl_m = sl_mult_above_s
        else:
            cur_l_tp_m = tp_mult_below_l
            cur_l_sl_m = sl_mult_below_l
            cur_s_tp_m = tp_mult_below_s
            cur_s_sl_m = sl_mult_below_s
            
        # --- LONG ---
        if not l_active:
            l_entry_price = op
            l_entry_idx = i
            l_active = True
            l_realized -= (l_entry_price * fee_rate)
            # Lock in params at entry
            l_current_tp_mult = cur_l_tp_m
            l_current_sl_mult = cur_l_sl_m
            
        l_tp = l_entry_price * l_current_tp_mult
        l_sl = l_entry_price * l_current_sl_mult
        
        l_pnl = 0.0
        l_status = ""
        l_exit_price = 0.0
        
        if lo <= l_sl:
            gross = l_sl - l_entry_price
            fee = l_sl * fee_rate
            l_pnl = gross - fee
            l_realized += l_pnl
            l_active = False
            l_status = "SL"
            l_exit_price = l_sl
        elif hi >= l_tp:
            gross = l_tp - l_entry_price
            fee = l_tp * fee_rate
            l_pnl = gross - fee
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
                'pnl': l_pnl - (l_entry_price * fee_rate),
                'status': l_status,
                'regime': 'Above' if smas[l_entry_idx] < opens[l_entry_idx] else 'Below'
            })
            long_equity[i] = l_realized
        else:
            long_equity[i] = l_realized + (cl - l_entry_price)

        # --- SHORT ---
        if not s_active:
            s_entry_price = op
            s_entry_idx = i
            s_active = True
            s_realized -= (s_entry_price * fee_rate)
            s_current_tp_mult = cur_s_tp_m
            s_current_sl_mult = cur_s_sl_m
            
        s_tp = s_entry_price * s_current_tp_mult
        s_sl = s_entry_price * s_current_sl_mult
        
        s_pnl = 0.0
        s_status = ""
        s_exit_price = 0.0
        
        if hi >= s_sl:
            gross = s_entry_price - s_sl
            fee = s_sl * fee_rate
            s_pnl = gross - fee
            s_realized += s_pnl
            s_active = False
            s_status = "SL"
            s_exit_price = s_sl
        elif lo <= s_tp:
            gross = s_entry_price - s_tp
            fee = s_tp * fee_rate
            s_pnl = gross - fee
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
                'pnl': s_pnl - (s_entry_price * fee_rate),
                'status': s_status,
                'regime': 'Above' if smas[s_entry_idx] < opens[s_entry_idx] else 'Below'
            })
            short_equity[i] = s_realized
        else:
            short_equity[i] = s_realized + (s_entry_price - cl)
            
    return long_equity, short_equity, trades

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
    _, _, trades = calculate_strategy_split(calc_data, tp_above, sl_above, tp_below, sl_below)
    
    rows = ""
    for t in reversed(trades[-50:]):
        color = "#00ff00" if t['pnl'] > 0 else "#ff0000"
        rows += f"""
        <tr>
            <td>{t['entry_time']}</td>
            <td>{t['exit_time']}</td>
            <td>{t['regime']}</td>
            <td style="color: {'cyan' if t['type']=='LONG' else 'orange'}">{t['type']}</td>
            <td>{t['entry_price']:.2f}</td>
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
                    document.getElementById('optimize-status').innerText = "Optimizing Separate Regimes... Please wait.";
                    fetch('/optimize_split')
                        .then(response => response.json())
                        .then(data => {{
                            document.getElementById('optimize-status').innerText = "Done! Above: TP " + data.tp_above + "/SL " + data.sl_above + " | Below: TP " + data.tp_below + "/SL " + data.sl_below;
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
                <h2>ETH/USDT Split Regime Strategy (0.02% Fee)</h2>
                
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
                            <input type="submit" value="Update" style="width: auto; cursor: pointer;">
                        </div>
                    </div>
                </form>
                
                <button onclick="runOptimization()">Run Split Optimization</button>
                <div id="optimize-status"></div>
                
                <br>
                <img src="/plot.png?sl_above={sl_above}&tp_above={tp_above}&sl_below={sl_below}&tp_below={tp_below}" style="border: 1px solid #555; max-width: 100%; margin-top: 20px;">
                
                <h3>Recent Trade History</h3>
                <div class="scroll-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Entry Time</th>
                                <th>Exit Time</th>
                                <th>Regime</th>
                                <th>Type</th>
                                <th>Entry</th>
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

@app.route('/optimize_split')
def optimize_split():
    global market_data, df
    
    if 'opens' not in market_data:
        calc_data = df.dropna(subset=['sma']).reset_index(drop=True)
        market_data['opens'] = calc_data['open'].values.astype(np.float64)
        market_data['highs'] = calc_data['high'].values.astype(np.float64)
        market_data['lows'] = calc_data['low'].values.astype(np.float64)
        market_data['closes'] = calc_data['close'].values.astype(np.float64)
        market_data['smas'] = calc_data['sma'].values.astype(np.float64)
    
    opens = market_data['opens']
    highs = market_data['highs']
    lows = market_data['lows']
    closes = market_data['closes']
    smas = market_data['smas']
    
    t0 = time.time()
    
    # Optimize Above Regime
    tp_a, sl_a, sharpe_a = optimize_regime_sharpe(opens, highs, lows, closes, smas, True)
    
    # Optimize Below Regime
    tp_b, sl_b, sharpe_b = optimize_regime_sharpe(opens, highs, lows, closes, smas, False)
    
    print(f"Split Optimization done in {time.time() - t0:.2f}s")
    
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
    
    long_eq, short_eq, _ = calculate_strategy_split(plot_data, tp_a, sl_a, tp_b, sl_b)
    
    plot_data['net_pnl'] = long_eq + short_eq
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), dpi=100, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
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
    
    ax2.plot(plot_data['timestamp'], plot_data['net_pnl'], color='cyan', linewidth=2, label='Net Equity')
    ax2.set_title(f'Split Regime Equity')
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
