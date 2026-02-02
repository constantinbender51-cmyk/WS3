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
    Optimizes for the new strategy:
    1. Open Long + Short every hour at Open.
    2. Close at SL, TP, or Hourly Close (whichever comes first).
    """
    n = len(opens)
    fee_rate = 0.0002
    
    best_sharpe = -np.inf
    best_tp = 5.0
    best_sl = 2.0
    
    # Search Range: 0.2% to 10.0%
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
            
            # Loop strictly for stats calculation
            for i in range(n):
                op = opens[i]
                sma = smas[i]
                
                # Check Regime
                is_above = op > sma
                if is_above != target_above_sma:
                    continue
                
                hi = highs[i]
                lo = lows[i]
                cl = closes[i]
                
                # --- Hourly Reset Logic ---
                
                # LONG Leg
                l_entry = op
                l_pnl = 0.0
                # Calc Exits
                l_sl_price = l_entry * l_sl_mult
                l_tp_price = l_entry * l_tp_mult
                
                # Logic: Did we hit SL or TP?
                # Assumption: If Low <= SL, we hit SL. If High >= TP, we hit TP.
                # Conflict: If both hit? Standard conservative: SL hit first.
                if lo <= l_sl_price:
                    l_exit = l_sl_price
                    l_pnl = (l_exit - l_entry) - (l_entry * fee_rate) - (l_exit * fee_rate)
                elif hi >= l_tp_price:
                    l_exit = l_tp_price
                    l_pnl = (l_exit - l_entry) - (l_entry * fee_rate) - (l_exit * fee_rate)
                else:
                    l_exit = cl
                    l_pnl = (l_exit - l_entry) - (l_entry * fee_rate) - (l_exit * fee_rate)
                    
                # SHORT Leg
                s_entry = op
                s_pnl = 0.0
                s_sl_price = s_entry * s_sl_mult
                s_tp_price = s_entry * s_tp_mult
                
                if hi >= s_sl_price:
                    s_exit = s_sl_price
                    s_pnl = (s_entry - s_exit) - (s_entry * fee_rate) - (s_exit * fee_rate)
                elif lo <= s_tp_price:
                    s_exit = s_tp_price
                    s_pnl = (s_entry - s_exit) - (s_entry * fee_rate) - (s_exit * fee_rate)
                else:
                    s_exit = cl
                    s_pnl = (s_entry - s_exit) - (s_entry * fee_rate) - (s_exit * fee_rate)
                    
                # Net PnL for this hour
                net_pnl = l_pnl + s_pnl
                
                sum_returns += net_pnl
                sum_returns_sq += (net_pnl * net_pnl)
                count += 1
            
            if count > 100:
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

def calculate_strategy_hourly(data, tp_above, sl_above, tp_below, sl_below):
    n = len(data)
    timestamps = data['timestamp'].values
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    smas = data['sma'].values
    
    # Cumulative PnL arrays for plotting
    net_equity = np.zeros(n)
    
    trades = []
    current_cum_pnl = 0.0
    fee_rate = 0.0002
    
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
        
        # Select Multipliers
        if is_above:
            l_tp_m, l_sl_m = tp_mult_above_l, sl_mult_above_l
            s_tp_m, s_sl_m = tp_mult_above_s, sl_mult_above_s
        else:
            l_tp_m, l_sl_m = tp_mult_below_l, sl_mult_below_l
            s_tp_m, s_sl_m = tp_mult_below_s, sl_mult_below_s
            
        # --- LONG CALC ---
        l_sl_price = op * l_sl_m
        l_tp_price = op * l_tp_m
        
        l_exit = 0.0
        l_status = ""
        
        if lo <= l_sl_price:
            l_exit = l_sl_price
            l_status = "SL"
        elif hi >= l_tp_price:
            l_exit = l_tp_price
            l_status = "TP"
        else:
            l_exit = cl
            l_status = "TIME" # Hourly Close
            
        l_pnl = (l_exit - op) - (op * fee_rate) - (l_exit * fee_rate)
        
        # --- SHORT CALC ---
        s_sl_price = op * s_sl_m
        s_tp_price = op * s_tp_m
        
        s_exit = 0.0
        s_status = ""
        
        if hi >= s_sl_price:
            s_exit = s_sl_price
            s_status = "SL"
        elif lo <= s_tp_price:
            s_exit = s_tp_price
            s_status = "TP"
        else:
            s_exit = cl
            s_status = "TIME"
            
        s_pnl = (op - s_exit) - (op * fee_rate) - (s_exit * fee_rate)
        
        # Net for this candle
        net_pnl = l_pnl + s_pnl
        current_cum_pnl += net_pnl
        net_equity[i] = current_cum_pnl
        
        # Log Interesting Trades (Only log if NOT Time/Time to reduce spam? 
        # Or log all? Let's log if at least one side triggered)
        if l_status != "TIME" or s_status != "TIME":
             trades.append({
                'entry_time': pd.to_datetime(ts),
                'exit_time': pd.to_datetime(ts) + pd.Timedelta(hours=1),
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
    net_equity, trades = calculate_strategy_hourly(calc_data, tp_above, sl_above, tp_below, sl_below)
    
    # Store equity in DF for plotting convenience
    # (Creating a temporary copy to avoid global state pollution)
    plot_df = calc_data.copy()
    plot_df['net_equity'] = net_equity
    
    rows = ""
    # Show last 50 triggered trades
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
                    document.getElementById('optimize-status').innerText = "Optimizing Hourly Reset Strategy... Please wait.";
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
                <h2>ETH/USDT Hourly Reset Strategy</h2>
                <p>Opens L+S every hour. Closes at TP, SL, or End of Hour.</p>
                
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
                
                <h3>Recent Triggered Events (Non-Time Exits)</h3>
                <div class="scroll-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Entry Time</th>
                                <th>Regime</th>
                                <th>Status (Long/Short)</th>
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
    tp_a, sl_a, sharpe_a = optimize_regime_sharpe(opens, highs, lows, closes, smas, True)
    tp_b, sl_b, sharpe_b = optimize_regime_sharpe(opens, highs, lows, closes, smas, False)
    
    print(f"Optimization done in {time.time() - t0:.2f}s")
    
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
    
    net_equity, _ = calculate_strategy_hourly(plot_data, tp_a, sl_a, tp_b, sl_b)
    
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
    
    ax2.plot(plot_data['timestamp'], net_equity, color='cyan', linewidth=2, label='Net Equity')
    ax2.set_title(f'Hourly Reset Equity (Cumulative PnL)')
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
