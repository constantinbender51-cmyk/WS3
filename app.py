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

# Try to import Numba for performance
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: Numba not installed. Grid search will be slow.")
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(n):
        return range(n)

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
def fast_backtest_sharpe(opens, highs, lows, closes, tp_pct, sl_pct):
    """
    Optimized core logic for Grid Search targeting Sharpe Ratio.
    Includes 0.02% fee per side.
    Returns: Sharpe Ratio (Mean / Std of hourly PnL)
    """
    n = len(opens)
    fee_rate = 0.0002
    
    # Multipliers
    l_tp_mult = 1.0 + tp_pct / 100.0
    l_sl_mult = 1.0 - sl_pct / 100.0
    s_tp_mult = 1.0 - tp_pct / 100.0
    s_sl_mult = 1.0 + sl_pct / 100.0
    
    # State
    l_active = False
    l_entry = 0.0
    l_realized = 0.0
    
    s_active = False
    s_entry = 0.0
    s_realized = 0.0
    
    # Statistics trackers for Sharpe (Welford's algorithm or simple sum of squares)
    # Using simple sum of x and sum of x^2 for variance
    sum_pnl = 0.0
    sum_pnl_sq = 0.0
    
    for i in range(n):
        op = opens[i]
        hi = highs[i]
        lo = lows[i]
        cl = closes[i]
        
        current_hour_pnl = 0.0
        
        # --- LONG ---
        if not l_active:
            l_entry = op
            # Pay Entry Fee
            l_realized -= (l_entry * fee_rate)
            current_hour_pnl -= (l_entry * fee_rate)
            l_active = True
            
        l_tp = l_entry * l_tp_mult
        l_sl = l_entry * l_sl_mult
        
        long_leg_pnl = 0.0
        
        if lo <= l_sl:
            # Exit at SL
            trade_pnl = (l_sl - l_entry)
            fee = l_sl * fee_rate
            net_change = trade_pnl - fee
            
            # Correction: We already deducted entry fee from realized. 
            # We need to add the diff between old realized and new realized to current_hour_pnl
            
            # Easier approach: calculate total equity change this hour
            prev_l_val = l_realized + (cl - l_entry) # Approx prev mark (wrong for intraday)
            # Strict logic:
            # PnL this hour = (Realized_End + Unrealized_End) - (Realized_Start + Unrealized_Start)
            
            l_realized += (trade_pnl - fee)
            l_active = False
        elif hi >= l_tp:
            # Exit at TP
            trade_pnl = (l_tp - l_entry)
            fee = l_tp * fee_rate
            l_realized += (trade_pnl - fee)
            l_active = False
            
        # --- SHORT ---
        if not s_active:
            s_entry = op
            s_realized -= (s_entry * fee_rate)
            s_active = True
            
        s_tp = s_entry * s_tp_mult
        s_sl = s_entry * s_sl_mult
        
        if hi >= s_sl:
            trade_pnl = (s_entry - s_sl)
            fee = s_sl * fee_rate
            s_realized += (trade_pnl - fee)
            s_active = False
        elif lo <= s_tp:
            trade_pnl = (s_entry - s_tp)
            fee = s_tp * fee_rate
            s_realized += (trade_pnl - fee)
            s_active = False
            
        # Calculate Total Floating Equity at end of hour to derive hourly return
        # Note: This is computationally expensive to get perfect, simplifying:
        # We track cumulative realized + current floating
        
        l_float = (cl - l_entry) if l_active else 0.0
        s_float = (s_entry - cl) if s_active else 0.0
        
        total_equity = l_realized + s_realized + l_float + s_float
        
        # To get Sharpe, we need the delta (PnL) of this hour
        # Store previous equity to get delta
        if i == 0:
            hourly_pnl = total_equity # Assume start at 0
        else:
            # We need to store prev_equity in the loop
            # To avoid array access, let's restructure variables slightly
            pass 

    # --- REWRITE LOOP FOR PRECISE RETURN TRACKING ---
    return 0.0 # Placeholder for logic below

@njit(fastmath=True)
def fast_backtest_sharpe_v2(opens, highs, lows, closes, tp_pct, sl_pct):
    n = len(opens)
    fee_rate = 0.0002
    
    l_tp_mult = 1.0 + tp_pct / 100.0
    l_sl_mult = 1.0 - sl_pct / 100.0
    s_tp_mult = 1.0 - tp_pct / 100.0
    s_sl_mult = 1.0 + sl_pct / 100.0
    
    l_active = False
    l_entry = 0.0
    l_cash = 0.0 # Realized PnL accumulator
    
    s_active = False
    s_entry = 0.0
    s_cash = 0.0
    
    prev_total_equity = 0.0
    
    sum_returns = 0.0
    sum_returns_sq = 0.0
    
    for i in range(n):
        op = opens[i]
        hi = highs[i]
        lo = lows[i]
        cl = closes[i]
        
        # --- LONG LOGIC ---
        if not l_active:
            l_entry = op
            l_cash -= (l_entry * fee_rate)
            l_active = True
            
        l_tp = l_entry * l_tp_mult
        l_sl = l_entry * l_sl_mult
        
        # Check Exits
        if lo <= l_sl:
            l_cash += (l_sl - l_entry) - (l_sl * fee_rate)
            l_active = False
            l_float = 0.0
        elif hi >= l_tp:
            l_cash += (l_tp - l_entry) - (l_tp * fee_rate)
            l_active = False
            l_float = 0.0
        else:
            l_float = cl - l_entry
            
        # --- SHORT LOGIC ---
        if not s_active:
            s_entry = op
            s_cash -= (s_entry * fee_rate)
            s_active = True
            
        s_tp = s_entry * s_tp_mult
        s_sl = s_entry * s_sl_mult
        
        if hi >= s_sl:
            s_cash += (s_entry - s_sl) - (s_sl * fee_rate)
            s_active = False
            s_float = 0.0
        elif lo <= s_tp:
            s_cash += (s_entry - s_tp) - (s_tp * fee_rate)
            s_active = False
            s_float = 0.0
        else:
            s_float = s_entry - cl
            
        # --- HOURLY STATS ---
        current_total_equity = l_cash + l_float + s_cash + s_float
        
        if i == 0:
            hourly_pnl = current_total_equity
        else:
            hourly_pnl = current_total_equity - prev_total_equity
            
        prev_total_equity = current_total_equity
        
        # Accumulate
        sum_returns += hourly_pnl
        sum_returns_sq += (hourly_pnl * hourly_pnl)
        
    # Finalize Sharpe
    # Mean = Sum / N
    # Var = (SumSq / N) - Mean^2
    # Std = Sqrt(Var)
    # Sharpe = Mean / Std
    
    mean = sum_returns / n
    var = (sum_returns_sq / n) - (mean * mean)
    
    if var <= 0.0000001:
        return 0.0
        
    std = np.sqrt(var)
    
    # Return Annualized Sharpe (approximate, assuming hourly)
    # Annualized = Hourly Sharpe * Sqrt(24*365)
    return (mean / std) * 93.59 # 93.59 is approx sqrt(8760)

def calculate_strategy_details(data, tp_pct, sl_pct):
    n = len(data)
    timestamps = data['timestamp'].values
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    
    long_equity = np.zeros(n)
    short_equity = np.zeros(n)
    trades = []
    
    fee_rate = 0.0002
    
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
        
        # LONG
        if not l_active:
            l_entry_price = op
            l_entry_idx = i
            l_active = True
            l_realized -= (l_entry_price * fee_rate) # Entry Fee
            
        l_tp = l_entry_price * l_tp_mult
        l_sl = l_entry_price * l_sl_mult
        
        l_pnl = 0.0
        l_status = ""
        l_exit_price = 0.0
        
        if lo <= l_sl:
            gross_pnl = l_sl - l_entry_price
            exit_fee = l_sl * fee_rate
            l_pnl = gross_pnl - exit_fee
            l_realized += (gross_pnl - exit_fee)
            l_active = False
            l_status = "SL"
            l_exit_price = l_sl
        elif hi >= l_tp:
            gross_pnl = l_tp - l_entry_price
            exit_fee = l_tp * fee_rate
            l_pnl = gross_pnl - exit_fee
            l_realized += (gross_pnl - exit_fee)
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
                'pnl': l_pnl - (l_entry_price * fee_rate), # Include entry fee in trade PnL report
                'status': l_status
            })
            long_equity[i] = l_realized
        else:
            long_equity[i] = l_realized + (cl - l_entry_price)

        # SHORT
        if not s_active:
            s_entry_price = op
            s_entry_idx = i
            s_active = True
            s_realized -= (s_entry_price * fee_rate)
            
        s_tp = s_entry_price * s_tp_mult
        s_sl = s_entry_price * s_sl_mult
        
        s_pnl = 0.0
        s_status = ""
        s_exit_price = 0.0
        
        if hi >= s_sl:
            gross_pnl = s_entry_price - s_sl
            exit_fee = s_sl * fee_rate
            s_pnl = gross_pnl - exit_fee
            s_realized += (gross_pnl - exit_fee)
            s_active = False
            s_status = "SL"
            s_exit_price = s_sl
        elif lo <= s_tp:
            gross_pnl = s_entry_price - s_tp
            exit_fee = s_tp * fee_rate
            s_pnl = gross_pnl - exit_fee
            s_realized += (gross_pnl - exit_fee)
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
        
    if df is None:
        return "Data not loaded.", 500

    calc_data = df.dropna(subset=['sma']).reset_index(drop=True)
    _, _, trades = calculate_strategy_details(calc_data, tp, sl)
    
    rows = ""
    for t in reversed(trades[-50:]):
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
                input, button {{ background: #333; color: white; border: 1px solid #555; padding: 5px; }}
                button {{ cursor: pointer; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 12px; }}
                th {{ background: #444; padding: 10px; text-align: left; position: sticky; top: 0; }}
                td {{ border-bottom: 1px solid #333; padding: 5px; text-align: left; }}
                .scroll-table {{ max_height: 500px; overflow-y: auto; border: 1px solid #444; margin-top: 20px; }}
                #optimize-status {{ margin-top: 10px; color: yellow; }}
            </style>
            <script>
                function runOptimization() {{
                    document.getElementById('optimize-status').innerText = "Running Sharpe Optimization (0-10%)... Please wait.";
                    fetch('/optimize_grid')
                        .then(response => response.json())
                        .then(data => {{
                            document.getElementById('optimize-status').innerText = 
                                "Optimization Complete! Best TP: " + data.best_tp + "%, SL: " + data.best_sl + "% (Sharpe: " + data.max_sharpe.toFixed(2) + ")";
                            document.querySelector('input[name="tp"]').value = data.best_tp;
                            document.querySelector('input[name="sl"]').value = data.best_sl;
                        }})
                        .catch(err => {{
                            document.getElementById('optimize-status').innerText = "Error: " + err;
                        }});
                }}
            </script>
        </head>
        <body>
            <div class="container">
                <h2>ETH/USDT Continuous Hedge Strategy (0.02% Fee)</h2>
                <div style="background: #333; padding: 15px; border-radius: 5px; display: inline-block;">
                    <form action="/" method="get" style="display: inline;">
                        <label>Stop Loss (%): <input type="number" step="0.1" name="sl" value="{sl}"></label>
                        <label style="margin-left: 20px;">Take Profit (%): <input type="number" step="0.1" name="tp" value="{tp}"></label>
                        <input type="submit" value="Update Chart" style="margin-left: 10px;">
                    </form>
                    <button onclick="runOptimization()" style="margin-left: 20px; background: #554400;">Optimize Sharpe</button>
                </div>
                <div id="optimize-status"></div>
                
                <br>
                <img src="/plot.png?sl={sl}&tp={tp}" style="border: 1px solid #555; max-width: 100%; margin-top: 20px;">
                
                <h3>Recent Trade History</h3>
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
                                <th>PnL (w/ Fee)</th>
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

@app.route('/optimize_grid')
def optimize_grid():
    global market_data, df
    
    if 'opens' not in market_data:
        calc_data = df.dropna(subset=['sma']).reset_index(drop=True)
        market_data['opens'] = calc_data['open'].values.astype(np.float64)
        market_data['highs'] = calc_data['high'].values.astype(np.float64)
        market_data['lows'] = calc_data['low'].values.astype(np.float64)
        market_data['closes'] = calc_data['close'].values.astype(np.float64)
    
    opens = market_data['opens']
    highs = market_data['highs']
    lows = market_data['lows']
    closes = market_data['closes']
    
    best_sharpe = -np.inf
    best_tp = 0.0
    best_sl = 0.0
    
    t0 = time.time()
    
    tp_range = np.arange(0.1, 10.1, 0.1)
    sl_range = np.arange(0.1, 10.1, 0.1)
    
    for tp in tp_range:
        for sl in sl_range:
            sharpe = fast_backtest_sharpe_v2(opens, highs, lows, closes, tp, sl)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_tp = tp
                best_sl = sl
                
    print(f"Optimization finished in {time.time() - t0:.2f}s. Best: TP {best_tp:.1f}, SL {best_sl:.1f}, Sharpe {best_sharpe:.2f}")
    
    return jsonify({
        'best_tp': round(best_tp, 1),
        'best_sl': round(best_sl, 1),
        'max_sharpe': best_sharpe
    })

@app.route('/plot.png')
def plot_png():
    global df
    try:
        sl_pct = float(request.args.get('sl', 20))
        tp_pct = float(request.args.get('tp', 50))
    except ValueError:
        sl_pct = 20
        tp_pct = 50

    plot_data = df.dropna(subset=['sma']).copy()
    plot_data = plot_data.reset_index(drop=True)
    
    if plot_data.empty:
        return "Not enough data", 400
    
    long_eq, short_eq, _ = calculate_strategy_details(plot_data, tp_pct, sl_pct)
    
    plot_data['long_pnl'] = long_eq
    plot_data['short_pnl'] = short_eq
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
    
    ax2.plot(plot_data['timestamp'], plot_data['long_pnl'], color='green', alpha=0.3, label='Long Leg PnL')
    ax2.plot(plot_data['timestamp'], plot_data['short_pnl'], color='red', alpha=0.3, label='Short Leg PnL')
    ax2.plot(plot_data['timestamp'], plot_data['net_pnl'], color='cyan', linewidth=2, label='Net Equity')
    ax2.set_title(f'Equity (SL: {sl_pct}%, TP: {tp_pct}%, Fee: 0.02%)')
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
