import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, send_file, request, render_template_string
import io
import numpy as np
import time
import threading

# Try importing Numba
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: Numba not installed. This will be extremely slow.")
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

app = Flask(__name__)

# --- CONFIG ---
ASSETS = [
    'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT', 'DOGE/USDT',
    'ADA/USDT', 'BCH/USDT', 'LINK/USDT', 'XLM/USDT', 'SUI/USDT',
    'AVAX/USDT', 'LTC/USDT', 'HBAR/USDT', 'SHIB/USDT', 'TON/USDT'
]
TIMEFRAMES = ['30m', '1h', '4h', '1d']
SPLIT_RATIO = 0.70  # 70% Train, 30% Test

# Global Stores
RAW_DATA_30M = {}   # Holds the raw 30m DF for each asset
TOP_5_RESULTS = []  # Stores the best optimized configs
IS_LOADING = True
LOADING_STATUS = "Initializing..."

# --- DATA FETCHING ---
def fetch_30m_history(symbol):
    """Fetches ~5 years of 30m data for a single symbol."""
    exchange = ccxt.binance({'enableRateLimit': True})
    limit = 1000
    now = exchange.milliseconds()
    since = now - (5 * 365 * 24 * 60 * 60 * 1000)
    all_ohlcv = []
    
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '30m', since, limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            # Move pointer forward (30m * 1000 candles)
            last_ts = ohlcv[-1][0]
            since = last_ts + 1800000 
        except Exception as e:
            print(f"Error {symbol}: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def resample_data(df_30m, target_tf):
    """Resamples 30m data to 1h, 4h, or 1d."""
    if target_tf == '30m':
        return df_30m.copy()
    
    # Map timeframe strings to pandas offsets
    tf_map = {'1h': '1h', '4h': '4h', '1d': '1D'}
    rule = tf_map.get(target_tf)
    
    if not rule:
        return df_30m
        
    df_res = df_30m.set_index('timestamp').resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    
    return df_res

# --- OPTIMIZER ENGINE ---
@njit(fastmath=True)
def run_grid_search(opens, highs, lows, closes):
    """
    Optimizes Single Strategy (Sharpe) on provided arrays.
    Returns: (best_tp, best_sl, best_sharpe)
    """
    n = len(opens)
    fee_rate = 0.0002
    
    best_sharpe = -np.inf
    best_tp = 5.0
    best_sl = 2.0
    
    # Grid 0.2% to 10.0%
    for tp_int in range(2, 101, 2):
        for sl_int in range(2, 101, 2):
            tp_pct = tp_int / 10.0
            sl_pct = sl_int / 10.0
            
            l_tp_mult = 1.0 + tp_pct/100.0
            l_sl_mult = 1.0 - sl_pct/100.0
            s_tp_mult = 1.0 - tp_pct/100.0
            s_sl_mult = 1.0 + sl_pct/100.0
            
            sum_ret = 0.0
            sum_ret_sq = 0.0
            
            l_closed = True
            s_closed = True
            
            for i in range(n):
                op = opens[i]
                hi = highs[i]
                lo = lows[i]
                cl = closes[i]
                
                # LONG
                l_entry_fee = op * fee_rate if l_closed else 0.0
                l_sl_price = op * l_sl_mult
                l_tp_price = op * l_tp_mult
                l_pnl = 0.0
                l_exit_fee = 0.0
                
                if lo <= l_sl_price:
                    l_pnl = l_sl_price - op
                    l_exit_fee = l_sl_price * fee_rate
                    l_closed = True
                elif hi >= l_tp_price:
                    l_pnl = l_tp_price - op
                    l_exit_fee = l_tp_price * fee_rate
                    l_closed = True
                else:
                    l_pnl = cl - op
                    l_exit_fee = 0.0
                    l_closed = False
                    
                # SHORT
                s_entry_fee = op * fee_rate if s_closed else 0.0
                s_sl_price = op * s_sl_mult
                s_tp_price = op * s_tp_mult
                s_pnl = 0.0
                s_exit_fee = 0.0
                
                if hi >= s_sl_price:
                    s_pnl = op - s_sl_price
                    s_exit_fee = s_sl_price * fee_rate
                    s_closed = True
                elif lo <= s_tp_price:
                    s_pnl = op - s_tp_price
                    s_exit_fee = s_tp_price * fee_rate
                    s_closed = True
                else:
                    s_pnl = op - cl
                    s_exit_fee = 0.0
                    s_closed = False
                    
                net = (l_pnl - l_entry_fee - l_exit_fee) + (s_pnl - s_entry_fee - s_exit_fee)
                sum_ret += net
                sum_ret_sq += net*net
                
            # Calc Sharpe
            if n > 50:
                mean = sum_ret / n
                var = (sum_ret_sq / n) - (mean*mean)
                if var > 1e-9:
                    std = np.sqrt(var)
                    sharpe = mean / std # Raw Sharpe per candle
                    # Store Raw Sharpe for ranking (Timeframe agnostic for raw comparison or annualize later)
                    # To be fair across timeframes, we should Annualize.
                    # However, numba function doesn't know timeframe.
                    # We will return raw sharpe * sqrt(candles_per_year) outside.
                    
                    if (mean / std) > best_sharpe:
                        best_sharpe = (mean / std)
                        best_tp = tp_pct
                        best_sl = sl_pct
                        
    return best_tp, best_sl, best_sharpe

# --- WORKER THREAD ---
def background_worker():
    global RAW_DATA_30M, TOP_5_RESULTS, IS_LOADING, LOADING_STATUS
    
    print("--- WORKER STARTED ---")
    
    # 1. Fetch Data
    for i, asset in enumerate(ASSETS):
        LOADING_STATUS = f"Fetching {asset} ({i+1}/{len(ASSETS)})..."
        print(LOADING_STATUS)
        df = fetch_30m_history(asset)
        RAW_DATA_30M[asset] = df
        
    LOADING_STATUS = "Data Fetched. Resampling & Optimizing..."
    print(LOADING_STATUS)
    
    results_list = []
    
    # 2. Resample & Optimize
    total_tasks = len(ASSETS) * len(TIMEFRAMES)
    completed = 0
    
    for asset in ASSETS:
        df_30m = RAW_DATA_30M.get(asset)
        if df_30m is None or df_30m.empty:
            continue
            
        for tf in TIMEFRAMES:
            # Resample
            df_tf = resample_data(df_30m, tf)
            if df_tf.empty:
                continue
            
            # Split Data (70% Train)
            cutoff_idx = int(len(df_tf) * SPLIT_RATIO)
            train_df = df_tf.iloc[:cutoff_idx]
            
            if len(train_df) < 100: 
                continue
                
            # Prepare Arrays
            opens = train_df['open'].values.astype(np.float64)
            highs = train_df['high'].values.astype(np.float64)
            lows = train_df['low'].values.astype(np.float64)
            closes = train_df['close'].values.astype(np.float64)
            
            # Run Grid Search
            best_tp, best_sl, raw_sharpe = run_grid_search(opens, highs, lows, closes)
            
            # Annualize Sharpe
            tf_annualizer = {
                '30m': np.sqrt(365*48),
                '1h': np.sqrt(365*24),
                '4h': np.sqrt(365*6),
                '1d': np.sqrt(365)
            }
            ann_sharpe = raw_sharpe * tf_annualizer.get(tf, 1.0)
            
            results_list.append({
                'asset': asset,
                'timeframe': tf,
                'tp': best_tp,
                'sl': best_sl,
                'sharpe': ann_sharpe,
                'cutoff_ts': df_tf.iloc[cutoff_idx]['timestamp'] # Save cutoff timestamp for plotting later
            })
            
            completed += 1
            if completed % 5 == 0:
                print(f"Optimized {completed}/{total_tasks} combinations...")

    # 3. Rank
    results_list.sort(key=lambda x: x['sharpe'], reverse=True)
    TOP_5_RESULTS = results_list[:5]
    
    IS_LOADING = False
    LOADING_STATUS = "Done"
    print("--- WORKER FINISHED ---")
    print("Top 5 Configs:")
    for r in TOP_5_RESULTS:
        print(r)

# Start Worker
t = threading.Thread(target=background_worker)
t.start()

# --- WEB SERVER ---
def calculate_equity_curve(df, tp, sl):
    """Calculates equity curve for plotting (Python logic, similar to Numba but returns array)"""
    n = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    timestamps = df['timestamp'].values
    
    equity = np.zeros(n)
    cum_pnl = 0.0
    fee_rate = 0.0002
    
    l_closed = True
    s_closed = True
    
    tp_mult_l = 1 + tp/100.0
    sl_mult_l = 1 - sl/100.0
    tp_mult_s = 1 - tp/100.0
    sl_mult_s = 1 + sl/100.0
    
    for i in range(n):
        op = opens[i]
        hi = highs[i]
        lo = lows[i]
        cl = closes[i]
        
        # Long
        l_entry_fee = op * fee_rate if l_closed else 0.0
        l_pnl = 0.0
        l_exit_fee = 0.0
        if lo <= op*sl_mult_l:
            l_pnl = (op*sl_mult_l) - op
            l_exit_fee = (op*sl_mult_l) * fee_rate
            l_closed = True
        elif hi >= op*tp_mult_l:
            l_pnl = (op*tp_mult_l) - op
            l_exit_fee = (op*tp_mult_l) * fee_rate
            l_closed = True
        else:
            l_pnl = cl - op
            l_closed = False
            
        # Short
        s_entry_fee = op * fee_rate if s_closed else 0.0
        s_pnl = 0.0
        s_exit_fee = 0.0
        if hi >= op*sl_mult_s:
            s_pnl = op - (op*sl_mult_s)
            s_exit_fee = (op*sl_mult_s) * fee_rate
            s_closed = True
        elif lo <= op*tp_mult_s:
            s_pnl = op - (op*tp_mult_s)
            s_exit_fee = (op*tp_mult_s) * fee_rate
            s_closed = True
        else:
            s_pnl = op - cl
            s_closed = False
            
        net = (l_pnl - l_entry_fee - l_exit_fee) + (s_pnl - s_entry_fee - s_exit_fee)
        cum_pnl += net
        equity[i] = cum_pnl
        
    return timestamps, equity

@app.route('/')
def dashboard():
    if IS_LOADING:
        return f"""
        <html>
        <head><meta http-equiv="refresh" content="5"></head>
        <body style='background:#222; color:#0f0; font-family:monospace; text-align:center; padding-top:100px;'>
            <h1>System Initializing...</h1>
            <p>{LOADING_STATUS}</p>
            <p>Please wait. Page will reload automatically.</p>
        </body>
        </html>
        """
        
    # Generate Table HTML
    rows = ""
    for idx, res in enumerate(TOP_5_RESULTS):
        rows += f"""
        <tr style="background: #333;">
            <td>#{idx+1}</td>
            <td style="color:cyan; font-weight:bold;">{res['asset']}</td>
            <td>{res['timeframe']}</td>
            <td>{res['tp']}%</td>
            <td>{res['sl']}%</td>
            <td style="color:gold;">{res['sharpe']:.2f}</td>
            <td>
                <a href="/plot?rank={idx}">
                    <button style="cursor:pointer; background:#554400; color:white; border:1px solid #776600;">View Chart</button>
                </a>
            </td>
        </tr>
        """
        
    html = f"""
    <html>
    <head>
        <title>Top 5 Strategies</title>
        <style>
            body {{ font-family: monospace; background: #111; color: #ddd; padding: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th {{ background: #444; padding: 10px; text-align: left; }}
            td {{ padding: 10px; border-bottom: 1px solid #222; }}
            h2 {{ color: #0f0; }}
        </style>
    </head>
    <body>
        <h2>Top 5 Optimized Strategies (70/30 Split)</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Asset</th>
                    <th>Timeframe</th>
                    <th>TP</th>
                    <th>SL</th>
                    <th>Train Sharpe</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        <p style="margin-top:20px; color:#888;">*Optimization performed on first 70% of data (Training Set).</p>
    </body>
    </html>
    """
    return html

@app.route('/plot')
def plot_chart():
    try:
        rank = int(request.args.get('rank', 0))
        config = TOP_5_RESULTS[rank]
    except:
        return "Invalid Rank"
    
    # Reconstruct Data
    df_30m = RAW_DATA_30M[config['asset']]
    df_tf = resample_data(df_30m, config['timeframe'])
    
    # Calculate Full Equity
    ts, equity = calculate_equity_curve(df_tf, config['tp'], config['sl'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(ts, equity, color='cyan', label='Total Equity')
    
    # Draw Split Line
    cutoff_ts = config['cutoff_ts']
    ax.axvline(cutoff_ts, color='yellow', linestyle='--', linewidth=2, label='Train/Test Split (70/30)')
    
    # Highlight Test Area
    ylim = ax.get_ylim()
    ax.fill_betweenx(ylim, cutoff_ts, ts[-1], color='white', alpha=0.1, label='Unseen Test Data')
    
    ax.set_title(f"#{rank+1} {config['asset']} {config['timeframe']} | TP:{config['tp']}% SL:{config['sl']}% | Train Sharpe: {config['sharpe']:.2f}")
    ax.legend(loc='upper left')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('#222')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    print("Server starting on 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)
