import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# Global Storage
current_df = None
market_data_cache = {} 
current_config = {
    'symbol': 'BTC/USDT',
    'timeframe': '1h'
}

ASSETS = [
    'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT', 'DOGE/USDT',
    'ADA/USDT', 'BCH/USDT', 'LINK/USDT', 'XLM/USDT', 'SUI/USDT',
    'AVAX/USDT', 'LTC/USDT', 'HBAR/USDT', 'SHIB/USDT', 'TON/USDT'
]

TIMEFRAMES = ['30m', '1h', '4h', '1d']

def fetch_data(symbol, timeframe):
    exchange = ccxt.binance({'enableRateLimit': True})
    limit = 1000
    now = exchange.milliseconds()
    since = now - (5 * 365 * 24 * 60 * 60 * 1000)
    
    all_ohlcv = []
    print(f"Fetching data for {symbol} {timeframe}...")
    
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            tf_ms = {'30m': 1800000, '1h': 3600000, '4h': 14400000, '1d': 86400000}
            duration = tf_ms.get(timeframe, 3600000)
            since = last_timestamp + duration
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

@njit(fastmath=True)
def optimize_single_sharpe(opens, highs, lows, closes):
    """
    Optimizes for Sharpe Ratio (Risk-Adjusted Return).
    """
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
            
            # Welford's or SumSq for variance
            sum_returns = 0.0
            sum_returns_sq = 0.0
            
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
                l_pnl = 0.0
                
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
                
                # SHORT
                s_entry_fee = op * fee_rate if s_closed_prev else 0.0
                s_sl_price = op * s_sl_mult
                s_tp_price = op * s_tp_mult
                s_exit_fee = 0.0
                s_pnl = 0.0
                
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
                
                # Accumulate Stats
                sum_returns += net_pnl
                sum_returns_sq += (net_pnl * net_pnl)
            
            # Calculate Sharpe
            if n > 100:
                mean = sum_returns / n
                var = (sum_returns_sq / n) - (mean * mean)
                if var > 1e-9:
                    std = np.sqrt(var)
                    # Annualize Sharpe (approx 8760 hours in year)
                    # If timeframe is different, this constant is just a scaler, doesn't change optimization rank
                    sharpe = (mean / std) * 93.59
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_tp = tp_pct
                        best_sl = sl_pct
                        
    return best_tp, best_sl, best_sharpe

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
        l_exit_fee = 0.0
        l_status = ""
        
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
        s_exit_fee = 0.0
        s_status = ""
        
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

@app.route('/load_data', methods=['POST'])
def load_data():
    global current_df, market_data_cache, current_config
    
    req = request.json
    symbol = req.get('symbol', 'BTC/USDT')
    timeframe = req.get('timeframe', '1h')
    
    current_df = fetch_data(symbol, timeframe)
    current_config['symbol'] = symbol
    current_config['timeframe'] = timeframe
    
    # Invalidate cache
    market_data_cache = {}
    
    return jsonify({'status': 'success', 'rows': len(current_df)})

@app.route('/')
def index():
    sl = request.args.get('sl', '2.0')
    tp = request.args.get('tp', '5.0')
    
    asset_options = "".join([f"<option value='{a}'>{a}</option>" for a in ASSETS])
    tf_options = "".join([f"<option value='{t}'>{t}</option>" for t in TIMEFRAMES])
    
    status_text = "No data loaded."
    if current_df is not None:
        status_text = f"Loaded: {current_config['symbol']} ({current_config['timeframe']})"

    html = f"""
    <html>
        <head>
            <title>Multi-Asset Optimization</title>
            <style>
                body {{ font-family: monospace; background: #222; color: #ddd; margin: 0; padding: 20px; }}
                .container {{ max_width: 1200px; margin: 0 auto; text-align: center; }}
                .controls {{ background: #333; padding: 15px; border-radius: 5px; margin-bottom: 20px; display: flex; justify-content: center; gap: 20px; align-items: center; }}
                select, input, button {{ background: #444; color: white; border: 1px solid #555; padding: 8px; border-radius: 4px; }}
                button {{ cursor: pointer; background: #554400; font-weight: bold; }}
                button:hover {{ background: #776600; }}
                #load-btn {{ background: #004400; }}
                .scroll-table {{ max_height: 500px; overflow-y: auto; border: 1px solid #444; margin-top: 20px; }}
                table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
                th {{ background: #444; padding: 10px; position: sticky; top: 0; }}
                td {{ border-bottom: 1px solid #333; padding: 5px; text-align: left; }}
                #status-bar {{ margin-bottom: 10px; color: cyan; }}
            </style>
            <script>
                function loadData() {{
                    const sym = document.getElementById('symbol').value;
                    const tf = document.getElementById('timeframe').value;
                    document.getElementById('status-bar').innerText = "Fetching data (this may take a few seconds)...";
                    
                    fetch('/load_data', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{symbol: sym, timeframe: tf}})
                    }})
                    .then(res => res.json())
                    .then(data => {{
                        document.getElementById('status-bar').innerText = "Loaded " + sym + " " + tf + " (" + data.rows + " candles)";
                        location.reload(); 
                    }});
                }}

                function runOptimizer() {{
                    document.getElementById('status-bar').innerText = "Optimizing Sharpe (Train <2026)... Please wait.";
                    fetch('/optimize')
                        .then(res => res.json())
                        .then(data => {{
                            document.getElementById('status-bar').innerText = "Optimized! SL: " + data.sl + " TP: " + data.tp + " Sharpe: " + data.sharpe.toFixed(2);
                            document.querySelector('input[name="sl"]').value = data.sl;
                            document.querySelector('input[name="tp"]').value = data.tp;
                            document.getElementById('plot-form').submit();
                        }});
                }}
            </script>
        </head>
        <body>
            <div class="container">
                <h2>Multi-Asset Strategy (Sharpe)</h2>
                
                <div class="controls">
                    <label>Asset: <select id="symbol">{asset_options}</select></label>
                    <label>Timeframe: <select id="timeframe">{tf_options}</select></label>
                    <button id="load-btn" onclick="loadData()">Load Data</button>
                </div>
                
                <div id="status-bar">{status_text}</div>
                
                <div class="controls">
                    <form id="plot-form" action="/" method="get" style="display:flex; gap:10px; align-items:center; margin:0;">
                        <label>SL % <input type="number" step="0.1" name="sl" value="{sl}"></label>
                        <label>TP % <input type="number" step="0.1" name="tp" value="{tp}"></label>
                        <input type="submit" value="Update Chart" style="background:#444;">
                    </form>
                    <button onclick="runOptimizer()">Optimize Sharpe</button>
                </div>

                <img src="/plot.png?sl={sl}&tp={tp}" style="border: 1px solid #555; max-width: 100%; width: 100%; height: auto;">
                
                <h3>Recent Trades</h3>
                <div class="scroll-table" id="trade-list">
                    {get_trade_table_html(float(tp), float(sl))}
                </div>
            </div>
        </body>
    </html>
    """
    return html

def get_trade_table_html(tp, sl):
    if current_df is None:
        return "<table><tr><td>No Data</td></tr></table>"
        
    calc_data = current_df.dropna(subset=['close']).reset_index(drop=True)
    _, trades = calculate_strategy_single(calc_data, tp, sl)
    
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
    return f"<table><thead><tr><th>Time</th><th>Status</th><th>PnL</th></tr></thead><tbody>{rows}</tbody></table>"

@app.route('/optimize')
def optimize():
    global market_data_cache, current_df
    
    if current_df is None:
        return jsonify({'error': 'No data'}), 400
        
    cutoff_ts = pd.Timestamp('2026-01-01')
    
    if 'opens' not in market_data_cache:
        calc_data = current_df.dropna(subset=['close']).reset_index(drop=True)
        market_data_cache['timestamps'] = calc_data['timestamp'].values
        market_data_cache['opens'] = calc_data['open'].values.astype(np.float64)
        market_data_cache['highs'] = calc_data['high'].values.astype(np.float64)
        market_data_cache['lows'] = calc_data['low'].values.astype(np.float64)
        market_data_cache['closes'] = calc_data['close'].values.astype(np.float64)

    train_mask = market_data_cache['timestamps'] < np.datetime64(cutoff_ts)
    
    opens = market_data_cache['opens'][train_mask]
    highs = market_data_cache['highs'][train_mask]
    lows = market_data_cache['lows'][train_mask]
    closes = market_data_cache['closes'][train_mask]
    
    best_tp, best_sl, best_sharpe = optimize_single_sharpe(opens, highs, lows, closes)
    
    return jsonify({
        'tp': round(best_tp, 1),
        'sl': round(best_sl, 1),
        'sharpe': best_sharpe
    })

@app.route('/plot.png')
def plot_png():
    global current_df, current_config
    
    try:
        sl = float(request.args.get('sl', 2.0))
        tp = float(request.args.get('tp', 5.0))
    except:
        sl, tp = 2.0, 5.0

    if current_df is None:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "Please Load Data First", ha='center', va='center')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close(fig)
        return send_file(img, mimetype='image/png')

    plot_data = current_df.dropna(subset=['close']).reset_index(drop=True).copy()
    net_equity, _ = calculate_strategy_single(plot_data, tp, sl)
    plot_data['net_equity'] = net_equity
    
    cutoff_ts = pd.Timestamp('2026-01-01')
    unseen_data = plot_data[plot_data['timestamp'] >= cutoff_ts].copy()
    if not unseen_data.empty:
        unseen_data['net_equity'] -= unseen_data['net_equity'].iloc[0]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), dpi=100, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    x = matplotlib.dates.date2num(plot_data['timestamp'])
    y = plot_data['close'].values
    
    ax1.plot(plot_data['timestamp'], y, color='white', linewidth=1)
    ax1.set_title(f"{current_config['symbol']} ({current_config['timeframe']}) Price")
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
    print("Starting server on 8080...")
    current_df = fetch_data('BTC/USDT', '1h')
    app.run(host='0.0.0.0', port=8080, debug=False)
