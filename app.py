import http.server
import socketserver
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import io
import base64
import time
import sys
import math
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import json
import threading
import os
import csv
import urllib.parse

# ---------------------------------------------------------
# Configuration & Assets
# ---------------------------------------------------------
# List of assets to track (mapped to Binance USDT pairs)
ASSETS = [
    "BTCUSDT",   # Bitcoin
    "ETHUSDT",   # Ethereum
    "XRPUSDT",   # XRP
    "SOLUSDT",   # Solana
    "DOGEUSDT",  # Dogecoin
    "ADAUSDT",   # Cardano
    "BCHUSDT",   # Bitcoin Cash
    "LINKUSDT",  # Chainlink
    "XLMUSDT",   # Stellar
    "SUIUSDT",   # Sui
    "AVAXUSDT",  # Avalanche
    "LTCUSDT",   # Litecoin
    "HBARUSDT",  # Hedera
    "SHIBUSDT",  # Shiba Inu
    "TONUSDT"    # Toncoin
]

INTERVAL = os.getenv("INTERVAL", '1h')
START_TIME = os.getenv("START_TIME", '2024-01-01')
END_TIME = os.getenv("END_TIME", '2026-01-01')
PORT = int(os.getenv("PORT", 8080))
TRAIN_SPLIT_RATIO = float(os.getenv("TRAIN_SPLIT_RATIO", 0.7))

# Parse Grid Search Values from Env (comma separated) or use default
_grid_env = os.getenv("GRID_SEARCH_VALUES")
if _grid_env:
    GRID_SEARCH_VALUES = [float(x.strip()) for x in _grid_env.split(',')]
else:
    # Default: 0.1% to 5%
    GRID_SEARCH_VALUES = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]

# Global Storage for Multi-Asset Data
# Structure: GLOBAL_RESULTS[symbol] = { 'best_result': ..., 'plot_b64': ..., 'model': ... }
GLOBAL_RESULTS = {}
GLOBAL_LIVE_LOG = [] # Shared log for all assets

# ---------------------------------------------------------
# 1. Fetch Data
# ---------------------------------------------------------
def fetch_binance_data(symbol, interval, start_str, end_str=None, limit=1000):
    base_url = "https://api.binance.com/api/v3/klines"
    
    if end_str:
        print(f"[{symbol}] Fetching data from {start_str} to {end_str}...")
        try:
            start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
            end_ts = int(pd.Timestamp(end_str).timestamp() * 1000)
        except Exception as e:
            print(f"Error parsing dates: {e}")
            return pd.DataFrame()
        
        all_data = []
        while start_ts < end_ts:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_ts,
                'endTime': end_ts,
                'limit': limit
            }
            try:
                response = requests.get(base_url, params=params, timeout=10)
                data = response.json()
                if not isinstance(data, list) or len(data) == 0:
                    break
                all_data.extend(data)
                last_close_time = data[-1][6]
                start_ts = last_close_time + 1
                time.sleep(0.05) 
            except Exception as e:
                print(f"[{symbol}] Error fetching data: {e}")
                break
        
        sys.stdout.write(f"[{symbol}] Downloaded {len(all_data)} candles.\n")
        
    else:
        # Live mode
        params = {'symbol': symbol, 'interval': interval, 'limit': 5}
        try:
            response = requests.get(base_url, params=params, timeout=5)
            all_data = response.json()
        except Exception as e:
            print(f"[{symbol}] Error fetching live data: {e}")
            return pd.DataFrame()

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    
    return df[['open_time', 'close']]

# ---------------------------------------------------------
# Helper: Sharpe Calculation
# ---------------------------------------------------------
def calculate_sharpe_ratio(returns_series, periods_per_year=24*365):
    if len(returns_series) < 2:
        return 0.0
    mean_ret = np.mean(returns_series)
    std_ret = np.std(returns_series)
    if std_ret == 0:
        return 0.0
    return (mean_ret / std_ret) * np.sqrt(periods_per_year)

# ---------------------------------------------------------
# Processing & Backtesting Logic
# ---------------------------------------------------------
def train_model(df, grid_size, needed_precision):
    # This function is used by the live loop to rebuild the model for the specific asset
    df = df.copy()
    df['rounded_close'] = ((df['close'] / grid_size).round() * grid_size).round(needed_precision)
    df['next_rounded'] = df['rounded_close'].shift(-1)
    
    conditions = [
        df['next_rounded'] > df['rounded_close'],
        df['next_rounded'] < df['rounded_close']
    ]
    choices = ['UP', 'DOWN']
    df['target_direction'] = np.select(conditions, choices, default='FLAT')
    
    df['t_0'] = df['rounded_close']
    df['t_1'] = df['rounded_close'].shift(1)
    df['t_2'] = df['rounded_close'].shift(2)
    
    data = df.dropna().copy()
    
    sequence_map = defaultdict(list)
    for _, row in data.iterrows():
        seq = (row['t_2'], row['t_1'], row['t_0'])
        sequence_map[seq].append(row['target_direction'])
        
    final_model = {}
    for seq, directions in sequence_map.items():
        counts = Counter(directions)
        most_common = counts.most_common(1)[0][0]
        final_model[seq] = most_common
        
    return final_model

def evaluate_strategy(df_original, grid_percent, verbose=False):
    if df_original.empty:
        return {'sharpe': -99, 'accuracy': 0, 'cumulative_pnl': 0}

    df = df_original.copy()
    first_close = df['close'].iloc[0]
    grid_size = first_close * grid_percent
    
    if grid_size == 0:
        needed_precision = 8
    else:
        needed_precision = int(math.ceil(-math.log10(grid_size))) + 2
    needed_precision = max(2, min(needed_precision, 10))
    
    df['rounded_close'] = ((df['close'] / grid_size).round() * grid_size).round(needed_precision)
    df['next_rounded'] = df['rounded_close'].shift(-1)
    df['next_close_raw'] = df['close'].shift(-1)
    
    conditions = [
        df['next_rounded'] > df['rounded_close'],
        df['next_rounded'] < df['rounded_close']
    ]
    choices = ['UP', 'DOWN']
    df['target_direction'] = np.select(conditions, choices, default='FLAT')
    
    df['t_0'] = df['rounded_close']
    df['t_1'] = df['rounded_close'].shift(1)
    df['t_2'] = df['rounded_close'].shift(2)

    df['raw_t_0'] = df['close']
    df['raw_t_1'] = df['close'].shift(1)
    df['raw_t_2'] = df['close'].shift(2)
    
    data = df.dropna().copy()
    split_idx = int(len(data) * TRAIN_SPLIT_RATIO)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    sequence_map = defaultdict(list)
    for _, row in train_df.iterrows():
        seq = (row['t_2'], row['t_1'], row['t_0'])
        sequence_map[seq].append(row['target_direction'])
        
    model = {}
    for seq, directions in sequence_map.items():
        counts = Counter(directions)
        model[seq] = counts.most_common(1)[0][0]
        
    correct_predictions = 0
    total_predictions = 0
    cumulative_pnl = 0.0
    
    test_results_list = []
    hourly_returns = []
    
    for idx, row in test_df.iterrows():
        seq = (row['t_2'], row['t_1'], row['t_0'])
        prediction = model.get(seq, 'FLAT') 
        actual = row['target_direction']
        
        if prediction != 'FLAT' and actual != 'FLAT':
            if prediction == actual:
                correct_predictions += 1
            total_predictions += 1
        
        curr_price = row['close']
        next_price = row['next_close_raw']
        
        trade_pnl = 0.0
        if prediction == 'UP':
            trade_pnl = next_price - curr_price
        elif prediction == 'DOWN':
            trade_pnl = curr_price - next_price
            
        cumulative_pnl += trade_pnl
        hourly_returns.append(trade_pnl / curr_price if curr_price > 0 else 0)
        
        test_results_list.append({
            'time_t': row['open_time'],
            'rnd_t_2': row['t_2'],
            'rnd_t_1': row['t_1'],
            'rnd_t_0': row['t_0'],
            'raw_t_2': row['raw_t_2'],
            'raw_t_1': row['raw_t_1'],
            'raw_t_0': row['raw_t_0'],
            'prediction': prediction,
            'actual': actual,
            'pnl': trade_pnl,
            'cum_pnl': cumulative_pnl,
            'next_price_raw': next_price
        })
        
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    sharpe = calculate_sharpe_ratio(hourly_returns)
    
    return {
        'sharpe': sharpe,
        'accuracy': accuracy,
        'cumulative_pnl': cumulative_pnl,
        'grid_size': grid_size,
        'needed_precision': needed_precision,
        'test_results': test_results_list,
        'grid_percent': grid_percent,
        'model': model
    }

def run_grid_search(df, symbol):
    print(f"\n--- Grid Search: {symbol} ---")
    best_sharpe = -float('inf')
    best_result = None
    
    for gp in GRID_SEARCH_VALUES:
        res = evaluate_strategy(df, gp, verbose=False)
        # print(f"  Grid: {gp*100:5.2f}% | Sharpe: {res['sharpe']:6.3f}")
        if res['sharpe'] > best_sharpe:
            best_sharpe = res['sharpe']
            best_result = res
            
    print(f"Best for {symbol}: Grid {best_result['grid_percent']*100}% | Sharpe: {best_result['sharpe']:.4f}")
    return best_result

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
def create_plot(df, test_results, accuracy, total_pnl, symbol, grid_percent):
    if not test_results:
        return ""
    plt.figure(figsize=(10, 5))
    plt.plot(df['open_time'], df['close'], label='Price', color='gray', alpha=0.3)
    
    test_times = [x['time_t'] for x in test_results]
    test_pnl = [x['cum_pnl'] for x in test_results]
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(test_times, test_pnl, label='Strategy PnL', color='blue', linewidth=1.5)
    
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Cumulative PnL (USDT)')
    plt.title(f'{symbol} Backtest | Grid={grid_percent*100}% | Acc: {accuracy:.2f}% | PnL: {total_pnl:.4f}')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return image_base64

# ---------------------------------------------------------
# Live Prediction Logic
# ---------------------------------------------------------
def save_live_prediction(symbol, timestamp, close_price, sequence, prediction):
    file_name = "live_prediction_log.csv"
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, "a") as f:
        if not file_exists:
            f.write("timestamp,symbol,close_price,sequence,prediction\n")
        seq_str = f"{sequence[0]}|{sequence[1]}|{sequence[2]}"
        f.write(f"{timestamp},{symbol},{close_price},{seq_str},{prediction}\n")
    
    GLOBAL_LIVE_LOG.append({
        'timestamp': str(timestamp),
        'symbol': symbol,
        'close_price': close_price,
        'sequence': str(sequence),
        'prediction': prediction
    })
    
    print(f"[{symbol}] Prediction: {prediction} (Seq: {seq_str})")

def live_prediction_loop():
    print("--- Live Prediction Service Started ---")
    while True:
        try:
            for symbol in ASSETS:
                # We need the optimized grid size for this symbol
                if symbol not in GLOBAL_RESULTS:
                    continue
                
                res = GLOBAL_RESULTS[symbol]['best_result']
                grid_size = res['grid_size']
                precision = res['needed_precision']
                model = res['model']
                
                # Fetch recent candles
                df_live = fetch_binance_data(symbol, INTERVAL, start_str=None) # Fetches last 5 candles
                
                if len(df_live) >= 3:
                    last_row = df_live.iloc[-1]
                    # Calculate rounding for last 3 candles to form sequence
                    # We need t-2, t-1, t-0
                    recent_closes = df_live['close'].tail(3).values
                    rounded = [round(round(x / grid_size) * grid_size, precision) for x in recent_closes]
                    
                    if len(rounded) == 3:
                        seq = (rounded[0], rounded[1], rounded[2])
                        prediction = model.get(seq, 'FLAT')
                        
                        # Only log if new hour/timestamp
                        last_ts = str(last_row['open_time'])
                        # Simple check to avoid duplicates: check if last log for this symbol has same TS
                        is_duplicate = False
                        for log in reversed(GLOBAL_LIVE_LOG):
                            if log['symbol'] == symbol and log['timestamp'] == last_ts:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            save_live_prediction(symbol, last_ts, recent_closes[-1], seq, prediction)
                
                time.sleep(1) # Small delay between symbols
            
            # Wait for next hour check (approximate, simpler logic here just waits 60s)
            # Real production would align with clock
            time.sleep(60) 
            
        except Exception as e:
            print(f"Live Loop Error: {e}")
            time.sleep(60)

# ---------------------------------------------------------
# Server & Dashboard
# ---------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Asset Strategy Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; background-color: #f8f9fa; color: #333; }}
        .sidebar {{ width: 220px; background: #343a40; color: white; position: fixed; height: 100%; overflow-y: auto; padding-top: 20px; }}
        .sidebar h3 {{ text-align: center; margin-bottom: 20px; font-size: 1.2em; }}
        .nav-item {{ padding: 10px 20px; display: block; color: #ccc; text-decoration: none; cursor: pointer; }}
        .nav-item:hover, .nav-item.active {{ background: #495057; color: white; }}
        .content {{ margin-left: 220px; padding: 20px; }}
        .card {{ background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); padding: 20px; margin-bottom: 20px; }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #e9ecef; font-weight: 600; }}
        tr:hover {{ background-color: #f1f1f1; }}
        
        .up {{ color: #28a745; font-weight: bold; }}
        .down {{ color: #dc3545; font-weight: bold; }}
        .flat {{ color: #6c757d; }}
        
        .summary-metric {{ display: inline-block; margin-right: 20px; font-size: 0.9em; }}
        .summary-val {{ font-size: 1.2em; font-weight: bold; display: block; }}
        
        #detailView {{ display: none; }}
        .loading {{ color: #666; font-style: italic; }}
        
        .live-tag {{ background: #dc3545; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; vertical-align: middle; margin-left: 5px; }}
    </style>
</head>
<body>

<div class="sidebar">
    <h3>Portfolio</h3>
    <a class="nav-item active" onclick="showSummary()">Overview</a>
    <div id="assetList">
        </div>
</div>

<div class="content">
    <div id="overviewSection">
        <div class="card">
            <h2>Portfolio Overview</h2>
            <p>Backtest Period: {start} to {end} | Interval: {interval}</p>
            <table>
                <thead>
                    <tr>
                        <th>Asset</th>
                        <th>Grid Size</th>
                        <th>Grid %</th>
                        <th>Sharpe</th>
                        <th>Accuracy</th>
                        <th>Total PnL</th>
                    </tr>
                </thead>
                <tbody id="summaryTableBody">
                    </tbody>
            </table>
        </div>
        
        <div class="card">
            <h3>Recent Live Predictions <span class="live-tag">LIVE</span></h3>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Price</th>
                        <th>Sequence</th>
                        <th>Prediction</th>
                    </tr>
                </thead>
                <tbody id="globalLiveLog">
                </tbody>
            </table>
        </div>
    </div>

    <div id="detailView">
        <h2 id="detailTitle">Asset Details</h2>
        
        <div class="card">
            <div id="detailStats"></div>
            <div id="detailPlot" style="text-align:center; margin-top:15px;"></div>
        </div>

        <div class="card">
            <h3>Backtest Log (Last 100 Trades)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Input Seq</th>
                        <th>Prediction</th>
                        <th>Actual</th>
                        <th>PnL</th>
                    </tr>
                </thead>
                <tbody id="detailLogBody"></tbody>
            </table>
        </div>
    </div>
</div>

<script>
    const assets = {assets_json}; // List of symbols
    
    // Fetch Summary Data on Load
    fetch('/api/summary')
        .then(response => response.json())
        .then(data => {{
            const tbody = document.getElementById('summaryTableBody');
            const assetList = document.getElementById('assetList');
            
            data.forEach(row => {{
                // Populate Summary Table
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><b>${{row.symbol}}</b></td>
                    <td>${{row.grid_size}}</td>
                    <td>${{row.grid_percent}}%</td>
                    <td>${{row.sharpe}}</td>
                    <td>${{row.accuracy}}%</td>
                    <td class="${{row.pnl >= 0 ? 'up' : 'down'}}">${{row.pnl}}</td>
                `;
                tbody.appendChild(tr);
                
                // Populate Sidebar
                const link = document.createElement('a');
                link.className = 'nav-item';
                link.innerText = row.symbol;
                link.onclick = () => loadAsset(row.symbol);
                assetList.appendChild(link);
            }});
            
            // Populate Global Live Log
            renderLiveLog();
        }});

    function renderLiveLog() {{
        fetch('/api/livelog')
        .then(r => r.json())
        .then(logs => {{
            const tbody = document.getElementById('globalLiveLog');
            tbody.innerHTML = '';
            // Show last 10
            logs.slice(0, 15).forEach(log => {{
                 const tr = document.createElement('tr');
                 const pClass = log.prediction === 'UP' ? 'up' : (log.prediction === 'DOWN' ? 'down' : 'flat');
                 tr.innerHTML = `
                    <td>${{log.timestamp}}</td>
                    <td><b>${{log.symbol}}</b></td>
                    <td>${{log.close_price}}</td>
                    <td>${{log.sequence}}</td>
                    <td class="${{pClass}}">${{log.prediction}}</td>
                 `;
                 tbody.appendChild(tr);
            }});
        }});
    }}

    function showSummary() {{
        document.getElementById('overviewSection').style.display = 'block';
        document.getElementById('detailView').style.display = 'none';
        document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
        document.querySelector('.sidebar a:first-child').classList.add('active');
    }}

    function loadAsset(symbol) {{
        document.getElementById('overviewSection').style.display = 'none';
        document.getElementById('detailView').style.display = 'block';
        document.getElementById('detailTitle').innerText = symbol;
        document.getElementById('detailStats').innerHTML = '<p class="loading">Loading data...</p>';
        document.getElementById('detailPlot').innerHTML = '';
        document.getElementById('detailLogBody').innerHTML = '';

        // Update Sidebar Active State
        document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
        event.target.classList.add('active');

        fetch('/api/details?symbol=' + symbol)
            .then(r => r.json())
            .then(data => {{
                // Stats
                const htmlStats = `
                    <div class="summary-metric"><span class="summary-val">${{data.sharpe}}</span>Sharpe Ratio</div>
                    <div class="summary-metric"><span class="summary-val">${{data.accuracy}}%</span>Accuracy</div>
                    <div class="summary-metric"><span class="summary-val" style="color:${{data.pnl >= 0 ? 'green':'red'}}">${{data.pnl}}</span>Total PnL</div>
                    <div class="summary-metric"><span class="summary-val">${{data.grid_percent}}%</span>Grid Size</div>
                `;
                document.getElementById('detailStats').innerHTML = htmlStats;
                
                // Plot
                const img = document.createElement('img');
                img.src = "data:image/png;base64," + data.plot;
                img.style.maxWidth = "100%";
                document.getElementById('detailPlot').appendChild(img);
                
                // Log (Limit to last 100 for perf)
                const tbody = document.getElementById('detailLogBody');
                const logs = data.logs.slice(-100).reverse();
                logs.forEach(row => {{
                    const tr = document.createElement('tr');
                    const pnlColor = row.pnl >= 0 ? 'green' : 'red';
                    const predClass = row.prediction === 'UP' ? 'up' : (row.prediction === 'DOWN' ? 'down' : '');
                    tr.innerHTML = `
                        <td>${{row.time_t}}</td>
                        <td>[${{row.rnd_t_2}}, ${{row.rnd_t_1}}, ${{row.rnd_t_0}}]</td>
                        <td class="${{predClass}}">${{row.prediction}}</td>
                        <td>${{row.actual}}</td>
                        <td style="color:${{pnlColor}}">${{row.pnl.toFixed(4)}}</td>
                    `;
                    tbody.appendChild(tr);
                }});
            }});
    }}
</script>

</body>
</html>
"""

class BacktestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = HTML_TEMPLATE.format(
                assets_json=json.dumps(ASSETS),
                start=START_TIME,
                end=END_TIME,
                interval=INTERVAL
            )
            self.wfile.write(html.encode('utf-8'))
            
        elif path == '/api/summary':
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            
            summary = []
            for sym in ASSETS:
                if sym in GLOBAL_RESULTS:
                    r = GLOBAL_RESULTS[sym]['best_result']
                    summary.append({
                        'symbol': sym,
                        'grid_size': f"{r['grid_size']:.{r['needed_precision']}f}",
                        'grid_percent': f"{r['grid_percent']*100:.2f}",
                        'sharpe': f"{r['sharpe']:.3f}",
                        'accuracy': f"{r['accuracy']:.1f}",
                        'pnl': f"{r['cumulative_pnl']:.2f}"
                    })
            self.wfile.write(json.dumps(summary).encode('utf-8'))
            
        elif path == '/api/livelog':
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            # Return reversed log (newest first)
            self.wfile.write(json.dumps(GLOBAL_LIVE_LOG[::-1]).encode('utf-8'))
            
        elif path == '/api/details':
            sym = query.get('symbol', [None])[0]
            if sym and sym in GLOBAL_RESULTS:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                
                data = GLOBAL_RESULTS[sym]
                res = data['best_result']
                
                # Format logs for JSON
                logs_out = []
                for r in res['test_results']:
                    logs_out.append({
                        'time_t': str(r['time_t']),
                        'rnd_t_2': r['rnd_t_2'], 'rnd_t_1': r['rnd_t_1'], 'rnd_t_0': r['rnd_t_0'],
                        'prediction': r['prediction'],
                        'actual': r['actual'],
                        'pnl': r['pnl']
                    })
                    
                resp_obj = {
                    'sharpe': f"{res['sharpe']:.3f}",
                    'accuracy': f"{res['accuracy']:.2f}",
                    'pnl': f"{res['cumulative_pnl']:.4f}",
                    'grid_percent': f"{res['grid_percent']*100:.2f}",
                    'plot': data['plot_b64'],
                    'logs': logs_out
                }
                self.wfile.write(json.dumps(resp_obj).encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()

def main_loop():
    print(f"Starting Multi-Asset Backtest for: {', '.join(ASSETS)}")
    
    # 1. Backtest Loop
    for sym in ASSETS:
        df = fetch_binance_data(sym, INTERVAL, START_TIME, END_TIME)
        if not df.empty:
            best_res = run_grid_search(df, sym)
            plot_b64 = create_plot(df, best_res['test_results'], best_res['accuracy'], 
                                   best_res['cumulative_pnl'], sym, best_res['grid_percent'])
            
            GLOBAL_RESULTS[sym] = {
                'best_result': best_res,
                'plot_b64': plot_b64
            }
        else:
            print(f"Skipping {sym} (No Data)")
            
    print("\nAll backtests complete. Starting Server & Live Loop...")
    
    # 2. Start Live Loop Thread
    t = threading.Thread(target=live_prediction_loop, daemon=True)
    t.start()
    
    # 3. Start Server
    handler = BacktestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving Dashboard at http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    main_loop()
