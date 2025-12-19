import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string, jsonify
from itertools import product
from datetime import datetime
import time
import threading

app = Flask(__name__)

# --- CONFIGURATION ---
SYMBOL = 'BTCUSDT'
START_YEAR = 2018
SMA_START = 10
SMA_END = 400
SMA_STEP = 10
# Weights are grid searched from 0 to 1.
WEIGHT_STEPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# --- GLOBAL STATE ---
# Stores the current progress and results of the background task
GLOBAL_STATE = {
    'status': 'Waiting to start...',
    'progress': 0,
    'done': False,
    'results': [],
    'plot': None,
    'error': None
}

# --- DATA FETCHING ---
def fetch_binance_data(symbol, interval, start_year):
    """
    Fetches historical OHLC data from Binance API with pagination.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime(start_year, 1, 1).timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    limit = 1000
    
    all_data = []
    current_start = start_ts
    
    req_count = 0
    while current_start < end_ts and req_count < 1000:
        params = {'symbol': symbol, 'interval': interval, 'startTime': current_start, 'limit': limit}
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            if not data: break
            all_data.extend(data)
            current_start = data[-1][0] + 1
            req_count += 1
            time.sleep(0.05)
        except: break
            
    df = pd.DataFrame(all_data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qav', 'nt', 'tbv', 'tqv', 'i'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    df.set_index('open_time', inplace=True)
    return df[['close']]

# --- BACKTESTING ENGINE ---
def calculate_sharpe_ratio(returns, periods_per_year):
    std = returns.std()
    if std == 0 or np.isnan(std): return -10.0
    return (returns.mean() / std) * np.sqrt(periods_per_year)

def run_grid_search(df, timeframe_label, progress_start, progress_end):
    """
    Runs grid search and updates GLOBAL_STATE progress.
    progress_start: Percentage to start at (e.g., 0)
    progress_end: Percentage to end at (e.g., 33)
    """
    prices = df['close'].values
    returns = df['close'].pct_change().fillna(0).values
    
    periods = list(range(SMA_START, SMA_END + 1, SMA_STEP))
    
    # Pre-calculate signals (Vectorized)
    signals_matrix = []
    for p in periods:
        sma = pd.Series(prices).rolling(window=p).mean().values
        sig = np.where(prices > sma, 1.0, -1.0)
        sig = np.concatenate(([0], sig[:-1]))
        signals_matrix.append(sig)
    
    signals_matrix = np.array(signals_matrix)
    
    # Space generation
    weight_opts = list(product(WEIGHT_STEPS, repeat=3))
    weight_opts = [w for w in weight_opts if sum(w) > 0]
    
    num_p = len(periods)
    sma_idx_opts = [(i, j, k) for i, j, k in product(range(num_p), repeat=3) if i < j < k]
    
    best_sharpe = -999
    best_params = None
    best_signal = None

    if timeframe_label == '1H': factor = 365*24
    elif timeframe_label == '4H': factor = 365*6
    else: factor = 365

    total_iters = len(sma_idx_opts)
    
    # Optimized search loop
    for idx, (i, j, k) in enumerate(sma_idx_opts):
        # Update progress roughly every 100 iterations to avoid locking overhead
        if idx % 100 == 0:
            current_chunk_progress = idx / total_iters
            total_progress = progress_start + (current_chunk_progress * (progress_end - progress_start))
            GLOBAL_STATE['progress'] = int(total_progress)
            GLOBAL_STATE['status'] = f"Optimizing {timeframe_label}... ({int(current_chunk_progress * 100)}%)"

        s1, s2, s3 = signals_matrix[i], signals_matrix[j], signals_matrix[k]
        p1, p2, p3 = periods[i], periods[j], periods[k]
        
        for w1, w2, w3 in weight_opts:
            total_w = w1 + w2 + w3
            pos = (w1*s1 + w2*s2 + w3*s3) / total_w
            
            strat_rets = pos * returns
            sharpe = calculate_sharpe_ratio(strat_rets, factor)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {
                    'smas': (p1, p2, p3),
                    'weights': (w1, w2, w3),
                    'sharpe': round(sharpe, 4),
                    'return': round((np.prod(1 + strat_rets) - 1) * 100, 2)
                }
                best_signal = pos

    best_curve = pd.Series(np.cumprod(1 + (best_signal * returns)), index=df.index)
    return best_params, best_curve

# --- PLOTTING ---
def create_plot(curves_data):
    fig, axes = plt.subplots(len(curves_data), 1, figsize=(12, 5 * len(curves_data)))
    if len(curves_data) == 1: axes = [axes]
    
    for ax, (label, base_df, strat_curve) in zip(axes, curves_data):
        base_curve = (1 + base_df['close'].pct_change().fillna(0)).cumprod()
        ax.plot(base_curve.index, base_curve, label='BTC Buy & Hold', color='black', alpha=0.3)
        ax.plot(strat_curve.index, strat_curve, label='Opt. Triple SMA', color='forestgreen', linewidth=1.5)
        ax.set_title(f"Equity Curve: {label}")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# --- WORKER THREAD ---
def background_worker():
    """
    Main function for the background thread.
    Fetches data and runs grid search sequentially.
    """
    try:
        GLOBAL_STATE['status'] = "Fetching Binance Data..."
        GLOBAL_STATE['progress'] = 5
        
        # 1. Get Base Data (1H)
        df_1h = fetch_binance_data(SYMBOL, '1h', START_YEAR)
        if df_1h.empty:
            raise Exception("No data fetched")
            
        # 2. Resample
        df_4h = df_1h.resample('4h').last().dropna()
        df_1d = df_1h.resample('1D').last().dropna()
        
        # 3. Run search (Split progress bar: 10%->40%, 40%->70%, 70%->95%)
        GLOBAL_STATE['status'] = "Starting 1H Grid Search..."
        res_1h, curve_1h = run_grid_search(df_1h, '1H', 10, 40)
        
        GLOBAL_STATE['status'] = "Starting 4H Grid Search..."
        res_4h, curve_4h = run_grid_search(df_4h, '4H', 40, 70)
        
        GLOBAL_STATE['status'] = "Starting 1D Grid Search..."
        res_1d, curve_1d = run_grid_search(df_1d, '1D', 70, 95)
        
        # 4. Create Plot
        GLOBAL_STATE['status'] = "Generating Plots..."
        plot_b64 = create_plot([
            ('1 Hour', df_1h, curve_1h),
            ('4 Hour', df_4h, curve_4h),
            ('1 Day', df_1d, curve_1d)
        ])
        
        GLOBAL_STATE['results'] = [
            {'tf': '1H', 'smas': str(res_1h['smas']), 'weights': str(res_1h['weights']), 'sharpe': res_1h['sharpe'], 'ret': res_1h['return']},
            {'tf': '4H', 'smas': str(res_4h['smas']), 'weights': str(res_4h['weights']), 'sharpe': res_4h['sharpe'], 'ret': res_4h['return']},
            {'tf': '1D', 'smas': str(res_1d['smas']), 'weights': str(res_1d['weights']), 'sharpe': res_1d['sharpe'], 'ret': res_1d['return']}
        ]
        GLOBAL_STATE['plot'] = plot_b64
        GLOBAL_STATE['progress'] = 100
        GLOBAL_STATE['status'] = "Optimization Completed"
        GLOBAL_STATE['done'] = True
        
    except Exception as e:
        GLOBAL_STATE['error'] = str(e)
        GLOBAL_STATE['status'] = "Error Occurred"
        print(f"Background Worker Error: {e}")

# --- FLASK APP ---
@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Triple SMA Optimization</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background: #121212; color: #e0e0e0; font-family: sans-serif; }
            .container { max-width: 1000px; margin-top: 50px; }
            .card { background: #1e1e1e; border: 1px solid #333; margin-bottom: 20px; }
            .progress { height: 25px; background-color: #333; }
            .progress-bar { transition: width 0.5s ease; }
            .table { color: #e0e0e0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 class="mb-4">Triple SMA Optimizer (2018 - Present)</h2>
            
            <div id="progressSection" class="card p-4">
                <h5 class="card-title">Optimization Status</h5>
                <p id="statusText">Initializing...</p>
                <div class="progress mb-3">
                    <div id="progressBar" class="progress-bar progress-bar-striped progress-bar-animated bg-primary" 
                         role="progressbar" style="width: 0%">0%</div>
                </div>
            </div>

            <div id="results" style="display:none;">
                <div class="card p-3">
                    <h5 class="mb-3">Optimization Results</h5>
                    <table class="table table-dark table-hover">
                        <thead>
                            <tr>
                                <th>Timeframe</th>
                                <th>Best SMA Periods</th>
                                <th>Best Weights (a, b, c)</th>
                                <th>Sharpe Ratio</th>
                                <th>Total Return (%)</th>
                            </tr>
                        </thead>
                        <tbody id="resBody"></tbody>
                    </table>
                </div>
                <div class="card p-2 text-center">
                    <img id="perfPlot" style="max-width:100%">
                </div>
            </div>
        </div>

        <script>
            function pollStatus() {
                fetch('/status')
                .then(response => response.json())
                .then(data => {
                    // Update Progress Bar
                    const progressBar = document.getElementById('progressBar');
                    const statusText = document.getElementById('statusText');
                    
                    progressBar.style.width = data.progress + '%';
                    progressBar.textContent = data.progress + '%';
                    statusText.textContent = data.status;

                    if (data.error) {
                        statusText.textContent = "Error: " + data.error;
                        progressBar.classList.add('bg-danger');
                        return; // Stop polling on error
                    }

                    if (data.done) {
                        // Show Results
                        document.getElementById('progressSection').style.display = 'none';
                        const results = document.getElementById('results');
                        const tbody = document.getElementById('resBody');
                        
                        if (results.style.display === 'none') {
                            tbody.innerHTML = '';
                            data.results.forEach(r => {
                                tbody.innerHTML += `<tr>
                                    <td>${r.tf}</td>
                                    <td>${r.smas}</td>
                                    <td>${r.weights}</td>
                                    <td>${r.sharpe}</td>
                                    <td>${r.ret}%</td>
                                </tr>`;
                            });
                            document.getElementById('perfPlot').src = 'data:image/png;base64,' + data.plot;
                            results.style.display = 'block';
                        }
                    } else {
                        // Continue polling every second
                        setTimeout(pollStatus, 1000);
                    }
                })
                .catch(err => console.error(err));
            }

            // Start polling on page load
            window.onload = pollStatus;
        </script>
    </body>
    </html>
    """)

@app.route('/status')
def get_status():
    return jsonify(GLOBAL_STATE)

if __name__ == '__main__':
    # Start the optimization thread before the server
    print("Starting Background Optimization Thread...")
    t = threading.Thread(target=background_worker)
    t.daemon = True # Thread dies when main app dies
    t.start()
    
    # use_reloader=False prevents the script (and thus the thread) from running twice in debug mode
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
