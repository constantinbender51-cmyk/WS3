import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string, request, jsonify
from itertools import product
from datetime import datetime
import time

app = Flask(__name__)

# --- CONFIGURATION (UPDATED PER REQUIREMENTS) ---
SYMBOL = 'BTCUSDT'
START_YEAR = 2018
SMA_START = 10
SMA_END = 400
SMA_STEP = 10 
# Weights are grid searched from 0 to 1. 
# We use a step of 0.2 to keep the search space manageable in a web environment.
WEIGHT_STEPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

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
    
    # We increase the request limit to capture full history since 2018
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
            time.sleep(0.05) # Tiny sleep for rate limits
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

def run_grid_search(df, timeframe_label):
    prices = df['close'].values
    returns = df['close'].pct_change().fillna(0).values
    
    periods = list(range(SMA_START, SMA_END + 1, SMA_STEP))
    
    # Pre-calculate signals (Vectorized)
    # signals_matrix[period_index] = array of +1/-1
    signals_matrix = []
    for p in periods:
        # Simple Moving Average
        sma = pd.Series(prices).rolling(window=p).mean().values
        sig = np.where(prices > sma, 1.0, -1.0)
        # Shift 1 to avoid lookahead bias
        sig = np.concatenate(([0], sig[:-1]))
        signals_matrix.append(sig)
    
    signals_matrix = np.array(signals_matrix) # Shape: (num_periods, num_bars)
    
    # Space generation
    weight_opts = list(product(WEIGHT_STEPS, repeat=3))
    # Filter weight_opts to remove all-zeros and redundant scales (0.5,0.5,0.5 is same as 1,1,1)
    weight_opts = [w for w in weight_opts if sum(w) > 0]
    
    # SMA Period Combinations
    num_p = len(periods)
    sma_idx_opts = [(i, j, k) for i, j, k in product(range(num_p), repeat=3) if i < j < k]
    
    best_sharpe = -999
    best_params = None
    best_signal = None

    # Determine annualization factor
    if timeframe_label == '1H': factor = 365*24
    elif timeframe_label == '4H': factor = 365*6
    else: factor = 365

    # Optimized search loop
    for i, j, k in sma_idx_opts:
        s1, s2, s3 = signals_matrix[i], signals_matrix[j], signals_matrix[k]
        p1, p2, p3 = periods[i], periods[j], periods[k]
        
        for w1, w2, w3 in weight_opts:
            total_w = w1 + w2 + w3
            # Combined position size based on weights and individual SMA signals
            # Requirements check: 120L, 400L, 40L -> size 1.0 | 120L, 400S, 40S -> size (1 - 0.33 - 0.33) / 3 ? 
            # Logic: Position = (w1*s1 + w2*s2 + w3*s3) / (w1 + w2 + w3)
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

    # Reconstruct best curve
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
            .btn-primary { background: #3d5afe; border: none; }
            .table { color: #e0e0e0; }
            .spinner-container { display: none; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 class="mb-4">Triple SMA Optimizer (2018 - Present)</h2>
            <div class="card p-4">
                <p>Search Space: 3 SMA Periods (10 to 400, step 10), 3 Weights (0.0 to 1.0). <br>
                Optimizing for <strong>Sharpe Ratio</strong> across 1H, 4H, and 1D timeframes.</p>
                <button id="runBtn" class="btn btn-primary" onclick="startProcess()">Start Deep Grid Search</button>
                <div id="loader" class="spinner-container text-center">
                    <div class="spinner-border text-primary" role="status"></div>
                    <p class="mt-2">Computing thousands of combinations... please wait.</p>
                </div>
            </div>

            <div id="results" style="display:none;">
                <div class="card p-3">
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
            async function startProcess() {
                const btn = document.getElementById('runBtn');
                const loader = document.getElementById('loader');
                const results = document.getElementById('results');
                
                btn.style.display = 'none';
                loader.style.display = 'block';
                results.style.display = 'none';

                try {
                    const resp = await fetch('/optimize');
                    const data = await resp.json();
                    
                    const tbody = document.getElementById('resBody');
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
                } catch (e) {
                    alert("Optimization failed. Check console.");
                } finally {
                    loader.style.display = 'none';
                    btn.style.display = 'block';
                }
            }
        </script>
    </body>
    </html>
    """)

@app.route('/optimize')
def optimize():
    # 1. Get Base Data (1H)
    df_1h = fetch_binance_data(SYMBOL, '1h', START_YEAR)
    
    # 2. Resample
    df_4h = df_1h.resample('4h').last().dropna()
    df_1d = df_1h.resample('1D').last().dropna()
    
    # 3. Run search
    res_1h, curve_1h = run_grid_search(df_1h, '1H')
    res_4h, curve_4h = run_grid_search(df_4h, '4H')
    res_1d, curve_1d = run_grid_search(df_1d, '1D')
    
    # 4. Create Plot
    plot_b64 = create_plot([
        ('1 Hour', df_1h, curve_1h),
        ('4 Hour', df_4h, curve_4h),
        ('1 Day', df_1d, curve_1d)
    ])
    
    # 5. Response
    output = {
        'results': [
            {'tf': '1H', 'smas': str(res_1h['smas']), 'weights': str(res_1h['weights']), 'sharpe': res_1h['sharpe'], 'ret': res_1h['return']},
            {'tf': '4H', 'smas': str(res_4h['smas']), 'weights': str(res_4h['weights']), 'sharpe': res_4h['sharpe'], 'ret': res_4h['return']},
            {'tf': '1D', 'smas': str(res_1d['smas']), 'weights': str(res_1d['weights']), 'sharpe': res_1d['sharpe'], 'ret': res_1d['return']}
        ],
        'plot': plot_b64
    }
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
