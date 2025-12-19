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

# --- CONFIGURATION ---
SYMBOL = 'BTCUSDT'
START_YEAR = 2018
# For demonstration purposes, we use coarser steps to prevent execution timeout.
# In a production environment, change SMA_STEP to 10 and WEIGHT_STEP to 0.1
SMA_START = 20
SMA_END = 200 # Extended search would go to 400
SMA_STEP = 50 
WEIGHT_STEPS = [0.5, 1.0] # Weights to test

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
    
    print(f"Fetching data for {symbol} starting from {start_year}...")
    
    # Safety limit to prevent infinite loops during dev/testing (max 100 requests)
    req_count = 0
    while current_start < end_ts and req_count < 100:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'limit': limit
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if not data:
                break
                
            all_data.extend(data)
            current_start = data[-1][0] + 1
            req_count += 1
            
            # Respect API rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    df.set_index('open_time', inplace=True)
    return df[['close']]

# --- BACKTESTING ENGINE ---
def calculate_sharpe_ratio(returns, periods_per_year=24*365):
    if returns.std() == 0:
        return 0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)

def run_grid_search(df, timeframe_label):
    """
    Performs grid search for 3 SMAs and 3 Weights.
    """
    prices = df['close']
    returns = prices.pct_change().fillna(0)
    
    # 1. Pre-calculate all possible SMAs and their signals
    # Range of periods to test
    periods = list(range(SMA_START, SMA_END + 1, SMA_STEP))
    
    # Dictionary to store signal series: {period: series (+1/-1)}
    sma_signals = {}
    
    for p in periods:
        sma = prices.rolling(window=p).mean()
        # Signal: 1 if Price > SMA, -1 if Price < SMA
        sig = np.where(prices > sma, 1.0, -1.0)
        # Shift signal by 1 because we trade at Open based on Close of prev candle
        # (Approximation: Use prev close signal for next return)
        sma_signals[p] = pd.Series(sig, index=prices.index).shift(1).fillna(0)

    results = []
    
    # Generate weight combinations
    # We normalize weights later, so (1,1,1) is same as (0.5,0.5,0.5)
    # To reduce space, we can fix one weight or iterate simplified steps
    weight_opts = list(product(WEIGHT_STEPS, repeat=3))
    
    # Generate SMA period combinations (p1, p2, p3)
    # Only take combinations where p1 < p2 < p3 to avoid duplicates and self-crossing redundancy logic
    # (Though strategy allows distinct weights, usually distinct periods are desired)
    sma_opts = [p for p in product(periods, repeat=3) if p[0] < p[1] < p[2]]
    
    print(f"Grid Search {timeframe_label}: Testing {len(sma_opts) * len(weight_opts)} combinations...")

    best_sharpe = -999
    best_curve = None
    best_params = None

    # Vectorized loop could be better, but we iterate for clarity and memory management
    for p1, p2, p3 in sma_opts:
        s1 = sma_signals[p1]
        s2 = sma_signals[p2]
        s3 = sma_signals[p3]
        
        for w1, w2, w3 in weight_opts:
            total_weight = w1 + w2 + w3
            if total_weight == 0: continue
            
            # Composite Signal
            # Position size = (w1*s1 + w2*s2 + w3*s3) / sum(w)
            composite_pos = (w1*s1 + w2*s2 + w3*s3) / total_weight
            
            # Strategy Returns
            strat_returns = composite_pos * returns
            
            # Calculate Metrics
            # Adjust annualization factor based on timeframe
            if timeframe_label == '1H': factor = 365*24
            elif timeframe_label == '4H': factor = 365*6
            elif timeframe_label == '1D': factor = 365
            else: factor = 365
            
            sharpe = calculate_sharpe_ratio(strat_returns, factor)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {
                    'SMA_Periods': (p1, p2, p3),
                    'Weights': (w1, w2, w3),
                    'Sharpe': round(sharpe, 4),
                    'Total Return': round((np.cumprod(1 + strat_returns)[-1] - 1) * 100, 2)
                }
                best_curve = (1 + strat_returns).cumprod()

    return best_params, best_curve

# --- PLOTTING ---
def create_plot(df_1h, curve_1h, df_4h, curve_4h, df_1d, curve_1d):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Helper to plot
    def plot_ax(ax, base_df, strat_curve, title):
        base_curve = (1 + base_df['close'].pct_change().fillna(0)).cumprod()
        # Normalize to start at 1
        strat_curve = strat_curve / strat_curve.iloc[0]
        base_curve = base_curve / base_curve.iloc[0]
        
        ax.plot(base_curve.index, base_curve, label='Buy & Hold (BTC)', color='gray', alpha=0.5)
        ax.plot(strat_curve.index, strat_curve, label='Optimized Strategy', color='blue')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    if curve_1h is not None: plot_ax(ax1, df_1h, curve_1h, "1 Hour Timeframe")
    if curve_4h is not None: plot_ax(ax2, df_4h, curve_4h, "4 Hour Timeframe")
    if curve_1d is not None: plot_ax(ax3, df_1d, curve_1d, "1 Day Timeframe")

    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Triple SMA Grid Search</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #f8f9fa; padding-top: 20px; }
            .card { margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .spinner-border { display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center mb-4">Binance Triple SMA Optimizer</h1>
            
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Configuration</h5>
                    <p><strong>Strategy:</strong> Long if Price > SMA, Short if Price < SMA.</p>
                    <p><strong>Composition:</strong> 3 SMAs mixed by weight.</p>
                    <p><strong>Grid Search:</strong> Optimizing SMA periods ({SMA_START}-{SMA_END}) and Weights ({WEIGHT_STEPS}) for Sharpe Ratio.</p>
                    <button id="runBtn" class="btn btn-primary btn-lg w-100" onclick="runBacktest()">
                        <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                        Run Optimization
                    </button>
                </div>
            </div>

            <div id="resultsArea" style="display:none;">
                <div class="card">
                    <div class="card-header">Optimization Results</div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-bordered table-striped" id="resultsTable">
                                <thead class="table-dark">
                                    <tr>
                                        <th>Timeframe</th>
                                        <th>Best SMAs</th>
                                        <th>Best Weights</th>
                                        <th>Sharpe Ratio</th>
                                        <th>Total Return %</th>
                                    </tr>
                                </thead>
                                <tbody></tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-body text-center">
                        <img id="plotImage" class="img-fluid" alt="Strategy Performance">
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function runBacktest() {
                const btn = document.getElementById('runBtn');
                const spinner = btn.querySelector('.spinner-border');
                const resultsArea = document.getElementById('resultsArea');
                
                btn.disabled = true;
                spinner.style.display = 'inline-block';
                btn.childNodes[2].textContent = ' Processing... (This may take 10-20 seconds)';
                
                try {
                    const response = await fetch('/run_optimization');
                    const data = await response.json();
                    
                    if (data.error) {
                        alert("Error: " + data.error);
                        return;
                    }

                    // Populate Table
                    const tbody = document.querySelector('#resultsTable tbody');
                    tbody.innerHTML = '';
                    
                    data.results.forEach(row => {
                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td>${row.timeframe}</td>
                            <td>${row.smas}</td>
                            <td>${row.weights}</td>
                            <td>${row.sharpe}</td>
                            <td>${row.return}</td>
                        `;
                        tbody.appendChild(tr);
                    });

                    // Set Image
                    document.getElementById('plotImage').src = 'data:image/png;base64,' + data.plot;
                    resultsArea.style.display = 'block';
                    
                } catch (e) {
                    alert("Request failed: " + e);
                } finally {
                    btn.disabled = false;
                    spinner.style.display = 'none';
                    btn.childNodes[2].textContent = 'Run Optimization';
                }
            }
        </script>
    </body>
    </html>
    """.replace('{SMA_START}', str(SMA_START))
       .replace('{SMA_END}', str(SMA_END))
       .replace('{WEIGHT_STEPS}', str(WEIGHT_STEPS)))

@app.route('/run_optimization')
def run_optimization():
    try:
        # 1. Fetch 1H Data (Base)
        df_1h = fetch_binance_data(SYMBOL, '1h', START_YEAR)
        if df_1h.empty:
            return jsonify({'error': 'No data fetched from Binance'}), 500

        # 2. Resample for 4H and 1D
        # Resampling logic: Close is last close, Open is first open. 
        # But we only kept 'close' for simplicity in this demo.
        # Ideally we resample OHL CV, but for SMA on Close, resampling 'close' with .last() is sufficient.
        df_4h = df_1h.resample('4h').last().dropna()
        df_1d = df_1h.resample('1D').last().dropna()

        # 3. Run Grid Search for each
        # 1H
        res_1h, curve_1h = run_grid_search(df_1h, '1H')
        
        # 4H
        res_4h, curve_4h = run_grid_search(df_4h, '4H')
        
        # 1D
        res_1d, curve_1d = run_grid_search(df_1d, '1D')

        # 4. Generate Plot
        plot_b64 = create_plot(df_1h, curve_1h, df_4h, curve_4h, df_1d, curve_1d)

        # 5. Prepare JSON response
        results = [
            {
                'timeframe': '1 Hour',
                'smas': str(res_1h['SMA_Periods']),
                'weights': str(res_1h['Weights']),
                'sharpe': res_1h['Sharpe'],
                'return': res_1h['Total Return']
            },
            {
                'timeframe': '4 Hour',
                'smas': str(res_4h['SMA_Periods']),
                'weights': str(res_4h['Weights']),
                'sharpe': res_4h['Sharpe'],
                'return': res_4h['Total Return']
            },
            {
                'timeframe': '1 Day',
                'smas': str(res_1d['SMA_Periods']),
                'weights': str(res_1d['Weights']),
                'sharpe': res_1d['Sharpe'],
                'return': res_1d['Total Return']
            }
        ]

        return jsonify({'results': results, 'plot': plot_b64})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Web Server on Port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=True)
