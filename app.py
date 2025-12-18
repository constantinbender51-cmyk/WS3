import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from flask import Flask
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import itertools

# --- INITIALIZE FLASK ---
app = Flask(__name__)

# -----------------------------------------------------------------------------
# 1. Data Acquisition & Processing
# -----------------------------------------------------------------------------
def fetch_binance_data(symbol='BTC/USDT', timeframe='1d', since_year=2018):
    print(f"Fetching data for {symbol} since {since_year}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(f'{since_year}-01-01T00:00:00Z')
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(0.1)
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(1)
            continue

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

def calculate_indicators(df):
    df = df.copy()
    df['SMA_120'] = df['close'].rolling(window=120).mean()
    df['SMA_40'] = df['close'].rolling(window=40).mean()
    
    window = 14
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0.0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0.0)
    
    alpha = 1 / window
    df['tr_s'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
    df['plus_dm_s'] = df['plus_dm'].ewm(alpha=alpha, adjust=False).mean()
    df['minus_dm_s'] = df['minus_dm'].ewm(alpha=alpha, adjust=False).mean()
    
    df['plus_di'] = 100 * (df['plus_dm_s'] / df['tr_s'])
    df['minus_di'] = 100 * (df['minus_dm_s'] / df['tr_s'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()
    
    return df.dropna()

# -----------------------------------------------------------------------------
# 2. Strategy Engine
# -----------------------------------------------------------------------------
def run_strategy(df, sma_col, decay_adx_threshold, decay_k, decay_days, 
                 entry_mode='immediate', entry_threshold=0):
    """
    Runs the strategy with variable decay parameters and entry logic.
    
    entry_mode: 
      - 'immediate': Enter as soon as SMA is crossed (SMA 120 logic).
      - 'distance': Enter only when (Price - SMA)/SMA > entry_threshold (SMA 40 logic).
    """
    closes = df['close'].values
    smas = df[sma_col].values
    adxs = df['adx'].values
    positions = np.zeros(len(df))
    
    current_signal = 0 
    state = 'WAIT_FOR_ENTRY' 
    decay_start_idx = 0
    
    # Pre-calculate signal direction based on pure SMA cross
    raw_signals = np.where(closes > smas, 1, -1)
    current_direction = 0
    
    # Optimization constant
    if decay_days > 0:
        inv_decay_days = 1.0 / decay_days
    else:
        inv_decay_days = 0 
    
    for i in range(1, len(df)):
        new_direction = raw_signals[i]
        price = closes[i]
        sma = smas[i]
        adx = adxs[i]
        
        # 1. Detect Core SMA Crossover (Reset Event)
        if new_direction != current_direction:
            current_direction = new_direction
            state = 'WAIT_FOR_ENTRY'
            
        # 2. State Machine
        if state == 'WAIT_FOR_ENTRY':
            entered = False
            
            if entry_mode == 'immediate':
                entered = True
            elif entry_mode == 'distance':
                # Check distance percent
                # If Long (1): (Price - SMA)/SMA > threshold
                # If Short (-1): (SMA - Price)/SMA > threshold
                # Unified: direction * (Price - SMA) / SMA > threshold
                dist_pct = (price - sma) / sma
                if current_direction * dist_pct > entry_threshold:
                    entered = True
            
            if entered:
                state = 'NORMAL'
                positions[i] = current_direction
            else:
                positions[i] = 0.0
                
        elif state == 'NORMAL':
            if adx > decay_adx_threshold:
                state = 'DECAY'
                decay_start_idx = i
                # Assume Day 0 of decay is full position or immediate start?
                # Using immediate start: 1 - 0 = 1.0
                positions[i] = current_direction 
            else:
                positions[i] = current_direction
                
        elif state == 'DECAY':
            day_count = i - decay_start_idx
            if day_count < decay_days:
                decay_factor = 1.0 - (day_count * inv_decay_days) ** decay_k
                positions[i] = current_direction * decay_factor
            else:
                state = 'LOCKED'
                positions[i] = 0.0
                
        elif state == 'LOCKED':
            positions[i] = 0.0
            
    # Calculate Returns
    market_returns = df['close'].pct_change()
    strategy_returns = market_returns * pd.Series(positions).shift(1).fillna(0).values
    
    cum_ret = (1 + strategy_returns).cumprod()
    total_ret = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0
    
    mean_daily_ret = strategy_returns.mean()
    std_daily_ret = strategy_returns.std()
    
    if std_daily_ret > 1e-9:
        sharpe = (mean_daily_ret / std_daily_ret) * np.sqrt(365)
    else:
        sharpe = 0.0
    
    return {
        'positions': positions,
        'equity_curve': cum_ret,
        'total_return': total_ret,
        'sharpe': sharpe
    }

# -----------------------------------------------------------------------------
# 3. Heavy Grid Search
# -----------------------------------------------------------------------------
def grid_search_sma120(df):
    print("Starting Grid Search for SMA 120 (Immediate Entry)...")
    start_time = time.time()
    
    # Param Space
    x_values = range(20, 85, 5)              
    k_values = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    d_values = range(5, 155, 10)             
    
    best_sharpe = -np.inf
    best_params = {}
    
    param_grid = list(itertools.product(x_values, k_values, d_values))
    total_iter = len(param_grid)
    
    for x, k, d in param_grid:
        res = run_strategy(df, 'SMA_120', 
                           decay_adx_threshold=x, 
                           decay_k=k, 
                           decay_days=d, 
                           entry_mode='immediate',
                           entry_threshold=0)
        
        if res['sharpe'] > best_sharpe:
            best_sharpe = res['sharpe']
            best_params = {'x': x, 'k': k, 'd': d}
            
    print(f"SMA 120 Complete. {total_iter} combinations. Time: {time.time()-start_time:.2f}s")
    return best_params

def grid_search_sma40(df):
    print("Starting Grid Search for SMA 40 (Distance Entry)...")
    start_time = time.time()
    
    # Param Space
    # Distance: 0.1%, 0.5%, 1%, 2%, 3%, 5%
    dist_values = [0.001, 0.005, 0.010, 0.020, 0.030, 0.050] 
    
    x_values = range(20, 85, 5)             # Decay ADX
    k_values = [1.0, 2.0, 3.0, 5.0]         # Decay K
    d_values = range(10, 100, 10)           # Decay Days
    
    best_sharpe = -np.inf
    best_params = {}
    
    param_grid = list(itertools.product(dist_values, x_values, k_values, d_values))
    total_iter = len(param_grid)
    
    for dist, x, k, d in param_grid:
        res = run_strategy(df, 'SMA_40', 
                           decay_adx_threshold=x, 
                           decay_k=k, 
                           decay_days=d, 
                           entry_mode='distance',
                           entry_threshold=dist)
        
        if res['sharpe'] > best_sharpe:
            best_sharpe = res['sharpe']
            best_params = {'dist': dist, 'x': x, 'k': k, 'd': d}
            
    print(f"SMA 40 Complete. {total_iter} combinations. Time: {time.time()-start_time:.2f}s")
    return best_params

# -----------------------------------------------------------------------------
# 4. Web Application Routes
# -----------------------------------------------------------------------------
CACHE = {'df': None}

def get_data():
    if CACHE['df'] is None:
        raw_df = fetch_binance_data()
        CACHE['df'] = calculate_indicators(raw_df)
    return CACHE['df']

@app.route('/')
def dashboard():
    df = get_data()
    
    # ---------------------------
    # Run Scenarios
    # ---------------------------
    
    # 1. Vanilla Benchmarks
    vanilla_120 = run_strategy(df, 'SMA_120', 999, 1, 10, 'immediate', 0)
    # Vanilla 40 also implies no distance check usually, or we can assume dist=0
    vanilla_40 = run_strategy(df, 'SMA_40', 999, 1, 10, 'distance', 0)
    
    # 2. Optimized SMA 120
    best_p_120 = grid_search_sma120(df)
    decay_120 = run_strategy(df, 'SMA_120', 
                             decay_adx_threshold=best_p_120['x'], 
                             decay_k=best_p_120['k'], 
                             decay_days=best_p_120['d'],
                             entry_mode='immediate',
                             entry_threshold=0)
    
    # 3. Optimized SMA 40 (Distance Logic)
    best_p_40 = grid_search_sma40(df)
    decay_40 = run_strategy(df, 'SMA_40', 
                            decay_adx_threshold=best_p_40['x'], 
                            decay_k=best_p_40['k'], 
                            decay_days=best_p_40['d'], 
                            entry_mode='distance',
                            entry_threshold=best_p_40['dist'])
    
    # ---------------------------
    # Plotting
    # ---------------------------
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=('SMA 120 Equity (Immediate Entry)', 'SMA 40 Equity (Distance Entry)'),
                        vertical_spacing=0.15)
    
    # SMA 120
    fig.add_trace(go.Scatter(x=df.index, y=vanilla_120['equity_curve'], 
                             name=f'Vanilla 120 (Sharpe: {vanilla_120["sharpe"]:.2f})', 
                             line=dict(color='gray', width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=decay_120['equity_curve'], 
                             name=f'Opt Decay 120 (Sharpe: {decay_120["sharpe"]:.2f})', 
                             line=dict(color='blue', width=2)), 1, 1)
    
    # SMA 40
    fig.add_trace(go.Scatter(x=df.index, y=vanilla_40['equity_curve'], 
                             name=f'Vanilla 40 (Sharpe: {vanilla_40["sharpe"]:.2f})', 
                             line=dict(color='silver', width=1)), 2, 1)
    fig.add_trace(go.Scatter(x=df.index, y=decay_40['equity_curve'], 
                             name=f'Opt Decay 40 (Sharpe: {decay_40["sharpe"]:.2f})', 
                             line=dict(color='orange', width=2)), 2, 1)
    
    fig.update_layout(height=1000, template="plotly_white", margin=dict(t=80, b=50, l=50, r=50))

    # ---------------------------
    # HTML Output
    # ---------------------------
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Distance Entry Analysis</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; padding: 20px; background: #fafafa; color: #333; }}
            .header {{ margin-bottom: 20px; border-bottom: 2px solid #eee; padding-bottom: 20px; text-align: center; }}
            .card-container {{ display: flex; gap: 20px; margin-bottom: 30px; justify-content: center; flex-wrap: wrap; }}
            .card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); width: 400px; border-top: 4px solid #333; }}
            .metric-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; align-items: center; }}
            .metric-val {{ font-weight: bold; font-family: monospace; font-size: 1.1em; }}
            .highlight {{ color: #007bff; font-weight: bold; font-size: 1.4em; }}
            .param-box {{ background: #f0f4f8; padding: 15px; border-radius: 6px; margin-top: 15px; font-size: 0.95em; }}
            .param-item {{ display: flex; justify-content: space-between; margin-bottom: 6px; border-bottom: 1px dotted #ccc; padding-bottom: 2px; }}
            .param-item:last-child {{ border-bottom: none; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>BTC/USDT Strategy: Distance Entry Logic</h1>
            <p>Target: <strong>Max Sharpe Ratio</strong> | Data: Jan 2018 - Present</p>
        </div>
        
        <div class="card-container">
            <!-- SMA 120 Card -->
            <div class="card" style="border-top-color: blue;">
                <h3>SMA 120 (Immediate)</h3>
                
                <div class="metric-row">
                    <span>Optimized Sharpe:</span>
                    <span class="highlight">{decay_120['sharpe']:.3f}</span>
                </div>
                <div class="metric-row">
                    <span>Total Return:</span>
                    <span class="metric-val">{decay_120['total_return']*100:.1f}%</span>
                </div>

                <div class="param-box">
                    <div style="font-weight:bold; margin-bottom:10px; color:#444;">Optimal Parameters:</div>
                    <div class="param-item"><span>Entry Type:</span> <span>Immediate</span></div>
                    <div class="param-item"><span>Decay Start (x):</span> <span>{best_p_120['x']}</span></div>
                    <div class="param-item"><span>Decay Power (k):</span> <span>{best_p_120['k']}</span></div>
                    <div class="param-item"><span>Decay Days (d):</span> <span>{best_p_120['d']}</span></div>
                </div>
            </div>

            <!-- SMA 40 Card -->
            <div class="card" style="border-top-color: orange;">
                <h3>SMA 40 (Distance)</h3>
                
                <div class="metric-row">
                    <span>Optimized Sharpe:</span>
                    <span class="highlight">{decay_40['sharpe']:.3f}</span>
                </div>
                 <div class="metric-row">
                    <span>Total Return:</span>
                    <span class="metric-val">{decay_40['total_return']*100:.1f}%</span>
                </div>

                <div class="param-box">
                    <div style="font-weight:bold; margin-bottom:10px; color:#444;">Optimal Parameters:</div>
                    <div class="param-item"><span>Entry Distance:</span> <span>{best_p_40['dist']*100:.1f}%</span></div>
                    <div class="param-item"><span>Decay Start (x):</span> <span>{best_p_40['x']}</span></div>
                    <div class="param-item"><span>Decay Power (k):</span> <span>{best_p_40['k']}</span></div>
                    <div class="param-item"><span>Decay Days (d):</span> <span>{best_p_40['d']}</span></div>
                </div>
            </div>
        </div>

        <div style="background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); overflow: hidden; padding: 10px;">
            {fig.to_html(full_html=False, include_plotlyjs='cdn')}
        </div>
        
        <div style="text-align:center; margin-top: 20px; font-size: 12px; color: #888;">
            Grid Search Scanned Combinations: ~3000 (SMA120) + ~3000 (SMA40).
        </div>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
