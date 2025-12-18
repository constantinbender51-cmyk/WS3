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
def run_strategy(df, sma_col, decay_adx_threshold, decay_k, decay_days, entry_adx_threshold=0):
    """
    Runs the strategy with variable decay parameters.
    
    sma_col: 'SMA_120' or 'SMA_40'
    decay_adx_threshold (x): Level where we start fading out.
    decay_k (k): Exponent for decay curve.
    decay_days (d): Duration of the decay.
    entry_adx_threshold (y): Minimum ADX required to ENTER a trade (used for SMA 40).
    """
    closes = df['close'].values
    smas = df[sma_col].values
    adxs = df['adx'].values
    positions = np.zeros(len(df))
    
    current_signal = 0 
    state = 'WAIT_FOR_ENTRY' # States: WAIT_FOR_ENTRY, NORMAL, DECAY, LOCKED
    decay_start_idx = 0
    
    # Pre-calculate signal direction to speed up loop slightly
    # 1 if Price > SMA, -1 if Price < SMA
    raw_signals = np.where(closes > smas, 1, -1)
    
    current_direction = 0
    
    for i in range(1, len(df)):
        new_direction = raw_signals[i]
        adx = adxs[i]
        
        # 1. Detect Core SMA Crossover (Reset Event)
        # "Only reenters on new signal" -> New SMA Cross resets everything
        if new_direction != current_direction:
            current_direction = new_direction
            # Reset state machine on crossover
            state = 'WAIT_FOR_ENTRY'
            
        # 2. State Machine
        if state == 'WAIT_FOR_ENTRY':
            # Check Entry Gate (y)
            # If entry_adx_threshold is 0 (SMA 120), this is always True immediately
            if adx > entry_adx_threshold:
                state = 'NORMAL'
                # Fall through to NORMAL logic immediately to check for instant decay trigger?
                # No, let's take position this bar.
                positions[i] = current_direction
            else:
                positions[i] = 0.0
                
        elif state == 'NORMAL':
            # Check Decay Trigger (x)
            if adx > decay_adx_threshold:
                state = 'DECAY'
                decay_start_idx = i
                # Immediate decay calc for day 0
                decay_factor = 1.0 - (0.0 / decay_days) ** decay_k
                positions[i] = current_direction * decay_factor
            else:
                positions[i] = current_direction
                
        elif state == 'DECAY':
            day_count = i - decay_start_idx
            if day_count < decay_days:
                decay_factor = 1.0 - (day_count / decay_days) ** decay_k
                positions[i] = current_direction * decay_factor
            else:
                # Decay finished, entering lockout
                state = 'LOCKED'
                positions[i] = 0.0
                
        elif state == 'LOCKED':
            # Remain out until SMA crossover event (handled at top of loop)
            positions[i] = 0.0
            
    # Calculate Returns
    market_returns = df['close'].pct_change()
    # Strategy return depends on position held YESTERDAY
    strategy_returns = market_returns * pd.Series(positions).shift(1).fillna(0).values
    
    cum_ret = (1 + strategy_returns).cumprod()
    total_ret = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0
    
    # Calculate Sharpe Ratio (Annualized for Crypto 365 days)
    # Using 0% risk free rate for simplicity
    mean_daily_ret = strategy_returns.mean()
    std_daily_ret = strategy_returns.std()
    
    if std_daily_ret > 1e-9: # Avoid division by zero
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
# 3. Optimization (Targeting Sharpe)
# -----------------------------------------------------------------------------
def grid_search_sma120(df):
    # SMA 120: Entry is immediate (y=0). Optimize x, k, d.
    x_values = [30, 40, 50, 60, 70]         # Decay Threshold (Start fading)
    k_values = [1.0, 2.0, 3.0]              # Decay Convexity
    d_values = [5, 10, 20, 30, 45, 60, 80, 100] # Decay Days
    
    best_sharpe = -np.inf
    best_params = {}
    
    # Itertools for cleaner nested loops
    param_grid = list(itertools.product(x_values, k_values, d_values))
    
    for x, k, d in param_grid:
        res = run_strategy(df, 'SMA_120', decay_adx_threshold=x, decay_k=k, decay_days=d, entry_adx_threshold=0)
        
        if res['sharpe'] > best_sharpe:
            best_sharpe = res['sharpe']
            best_params = {'x': x, 'k': k, 'd': d}
            
    return best_params

def grid_search_sma40(df):
    # SMA 40: Entry depends on y. Optimize y, x, k, d.
    # Constraints: x (Decay) should ideally be > y (Entry), otherwise we decay immediately upon entry.
    
    y_values = [15, 20, 25, 30]             # Entry Threshold
    x_values = [40, 50, 60, 70]             # Decay Threshold
    k_values = [1.0, 3.0]                   # Decay Convexity (Reduced set for speed)
    d_values = [10, 30, 60, 90]             # Decay Days (Reduced set for speed)
    
    best_sharpe = -np.inf
    best_params = {}
    
    param_grid = list(itertools.product(y_values, x_values, k_values, d_values))
    
    for y, x, k, d in param_grid:
        # Optimization: Don't test if Decay Trigger (x) <= Entry Trigger (y)
        # Because we would enter DECAY state immediately, skipping NORMAL.
        if x <= y:
            continue
            
        res = run_strategy(df, 'SMA_40', decay_adx_threshold=x, decay_k=k, decay_days=d, entry_adx_threshold=y)
        
        if res['sharpe'] > best_sharpe:
            best_sharpe = res['sharpe']
            best_params = {'y': y, 'x': x, 'k': k, 'd': d}
            
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
    
    # 1. Vanilla Benchmarks (No Decay, Immediate Entry)
    vanilla_120 = run_strategy(df, 'SMA_120', 999, 1, 10, 0)
    vanilla_40 = run_strategy(df, 'SMA_40', 999, 1, 10, 0)
    
    # 2. Optimized SMA 120
    # Search space: x, k, d
    best_p_120 = grid_search_sma120(df)
    decay_120 = run_strategy(df, 'SMA_120', 
                             decay_adx_threshold=best_p_120['x'], 
                             decay_k=best_p_120['k'], 
                             decay_days=best_p_120['d'],
                             entry_adx_threshold=0)
    
    # 3. Optimized SMA 40
    # Search space: y, x, k, d
    best_p_40 = grid_search_sma40(df)
    decay_40 = run_strategy(df, 'SMA_40', 
                            decay_adx_threshold=best_p_40['x'], 
                            decay_k=best_p_40['k'], 
                            decay_days=best_p_40['d'], 
                            entry_adx_threshold=best_p_40['y'])
    
    # ---------------------------
    # Plotting
    # ---------------------------
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=('SMA 120 Equity Curve', 'SMA 40 Equity Curve'),
                        vertical_spacing=0.15)
    
    # Plot SMA 120
    fig.add_trace(go.Scatter(x=df.index, y=vanilla_120['equity_curve'], 
                             name=f'Vanilla 120 (Sharpe: {vanilla_120["sharpe"]:.2f})', 
                             line=dict(color='gray', width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=decay_120['equity_curve'], 
                             name=f'Opt Decay 120 (Sharpe: {decay_120["sharpe"]:.2f})', 
                             line=dict(color='blue', width=2)), 1, 1)
    
    # Plot SMA 40
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
        <title>Fine-Grained Strategy Analysis</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; padding: 20px; background: #fafafa; color: #333; }}
            .header {{ margin-bottom: 20px; border-bottom: 2px solid #eee; padding-bottom: 20px; text-align: center; }}
            .card-container {{ display: flex; gap: 20px; margin-bottom: 30px; justify-content: center; flex-wrap: wrap; }}
            .card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); width: 350px; border-top: 4px solid #333; }}
            .metric-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; align-items: center; }}
            .metric-val {{ font-weight: bold; font-family: monospace; font-size: 1.1em; }}
            .highlight {{ color: #007bff; font-weight: bold; font-size: 1.2em; }}
            .param-box {{ background: #f0f4f8; padding: 10px; border-radius: 6px; margin-top: 10px; font-size: 0.9em; }}
            .param-item {{ display: flex; justify-content: space-between; margin-bottom: 4px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>BTC/USDT Fine-Grained Analysis</h1>
            <p>Target: <strong>Max Sharpe Ratio</strong> | Data: Jan 2018 - Present</p>
        </div>
        
        <div class="card-container">
            <!-- SMA 120 Card -->
            <div class="card" style="border-top-color: blue;">
                <h3>SMA 120 Strategy</h3>
                
                <div class="metric-row">
                    <span>Optimized Sharpe:</span>
                    <span class="highlight">{decay_120['sharpe']:.2f}</span>
                </div>
                <div class="metric-row">
                    <span>Vanilla Sharpe:</span>
                    <span class="metric-val" style="color: #666;">{vanilla_120['sharpe']:.2f}</span>
                </div>
                <div class="metric-row">
                    <span>Total Return:</span>
                    <span class="metric-val">{decay_120['total_return']*100:.1f}%</span>
                </div>

                <div class="param-box">
                    <div style="font-weight:bold; margin-bottom:5px;">Best Parameters:</div>
                    <div class="param-item"><span>Entry ADX (y):</span> <span>N/A (0)</span></div>
                    <div class="param-item"><span>Decay Start (x):</span> <span>{best_p_120['x']}</span></div>
                    <div class="param-item"><span>Decay Power (k):</span> <span>{best_p_120['k']}</span></div>
                    <div class="param-item"><span>Decay Days (d):</span> <span>{best_p_120['d']}</span></div>
                </div>
            </div>

            <!-- SMA 40 Card -->
            <div class="card" style="border-top-color: orange;">
                <h3>SMA 40 Strategy</h3>
                
                <div class="metric-row">
                    <span>Optimized Sharpe:</span>
                    <span class="highlight">{decay_40['sharpe']:.2f}</span>
                </div>
                <div class="metric-row">
                    <span>Vanilla Sharpe:</span>
                    <span class="metric-val" style="color: #666;">{vanilla_40['sharpe']:.2f}</span>
                </div>
                <div class="metric-row">
                    <span>Total Return:</span>
                    <span class="metric-val">{decay_40['total_return']*100:.1f}%</span>
                </div>

                <div class="param-box">
                    <div style="font-weight:bold; margin-bottom:5px;">Best Parameters:</div>
                    <div class="param-item"><span>Entry ADX (y):</span> <span>{best_p_40['y']}</span></div>
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
            Grid Search Complete. SMA120 params scanned: {len(list(itertools.product([1]*5, [1]*3, [1]*8)))} | 
            SMA40 params scanned: {len(list(itertools.product([1]*4, [1]*4, [1]*2, [1]*4)))} (approx)
        </div>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
