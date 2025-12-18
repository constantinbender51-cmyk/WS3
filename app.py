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
# 2. Strategy Engine (Chop Filter Logic)
# -----------------------------------------------------------------------------
def run_strategy(df, sma_col, 
                 thresh_low, thresh_high, decay_rate, 
                 entry_mode='immediate', entry_dist=0):
    """
    thresh_low: ADX below this triggers decay state.
    thresh_high (v): ADX above this restores full position.
    decay_rate (p): Daily percent reduction (0.05 = 5%).
    entry_mode: 'immediate' or 'distance'.
    entry_dist: % distance required to enter (0.01 = 1%).
    """
    closes = df['close'].values
    smas = df[sma_col].values
    adxs = df['adx'].values
    positions = np.zeros(len(df))
    
    # Pre-calculate pure SMA signal direction
    # 1 (Long), -1 (Short)
    raw_signals = np.where(closes > smas, 1, -1)
    
    current_direction = 0
    current_size = 1.0
    
    # State Machine: 'NORMAL' (Size 1.0) or 'DECAY' (Size reducing)
    # Hysteresis logic:
    # If in NORMAL and ADX < thresh_low -> switch to DECAY
    # If in DECAY and ADX > thresh_high -> switch to NORMAL
    state = 'NORMAL' 
    
    # Signal State (for entry logic)
    signal_state = 'WAIT_FOR_ENTRY' 
    
    for i in range(1, len(df)):
        new_raw_direction = raw_signals[i]
        price = closes[i]
        sma = smas[i]
        adx = adxs[i]
        
        # --- 1. Signal & Direction Logic ---
        
        # New SMA Cross Detected?
        if new_raw_direction != current_direction:
            current_direction = new_raw_direction
            signal_state = 'WAIT_FOR_ENTRY'
            # Reset sizing state on new trend attempt?
            # Usually yes. A new cross is a new attempt.
            state = 'NORMAL' 
            current_size = 1.0 
        
        # Check Entry Conditions
        in_market = False
        if signal_state == 'WAIT_FOR_ENTRY':
            if entry_mode == 'immediate':
                signal_state = 'IN_MARKET'
                in_market = True
            elif entry_mode == 'distance':
                # Check distance
                dist_pct = (price - sma) / sma
                # Direction * Dist > Threshold
                if current_direction * dist_pct > entry_dist:
                    signal_state = 'IN_MARKET'
                    in_market = True
        elif signal_state == 'IN_MARKET':
            in_market = True

        # --- 2. ADX Decay/Restore Logic (Chop Filter) ---
        
        if in_market:
            # Hysteresis State Machine
            if state == 'NORMAL':
                if adx < thresh_low:
                    state = 'DECAY'
                    # Apply first day decay immediately? Or wait? 
                    # Let's apply immediately to react fast to chop.
                    current_size = current_size * (1.0 - decay_rate)
                else:
                    current_size = 1.0
            
            elif state == 'DECAY':
                if adx > thresh_high:
                    state = 'NORMAL'
                    current_size = 1.0
                else:
                    # Continue decaying
                    current_size = current_size * (1.0 - decay_rate)
            
            # Cap size at 1.0 (safety) and floor at 0.0
            current_size = max(0.0, min(1.0, current_size))
            
            positions[i] = current_direction * current_size
        else:
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
# 3. Grid Search
# -----------------------------------------------------------------------------
def grid_search_sma120(df):
    print("Starting Grid Search for SMA 120 (Chop Filter)...")
    start_time = time.time()
    
    # Params:
    # low: when to start fading
    # high (v): when to restore
    # p: how fast to fade
    
    low_values = [10, 15, 20, 25, 30]
    high_values = [20, 25, 30, 40, 50]
    p_values = [0.01, 0.03, 0.05, 0.10, 0.20] # 1% to 20% daily decay
    
    best_sharpe = -np.inf
    best_params = {}
    
    param_grid = list(itertools.product(low_values, high_values, p_values))
    count = 0
    
    for l, h, p in param_grid:
        # Constraint: High threshold (restore) must be >= Low threshold (decay)
        if h <= l:
            continue
            
        count += 1
        res = run_strategy(df, 'SMA_120', 
                           thresh_low=l, 
                           thresh_high=h, 
                           decay_rate=p, 
                           entry_mode='immediate',
                           entry_dist=0)
        
        if res['sharpe'] > best_sharpe:
            best_sharpe = res['sharpe']
            best_params = {'low': l, 'high': h, 'p': p}
            
    print(f"SMA 120 Complete. {count} combinations. Time: {time.time()-start_time:.2f}s")
    return best_params

def grid_search_sma40(df):
    print("Starting Grid Search for SMA 40 (Chop Filter + Dist Entry)...")
    start_time = time.time()
    
    # Distance: 0.5% to 5%
    dist_values = [0.005, 0.01, 0.02, 0.03, 0.05]
    
    low_values = [10, 15, 20, 25]
    high_values = [20, 25, 30, 40]
    p_values = [0.01, 0.05, 0.10, 0.20]
    
    best_sharpe = -np.inf
    best_params = {}
    
    param_grid = list(itertools.product(dist_values, low_values, high_values, p_values))
    count = 0
    
    for d, l, h, p in param_grid:
        if h <= l:
            continue
            
        count += 1
        res = run_strategy(df, 'SMA_40', 
                           thresh_low=l, 
                           thresh_high=h, 
                           decay_rate=p, 
                           entry_mode='distance',
                           entry_dist=d)
        
        if res['sharpe'] > best_sharpe:
            best_sharpe = res['sharpe']
            best_params = {'dist': d, 'low': l, 'high': h, 'p': p}
            
    print(f"SMA 40 Complete. {count} combinations. Time: {time.time()-start_time:.2f}s")
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
    
    # 1. Vanilla Benchmarks (No Decay p=0, Immediate/Distance)
    vanilla_120 = run_strategy(df, 'SMA_120', 0, 0, 0, 'immediate', 0)
    vanilla_40 = run_strategy(df, 'SMA_40', 0, 0, 0, 'distance', 0) 
    # Note: Vanilla 40 usually doesn't have distance, but comparing apples to apples
    # if we want pure vanilla, dist=0. Let's do pure vanilla for baseline.
    vanilla_40_pure = run_strategy(df, 'SMA_40', 0, 0, 0, 'immediate', 0)
    
    # 2. Optimized SMA 120
    best_p_120 = grid_search_sma120(df)
    decay_120 = run_strategy(df, 'SMA_120', 
                             thresh_low=best_p_120['low'], 
                             thresh_high=best_p_120['high'], 
                             decay_rate=best_p_120['p'],
                             entry_mode='immediate',
                             entry_dist=0)
    
    # 3. Optimized SMA 40
    best_p_40 = grid_search_sma40(df)
    decay_40 = run_strategy(df, 'SMA_40', 
                            thresh_low=best_p_40['low'], 
                            thresh_high=best_p_40['high'], 
                            decay_rate=best_p_40['p'], 
                            entry_mode='distance',
                            entry_dist=best_p_40['dist'])
    
    # ---------------------------
    # Plotting
    # ---------------------------
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=('SMA 120 Equity (Low ADX Decay)', 'SMA 40 Equity (Dist Entry + Low ADX Decay)'),
                        vertical_spacing=0.15)
    
    # SMA 120
    fig.add_trace(go.Scatter(x=df.index, y=vanilla_120['equity_curve'], 
                             name=f'Vanilla 120 (Sharpe: {vanilla_120["sharpe"]:.2f})', 
                             line=dict(color='gray', width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=decay_120['equity_curve'], 
                             name=f'Opt Filter 120 (Sharpe: {decay_120["sharpe"]:.2f})', 
                             line=dict(color='blue', width=2)), 1, 1)
    
    # SMA 40
    fig.add_trace(go.Scatter(x=df.index, y=vanilla_40_pure['equity_curve'], 
                             name=f'Vanilla 40 (Sharpe: {vanilla_40_pure["sharpe"]:.2f})', 
                             line=dict(color='silver', width=1)), 2, 1)
    fig.add_trace(go.Scatter(x=df.index, y=decay_40['equity_curve'], 
                             name=f'Opt Filter 40 (Sharpe: {decay_40["sharpe"]:.2f})', 
                             line=dict(color='orange', width=2)), 2, 1)
    
    fig.update_layout(height=1000, template="plotly_white", margin=dict(t=80, b=50, l=50, r=50))

    # ---------------------------
    # HTML Output
    # ---------------------------
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chop Filter Analysis</title>
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
            <h1>BTC/USDT Strategy: Chop Filter Logic</h1>
            <p>Target: <strong>Max Sharpe Ratio</strong> | Data: Jan 2018 - Present</p>
            <p style="font-size: 0.9em; color: #666;">Logic: Decay position by <em>p</em>% daily when ADX < <em>Low</em>. Restore to 100% when ADX > <em>High</em>.</p>
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
                    <div class="param-item"><span>Decay Trigger (Low):</span> <span>{best_p_120['low']}</span></div>
                    <div class="param-item"><span>Restore Trigger (v):</span> <span>{best_p_120['high']}</span></div>
                    <div class="param-item"><span>Decay Rate (p):</span> <span>{best_p_120['p']*100:.1f}%</span></div>
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
                    <div class="param-item"><span>Decay Trigger (Low):</span> <span>{best_p_40['low']}</span></div>
                    <div class="param-item"><span>Restore Trigger (v):</span> <span>{best_p_40['high']}</span></div>
                    <div class="param-item"><span>Decay Rate (p):</span> <span>{best_p_40['p']*100:.1f}%</span></div>
                </div>
            </div>
        </div>

        <div style="background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); overflow: hidden; padding: 10px;">
            {fig.to_html(full_html=False, include_plotlyjs='cdn')}
        </div>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
