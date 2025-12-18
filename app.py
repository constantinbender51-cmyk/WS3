import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from flask import Flask
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

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
def run_strategy(df, sma_col, adx_threshold, decay_k):
    closes = df['close'].values
    smas = df[sma_col].values
    adxs = df['adx'].values
    positions = np.zeros(len(df))
    
    current_signal = 0 
    state = 'NORMAL'   
    decay_start_idx = 0
    
    for i in range(1, len(df)):
        price = closes[i]
        sma = smas[i]
        adx = adxs[i]
        
        # Core Signal
        new_signal = 1 if price > sma else -1
        crossover_event = (new_signal != current_signal)
        
        if crossover_event:
            current_signal = new_signal
            state = 'NORMAL'
            
        if state == 'NORMAL':
            if adx > adx_threshold:
                state = 'DECAY'
                decay_start_idx = i
                positions[i] = current_signal * (1 - (0/10)**decay_k) # Day 0 decay
            else:
                positions[i] = current_signal
        elif state == 'DECAY':
            day_count = i - decay_start_idx
            if day_count < 10:
                positions[i] = current_signal * (1 - (day_count/10)**decay_k)
            else:
                state = 'LOCKED'
                positions[i] = 0.0
        elif state == 'LOCKED':
            positions[i] = 0.0
            
    # Calculate Returns
    market_returns = df['close'].pct_change()
    # Strategy return depends on position held YESTERDAY
    strategy_returns = market_returns * pd.Series(positions).shift(1).fillna(0).values
    
    cum_ret = (1 + strategy_returns).cumprod()
    total_ret = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0
    
    # Calculate Sharpe Ratio (Annualized for Crypto 365 days)
    # Using 0% risk free rate
    mean_daily_ret = strategy_returns.mean()
    std_daily_ret = strategy_returns.std()
    
    if std_daily_ret > 0:
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
def grid_search(df, sma_col):
    x_values = [20, 25, 30, 40, 50]
    k_values = [0.5, 1.0, 2.0, 3.0]
    
    best_sharpe = -np.inf
    best_params = (None, None)
    
    for x in x_values:
        for k in k_values:
            res = run_strategy(df, sma_col, x, k)
            
            # Optimization Metric: Sharpe Ratio
            if res['sharpe'] > best_sharpe:
                best_sharpe = res['sharpe']
                best_params = (x, k)
                
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
    
    # Run Scenarios
    # 1. Vanilla (Benchmarks)
    vanilla_120 = run_strategy(df, 'SMA_120', 999, 1) # Threshold 999 = never decay
    vanilla_40 = run_strategy(df, 'SMA_40', 999, 1)
    
    # 2. Optimized Decay (Targeting Sharpe)
    best_p_120 = grid_search(df, 'SMA_120')
    decay_120 = run_strategy(df, 'SMA_120', best_p_120[0], best_p_120[1])
    
    best_p_40 = grid_search(df, 'SMA_40')
    decay_40 = run_strategy(df, 'SMA_40', best_p_40[0], best_p_40[1])
    
    # Subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=('SMA 120 Equity Curve', 'SMA 40 Equity Curve'))
    
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
    
    fig.update_layout(height=800, template="plotly_white", margin=dict(t=50, b=50, l=50, r=50))

    # Construct HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sharpe Optimization Analysis</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; padding: 40px; background: #fafafa; color: #333; }}
            .header {{ margin-bottom: 30px; border-bottom: 2px solid #eee; padding-bottom: 20px; }}
            .card-container {{ display: flex; gap: 20px; margin-bottom: 30px; }}
            .card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); flex: 1; border-top: 4px solid #333; }}
            .metric-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; }}
            .metric-val {{ font-weight: bold; }}
            .highlight {{ color: #007bff; font-weight: bold; font-size: 1.1em; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>BTC/USDT Strategy Analysis</h1>
            <p>Optimization Target: <strong>Sharpe Ratio (Annualized 365 days)</strong></p>
        </div>
        
        <div class="card-container">
            <!-- SMA 120 Card -->
            <div class="card" style="border-top-color: blue;">
                <h3>SMA 120 Strategy</h3>
                <div class="metric-row">
                    <span>Best Params:</span>
                    <span class="metric-val">x={best_p_120[0]}, k={best_p_120[1]}</span>
                </div>
                <hr style="border: 0; border-top: 1px solid #eee; margin: 10px 0;">
                <div class="metric-row">
                    <span>Decay Sharpe:</span>
                    <span class="highlight">{decay_120['sharpe']:.2f}</span>
                </div>
                <div class="metric-row">
                    <span>Vanilla Sharpe:</span>
                    <span class="metric-val">{vanilla_120['sharpe']:.2f}</span>
                </div>
                <div class="metric-row">
                    <span>Decay Return:</span>
                    <span class="metric-val">{decay_120['total_return']*100:.1f}%</span>
                </div>
            </div>

            <!-- SMA 40 Card -->
            <div class="card" style="border-top-color: orange;">
                <h3>SMA 40 Strategy</h3>
                <div class="metric-row">
                    <span>Best Params:</span>
                    <span class="metric-val">x={best_p_40[0]}, k={best_p_40[1]}</span>
                </div>
                <hr style="border: 0; border-top: 1px solid #eee; margin: 10px 0;">
                 <div class="metric-row">
                    <span>Decay Sharpe:</span>
                    <span class="highlight">{decay_40['sharpe']:.2f}</span>
                </div>
                <div class="metric-row">
                    <span>Vanilla Sharpe:</span>
                    <span class="metric-val">{vanilla_40['sharpe']:.2f}</span>
                </div>
                <div class="metric-row">
                    <span>Decay Return:</span>
                    <span class="metric-val">{decay_40['total_return']*100:.1f}%</span>
                </div>
            </div>
        </div>

        <div style="background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); overflow: hidden;">
            {fig.to_html(full_html=False, include_plotlyjs='cdn')}
        </div>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
