import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from flask import Flask, render_template_string
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- INITIALIZE FLASK ---
# Railway expects a variable 'app' or 'server' to bind to.
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
        
        new_signal = 1 if price > sma else -1
        crossover_event = (new_signal != current_signal)
        
        if crossover_event:
            current_signal = new_signal
            state = 'NORMAL'
            
        if state == 'NORMAL':
            if adx > adx_threshold:
                state = 'DECAY'
                decay_start_idx = i
                positions[i] = current_signal * (1 - (0/10)**decay_k)
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
            
    market_returns = df['close'].pct_change()
    strategy_returns = market_returns * pd.Series(positions).shift(1).fillna(0).values
    cum_ret = (1 + strategy_returns).cumprod()
    total_ret = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0
    
    return {'positions': positions, 'equity_curve': cum_ret, 'total_return': total_ret}

# -----------------------------------------------------------------------------
# 3. Optimization
# -----------------------------------------------------------------------------
def grid_search(df, sma_col):
    x_values = [20, 25, 30, 40, 50]
    k_values = [0.5, 1.0, 2.0, 3.0]
    best_perf = -np.inf
    best_params = (None, None)
    
    for x in x_values:
        for k in k_values:
            res = run_strategy(df, sma_col, x, k)
            if res['total_return'] > best_perf:
                best_perf = res['total_return']
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
    vanilla_120 = run_strategy(df, 'SMA_120', 999, 1)
    best_params_120 = grid_search(df, 'SMA_120')
    decay_120 = run_strategy(df, 'SMA_120', best_params_120[0], best_params_120[1])
    
    vanilla_40 = run_strategy(df, 'SMA_40', 999, 1)
    best_params_40 = grid_search(df, 'SMA_40')
    decay_40 = run_strategy(df, 'SMA_40', best_params_40[0], best_params_40[1])
    
    # Subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('SMA 120 Performance', 'SMA 40 Performance'))
    fig.add_trace(go.Scatter(x=df.index, y=vanilla_120['equity_curve'], name='Vanilla 120', line=dict(color='gray')), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=decay_120['equity_curve'], name=f'Decay 120 (x={best_params_120[0]}, k={best_params_120[1]})', line=dict(color='blue')), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=vanilla_40['equity_curve'], name='Vanilla 40', line=dict(color='silver')), 2, 1)
    fig.add_trace(go.Scatter(x=df.index, y=decay_40['equity_curve'], name=f'Decay 40 (x={best_params_40[0]}, k={best_params_40[1]})', line=dict(color='orange')), 2, 1)
    fig.update_layout(height=800, template="plotly_white")

    html = f"""
    <html>
    <body style="font-family: sans-serif; padding: 20px;">
        <h2>BTC Strategy Optimization</h2>
        <div style="display: flex; gap: 20px; margin-bottom: 20px;">
            <div style="padding: 15px; border: 1px solid #ccc; border-radius: 8px;">
                <strong>SMA 120:</strong> Vanilla: {vanilla_120['total_return']*100:.1f}% | Decay: {decay_120['total_return']*100:.1f}%
            </div>
            <div style="padding: 15px; border: 1px solid #ccc; border-radius: 8px;">
                <strong>SMA 40:</strong> Vanilla: {vanilla_40['total_return']*100:.1f}% | Decay: {decay_40['total_return']*100:.1f}%
            </div>
        </div>
        {fig.to_html(full_html=False)}
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    # When running locally or via Railway's start command
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
