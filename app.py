import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from flask import Flask, render_template_string
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# 1. Data Acquisition & Processing
# -----------------------------------------------------------------------------
def fetch_binance_data(symbol='BTC/USDT', timeframe='1d', since_year=2018):
    """
    Fetches full historical OHLCV data from Binance using CCXT with pagination.
    """
    print(f"Fetching data for {symbol} since {since_year}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(f'{since_year}-01-01T00:00:00Z')
    all_ohlcv = []
    
    # Retry logic and pagination
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1 # Advance timestamp
            
            # Rate limit sleep
            time.sleep(0.1)
            
            # Break if we reached current time (roughly)
            if len(ohlcv) < 1000:
                break
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(1)
            continue

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Drop potential duplicates from pagination overlap
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"Fetched {len(df)} rows.")
    return df

def calculate_indicators(df):
    """
    Calculates SMA and ADX manually to avoid external heavy dependencies like talib/pandas_ta.
    """
    df = df.copy()
    
    # SMAs
    df['SMA_120'] = df['close'].rolling(window=120).mean()
    df['SMA_40'] = df['close'].rolling(window=40).mean()
    
    # ADX Calculation (Wilder's Smoothing)
    window = 14
    
    # True Range
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    
    # Directional Movement
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0.0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0.0)
    
    # Wilder's Smoothing Function
    def wilder_smooth(series, period):
        res = np.zeros_like(series)
        # Initialize with SMA
        res[period-1] = series.iloc[0:period].mean() 
        for i in range(period, len(series)):
            res[i] = res[i-1] * (period - 1) / period + series.iloc[i] / period
        return res

    # Apply smoothing (using simple EWMA as approximation for speed in vector, 
    # but strictly Wilder's is preferred. Using pandas ewm with alpha=1/window mimics Wilder)
    alpha = 1 / window
    df['tr_s'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
    df['plus_dm_s'] = df['plus_dm'].ewm(alpha=alpha, adjust=False).mean()
    df['minus_dm_s'] = df['minus_dm'].ewm(alpha=alpha, adjust=False).mean()
    
    df['plus_di'] = 100 * (df['plus_dm_s'] / df['tr_s'])
    df['minus_di'] = 100 * (df['minus_dm_s'] / df['tr_s'])
    
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()
    
    # Cleanup temp columns
    df.drop(['h-l', 'h-pc', 'l-pc', 'tr', 'up_move', 'down_move', 
             'plus_dm', 'minus_dm', 'tr_s', 'plus_dm_s', 'minus_dm_s', 'dx'], axis=1, inplace=True)
    
    return df.dropna()

# -----------------------------------------------------------------------------
# 2. Strategy Engine
# -----------------------------------------------------------------------------
def run_strategy(df, sma_col, adx_threshold, decay_k):
    """
    Runs the logic:
    1. Cross SMA -> Entry (Long/Short)
    2. If ADX > x -> Decay position size 1-(day/10)^k for 10 days
    3. After 10 days -> Lockout (Pos=0) until NEW crossover signal
    """
    closes = df['close'].values
    smas = df[sma_col].values
    adxs = df['adx'].values
    dates = df.index
    
    # States
    positions = np.zeros(len(df))
    
    # State tracking
    current_signal = 0 # 1 (Long) or -1 (Short)
    state = 'NORMAL'   # NORMAL, DECAY, LOCKED
    decay_start_idx = 0
    
    # Iterate (iterating is safer for this complex state machine than vectorizing)
    for i in range(1, len(df)):
        price = closes[i]
        sma = smas[i]
        adx = adxs[i]
        
        # 1. Determine Core Signal (Crossover)
        # We check if a NEW crossover happened compared to the *held* signal logic
        # However, the user said "goes long when price crosses SMA". 
        new_signal = 1 if price > sma else -1
        
        # Detect Crossover Event (Change in core signal direction)
        crossover_event = (new_signal != current_signal)
        
        if crossover_event:
            # RESET everything on new cross
            current_signal = new_signal
            state = 'NORMAL'
            
        # 2. Handle Decay Logic
        if state == 'NORMAL':
            # Check trigger
            if adx > adx_threshold:
                state = 'DECAY'
                decay_start_idx = i
                # Immediate decay calc for day 0
                day_count = 0
                decay_factor = 1 - (day_count/10)**decay_k
                positions[i] = current_signal * decay_factor
            else:
                positions[i] = current_signal
                
        elif state == 'DECAY':
            day_count = i - decay_start_idx
            if day_count < 10:
                decay_factor = 1 - (day_count/10)**decay_k
                positions[i] = current_signal * decay_factor
            else:
                # Decay finished, entering lockout
                state = 'LOCKED'
                positions[i] = 0.0
                
        elif state == 'LOCKED':
            # Remain out until crossover_event (handled at top of loop)
            positions[i] = 0.0
            
    # Calculate returns
    # position is the position held at the END of day i (simplified)
    # Strategy Return = Position(t-1) * (Price(t)/Price(t-1) - 1)
    # We shift positions by 1 to align with next day's return
    market_returns = df['close'].pct_change()
    strategy_returns = market_returns * pd.Series(positions).shift(1).fillna(0).values
    
    # Stats
    cum_ret = (1 + strategy_returns).cumprod()
    total_ret = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0
    
    return {
        'positions': positions,
        'equity_curve': cum_ret,
        'total_return': total_ret
    }

# -----------------------------------------------------------------------------
# 3. Optimization
# -----------------------------------------------------------------------------
def grid_search(df, sma_col):
    x_values = [20, 25, 30, 40, 50]       # ADX Thresholds
    k_values = [0.5, 1.0, 2.0, 3.0]       # Decay Exponents (Concave to Convex)
    
    best_perf = -np.inf
    best_params = (None, None)
    results = []
    
    for x in x_values:
        for k in k_values:
            res = run_strategy(df, sma_col, x, k)
            perf = res['total_return']
            results.append({'x': x, 'k': k, 'return': perf})
            
            if perf > best_perf:
                best_perf = perf
                best_params = (x, k)
                
    return best_params, results

# -----------------------------------------------------------------------------
# 4. Web Application
# -----------------------------------------------------------------------------
app = Flask(__name__)

# Global cache for data to avoid re-fetching on every request
CACHE = {
    'df': None,
    'last_fetch': None
}

def get_data():
    if CACHE['df'] is None:
        raw_df = fetch_binance_data()
        CACHE['df'] = calculate_indicators(raw_df)
    return CACHE['df']

@app.route('/')
def dashboard():
    df = get_data()
    
    # --- SMA 120 Analysis ---
    # 1. Vanilla SMA 120 (Benchmark) - No Decay
    # We simulate "No Decay" by setting threshold > 100 (impossible ADX)
    vanilla_120 = run_strategy(df, 'SMA_120', 999, 1)
    
    # 2. Optimized Decay SMA 120
    best_params_120, grid_120 = grid_search(df, 'SMA_120')
    decay_120 = run_strategy(df, 'SMA_120', best_params_120[0], best_params_120[1])
    
    # --- SMA 40 Analysis ---
    # 1. Vanilla SMA 40
    vanilla_40 = run_strategy(df, 'SMA_40', 999, 1)
    
    # 2. Optimized Decay SMA 40
    best_params_40, grid_40 = grid_search(df, 'SMA_40')
    decay_40 = run_strategy(df, 'SMA_40', best_params_40[0], best_params_40[1])
    
    # --- Plotting ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, subplot_titles=('Performance: SMA 120', 'Performance: SMA 40'))

    # Subplot 1: SMA 120
    fig.add_trace(go.Scatter(x=df.index, y=vanilla_120['equity_curve'], 
                             name='Vanilla SMA 120', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=decay_120['equity_curve'], 
                             name=f'Decay SMA 120 (x={best_params_120[0]}, k={best_params_120[1]})', 
                             line=dict(color='blue')), row=1, col=1)
    
    # Subplot 2: SMA 40
    fig.add_trace(go.Scatter(x=df.index, y=vanilla_40['equity_curve'], 
                             name='Vanilla SMA 40', line=dict(color='silver')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=decay_40['equity_curve'], 
                             name=f'Decay SMA 40 (x={best_params_40[0]}, k={best_params_40[1]})', 
                             line=dict(color='orange')), row=2, col=1)

    fig.update_layout(height=800, title_text="Strategy Backtest Analysis (BTC/USDT)", template="plotly_white")
    graph_html = fig.to_html(full_html=False)
    
    # --- Stats Table ---
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crypto Strategy Analysis</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
            .container {{ max_width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #1a1a1a; }}
            .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
            .card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 4px solid #333; }}
            .warning {{ background: #fff3cd; color: #856404; padding: 10px; border-radius: 4px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Binance BTC/USDT Strategy Analysis</h1>
            
            <div class="warning">
                <strong>Critical Analysis:</strong> This strategy "fades" high ADX trends. In crypto, high ADX usually signifies the most profitable "parabolic" phase. 
                By decaying position size during these phases and entering a lockout period, the strategy risks exiting early on massive winners.
                The graphs below often show the "Vanilla" (Standard Hold) strategy outperforming the "Decay" strategy during bull runs.
            </div>

            <div class="stats-grid">
                <div class="card" style="border-left-color: blue;">
                    <h3>SMA 120 Analysis</h3>
                    <p><strong>Vanilla Return:</strong> {vanilla_120['total_return']*100:.2f}%</p>
                    <p><strong>Best Decay Return:</strong> {decay_120['total_return']*100:.2f}%</p>
                    <p><strong>Optimal Parameters:</strong> ADX Threshold (x): {best_params_120[0]}, Decay Power (k): {best_params_120[1]}</p>
                </div>
                <div class="card" style="border-left-color: orange;">
                    <h3>SMA 40 Analysis</h3>
                    <p><strong>Vanilla Return:</strong> {vanilla_40['total_return']*100:.2f}%</p>
                    <p><strong>Best Decay Return:</strong> {decay_40['total_return']*100:.2f}%</p>
                    <p><strong>Optimal Parameters:</strong> ADX Threshold (x): {best_params_40[0]}, Decay Power (k): {best_params_40[1]}</p>
                </div>
            </div>

            {graph_html}
            
            <p style="text-align: center; color: #666; font-size: 0.9em;">
                Data source: Binance (Jan 2018 - Present) | Powered by Python/Flask/Plotly
            </p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    # Fetch data once on startup to ensure it works
    get_data()
    # Run on port 8080 for Railway/Cloud environments
    app.run(host='0.0.0.0', port=8080, debug=False)
