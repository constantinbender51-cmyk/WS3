import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import io
import base64
from dash import Dash, dcc, html
import datetime
import time

# --- Configuration ---
DAYS_BACK = 30           # Data fetch duration (kept at 30 days for context)
DAYS_DURATION = 7        # **STRATEGY DURATION: 7 DAYS**
SYMBOL = 'BTC/USDT'
TIMEFRAME = '30m'
PORT = 8080

# -----------------------------------------------------------------------------
# 1. Data Fetching
# -----------------------------------------------------------------------------
def fetch_binance_data(symbol=SYMBOL, timeframe=TIMEFRAME, days_back=DAYS_BACK):
    print(f"Fetching {timeframe} data for {symbol} for the last {days_back} days...")
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Calculate start time dynamically
    start_date = datetime.datetime.now() - datetime.timedelta(days=days_back)
    since = int(start_date.timestamp() * 1000)
    
    all_ohlcv = []
    limit = 1000
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            
            if len(all_ohlcv) % 5000 == 0:
                print(f"Fetched {len(all_ohlcv)} candles...")
            
            if last_timestamp > (time.time() * 1000) - 60000:
                break
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
            
    print(f"\nData fetch complete. Total candles: {len(all_ohlcv)}")
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # CRITICAL: Ensure all numeric columns are floats
    cols = ['open', 'high', 'low', 'close', 'volume']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    
    return df

# -----------------------------------------------------------------------------
# 2. Strategy Logic
# -----------------------------------------------------------------------------
def calculate_rsi(series, period=14):
    """Calculates RSI using Wilder's Smoothing (EWMA)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)

def apply_strategy(df):
    print(f"Calculating strategy with {DAYS_DURATION}-day decay...")
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    long_signals = df['rsi'] < 30
    short_signals = df['rsi'] > 70
    
    print(f"Long Signals found: {long_signals.sum()}")
    print(f"Short Signals found: {short_signals.sum()}")
    
    n = len(df)
    net_position = np.zeros(n)
    
    intervals_per_day = 48 # 30m intervals
    # --- UPDATED: 7 days * 48 intervals/day = 336 steps ---
    total_steps = DAYS_DURATION * intervals_per_day
    
    steps = np.arange(total_steps)
    days_elapsed = steps / intervals_per_day
    
    # Weight decay: 1 - (day/DAYS_DURATION)^2
    weights = 1 - (days_elapsed / DAYS_DURATION)**2
    weights = np.maximum(weights, 0)
    
    # Apply Longs
    long_indices = np.where(long_signals)[0]
    for idx in long_indices:
        end_idx = min(idx + total_steps, n)
        length = end_idx - idx
        net_position[idx:end_idx] += weights[:length]

    # Apply Shorts
    short_indices = np.where(short_signals)[0]
    for idx in short_indices:
        end_idx = min(idx + total_steps, n)
        length = end_idx - idx
        net_position[idx:end_idx] -= weights[:length]
        
    df['position'] = net_position
    
    print(f"Non-zero position candles: {np.count_nonzero(net_position)}")
    
    # Returns & Equity
    df['returns'] = df['close'].pct_change().fillna(0)
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['strategy_returns'] = df['strategy_returns'].fillna(0)
    df['equity'] = (1 + df['strategy_returns']).cumprod() * 100
    
    return df, total_steps

# -----------------------------------------------------------------------------
# 3. Event Study
# -----------------------------------------------------------------------------
def calculate_event_study(df, period_steps):
    """Calculates the average price movement path over the 7-day period."""
    long_moves = []
    short_moves = []
    
    long_indices = np.where(df['rsi'] < 30)[0]
    short_indices = np.where(df['rsi'] > 70)[0]
    price_arr = df['close'].values
    n = len(price_arr)
    
    for idx in long_indices:
        # NOTE: period_steps is now 7 days, aligning with the trade duration.
        if idx + period_steps < n:
            slice_price = price_arr[idx : idx + period_steps]
            start_price = slice_price[0] if slice_price[0] != 0 else 1.0
            long_moves.append((slice_price - start_price) / start_price)
            
    for idx in short_indices:
        if idx + period_steps < n:
            slice_price = price_arr[idx : idx + period_steps]
            start_price = slice_price[0] if slice_price[0] != 0 else 1.0
            short_moves.append((slice_price - start_price) / start_price)
            
    avg_long = np.mean(long_moves, axis=0) if long_moves else np.zeros(period_steps)
    avg_short = np.mean(short_moves, axis=0) if short_moves else np.zeros(period_steps)
    return avg_long, avg_short

# -----------------------------------------------------------------------------
# 4. Optimized Matplotlib Functions
# -----------------------------------------------------------------------------
def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) 
    return f'data:image/png;base64,{img_base64}'

def create_main_plot(df):
    plt.style.use('ggplot') 
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 12), sharex=True, 
                             gridspec_kw={'height_ratios': [2, 1, 1], 'hspace': 0.1})
    
    # --- Row 1: Price and Equity ---
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='BTC Price', color='#333333', linewidth=1)
    ax1.set_ylabel('Price (USDT)', color='#333333')
    ax1.legend(loc='upper left')

    ax1b = ax1.twinx()
    ax1b.plot(df.index, df['equity'], label='Strategy Equity', color='#0077B6', linewidth=1.5)
    ax1b.set_ylabel('Equity', color='#0077B6')
    ax1b.legend(loc='upper right')
    
    # Optimized Background Shading
    trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
    is_long = df['position'] > 0.1
    is_short = df['position'] < -0.1
    
    ax1.fill_between(df.index, 0, 1, where=is_long, transform=trans, color='green', alpha=0.1, linewidth=0)
    ax1.fill_between(df.index, 0, 1, where=is_short, transform=trans, color='red', alpha=0.1, linewidth=0)

    ax1.set_title(f'{SYMBOL} 30m Strategy ({DAYS_DURATION}-Day Decay | Last {DAYS_BACK} Days)', fontsize=14)

    # --- Row 2: RSI ---
    ax2 = axes[1]
    ax2.plot(df.index, df['rsi'], color='#724C9F', linewidth=1)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.fill_between(df.index, 70, 100, color='red', alpha=0.1)
    ax2.fill_between(df.index, 0, 30, color='green', alpha=0.1)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI')

    # --- Row 3: Net Position ---
    ax3 = axes[2]
    # Fast Step Plot
    ax3.fill_between(df.index, df['position'], 0, where=(df['position']>=0), color='green', alpha=0.6, step='mid')
    ax3.fill_between(df.index, df['position'], 0, where=(df['position']<0), color='red', alpha=0.6, step='mid')
    
    ax3.set_ylabel('Net Position')
    ax3.set_xlabel('Date')
    ax3.axhline(0, color='black', linewidth=0.5)

    fig.align_ylabels(axes)
    return fig

def create_event_plot(avg_long, avg_short):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # X-axis now reflects the 7-day duration (336 steps / 48 steps/day)
    days = np.arange(len(avg_long)) / 48 
    ax.plot(days, avg_long * 100, label='Long', color='green', linewidth=2)
    ax.plot(days, avg_short * 100, label='Short', color='red', linewidth=2)
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_title(f'Avg Price Move ({DAYS_DURATION} Days Post-Signal)', fontsize=14)
    ax.set_xlabel('Days')
    ax.set_ylabel('% Change')
    ax.legend()
    return fig

# -----------------------------------------------------------------------------
# 5. Execution & Server
# -----------------------------------------------------------------------------
start_time = time.time()

# 1. Fetch
df = fetch_binance_data(days_back=DAYS_BACK)

# 2. Strategy
if not df.empty:
    df, trade_duration_steps = apply_strategy(df)
    avg_long_path, avg_short_path = calculate_event_study(df, trade_duration_steps)
else:
    print("CRITICAL ERROR: No data fetched. Dashboard will be empty.")
    df = pd.DataFrame({'close': [], 'equity': [], 'rsi': [], 'position': []})
    avg_long_path, avg_short_path = np.zeros(1), np.zeros(1)

# 3. Plots
print("Generating plots...")
try:
    if not df.empty:
        main_plot_fig = create_main_plot(df)
        main_plot_base64 = plot_to_base64(main_plot_fig)
        
        event_plot_fig = create_event_plot(avg_long_path, avg_short_path)
        event_plot_base64 = plot_to_base64(event_plot_fig)
    else:
        main_plot_base64 = ""
        event_plot_base64 = ""
except Exception as e:
    print(f"Plotting Error: {e}")
    main_plot_base64 = ""
    event_plot_base64 = ""

print(f"Initialization complete in {time.time() - start_time:.2f}s")

# 4. Server
app = Dash(__name__)
server = app.server 

app.layout = html.Div(style={'fontFamily': 'sans-serif', 'padding': '20px'}, children=[
    html.H1("RSI Strategy Backtest", style={'textAlign': 'center'}),
    html.Div([
        html.Img(src=main_plot_base64, style={'width': '100%', 'borderRadius': '8px'})
    ], style={'marginBottom': '20px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.1)'}),
    html.Div([
        html.Img(src=event_plot_base64, style={'width': '100%', 'borderRadius': '8px'})
    ], style={'boxShadow': '0 4px 12px rgba(0,0,0,0.1)'})
])

if __name__ == '__main__':
    app.run_server(debug=True, port=PORT)
