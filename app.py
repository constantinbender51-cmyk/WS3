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
DAYS_BACK = 30           
LONG_DURATION_DAYS = 2   
SHORT_DURATION_DAYS = 5  
SYMBOL = 'BTC/USDT'
TIMEFRAME = '30m'
PORT = 8080

# -----------------------------------------------------------------------------
# 1. Data Fetching
# -----------------------------------------------------------------------------
def fetch_binance_data(symbol=SYMBOL, timeframe=TIMEFRAME, days_back=DAYS_BACK):
    print(f"Fetching {timeframe} data for {symbol} for the last {days_back} days...")
    exchange = ccxt.binance({'enableRateLimit': True})
    
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

def calculate_decay_weight(elapsed_steps, max_steps):
    """Calculates the decaying weight: 1 - (day/max_days)^2"""
    # Note: elapsed_steps / steps_per_day = days_elapsed
    days_elapsed = elapsed_steps / 48 # 48 intervals per day
    decay_days = max_steps / 48
    
    weight = 1 - (days_elapsed / decay_days)**2
    return np.maximum(weight, 0) # Ensure weight is not negative

def apply_strategy(df):
    print(f"Calculating strategy: Long={LONG_DURATION_DAYS}d, Short={SHORT_DURATION_DAYS}d (Mutually Exclusive)...")
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    # --- CROSSOVER SIGNALS ---
    long_signals = (df['rsi'] < 30) & (df['rsi'].shift(1) >= 30)
    short_signals = (df['rsi'] > 70) & (df['rsi'].shift(1) <= 70)
    
    n = len(df)
    intervals_per_day = 48
    
    long_steps_max = LONG_DURATION_DAYS * intervals_per_day
    short_steps_max = SHORT_DURATION_DAYS * intervals_per_day
    
    # Initialize position tracking arrays
    position_weights = np.zeros(n)
    long_timer = 0  # Tracks elapsed steps since last long signal/reset
    short_timer = 0 # Tracks elapsed steps since last short signal/reset

    # --- STATE MACHINE LOOP (Required for mutually exclusive logic) ---
    for i in range(1, n):
        # 1. Handle Crossover Signals (Forces reset and position flip)
        if long_signals.iloc[i]:
            long_timer = 1      # Start decay from 1.0 (elapsed is 1)
            short_timer = 0     # Force short position/decay to zero
        elif short_signals.iloc[i]:
            short_timer = 1     # Start decay from 1.0 (elapsed is 1)
            long_timer = 0      # Force long position/decay to zero
        else:
            # 2. If no new signal, increment active timer
            if long_timer > 0:
                long_timer += 1
            if short_timer > 0:
                short_timer += 1

        # 3. Calculate Weight (Weight is 0 if timer is 0)
        long_weight = 0
        if long_timer > 0 and long_timer <= long_steps_max:
            long_weight = calculate_decay_weight(long_timer, long_steps_max)

        short_weight = 0
        if short_timer > 0 and short_timer <= short_steps_max:
            short_weight = calculate_decay_weight(short_timer, short_steps_max)

        # 4. Set Net Position
        # If both are somehow active (shouldn't happen with the resets above, but safe to check)
        # we prioritize the latest signal or treat them as exclusive. 
        # Since the resets happen immediately, we just take the difference.
        position_weights[i] = long_weight - short_weight

    df['position'] = position_weights
    
    print(f"Long Crossovers found: {long_signals.sum()}")
    print(f"Short Crossovers found: {short_signals.sum()}")
    print(f"Active Position Candles: {np.count_nonzero(df['position'])}")
    
    # Returns & Equity
    df['returns'] = df['close'].pct_change().fillna(0)
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['strategy_returns'] = df['strategy_returns'].fillna(0)
    df['equity'] = (1 + df['strategy_returns']).cumprod() * 100
    
    return df

# -----------------------------------------------------------------------------
# 3. Event Study
# -----------------------------------------------------------------------------
def calculate_event_study(df):
    """Calculates avg price movement relative to crossover signals."""
    max_duration_steps = max(LONG_DURATION_DAYS, SHORT_DURATION_DAYS) * 48
    
    long_moves = []
    short_moves = []
    
    # Use Crossover Signals for analysis
    long_indices = np.where((df['rsi'] < 30) & (df['rsi'].shift(1) >= 30))[0]
    short_indices = np.where((df['rsi'] > 70) & (df['rsi'].shift(1) <= 70))[0]
    
    price_arr = df['close'].values
    n = len(price_arr)
    
    for idx in long_indices:
        if idx + max_duration_steps < n:
            slice_price = price_arr[idx : idx + max_duration_steps]
            start_price = slice_price[0] if slice_price[0] != 0 else 1.0
            long_moves.append((slice_price - start_price) / start_price)
            
    for idx in short_indices:
        if idx + max_duration_steps < n:
            slice_price = price_arr[idx : idx + max_duration_steps]
            start_price = slice_price[0] if slice_price[0] != 0 else 1.0
            short_moves.append((slice_price - start_price) / start_price)
            
    avg_long = np.mean(long_moves, axis=0) if long_moves else np.zeros(max_duration_steps)
    avg_short = np.mean(short_moves, axis=0) if short_moves else np.zeros(max_duration_steps)
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
    
    # Background Shading based on ACTIVE position
    trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
    is_long = df['position'] > 0.05
    is_short = df['position'] < -0.05
    
    ax1.fill_between(df.index, 0, 1, where=is_long, transform=trans, color='green', alpha=0.1, linewidth=0)
    ax1.fill_between(df.index, 0, 1, where=is_short, transform=trans, color='red', alpha=0.1, linewidth=0)

    title_str = f'{SYMBOL} Strategy: Long {LONG_DURATION_DAYS}d, Short {SHORT_DURATION_DAYS}d (Mutually Exclusive)'
    ax1.set_title(title_str, fontsize=14)

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
    # Position weight decay visualization
    ax3.fill_between(df.index, df['position'], 0, where=(df['position']>=0), color='green', alpha=0.6, step='mid', label='Long Weight')
    ax3.fill_between(df.index, df['position'], 0, where=(df['position']<0), color='red', alpha=0.6, step='mid', label='Short Weight')
    
    ax3.set_ylabel('Position Weight')
    ax3.set_xlabel('Date')
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.legend(loc='upper right')

    fig.align_ylabels(axes)
    return fig

def create_event_plot(avg_long, avg_short):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 6))
    
    max_days = max(LONG_DURATION_DAYS, SHORT_DURATION_DAYS)
    days = np.arange(len(avg_long)) / 48 
    
    ax.plot(days, avg_long * 100, label=f'Long Signal (2d decay)', color='green', linewidth=2)
    ax.plot(days, avg_short * 100, label=f'Short Signal (5d decay)', color='red', linewidth=2)
    
    ax.axvline(LONG_DURATION_DAYS, color='green', linestyle=':', label='Long Exit')
    ax.axvline(SHORT_DURATION_DAYS, color='red', linestyle=':', label='Short Exit')
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_title(f'Avg Price Move Post-Crossover Signal (Up to {max_days} Days)', fontsize=14)
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
    df = apply_strategy(df)
    avg_long_path, avg_short_path = calculate_event_study(df)
else:
    print("CRITICAL ERROR: No data fetched. Dashboard will be empty.")
    df = pd.DataFrame({'close': [], 'equity': [], 'rsi': [], 'position': []})
    max_steps = max(LONG_DURATION_DAYS, SHORT_DURATION_DAYS) * 48
    avg_long_path, avg_short_path = np.zeros(max_steps), np.zeros(max_steps)

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
