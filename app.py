import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from dash import Dash, dcc, html
import datetime
import time

# --- Configuration ---
START_YEAR = 2018 # STARTING DATA FETCH FROM 2018
SYMBOL = 'BTC/USDT'
TIMEFRAME = '30m'
PORT = 8080

# -----------------------------------------------------------------------------
# 1. Data Fetching
# -----------------------------------------------------------------------------
def fetch_binance_data(symbol=SYMBOL, timeframe=TIMEFRAME, start_year=START_YEAR):
    print(f"Executing heavy data fetch: {timeframe} for {symbol} starting from {start_year}...")
    print("WARNING: This pre-loading step is slow and risks deployment timeouts.")
    exchange = ccxt.binance({'enableRateLimit': True})
    
    start_date = datetime.datetime(start_year, 1, 1)
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
            
            if len(all_ohlcv) % 10000 == 0:
                current_date = datetime.datetime.fromtimestamp(last_timestamp / 1000)
                print(f"Fetched {len(all_ohlcv)} candles up to {current_date}...")
            
            # Break if we are within 1 minute of current time
            if last_timestamp > (time.time() * 1000) - 60000:
                break
                
            time.sleep(0.1) # Be respectful of API rate limits
            
        except Exception as e:
            print(f"\nError fetching data: {e}. Stopping fetch.")
            break
            
    print(f"\nData fetch complete. Total candles: {len(all_ohlcv)}")
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# -----------------------------------------------------------------------------
# 2. Strategy Logic 
# -----------------------------------------------------------------------------
def calculate_rsi(series, period=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def apply_strategy(df):
    """
    Applies the strategy logic: 
    1. RSI Signal generation (Long < 30, Short > 70).
    2. Weighted position calculation (1 - (day/30)^2 decay).
    3. Equity calculation.
    """
    print("Calculating strategy...")
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    long_signals = df['rsi'] < 30
    short_signals = df['rsi'] > 70
    
    n = len(df)
    net_position = np.zeros(n)
    
    # 30 days * 24 hours * 2 (30m intervals) = 1440 intervals
    days_duration = 30
    intervals_per_day = 48
    total_steps = days_duration * intervals_per_day
    
    # Position Weight Curve: 1 - (day/30)^2
    steps = np.arange(total_steps)
    days_elapsed = steps / intervals_per_day
    weights = 1 - (days_elapsed / days_duration)**2
    weights = np.maximum(weights, 0) 
    
    # Longs are positive exposure
    long_indices = np.where(long_signals)[0]
    for idx in long_indices:
        end_idx = min(idx + total_steps, n)
        length = end_idx - idx
        net_position[idx:end_idx] += weights[:length]

    # Shorts are negative exposure
    short_indices = np.where(short_signals)[0]
    for idx in short_indices:
        end_idx = min(idx + total_steps, n)
        length = end_idx - idx
        net_position[idx:end_idx] -= weights[:length]
        
    df['position'] = net_position
    
    # Calculate Equity
    df['returns'] = df['close'].pct_change().fillna(0)
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['strategy_returns'] = df['strategy_returns'].fillna(0)
    
    df['equity'] = (1 + df['strategy_returns']).cumprod() * 100
    
    return df, total_steps

# -----------------------------------------------------------------------------
# 3. Event Study Analysis 
# -----------------------------------------------------------------------------
def calculate_event_study(df, period_steps):
    """Calculates the average price movement path following signals."""
    long_moves = []
    short_moves = []
    
    long_indices = np.where(df['rsi'] < 30)[0]
    short_indices = np.where(df['rsi'] > 70)[0]
    
    price_arr = df['close'].values
    n = len(price_arr)
    
    # Path for long signals
    for idx in long_indices:
        if idx + period_steps < n:
            slice_price = price_arr[idx : idx + period_steps]
            norm_move = (slice_price - slice_price[0]) / slice_price[0]
            long_moves.append(norm_move)
            
    # Path for short signals
    for idx in short_indices:
        if idx + period_steps < n:
            slice_price = price_arr[idx : idx + period_steps]
            norm_move = (slice_price - slice_price[0]) / slice_price[0]
            short_moves.append(norm_move)
            
    avg_long = np.mean(long_moves, axis=0) if long_moves else np.zeros(period_steps)
    avg_short = np.mean(short_moves, axis=0) if short_moves else np.zeros(period_steps)
    
    return avg_long, avg_short

# -----------------------------------------------------------------------------
# 4. Matplotlib Plotting Functions
# -----------------------------------------------------------------------------

def plot_to_base64(fig):
    """Saves a Matplotlib figure to a PNG buffer and encodes it as Base64 for web display."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) 
    return f'data:image/png;base64,{img_base64}'


def create_main_plot(df):
    """Creates the main 3-row strategy plot using Matplotlib."""
    plt.style.use('ggplot') 
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 12), sharex=True, 
                             gridspec_kw={'height_ratios': [2, 1, 1], 'hspace': 0.1})
    
    # --- Row 1: Price and Equity ---
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='BTC Price', color='#333333', linewidth=1)
    ax1.set_ylabel('Price (USDT)', color='#333333')

    ax1b = ax1.twinx()
    ax1b.plot(df.index, df['equity'], label='Strategy Equity', color='#0077B6', linewidth=2, alpha=0.8)
    ax1b.set_ylabel('Normalized Equity (100=Start)', color='#0077B6')
    
    # Background color based on net position
    pos_mask = df['position'].apply(lambda x: 1 if x > 0.1 else (-1 if x < -0.1 else 0))
    long_regions = pos_mask[pos_mask == 1].index
    short_regions = pos_mask[pos_mask == -1].index
    
    # Shading the background for long/short signals
    # We iterate over the index points to apply the vertical shading
    df_temp = df.copy()
    df_temp['long_color'] = np.where(df_temp['position'] > 0.1, 1, 0)
    df_temp['short_color'] = np.where(df_temp['position'] < -0.1, 1, 0)

    for i in range(len(df_temp) - 1):
        start_time = df_temp.index[i]
        end_time = df_temp.index[i+1]
        
        if df_temp['long_color'].iloc[i] == 1:
            ax1.axvspan(start_time, end_time, color='green', alpha=0.1, zorder=0)
        elif df_temp['short_color'].iloc[i] == 1:
            ax1.axvspan(start_time, end_time, color='red', alpha=0.1, zorder=0)

    ax1.set_title(f'{SYMBOL} 30m Strategy Performance ({START_YEAR}-Present)', fontsize=16)

    # --- Row 2: RSI ---
    ax2 = axes[1]
    ax2.plot(df.index, df['rsi'], label='RSI (14)', color='#724C9F', linewidth=1)
    ax2.axhline(70, color='red', linestyle='-', alpha=0.9, label='Sell Threshold')
    ax2.axhline(30, color='green', linestyle='-', alpha=0.9, label='Buy Threshold')
    ax2.fill_between(df.index, 70, 100, color='red', alpha=0.15)
    ax2.fill_between(df.index, 0, 30, color='green', alpha=0.15)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI')
    ax2.legend(loc='lower left')

    # --- Row 3: Net Position ---
    ax3 = axes[2]
    colors = np.where(df['position'] >= 0, 'green', 'red')
    ax3.bar(df.index, df['position'], width=pd.Timedelta(TIMEFRAME) * 0.8, color=colors, alpha=0.7)
    ax3.set_ylabel('Net Position Size')
    ax3.set_xlabel('Date')
    ax3.axhline(0, color='black', linewidth=0.5)

    for ax in axes:
        ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    return fig


def create_event_plot(avg_long_path, avg_short_path, trade_duration_steps):
    """Creates the Event Study plot."""
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Convert steps (30m intervals) to days
    days_axis = np.arange(len(avg_long_path)) / 48 

    ax.plot(days_axis, avg_long_path * 100, label='Avg Post-Long Move', color='green', linewidth=3)
    ax.plot(days_axis, avg_short_path * 100, label='Avg Post-Short Move', color='red', linewidth=3)
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.6)
    ax.set_title('Average Price Movement (30 Days Post-Signal)', fontsize=16)
    ax.set_xlabel('Days After Signal')
    ax.set_ylabel('Average Cumulative Price Change (%)')
    ax.legend(title="Signal Type")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# 5. EXECUTION: RUNS ONCE WHEN MODULE IS IMPORTED (PRE-SERVER START)
# -----------------------------------------------------------------------------

# 1. Fetch and Process Data (This block runs before Gunicorn starts serving)
df = fetch_binance_data() 
df, trade_duration_steps = apply_strategy(df)
avg_long_path, avg_short_path = calculate_event_study(df, trade_duration_steps)

# 2. Generate and Encode Plots (Matplotlib needs to generate the images now)
print("Generating and encoding Matplotlib plots...")
main_plot_fig = create_main_plot(df)
main_plot_base64 = plot_to_base64(main_plot_fig)

event_plot_fig = create_event_plot(avg_long_path, avg_short_path, trade_duration_steps)
event_plot_base64 = plot_to_base64(event_plot_fig)


# 3. Dash Web Server Setup
app = Dash(__name__)

# CRITICAL FOR GUNICORN: Expose the Flask server object
server = app.server 

app.layout = html.Div(style={'fontFamily': 'Roboto, sans-serif', 'padding': '20px', 'backgroundColor': '#f0f2f5'}, children=[
    html.H1("RSI Trading Strategy Backtest (Matplotlib)", style={'textAlign': 'center', 'color': '#1f2937', 'marginBottom': '10px'}),
    html.P(f"Data Source: Binance {SYMBOL} {TIMEFRAME} | Since {START_YEAR} | Total Candles: {len(df)}", 
           style={'textAlign': 'center', 'color': '#4b5563', 'fontSize': '0.9rem', 'marginBottom': '30px'}),
    
    html.Div([
        html.H3("Strategy Performance & Indicators", style={'color': '#1f2937', 'borderBottom': '1px solid #e5e7eb', 'paddingBottom': '10px'}),
        html.P("Equity (Blue Line) is plotted against Price (Black Line). Position background: Green=Long, Red=Short.", style={'color': '#6b7280'}),
        html.Img(src=main_plot_base64, style={'width': '100%', 'height': 'auto', 'borderRadius': '8px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.08)', 'marginTop': '15px'})
    ], style={'marginBottom': '50px', 'padding': '25px', 'backgroundColor': 'white', 'borderRadius': '12px', 'boxShadow': '0 8px 16px rgba(0,0,0,0.1)'}),
    
    html.Div([
        html.H3("Event Study: 30-Day Price Trajectory", style={'color': '#1f2937', 'borderBottom': '1px solid #e5e7eb', 'paddingBottom': '10px'}),
        html.P("Average cumulative price change following entry signals, demonstrating average expected move over the trade duration.", style={'color': '#6b7280'}),
        html.Img(src=event_plot_base64, style={'width': '100%', 'height': 'auto', 'borderRadius': '8px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.08)', 'marginTop': '15px'})
    ], style={'padding': '25px', 'backgroundColor': 'white', 'borderRadius': '12px', 'boxShadow': '0 8px 16px rgba(0,0,0,0.1)'})
])

if __name__ == '__main__':
    # Local development run
    print(f"Starting Dash development server locally on port {PORT}...")
    app.run_server(debug=True, port=PORT)
