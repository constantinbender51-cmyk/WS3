import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
import datetime
import time

# -----------------------------------------------------------------------------
# 1. Data Fetching
# -----------------------------------------------------------------------------
def fetch_binance_data(symbol='BTC/USDT', timeframe='30m', start_year=2018):
    print(f"Fetching {timeframe} data for {symbol} starting from {start_year}...")
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Calculate start timestamp in milliseconds
    start_date = datetime.datetime(start_year, 1, 1)
    since = int(start_date.timestamp() * 1000)
    
    all_ohlcv = []
    limit = 1000  # Binance limit per request
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Update 'since' to the last timestamp + 1 timeframe duration
            # But safer to just take the last timestamp + 1ms to avoid overlap/gaps logic manually
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            
            # Progress indicator
            current_date = datetime.datetime.fromtimestamp(last_timestamp / 1000)
            print(f"Fetched up to {current_date}", end='\r')
            
            # Break if we reached the present (allow a small buffer)
            if last_timestamp > (time.time() * 1000) - 60000:
                break
                
            # Respect rate limits is handled by enableRateLimit=True, but a small sleep is safe
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
            
    print("\nData fetch complete.")
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# -----------------------------------------------------------------------------
# 2. Strategy Logic
# -----------------------------------------------------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # Fill initial NaNs
    return rsi.fillna(50)

def apply_strategy(df):
    print("Calculating strategy...")
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    # Identify Signals
    # Long: RSI < 30
    # Short: RSI > 70
    long_signals = df['rsi'] < 30
    short_signals = df['rsi'] > 70
    
    # Initialize Net Position Vector
    # We will sum overlapping trades. 
    # Length of df
    n = len(df)
    net_position = np.zeros(n)
    
    # Pre-compute the weight decay curve for 30 days
    # 30 days * 24 hours * 2 (30m intervals) = 1440 intervals
    days_duration = 30
    intervals_per_day = 48
    total_steps = days_duration * intervals_per_day
    
    # Curve: 1 - (day/30)^2
    # We create an array of steps 0 to total_steps
    steps = np.arange(total_steps)
    days_elapsed = steps / intervals_per_day
    weights = 1 - (days_elapsed / days_duration)**2
    weights = np.maximum(weights, 0) # Ensure no negative weights
    
    # Apply Long Signals
    # Get indices where long signals occur
    long_indices = np.where(long_signals)[0]
    for idx in long_indices:
        # Determine the slice range
        end_idx = min(idx + total_steps, n)
        length = end_idx - idx
        
        # Add weights to position
        net_position[idx:end_idx] += weights[:length]

    # Apply Short Signals (Negative Weights)
    short_indices = np.where(short_signals)[0]
    for idx in short_indices:
        end_idx = min(idx + total_steps, n)
        length = end_idx - idx
        
        # Subtract weights from position
        net_position[idx:end_idx] -= weights[:length]
        
    df['position'] = net_position
    
    # Calculate Equity
    # Price returns
    df['returns'] = df['close'].pct_change().fillna(0)
    
    # Strategy returns = Position(t-1) * Returns(t)
    # Shift position because we enter AFTER the signal is observed (or at the close of the signal candle)
    # Assuming execution at the close of the signal bar implies exposure starts next bar
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['strategy_returns'] = df['strategy_returns'].fillna(0)
    
    # Normalized Equity Curve (start at 100)
    df['equity'] = (1 + df['strategy_returns']).cumprod() * 100
    
    return df, total_steps

# -----------------------------------------------------------------------------
# 3. Event Study Analysis
# -----------------------------------------------------------------------------
def calculate_event_study(df, period_steps):
    """
    Calculates average price movement following Long and Short signals.
    """
    long_moves = []
    short_moves = []
    
    long_indices = np.where(df['rsi'] < 30)[0]
    short_indices = np.where(df['rsi'] > 70)[0]
    
    price_arr = df['close'].values
    n = len(price_arr)
    
    for idx in long_indices:
        if idx + period_steps < n:
            # Extract price slice
            slice_price = price_arr[idx : idx + period_steps]
            # Normalize to start at 0% change
            norm_move = (slice_price - slice_price[0]) / slice_price[0]
            long_moves.append(norm_move)
            
    for idx in short_indices:
        if idx + period_steps < n:
            slice_price = price_arr[idx : idx + period_steps]
            norm_move = (slice_price - slice_price[0]) / slice_price[0]
            short_moves.append(norm_move)
            
    avg_long = np.mean(long_moves, axis=0) if long_moves else np.zeros(period_steps)
    avg_short = np.mean(short_moves, axis=0) if short_moves else np.zeros(period_steps)
    
    return avg_long, avg_short

# -----------------------------------------------------------------------------
# 4. Main Execution & Server
# -----------------------------------------------------------------------------

# Fetch and Process
# NOTE: Fetching 5+ years of 30m data takes time. 
# For demonstration, if you want it faster, change start_year to 2023 or 2024.
df = fetch_binance_data(start_year=2018) 
df, trade_duration_steps = apply_strategy(df)
avg_long_path, avg_short_path = calculate_event_study(df, trade_duration_steps)

# Generate Plotly Figures
print("Generating Plots...")

# --- Main Dashboard Figure ---
fig_main = make_subplots(
    rows=3, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05,
    row_heights=[0.5, 0.25, 0.25],
    specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
)

# 1. Price & Equity (Row 1)
fig_main.add_trace(go.Scatter(x=df.index, y=df['close'], name='BTC Price', line=dict(color='black', width=1)), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df.index, y=df['equity'], name='Strategy Equity', line=dict(color='blue', width=2)), row=1, col=1, secondary_y=True)

# Add Background Colors for Position
# To optimize rendering, we won't draw a rect for every single bar. 
# We'll identify continuous regions where Position > 0 or Position < 0.
# However, since position is continuous (float), we simplify to just tinting regions.
# A more performant way for large datasets in Plotly is using a Heatmap or filled Scatter behind.
# We will use a boolean mask to create colored bands.

# Long Zones (Green)
df['pos_type'] = 0
df.loc[df['position'] > 0.1, 'pos_type'] = 1  # Threshold to avoid noise
df.loc[df['position'] < -0.1, 'pos_type'] = -1

# We can plot 'pos_type' as a filled area on a separate axis or just use Shapes.
# For 100k points, Shapes are heavy. Let's use a step chart that fills to zero on the Price chart background? 
# Actually, the user asked for "Price position(red green background)".
# The most efficient way for web is a Heatmap strip at the bottom or top, 
# but let's try to map it to the plot area. 
# We will use a separate trace with `fill='tozeroy'` scaled to the price axis limits? No, too complex.
# We will just plot the Net Position in Row 3 and color it.

# 2. RSI (Row 2)
fig_main.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')), row=2, col=1)
fig_main.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig_main.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

# 3. Net Position (Row 3)
# Color based on value
pos_colors = np.where(df['position'] >= 0, 'green', 'red')
fig_main.add_trace(go.Bar(x=df.index, y=df['position'], name='Net Position', marker_color=pos_colors), row=3, col=1)

fig_main.update_layout(
    title="Binance BTC/USDT 30m Strategy (2018-Present)",
    template="plotly_white",
    height=800,
    xaxis_rangeslider_visible=False
)
fig_main.update_yaxes(title_text="Price", row=1, col=1)
fig_main.update_yaxes(title_text="Equity", secondary_y=True, row=1, col=1)
fig_main.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
fig_main.update_yaxes(title_text="Position Size", row=3, col=1)


# --- Event Study Figure ---
days_axis = np.arange(len(avg_long_path)) / 48 # Convert steps back to days
fig_event = go.Figure()
fig_event.add_trace(go.Scatter(x=days_axis, y=avg_long_path * 100, name='Avg Post-Long Move', line=dict(color='green')))
fig_event.add_trace(go.Scatter(x=days_axis, y=avg_short_path * 100, name='Avg Post-Short Move', line=dict(color='red')))
fig_event.update_layout(
    title=f"Average Price Movement (30 Days Post-Signal)",
    xaxis_title="Days After Signal",
    yaxis_title="Price Change (%)",
    template="plotly_white"
)

# -----------------------------------------------------------------------------
# 5. Dash Web Server
# -----------------------------------------------------------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Algorithmic Trading Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.H3("Strategy Performance & Indicators"),
        dcc.Graph(figure=fig_main)
    ]),
    
    html.Div([
        html.H3("Event Study: 30-Day Price Trajectory"),
        html.P("Average cumulative return starting from the moment a signal is triggered."),
        dcc.Graph(figure=fig_event)
    ], style={'marginTop': '50px'})
])

if __name__ == '__main__':
    print("Starting server on port 8080...")
    app.run_server(debug=True, port=8080, use_reloader=False)
