import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
import datetime
import time
import os

# -----------------------------------------------------------------------------
# 1. Data Fetching
# -----------------------------------------------------------------------------
def fetch_binance_data(symbol='BTC/USDT', timeframe='30m', start_year=2018):
    print(f"Fetching {timeframe} data for {symbol} starting from {start_year}...")
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
            
            # Progress indicator (simple print to avoid log clutter on server)
            if len(all_ohlcv) % 10000 == 0:
                print(f"Fetched {len(all_ohlcv)} candles...")
            
            # Break if we are within 1 minute of current time
            if last_timestamp > (time.time() * 1000) - 60000:
                break
                
            time.sleep(0.1) # Respect rate limits
            
        except Exception as e:
            print(f"\nError fetching data: {e}")
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
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def apply_strategy(df):
    print("Calculating strategy...")
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    # Identify Signals
    long_signals = df['rsi'] < 30
    short_signals = df['rsi'] > 70
    
    n = len(df)
    net_position = np.zeros(n)
    
    # 30 days * 24 hours * 2 (30m intervals) = 1440 intervals
    days_duration = 30
    intervals_per_day = 48
    total_steps = days_duration * intervals_per_day
    
    # Curve: 1 - (day/30)^2
    steps = np.arange(total_steps)
    days_elapsed = steps / intervals_per_day
    weights = 1 - (days_elapsed / days_duration)**2
    weights = np.maximum(weights, 0)
    
    # Apply Long Signals
    long_indices = np.where(long_signals)[0]
    for idx in long_indices:
        end_idx = min(idx + total_steps, n)
        length = end_idx - idx
        net_position[idx:end_idx] += weights[:length]

    # Apply Short Signals
    short_indices = np.where(short_signals)[0]
    for idx in short_indices:
        end_idx = min(idx + total_steps, n)
        length = end_idx - idx
        net_position[idx:end_idx] -= weights[:length]
        
    df['position'] = net_position
    
    # Strategy returns = Position(t-1) * Returns(t)
    df['returns'] = df['close'].pct_change().fillna(0)
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['strategy_returns'] = df['strategy_returns'].fillna(0)
    
    df['equity'] = (1 + df['strategy_returns']).cumprod() * 100
    
    return df, total_steps

# -----------------------------------------------------------------------------
# 3. Event Study Analysis
# -----------------------------------------------------------------------------
def calculate_event_study(df, period_steps):
    long_moves = []
    short_moves = []
    
    long_indices = np.where(df['rsi'] < 30)[0]
    short_indices = np.where(df['rsi'] > 70)[0]
    
    price_arr = df['close'].values
    n = len(price_arr)
    
    for idx in long_indices:
        if idx + period_steps < n:
            slice_price = price_arr[idx : idx + period_steps]
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
# 4. Generate Figures
# -----------------------------------------------------------------------------
# For Railway, we should fetch data only once on startup or use a simpler dataset
# to prevent timeouts during deployment. 
# Here we run it; if it takes too long, Railway might kill the health check.
# Consider reducing start_year to 2023 for faster boot in production.
df = fetch_binance_data(start_year=2023) 
df, trade_duration_steps = apply_strategy(df)
avg_long_path, avg_short_path = calculate_event_study(df, trade_duration_steps)

print("Generating Plots...")

# Main Figure
fig_main = make_subplots(
    rows=3, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05,
    row_heights=[0.5, 0.25, 0.25],
    specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
)

# Row 1: Price & Equity
fig_main.add_trace(go.Scatter(x=df.index, y=df['close'], name='BTC Price', line=dict(color='black', width=1)), row=1, col=1)
fig_main.add_trace(go.Scatter(x=df.index, y=df['equity'], name='Strategy Equity', line=dict(color='blue', width=2)), row=1, col=1, secondary_y=True)

# Row 2: RSI
fig_main.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')), row=2, col=1)
fig_main.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig_main.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

# Row 3: Net Position
pos_colors = np.where(df['position'] >= 0, 'green', 'red')
fig_main.add_trace(go.Bar(x=df.index, y=df['position'], name='Net Position', marker_color=pos_colors), row=3, col=1)

fig_main.update_layout(
    title="Binance BTC/USDT 30m Strategy",
    template="plotly_white",
    height=800,
    xaxis_rangeslider_visible=False
)

# Event Study Figure
days_axis = np.arange(len(avg_long_path)) / 48 
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

# CRITICAL FOR RAILWAY/GUNICORN: Expose the Flask server object
server = app.server 

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
    # Local development
    app.run_server(debug=True, port=8080)
