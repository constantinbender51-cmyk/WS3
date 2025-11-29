import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template_string
import datetime
import time

app = Flask(__name__)

def fetch_binance_data(symbol="BTCUSDT", interval="1d", start_str="2018-01-01"):
    """
    Fetches historical kline data from Binance Public API.
    Handles pagination to get data from 2018 to present.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Convert start string to millisecond timestamp
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    
    data = []
    limit = 1000  # Binance max limit per request
    
    print(f"Fetching data for {symbol} starting from {start_str}...")
    
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ts,
            'limit': limit
        }
        
        try:
            response = requests.get(base_url, params=params)
            temp_data = response.json()
            
            if not temp_data or isinstance(temp_data, dict) and 'code' in temp_data:
                # Handle empty response or error
                break
                
            data.extend(temp_data)
            
            # Update start_ts to the close time of the last candle + 1ms
            last_close_time = temp_data[-1][6]
            start_ts = last_close_time + 1
            
            # Stop if we fetched fewer candles than the limit (reached present)
            if len(temp_data) < limit:
                break
                
            # Respect API rate limits slightly
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    # Columns based on Binance API documentation
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
            
    df = pd.DataFrame(data, columns=cols)
    
    # Type conversion
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    
    # Drop duplicates just in case
    df = df.drop_duplicates(subset=['open_time'])
    
    print(f"Total records fetched: {len(df)}")
    return df

def calculate_indicators(df):
    """
    Calculates SMAs, EMAs and Strategy Returns.
    """
    # SMAs
    df['SMA_365'] = df['close'].rolling(window=365).mean()
    df['SMA_120'] = df['close'].rolling(window=120).mean()
    df['SMA_90'] = df['close'].rolling(window=90).mean()
    df['SMA_60'] = df['close'].rolling(window=60).mean()
    
    # EMAs
    df['EMA_120'] = df['close'].ewm(span=120, adjust=False).mean()
    df['EMA_90'] = df['close'].ewm(span=90, adjust=False).mean()
    df['EMA_60'] = df['close'].ewm(span=60, adjust=False).mean()
    
    # Strategy: Long if Price > SMA 365, Short if Price < SMA 365
    # We use shift(1) to avoid lookahead bias (decision made on yesterday's close implies trade at today's open/close)
    # 1 for Long, -1 for Short. 
    # Note: Logic assumes we can instantly flip.
    
    conditions = [
        (df['close'] > df['SMA_365']),
        (df['close'] < df['SMA_365'])
    ]
    choices = [1, -1] # 1 = Long, -1 = Short
    
    df['position'] = np.select(conditions, choices, default=0)
    
    # Calculate daily returns of the asset
    df['asset_returns'] = df['close'].pct_change()
    
    # Strategy returns: Position from previous day * today's return
    df['strategy_returns'] = df['position'].shift(1) * df['asset_returns']
    
    # Capital Curve (Cumulative Returns starting at 100)
    df['capital_curve'] = 100 * (1 + df['strategy_returns']).cumprod()
    
    return df

def generate_plot(df):
    """
    Generates a Plotly HTML string.
    """
    # Create single subplot with secondary y-axis for capital curve
    fig = make_subplots(rows=1, cols=1, 
                        subplot_titles=('BTCUSDT Price Analysis with Capital Development'),
                        specs=[[{"secondary_y": True}]])

    # --- PRICE CHART WITH CAPITAL CURVE ---
    
    # 1. Price Line
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Price', 
                             line=dict(color='black', width=1)), secondary_y=False)
    
    # 2. SMAs
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_365'], mode='lines', name='SMA 365', 
                             line=dict(color='purple', width=2)), secondary_y=False)
    
    sma_colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Default plotly colors
    for i, period in enumerate([120, 90, 60]):
        fig.add_trace(go.Scatter(x=df['date'], y=df[f'SMA_{period}'], mode='lines', 
                                 name=f'SMA {period}', line=dict(width=1, dash='dot')), secondary_y=False)

    # 3. EMAs
    for i, period in enumerate([120, 90, 60]):
        fig.add_trace(go.Scatter(x=df['date'], y=df[f'EMA_{period}'], mode='lines', 
                                 name=f'EMA {period}', line=dict(width=1, dash='dash')), secondary_y=False)

    # 4. Capital Curve on secondary y-axis
    fig.add_trace(go.Scatter(x=df['date'], y=df['capital_curve'], mode='lines', name='Capital', 
                             line=dict(color='green', width=2)), secondary_y=True)

    # 5. Background Colors (Blue for long positions, Orange for short positions)
    # Create a mask for position: 1 = Long, -1 = Short
    df_clean = df.copy()
    df_clean['group'] = (df_clean['position'] != df_clean['position'].shift()).cumsum()
    
    shapes = []
    
    # Group by consecutive positions to minimize number of shapes
    agg = df_clean.groupby('group').agg(
        start_date=('date', 'first'),
        end_date=('date', 'last'),
        position=('position', 'first')
    )
    
    for _, row in agg.iterrows():
        if row['position'] == 1:  # Long position
            color = "rgba(0, 0, 255, 0.1)"
        elif row['position'] == -1:  # Short position
            color = "rgba(255, 165, 0, 0.1)"
        else:
            continue  # Skip neutral positions
        
        # Add shape to layout
        shapes.append(dict(
            type="rect",
            xref="x", yref="y",
            x0=row['start_date'],
            x1=row['end_date'],
            y0=df_clean['close'].min(), y1=df_clean['close'].max(),
            fillcolor=color,
            opacity=0.1,
            layer="below",
            line_width=0
        ))

    # --- LAYOUT SETTINGS ---
    fig.update_layout(
        height=900,
        template="plotly_white",
        title_text="Binance BTCUSDT Daily Analysis with Capital Development (2018 - Present)",
        hovermode="x unified",
        shapes=shapes # Add the background rectangles
    )
    
    fig.update_yaxes(type="log", title="Price (USDT)", secondary_y=False)
    fig.update_yaxes(title="Capital", secondary_y=True)

    return fig.to_html(full_html=False)

@app.route('/')
def dashboard():
    # Fetch Data
    df = fetch_binance_data(symbol="BTCUSDT", start_str="2018-01-01")
    
    # Process Data
    df = calculate_indicators(df)
    
    # Generate Plot
    plot_html = generate_plot(df)
    
    # Simple HTML Template
    html = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Crypto Quant Dashboard</title>
            <style>
                body {{ font-family: sans-serif; margin: 0; padding: 20px; background-color: #f4f4f9; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1 {{ text-align: center; color: #333; }}
                .stats {{ display: flex; justify-content: space-around; margin-bottom: 20px; padding: 10px; background: #eee; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Algo Trading Dashboard</h1>
                <div class="stats">
                    <div><strong>Current Price:</strong> ${df['close'].iloc[-1]:,.2f}</div>
                    <div><strong>365 SMA:</strong> ${df['SMA_365'].iloc[-1]:,.2f}</div>
                    <div><strong>Strategy Return:</strong> {((df['capital_curve'].iloc[-1]/100 - 1)*100):.2f}%</div>
                    <div><strong>Signal:</strong> {'LONG' if df['close'].iloc[-1] > df['SMA_365'].iloc[-1] else 'SHORT'}</div>
                </div>
                <div>
                    {plot_html}
                </div>
                <p style="text-align: center; color: #777;">Data sourced from Binance Public API. Not financial advice.</p>
            </div>
        </body>
    </html>
    """
    return html

if __name__ == "__main__":
    print("Starting server on 0.0.0.0:8080...")
    app.run(host='0.0.0.0', port=8080, debug=True)
