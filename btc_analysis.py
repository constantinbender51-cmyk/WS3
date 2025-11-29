import pandas as pd
from flask import Flask, render_template_string
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from binance.client import Client
from datetime import datetime
import os

app = Flask(__name__)

# Initialize Binance client (no API keys needed for public data)
client = Client()

def fetch_btc_data():
    """Fetch daily BTC OHLCV data from Binance starting from 2021-01-01."""
    symbol = 'BTCUSDT'
    interval = Client.KLINE_INTERVAL_1DAY
    start_date = '2021-01-01'
    
    klines = client.get_historical_klines(symbol, interval, start_date)
    
    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                       'close_time', 'quote_asset_volume', 'number_of_trades', 
                                       'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    
    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Convert price columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    return df

def calculate_smas(df):
    """Calculate 120-day and 365-day Simple Moving Averages."""
    df['sma_120'] = df['close'].rolling(window=120).mean()
    df['sma_365'] = df['close'].rolling(window=365).mean()
    return df

def determine_position_status(df):
    """Determine current position status and historical positions."""
    positions = []
    current_position = None
    
    for i in range(len(df)):
        row = df.iloc[i]
        sma_365 = row['sma_365']
        sma_120 = row['sma_120']
        price = row['close']
        
        if pd.isna(sma_365) or pd.isna(sma_120):
            position = 'neutral'
        elif price > sma_365:
            if price < sma_120:
                position = 'short'
            else:
                position = 'long'
        else:
            position = 'short'
        
        positions.append({
            'date': df.index[i],
            'position': position,
            'price': price,
            'sma_365': sma_365,
            'sma_120': sma_120
        })
        
        if i == len(df) - 1:
            current_position = position
    
    return {
        'current_position': current_position,
        'positions': positions
    }

def calculate_capital(df, initial_capital=1000):
    """Calculate capital for short/long positions and buy-and-hold."""
    # Buy and hold strategy
    buy_price = df['close'].iloc[0]
    current_price = df['close'].iloc[-1]
    buy_hold_capital = initial_capital * (current_price / buy_price)
    
    # Trading strategy based on SMA logic
    capital = initial_capital
    position = None  # 'long', 'short', or None
    buy_price_long = None
    buy_price_short = None
    position_history = []
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # Determine signal based on previous day's SMAs and price
        sma_365_prev = prev_row['sma_365']
        sma_120_prev = prev_row['sma_120']
        price_prev = prev_row['close']
        
        if pd.isna(sma_365_prev) or pd.isna(sma_120_prev):
            signal = 'neutral'
        elif price_prev > sma_365_prev:
            if price_prev < sma_120_prev:
                signal = 'short'
            else:
                signal = 'long'
        else:
            signal = 'short'
        
        # Execute trades
        if signal == 'long' and position != 'long':
            if position == 'short':
                # Close short position
                capital *= (2-(row['close'] / buy_price_short))
            # Open long position
            position = 'long'
            buy_price_long = row['close']
        elif signal == 'short' and position != 'short':
            if position == 'long':
                # Close long position
                capital *= (row['close'] / buy_price_long)
            # Open short position
            position = 'short'
            buy_price_short = row['close']
        
        position_history.append({
            'date': df.index[i],
            'position': position,
            'capital': capital
        })
    
    # Close final position if any
    if position == 'long':
        capital *= (current_price / buy_price_long)
    elif position == 'short':
        capital *= (2-(current_price/buy_price_short))
    
    return {
        'trading_capital': capital,
        'buy_hold_capital': buy_hold_capital,
        'position_history': position_history
    }

@app.route('/')
def index():
    """Main route to display the plot and capital information."""
    df = fetch_btc_data()
    df = calculate_smas(df)
    
    # Determine position status
    position_info = determine_position_status(df)
    current_position = position_info['current_position']
    positions = position_info['positions']
    
    # Calculate capital
    capital_info = calculate_capital(df)
    trading_capital = capital_info['trading_capital']
    buy_hold_capital = capital_info['buy_hold_capital']
    position_history = capital_info['position_history']
    
    # Create plot with subplots for price and capital
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=('BTC Price with SMAs', 'Capital Over Time'))
    
    # Add price trace
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='BTC Price', line=dict(color='blue', width=2)), row=1, col=1)
    
    # Add SMA traces
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_120'], mode='lines', name='120 SMA', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_365'], mode='lines', name='365 SMA', line=dict(color='purple', width=2)), row=1, col=1)
    
    # Add capital trace
    capital_dates = [entry['date'] for entry in position_history]
    capital_values = [entry['capital'] for entry in position_history]
    fig.add_trace(go.Scatter(x=capital_dates, y=capital_values, mode='lines', name='Trading Capital', line=dict(color='green', width=2)), row=2, col=1)
    
    # Add shaded regions for long/short positions
    current_region = None
    region_start = None
    
    for i, pos in enumerate(positions):
        if pos['position'] != current_region:
            if current_region is not None and region_start is not None:
                # Close previous region
                color = 'rgba(0, 255, 0, 0.2)' if current_region == 'long' else 'rgba(255, 0, 0, 0.2)'
                fig.add_vrect(
                    x0=region_start, x1=pos['date'],
                    fillcolor=color, 
                    layer="below", line_width=0,
                    annotation_text=current_region.upper(),
                    annotation_position="top left"
                )
            
            current_region = pos['position']
            region_start = pos['date']
    
    # Close final region
    if current_region is not None and region_start is not None:
        color = 'rgba(0, 255, 0, 0.2)' if current_region == 'long' else 'rgba(255, 0, 0, 0.2)'
        fig.add_vrect(
            x0=region_start, x1=positions[-1]['date'],
            fillcolor=color, 
            layer="below", line_width=0,
            annotation_text=current_region.upper(),
            annotation_position="top left"
        )
    
    # Update layout
    fig.update_layout(
        title=f'BTC Daily Price with SMAs and Capital - Current Position: {current_position.upper()}',
        xaxis_title='Date',
        yaxis_title='Price (USDT)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=800
    )
    fig.update_yaxes(title_text='Capital (USD)', row=2, col=1)
    
    # Convert plot to HTML
    plot_html = fig.to_html(full_html=False)
    
    # HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BTC Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .info { margin-bottom: 20px; padding: 15px; border-radius: 5px; }
            .long { background-color: rgba(0, 255, 0, 0.1); border-left: 4px solid green; }
            .short { background-color: rgba(255, 0, 0, 0.1); border-left: 4px solid red; }
            .neutral { background-color: rgba(128, 128, 128, 0.1); border-left: 4px solid gray; }
            .position-indicator { font-weight: bold; font-size: 18px; margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <h1>BTC Daily Price Analysis</h1>
        <div class="info {{ current_position }}">
            <div class="position-indicator">
                Current Position: <span style="color: {{ 'green' if current_position == 'long' else 'red' if current_position == 'short' else 'gray' }}">{{ current_position.upper() }}</span>
            </div>
            <p><strong>Trading Strategy Capital:</strong> ${{ "%.2f"|format(trading_capital) }}</p>
            <p><strong>Buy and Hold Capital:</strong> ${{ "%.2f"|format(buy_hold_capital) }}</p>
            <p><em>Strategy: LONG when price > 365 SMA and price >= 120 SMA, SHORT otherwise</em></p>
            <p><em>Green shaded areas = Long positions, Red shaded areas = Short positions</em></p>
        </div>
        {{ plot|safe }}
    </body>
    </html>
    """
    
    return render_template_string(html_template, current_position=current_position, trading_capital=trading_capital, buy_hold_capital=buy_hold_capital, plot=plot_html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
