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
    """Fetch daily BTC OHLCV data from Binance starting from 2022-01-01."""
    symbol = 'BTCUSDT'
    interval = Client.KLINE_INTERVAL_1DAY
    start_date = '2022-01-01'
    
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

def determine_background_color(df):
    """Determine background color based on SMA logic."""
    latest = df.iloc[-1]
    sma_365 = latest['sma_365']
    price = latest['close']
    
    if price > sma_365:
        return 'green'
    else:
        return 'red'

def calculate_capital(df, initial_capital=1000):
    """Calculate capital for short/long positions and buy-and-hold."""
    # Buy and hold strategy
    buy_price = df['close'].iloc[0]
    current_price = df['close'].iloc[-1]
    buy_hold_capital = initial_capital * (current_price / buy_price)
    
    # Trading strategy based on background color logic
    capital = initial_capital
    position = None  # 'long', 'short', or None
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # Determine signal based on previous day's SMAs and price
        sma_365_prev = prev_row['sma_365']
        price_prev = prev_row['close']
        
        if price_prev > sma_365_prev:
            signal = 'long'
        else:
            signal = 'short'
        
        # Execute trades
        if signal == 'long' and position != 'long':
            if position == 'short':
                # Close short position (assume no leverage for simplicity)
                capital *= (prev_row['close'] / buy_price_short)  # Simplified P&L
            # Open long position
            position = 'long'
            buy_price_long = row['close']
        elif signal == 'short' and position != 'short':
            if position == 'long':
                # Close long position
                capital *= (row['close'] / buy_price_long)  # Simplified P&L
            # Open short position (assume no leverage for simplicity)
            position = 'short'
            buy_price_short = row['close']
    
    # Close final position if any
    if position == 'long':
        capital *= (current_price / buy_price_long)
    elif position == 'short':
        capital *= (buy_price_short / current_price)  # Simplified for short
    
    return {
        'trading_capital': capital,
        'buy_hold_capital': buy_hold_capital
    }

@app.route('/')
def index():
    """Main route to display the plot and capital information."""
    df = fetch_btc_data()
    df = calculate_smas(df)
    
    # Determine background color
    bg_color = determine_background_color(df)
    
    # Calculate capital
    capital_info = calculate_capital(df)
    trading_capital = capital_info['trading_capital']
    buy_hold_capital = capital_info['buy_hold_capital']
    
    # Create plot
    fig = make_subplots(rows=1, cols=1)
    
    # Add price trace
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='BTC Price', line=dict(color='blue')), row=1, col=1)
    
    # Add SMA traces
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_120'], mode='lines', name='120 SMA', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_365'], mode='lines', name='365 SMA', line=dict(color='purple')), row=1, col=1)
    
    # Update layout with dynamic background color
    fig.update_layout(
        title='BTC Daily Price with SMAs',
        xaxis_title='Date',
        yaxis_title='Price (USDT)',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color
    )
    
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
            .info { margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>BTC Daily Price Analysis</h1>
        <div class="info">
            <p><strong>Background Color:</strong> {{ bg_color }} (based on SMA logic)</p>
            <p><strong>Trading Strategy Capital:</strong> ${{ "%.2f"|format(trading_capital) }}</p>
            <p><strong>Buy and Hold Capital:</strong> ${{ "%.2f"|format(buy_hold_capital) }}</p>
        </div>
        {{ plot|safe }}
    </body>
    </html>
    """
    
    return render_template_string(html_template, bg_color=bg_color, trading_capital=trading_capital, buy_hold_capital=buy_hold_capital, plot=plot_html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
