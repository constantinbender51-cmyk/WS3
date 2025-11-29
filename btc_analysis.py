import pandas as pd
import requests
from flask import Flask, render_template_string
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)

def fetch_btc_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1d',
        'startTime': int(datetime(2018, 1, 1).timestamp() * 1000),
        'limit': 1000
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    return df

def calculate_indicators(df):
    df['SMA_120'] = df['close'].rolling(window=120).mean()
    df['SMA_365'] = df['close'].rolling(window=365).mean()
    df['position'] = 0  # 0 for neutral, 1 for long, -1 for short
    df.loc[df['close'] > df['SMA_365'], 'position'] = 1
    df.loc[df['close'] < df['SMA_365'], 'position'] = -1
    df.loc[df['close'] < df['SMA_120'], 'position'] = 0
    df['capital'] = 100.0  # Starting capital
    for i in range(1, len(df)):
        if df.iloc[i]['position'] == 1:  # Long position
            df.iloc[i, df.columns.get_loc('capital')] = df.iloc[i-1]['capital'] * (df.iloc[i]['close'] / df.iloc[i-1]['close'])
        elif df.iloc[i]['position'] == -1:  # Short position
            df.iloc[i, df.columns.get_loc('capital')] = df.iloc[i-1]['capital'] * (2 - (df.iloc[i]['close'] / df.iloc[i-1]['close']))
        else:  # Neutral position
            df.iloc[i, df.columns.get_loc('capital')] = df.iloc[i-1]['capital']
    return df

@app.route('/')
def index():
    df = fetch_btc_data()
    df = calculate_indicators(df)
    
    # Plot 1: Price and SMAs
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['close'], label='BTC Price', color='blue')
    plt.plot(df['timestamp'], df['SMA_120'], label='SMA 120', color='orange')
    plt.plot(df['timestamp'], df['SMA_365'], label='SMA 365', color='green')
    plt.title('BTC Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True)
    img1 = io.BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)
    plot_url1 = base64.b64encode(img1.getvalue()).decode()
    plt.close()
    
    # Plot 2: Positions
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['position'], label='Position', color='red', drawstyle='steps-post')
    plt.title('Trading Positions (Long: 1, Short: -1, Neutral: 0)')
    plt.xlabel('Date')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    img2 = io.BytesIO()
    plt.savefig(img2, format='png')
    img2.seek(0)
    plot_url2 = base64.b64encode(img2.getvalue()).decode()
    plt.close()
    
    # Plot 3: Capital
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['capital'], label='Capital', color='purple')
    plt.title('Capital Over Time')
    plt.xlabel('Date')
    plt.ylabel('Capital')
    plt.legend()
    plt.grid(True)
    img3 = io.BytesIO()
    plt.savefig(img3, format='png')
    img3.seek(0)
    plot_url3 = base64.b64encode(img3.getvalue()).decode()
    plt.close()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BTC Analysis</title>
    </head>
    <body>
        <h1>BTC Trading Analysis</h1>
        <h2>Price and Moving Averages</h2>
        <img src="data:image/png;base64,{plot_url1}" alt="Price and SMAs">
        <h2>Positions</h2>
        <img src="data:image/png;base64,{plot_url2}" alt="Positions">
        <h2>Capital</h2>
        <img src="data:image/png;base64,{plot_url3}" alt="Capital">
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)