import pandas as pd
import requests
from flask import Flask, render_template_string
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)

def fetch_btc_data():
    import time
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    start_time = int(datetime(2018, 1, 1).timestamp() * 1000)
    while True:
        params = {
            'symbol': 'BTCUSDT',
            'interval': '1d',
            'startTime': start_time,
            'limit': 1000
        }
        response = requests.get(url, params=params)
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        start_time = data[-1][6]  # Use close_time of last entry as next startTime
        time.sleep(1)  # Respect rate limits with 1-second delay
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
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
    # Update neutral positions based on SMA_120 relative to SMA_365 condition
    df.loc[(df['close'] > df['SMA_365']) & (df['close'] < df['SMA_120']), 'position'] = 0
    df.loc[(df['close'] < df['SMA_365']) & (df['close'] > df['SMA_120']), 'position'] = 0
    df['capital'] = 100.0  # Starting capital
    df['entry_price'] = 0.0  # Track entry price for stop loss
    df['stop_loss_hit'] = 0  # 0 for no hit, 1 for hit
    stop_loss_percent = 0.10  # 10% stop loss
    for i in range(1, len(df)):
        current_position = df.iloc[i]['position']
        prev_capital = df.iloc[i-1]['capital']
        current_close = df.iloc[i]['close']
        prev_close = df.iloc[i-1]['close']
        
        # Set entry price to the previous close for any given day
        df.iloc[i, df.columns.get_loc('entry_price')] = prev_close
        
        entry_price = df.iloc[i]['entry_price']
        
        # Check stop loss conditions
        current_high = df.iloc[i]['high']
        current_low = df.iloc[i]['low']
        stop_loss_hit = False
        if current_position == 1 and entry_price > 0 and current_low <= entry_price * (1 - stop_loss_percent):
            df.iloc[i, df.columns.get_loc('position')] = 0  # Close long position
            df.iloc[i, df.columns.get_loc('stop_loss_hit')] = 1  # Mark stop loss hit
            current_position = 0
            stop_loss_hit = True
        elif current_position == -1 and entry_price > 0 and current_high >= entry_price * (1 + stop_loss_percent):
            df.iloc[i, df.columns.get_loc('position')] = 0  # Close short position
            df.iloc[i, df.columns.get_loc('stop_loss_hit')] = 1  # Mark stop loss hit
            current_position = 0
            stop_loss_hit = True
        
        # Calculate capital based on position
        if stop_loss_hit:
            # Apply stop loss penalty
            df.iloc[i, df.columns.get_loc('capital')] = prev_capital * (1 - stop_loss_percent)
        elif current_position == 1:  # Long position
            df.iloc[i, df.columns.get_loc('capital')] = prev_capital * (current_close / prev_close)
        elif current_position == -1:  # Short position
            df.iloc[i, df.columns.get_loc('capital')] = prev_capital * (2 - (current_close / prev_close))
        else:  # Neutral position
            df.iloc[i, df.columns.get_loc('capital')] = prev_capital
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
    
    # Plot 4: Stop Loss Events
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['close'], label='BTC Price', color='blue')
    stop_loss_dates = df[df['stop_loss_hit'] == 1]['timestamp']
    stop_loss_prices = df[df['stop_loss_hit'] == 1]['close']
    plt.scatter(stop_loss_dates, stop_loss_prices, color='red', marker='x', s=100, label='Stop Loss Hit', zorder=5)
    plt.title('BTC Price with Stop Loss Events')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True)
    img4 = io.BytesIO()
    plt.savefig(img4, format='png')
    img4.seek(0)
    plot_url4 = base64.b64encode(img4.getvalue()).decode()
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
        <h2>Stop Loss Events</h2>
        <img src="data:image/png;base64,{plot_url4}" alt="Stop Loss Events">
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)