import ccxt
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from datetime import datetime, timedelta
import time

def fetch_data():
    exchange = ccxt.binance({'enableRateLimit': True})
    symbol = 'ETH/USDT'
    timeframe = '1h'
    limit = 1000
    
    # Calculate start time: 5 years ago
    now = exchange.milliseconds()
    since = now - (5 * 365 * 24 * 60 * 60 * 1000)
    
    all_ohlcv = []
    
    print(f"Fetching 5 years of {timeframe} data for {symbol}...")
    
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Update 'since' to the timestamp of the last candle + 1 timeframe duration
            # 1h in ms = 3600000
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 3600000
            
            # Print progress
            print(f"Fetched up to {pd.to_datetime(last_timestamp, unit='ms')}")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def serve_plot(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])
    
    fig.update_layout(
        title='ETH/USDT - 5 Year 1H OHLC',
        yaxis_title='Price (USDT)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(figure=fig, style={'height': '95vh'})
    ])
    
    print("Serving plot on port 8080...")
    app.run(debug=False, port=8080, host='0.0.0.0')

if __name__ == "__main__":
    df = fetch_data()
    serve_plot(df)
