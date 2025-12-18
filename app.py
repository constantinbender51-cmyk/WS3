import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import http.server
import socketserver
import webbrowser
import os
import sys

# --- Configuration ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1d'  # Daily candles
START_DATE = '2018-01-01'
SMA_PERIOD = 120
ADX_PERIOD = 14
PORT = 8080

def fetch_binance_data(symbol, interval, start_str):
    """
    Fetches historical OHLCV data from Binance Public API using pagination.
    """
    print(f"Fetching data for {symbol} starting from {start_str}...")
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Convert start string to millisecond timestamp
    dt_obj = datetime.strptime(start_str, '%Y-%m-%d')
    start_ts = int(dt_obj.timestamp() * 1000)
    
    all_data = []
    limit = 1000
    
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ts,
            'limit': limit
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            # Check for API errors or empty response
            if not isinstance(data, list) or len(data) == 0:
                break
                
            all_data.extend(data)
            
            # Update start_ts to the last candle's close time + 1ms
            start_ts = data[-1][6] + 1
            
            # If we fetched fewer than limit, we reached the end
            if len(data) < limit:
                break
                
            print(f"Fetched {len(all_data)} candles so far...")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    if not all_data:
        print("No data fetched.")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    # Type conversion
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('date', inplace=True)
    
    print(f"Total rows fetched: {len(df)}")
    return df[['open', 'high', 'low', 'close', 'volume']]

def calculate_adx(df, period=14):
    """
    Calculates ADX manually to ensure no external lib dependency issues.
    """
    df = df.copy()
    
    # True Range
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    
    # Directional Movement
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # Wilder's Smoothing (alpha = 1/n)
    alpha = 1/period
    df['tr_smooth'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
    df['plus_dm_smooth'] = df['plus_dm'].ewm(alpha=alpha, adjust=False).mean()
    df['minus_dm_smooth'] = df['minus_dm'].ewm(alpha=alpha, adjust=False).mean()
    
    # Directional Indicators
    df['plus_di'] = 100 * (df['plus_dm_smooth'] / df['tr_smooth'])
    df['minus_di'] = 100 * (df['minus_dm_smooth'] / df['tr_smooth'])
    
    # DX
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    
    # ADX
    df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()
    
    return df['adx']

def apply_strategy(df):
    """
    Applies the strategy logic:
    1. SMA 120 Trend
    2. Flat if ADX derivative < 0
    """
    # 1. Calculate Indicators
    df['sma120'] = df['close'].rolling(window=SMA_PERIOD).mean()
    df['adx'] = calculate_adx(df, ADX_PERIOD)
    df['adx_deriv'] = df['adx'].diff()
    
    # 2. Define Logic
    # Base Signal: 1 (Long) if Close > SMA, -1 (Short) if Close < SMA
    # We use np.select for cleaner logic
    conditions = [
        (df['close'] > df['sma120']),
        (df['close'] < df['sma120'])
    ]
    choices = [1, -1]
    df['trend_signal'] = np.select(conditions, choices, default=0)
    
    # Filter: If ADX derivative is negative, stay FLAT (0)
    # If ADX derivative is positive, take the Trend Signal
    df['position'] = np.where(df['adx_deriv'] < 0, 0, df['trend_signal'])
    
    # Shift position by 1 to simulate trading at the Open of the NEXT candle based on Close logic
    # (Or simply, we can't trade the close of the candle used to calculate the signal instantly)
    df['strategy_pos'] = df['position'].shift(1)
    
    # 3. Calculate Returns
    df['market_returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['market_returns'] * df['strategy_pos']
    
    # 4. Cumulative Returns
    df['cumulative_market'] = (1 + df['market_returns']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()
    
    return df.dropna()

def generate_html_report(df):
    """
    Generates a Plotly HTML report with table and charts.
    """
    print("Generating report...")
    
    # --- Statistics ---
    total_days = (df.index[-1] - df.index[0]).days
    total_return = (df['cumulative_strategy'].iloc[-1] - 1) * 100
    market_return = (df['cumulative_market'].iloc[-1] - 1) * 100
    annualized_return = ((1 + total_return/100) ** (365/total_days) - 1) * 100
    
    # Drawdown
    cum_max = df['cumulative_strategy'].cummax()
    drawdown = (df['cumulative_strategy'] - cum_max) / cum_max
    max_drawdown = drawdown.min() * 100
    
    stats_html = f"""
    <div style="font-family: sans-serif; padding: 20px; background-color: #f4f4f4; border-radius: 5px; margin-bottom: 20px;">
        <h2>Backtest Results ({SYMBOL})</h2>
        <p><strong>Period:</strong> {START_DATE} to {df.index[-1].strftime('%Y-%m-%d')}</p>
        <table border="1" style="border-collapse: collapse; width: 50%; background-color: white;">
            <tr><th style="padding: 8px; text-align: left;">Metric</th><th style="padding: 8px; text-align: left;">Value</th></tr>
            <tr><td style="padding: 8px;">Total Return</td><td style="padding: 8px;">{total_return:.2f}%</td></tr>
            <tr><td style="padding: 8px;">Annualized Return</td><td style="padding: 8px;">{annualized_return:.2f}%</td></tr>
            <tr><td style="padding: 8px;">Buy & Hold Return</td><td style="padding: 8px;">{market_return:.2f}%</td></tr>
            <tr><td style="padding: 8px;">Max Drawdown</td><td style="padding: 8px; color: red;">{max_drawdown:.2f}%</td></tr>
        </table>
    </div>
    """

    # --- Plotting ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=('Price & SMA120 (Green=Long, Red=Short, Gray=Flat)', 'Strategy vs Market Equity', 'ADX & Derivative'))

    # Subplot 1: Price and SMA with coloring based on position
    # We color the price line based on the position held
    # Constructing segments for coloring is heavy in plotly, so we simply plot Price and markers for entries
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Price', line=dict(color='black', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma120'], name='SMA 120', line=dict(color='orange', width=2)), row=1, col=1)
    
    # Add Markers for Position Changes
    # 1 (Long), -1 (Short), 0 (Flat)
    # Filter for changes
    df['pos_change'] = df['position'].diff()
    entries_long = df[df['pos_change'] == 1] # Entered Long (from 0 or -1) - technically logic simplifies to from 0 mostly due to ADX
    entries_short = df[df['pos_change'] == -1] # Entered Short
    exits = df[(df['position'] == 0) & (df['position'].shift(1) != 0)] # Went Flat
    
    fig.add_trace(go.Scatter(x=entries_long.index, y=entries_long['close'], mode='markers', 
                             marker=dict(color='green', symbol='triangle-up', size=10), name='Buy Signal'), row=1, col=1)
    fig.add_trace(go.Scatter(x=entries_short.index, y=entries_short['close'], mode='markers', 
                             marker=dict(color='red', symbol='triangle-down', size=10), name='Sell Signal'), row=1, col=1)
    fig.add_trace(go.Scatter(x=exits.index, y=exits['close'], mode='markers', 
                             marker=dict(color='gray', symbol='circle-x', size=8), name='Go Flat'), row=1, col=1)

    # Subplot 2: Equity Curves
    fig.add_trace(go.Scatter(x=df.index, y=df['cumulative_strategy'], name='Strategy Equity', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['cumulative_market'], name='Buy & Hold', line=dict(color='gray', dash='dot')), row=2, col=1)

    # Subplot 3: ADX
    fig.add_trace(go.Scatter(x=df.index, y=df['adx'], name='ADX', line=dict(color='purple')), row=3, col=1)
    # Highlight area where ADX derivative is negative (Flat Zone)
    # We can use a shape or just plot the derivative
    fig.add_trace(go.Bar(x=df.index, y=df['adx_deriv'], name='ADX Slope', marker=dict(color=np.where(df['adx_deriv']<0, 'red', 'green'))), row=3, col=1)

    fig.update_layout(height=1000, title_text="Strategy Analysis")
    
    # --- Generate HTML file ---
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Recent Trade Table
    recent_data = df.tail(10)[['close', 'sma120', 'adx', 'adx_deriv', 'position']].copy()
    recent_data['position_label'] = recent_data['position'].map({1: 'LONG', -1: 'SHORT', 0: 'FLAT'})
    table_html = "<h3>Last 10 Days Data</h3>" + recent_data.to_html(classes='table')

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bot Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .table {{ border-collapse: collapse; width: 100%; }}
            .table td, .table th {{ border: 1px solid #ddd; padding: 8px; }}
            .table tr:nth-child(even){{background-color: #f2f2f2;}}
            .table th {{ padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: #04AA6D; color: white; }}
        </style>
    </head>
    <body>
        {stats_html}
        {plot_html}
        {table_html}
    </body>
    </html>
    """
    
    with open("backtest_report.html", "w", encoding='utf-8') as f:
        f.write(full_html)
    
    print("Report generated: backtest_report.html")

def start_server():
    """
    Starts a simple HTTP server to serve the report.
    """
    # Helper to always serve backtest_report.html on root
    class ReportHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.path = '/backtest_report.html'
            return http.server.SimpleHTTPRequestHandler.do_GET(self, self)

    print(f"\nStarting server at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    
    # Open browser automatically
    webbrowser.open(f'http://localhost:{PORT}')
    
    with socketserver.TCPServer(("", PORT), ReportHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            httpd.shutdown()

def main():
    # 1. Fetch
    df = fetch_binance_data(SYMBOL, INTERVAL, START_DATE)
    if df.empty:
        print("Exiting...")
        return

    # 2. Strategy
    print("Calculating indicators and running strategy...")
    df = apply_strategy(df)

    # 3. Report
    generate_html_report(df)

    # 4. Server
    start_server()

if __name__ == "__main__":
    main()
