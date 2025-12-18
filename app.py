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
import time

# --- Configuration ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1d'
START_DATE = '2018-01-01'
SMA_PERIOD = 120
ADX_PERIOD = 14
# Port for local and cloud deployment
PORT = int(os.environ.get("PORT", 8080))
REPORT_FILE = "backtest_report.html"

def fetch_binance_data(symbol, interval, start_str):
    print(f"Fetching data for {symbol} starting from {start_str}...")
    base_url = "https://api.binance.com/api/v3/klines"
    dt_obj = datetime.strptime(start_str, '%Y-%m-%d')
    start_ts = int(dt_obj.timestamp() * 1000)
    
    all_data = []
    limit = 1000
    
    while True:
        params = {'symbol': symbol, 'interval': interval, 'startTime': start_ts, 'limit': limit}
        try:
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            if not isinstance(data, list) or len(data) == 0:
                break
            all_data.extend(data)
            start_ts = data[-1][6] + 1
            if len(data) < limit:
                break
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('date', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]

def calculate_adx(df, period=14):
    df = df.copy()
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    alpha = 1/period
    df['tr_smooth'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
    df['plus_dm_smooth'] = df['plus_dm'].ewm(alpha=alpha, adjust=False).mean()
    df['minus_dm_smooth'] = df['minus_dm'].ewm(alpha=alpha, adjust=False).mean()
    df['plus_di'] = 100 * (df['plus_dm_smooth'] / df['tr_smooth'])
    df['minus_di'] = 100 * (df['minus_dm_smooth'] / df['tr_smooth'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    return df['dx'].ewm(alpha=alpha, adjust=False).mean()

def apply_strategy(df):
    df['sma120'] = df['close'].rolling(window=SMA_PERIOD).mean()
    df['adx'] = calculate_adx(df, ADX_PERIOD)
    df['adx_deriv'] = df['adx'].diff()
    
    # Logic
    cond_long = (df['close'] > df['sma120'])
    cond_short = (df['close'] < df['sma120'])
    
    df['trend_signal'] = 0
    df.loc[cond_long, 'trend_signal'] = 1
    df.loc[cond_short, 'trend_signal'] = -1
    
    # Apply Flat filter
    df['position'] = np.where(df['adx_deriv'] < 0, 0, df['trend_signal'])
    df['strategy_pos'] = df['position'].shift(1)
    
    df['market_returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['market_returns'] * df['strategy_pos'].fillna(0)
    df['cumulative_market'] = (1 + df['market_returns'].fillna(0)).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_returns'].fillna(0)).cumprod()
    return df.dropna()

def generate_html_report(df):
    print("Generating report...")
    total_days = (df.index[-1] - df.index[0]).days
    total_return = (df['cumulative_strategy'].iloc[-1] - 1) * 100
    market_return = (df['cumulative_market'].iloc[-1] - 1) * 100
    annualized_return = ((1 + total_return/100) ** (365/total_days) - 1) * 100
    max_drawdown = ((df['cumulative_strategy'] / df['cumulative_strategy'].cummax()) - 1).min() * 100

    stats_html = f"""
    <div style="font-family: sans-serif; padding: 20px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6; margin-bottom: 20px;">
        <h2 style="margin-top:0;">Backtest Summary: {SYMBOL}</h2>
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 200px; padding: 15px; background: white; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <small style="color: #6c757d;">Total Return</small><br><span style="font-size: 24px; font-weight: bold; color: {'#28a745' if total_return > 0 else '#dc3545'};">{total_return:.2f}%</span>
            </div>
            <div style="flex: 1; min-width: 200px; padding: 15px; background: white; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <small style="color: #6c757d;">Max Drawdown</small><br><span style="font-size: 24px; font-weight: bold; color: #dc3545;">{max_drawdown:.2f}%</span>
            </div>
            <div style="flex: 1; min-width: 200px; padding: 15px; background: white; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <small style="color: #6c757d;">Market Return</small><br><span style="font-size: 24px; font-weight: bold;">{market_return:.2f}%</span>
            </div>
        </div>
    </div>
    """

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='BTC Price', line=dict(color='#333', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma120'], name='SMA 120', line=dict(color='orange', width=2)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['cumulative_strategy'], name='Strategy', line=dict(color='#007bff')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['cumulative_market'], name='Market', line=dict(color='#adb5bd', dash='dot')), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['adx'], name='ADX', line=dict(color='purple')), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['adx_deriv'], name='ADX Slope', marker_color=np.where(df['adx_deriv']<0, '#dc3545', '#28a745')), row=3, col=1)

    fig.update_layout(height=800, template="plotly_white", margin=dict(t=50, b=50, l=50, r=50))
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    recent_table = df.tail(15)[['close', 'sma120', 'adx', 'adx_deriv', 'position']].copy()
    recent_table['position'] = recent_table['position'].map({1: 'LONG', -1: 'SHORT', 0: 'FLAT'})
    table_html = f"<div style='margin-top:20px;'><h3>Recent Activity</h3>{recent_table.to_html(classes='table')}</div>"

    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Backtest: {SYMBOL}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #fff; }}
            .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }}
            .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
            .table th {{ background-color: #f8f9fa; font-weight: 600; }}
            .table tr:hover {{ background-color: #f1f3f5; }}
        </style>
    </head>
    <body>
        {stats_html}
        {plot_html}
        {table_html}
    </body>
    </html>
    """
    with open(REPORT_FILE, "w", encoding='utf-8') as f:
        f.write(full_html)

class RobustHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '':
            self.path = '/' + REPORT_FILE
        try:
            return super().do_GET()
        except Exception as e:
            print(f"Request error: {e}")

def main():
    try:
        df = fetch_binance_data(SYMBOL, INTERVAL, START_DATE)
        if df.empty:
            print("No data found.")
            return
        df = apply_strategy(df)
        generate_html_report(df)

        print(f"Server starting on port {PORT}...")
        with socketserver.TCPServer(("", PORT), RobustHandler) as httpd:
            httpd.allow_reuse_address = True
            if os.environ.get("PORT") is None: # Only auto-open on local
                webbrowser.open(f'http://localhost:{PORT}')
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
