import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import http.server
import socketserver
import os
import sys
import webbrowser

# --- Configuration ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1d'
START_DATE = '2018-01-01'
SMA_PERIOD = 120
ADX_PERIOD = 14
ADX_THRESHOLD = 30 # Only flatten if ADX > 30, only enter if ADX < 30
PORT = int(os.environ.get("PORT", 8080))
REPORT_FILE = "backtest_report.html"

def fetch_binance_data(symbol, interval, start_str):
    """Fetches OHLCV data from Binance Public API."""
    print(f"Fetching {symbol} data from {start_str}...")
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_str, '%Y-%m-%d').timestamp() * 1000)
    
    all_data = []
    limit = 1000
    
    while True:
        params = {'symbol': symbol, 'interval': interval, 'startTime': start_ts, 'limit': limit}
        try:
            response = requests.get(base_url, params=params, timeout=15)
            data = response.json()
            if not data or not isinstance(data, list): break
            all_data.extend(data)
            start_ts = data[-1][6] + 1
            if len(data) < limit: break
        except Exception as e:
            print(f"Fetch error: {e}")
            break
            
    if not all_data: return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'
    ])
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df.set_index('date')[['open', 'high', 'low', 'close', 'volume']]

def calculate_adx_wilder(df, period=14):
    """Calculates ADX using Wilder's Smoothing."""
    df = df.copy()
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['dn_move'] = df['low'].shift(1) - df['low']
    df['plus_dm'] = np.where((df['up_move'] > df['dn_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['dn_move'] > df['up_move']) & (df['dn_move'] > 0), df['dn_move'], 0)
    
    alpha = 1 / period
    df['tr_s'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
    df['pdm_s'] = df['plus_dm'].ewm(alpha=alpha, adjust=False).mean()
    df['mdm_s'] = df['minus_dm'].ewm(alpha=alpha, adjust=False).mean()
    
    df['plus_di'] = 100 * (df['pdm_s'] / df['tr_s'])
    df['minus_di'] = 100 * (df['mdm_s'] / df['tr_s'])
    
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].fillna(0).ewm(alpha=alpha, adjust=False).mean()
    return df['adx']

def apply_strategy(df):
    """
    Implements:
    - Long if Price > SMA 120, Short if Price < SMA 120
    - Only enter new position if ADX < 30
    - Only flatten if ADX > 30 AND derivative is negative
    """
    df = df.copy()
    df['sma120'] = df['close'].rolling(window=SMA_PERIOD).mean()
    df['adx'] = calculate_adx_wilder(df, ADX_PERIOD)
    df['adx_deriv'] = df['adx'].diff()
    
    # Base signal from SMA
    df['base_signal'] = np.where(df['close'] > df['sma120'], 1, -1)
    
    # State machine for positions
    positions = []
    current_pos = 0 # 0: Flat, 1: Long, -1: Short
    
    for i in range(len(df)):
        row = df.iloc[i]
        adx = row['adx']
        deriv = row['adx_deriv']
        base = row['base_signal']
        
        # 1. Exit Logic (Flatten)
        # Only flatten if ADX > 30 and derivative is negative
        if current_pos != 0 and adx > ADX_THRESHOLD and deriv < 0:
            current_pos = 0
            
        # 2. Entry Logic
        # Only enter if we are currently Flat and ADX is below 30
        elif current_pos == 0 and adx < ADX_THRESHOLD:
            current_pos = base
            
        # 3. Position Maintenance / Flip
        # If we are already in a position and it's still "trending" (ADX < 30 or deriv >= 0)
        # we check if the SMA signal has flipped.
        elif current_pos != 0:
            # If the underlying trend changes while we are in a position
            if current_pos != base:
                # Flip only if ADX is still in "entry" range
                if adx < ADX_THRESHOLD:
                    current_pos = base
        
        positions.append(current_pos)
        
    df['position'] = positions
    df['strategy_pos'] = df['position'].shift(1)
    
    # Performance Calculation
    df['mkt_ret'] = df['close'].pct_change()
    df['strat_ret'] = df['mkt_ret'] * df['strategy_pos'].fillna(0)
    df['cum_mkt'] = (1 + df['mkt_ret'].fillna(0)).cumprod()
    df['cum_strat'] = (1 + df['strat_ret'].fillna(0)).cumprod()
    
    return df.dropna()

def generate_report(df):
    """Generates the HTML dashboard."""
    print("Building report...")
    total_ret = (df['cum_strat'].iloc[-1] - 1) * 100
    mkt_ret = (df['cum_mkt'].iloc[-1] - 1) * 100
    max_dd = ((df['cum_strat'] / df['cum_strat'].cummax()) - 1).min() * 100

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=(f"{SYMBOL} Price & SMA", "Equity Curve", "ADX (14) & Logic Regions"))

    # Price & SMA
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Price', line=dict(color='#2c3e50', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma120'], name='SMA 120', line=dict(color='#f39c12', width=2)), row=1, col=1)
    
    # Equity
    fig.add_trace(go.Scatter(x=df.index, y=df['cum_strat'], name='Strategy', fill='tozeroy', line=dict(color='#27ae60')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['cum_mkt'], name='Buy & Hold', line=dict(color='#95a5a6', dash='dot')), row=2, col=1)
    
    # ADX with Threshold Line
    fig.add_trace(go.Scatter(x=df.index, y=df['adx'], name='ADX', line=dict(color='#8e44ad')), row=3, col=1)
    fig.add_hline(y=ADX_THRESHOLD, line_dash="dash", line_color="gray", row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['adx_deriv'], name='ADX Slope', 
                         marker_color=np.where(df['adx_deriv'] < 0, '#e74c3c', '#2ecc71')), row=3, col=1)

    fig.update_layout(height=900, template="plotly_white", title=f"Strategy: SMA120 Crossover + ADX Threshold ({ADX_THRESHOLD}) Filter")
    
    stats_html = f"""
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:15px; margin-bottom:20px; font-family:sans-serif;">
        <div style="padding:20px; border-radius:10px; background:#f8f9fa; border-top:5px solid #27ae60; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color:#7f8c8d; font-size:12px; text-transform:uppercase;">Strategy Return</div>
            <div style="font-size:24px; font-weight:bold; color:#27ae60;">{total_ret:.2f}%</div>
        </div>
        <div style="padding:20px; border-radius:10px; background:#f8f9fa; border-top:5px solid #e74c3c; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color:#7f8c8d; font-size:12px; text-transform:uppercase;">Max Drawdown</div>
            <div style="font-size:24px; font-weight:bold; color:#e74c3c;">{max_dd:.2f}%</div>
        </div>
        <div style="padding:20px; border-radius:10px; background:#f8f9fa; border-top:5px solid #95a5a6; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color:#7f8c8d; font-size:12px; text-transform:uppercase;">Market Return</div>
            <div style="font-size:24px; font-weight:bold;">{mkt_ret:.2f}%</div>
        </div>
    </div>
    """

    recent_df = df.tail(20)[['close', 'sma120', 'adx', 'adx_deriv', 'position']].copy()
    recent_df['position'] = recent_df['position'].replace({1: 'LONG', -1: 'SHORT', 0: 'FLAT'})
    table_html = f"<h3 style='font-family:sans-serif;'>Recent Activity Log</h3>{recent_df.to_html(classes='table')}"

    full_page = f"""
    <html>
    <head>
        <title>Strategy Report</title>
        <style>
            body {{ padding: 30px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background:#fff; color:#333; }}
            .table {{ width: 100%; border-collapse: collapse; margin-top: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            .table th, .table td {{ padding: 12px; border: 1px solid #eee; text-align: left; }}
            .table th {{ background: #fcfcfc; color: #666; font-weight:600; }}
            .table tr:nth-child(even) {{ background: #fafafa; }}
        </style>
    </head>
    <body>
        {stats_html}
        {fig.to_html(full_html=False, include_plotlyjs='cdn')}
        {table_html}
    </body>
    </html>
    """
    
    with open(REPORT_FILE, "w", encoding='utf-8') as f:
        f.write(full_page)

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/': self.path = f"/{REPORT_FILE}"
        return super().do_GET()

def run_server():
    print(f"Starting web server on port {PORT}...")
    with socketserver.TCPServer(("", PORT), RequestHandler) as httpd:
        httpd.allow_reuse_address = True
        if "PORT" not in os.environ:
            webbrowser.open(f"http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    raw_df = fetch_binance_data(SYMBOL, INTERVAL, START_DATE)
    if not raw_df.empty:
        processed_df = apply_strategy(raw_df)
        generate_report(processed_df)
        run_server()
