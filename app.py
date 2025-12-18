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
ADX_PERIOD = 14  # As requested: ADX smoothing 14
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
    """Calculates ADX using Wilder's Smoothing (Industry Standard)."""
    df = df.copy()
    
    # Calculate True Range
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Directional Movement
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['dn_move'] = df['low'].shift(1) - df['low']
    
    df['plus_dm'] = np.where((df['up_move'] > df['dn_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['dn_move'] > df['up_move']) & (df['dn_move'] > 0), df['dn_move'], 0)
    
    # Wilder's Smoothing: 1st value is sum, subsequent are (prev_val - prev_val/n) + current
    def wilders_smoothing(series, n):
        result = np.zeros(len(series))
        # First non-NaN value position
        start_idx = series.first_valid_index()
        if start_idx is None: return result
        
        # Start calculating from the nth value
        idx = df.index.get_loc(start_idx) + n
        if idx >= len(series): return result
        
        # Initial sum
        result[idx-1] = series.iloc[idx-n:idx].sum()
        
        # Cumulative moving average
        for i in range(idx, len(series)):
            result[i] = result[i-1] - (result[i-1] / n) + series.iloc[i]
        return result

    df['smoothed_tr'] = wilders_smoothing(df['tr'], period)
    df['smoothed_pdm'] = wilders_smoothing(df['plus_dm'], period)
    df['smoothed_mdm'] = wilders_smoothing(df['minus_dm'], period)
    
    df['plus_di'] = 100 * (df['smoothed_pdm'] / df['smoothed_tr'])
    df['minus_di'] = 100 * (df['smoothed_mdm'] / df['smoothed_tr'])
    
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['dx'] = df['dx'].fillna(0)
    
    # ADX is the Wilder's smoothing of DX
    # (Using the same logic function)
    df['adx'] = wilders_smoothing(df['dx'], period) / period # Divided by period because initial sum wasn't averaged
    
    # Correction for initial ADX average
    # The helper above returns the running sum for DI, but ADX needs to be the average
    # Let's use EWM with alpha=1/N which is equivalent to Wilder's
    alpha = 1 / period
    df['adx_final'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()
    
    return df['adx_final']

def apply_strategy(df):
    """Implements the crossover and ADX derivative logic."""
    df = df.copy()
    df['sma120'] = df['close'].rolling(window=SMA_PERIOD).mean()
    df['adx'] = calculate_adx_wilder(df, ADX_PERIOD)
    df['adx_deriv'] = df['adx'].diff()
    
    # 1. Price vs SMA Logic
    # Long when Price > SMA, Short when Price < SMA
    df['base_signal'] = np.where(df['close'] > df['sma120'], 1, -1)
    
    # 2. ADX Derivative Filter
    # Flat (0) when derivative < 0, otherwise follow base_signal
    df['position'] = np.where(df['adx_deriv'] < 0, 0, df['base_signal'])
    
    # Execution: Trade on next open
    df['strategy_pos'] = df['position'].shift(1)
    
    # Returns
    df['mkt_ret'] = df['close'].pct_change()
    df['strat_ret'] = df['mkt_ret'] * df['strategy_pos'].fillna(0)
    df['cum_mkt'] = (1 + df['mkt_ret'].fillna(0)).cumprod()
    df['cum_strat'] = (1 + df['strat_ret'].fillna(0)).cumprod()
    
    return df.dropna()

def generate_report(df):
    """Generates an interactive HTML dashboard."""
    print("Building report...")
    
    total_ret = (df['cum_strat'].iloc[-1] - 1) * 100
    mkt_ret = (df['cum_mkt'].iloc[-1] - 1) * 100
    max_dd = ((df['cum_strat'] / df['cum_strat'].cummax()) - 1).min() * 100

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=(f"{SYMBOL} Price & SMA", "Equity Curve", "ADX (14) & Slope"))

    # Price & SMA
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Price', line=dict(color='#2c3e50', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma120'], name='SMA 120', line=dict(color='#f39c12', width=2)), row=1, col=1)
    
    # Equity
    fig.add_trace(go.Scatter(x=df.index, y=df['cum_strat'], name='Strategy', fill='tozeroy', line=dict(color='#27ae60')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['cum_mkt'], name='Buy & Hold', line=dict(color='#95a5a6', dash='dot')), row=2, col=1)
    
    # ADX
    fig.add_trace(go.Scatter(x=df.index, y=df['adx'], name='ADX', line=dict(color='#8e44ad')), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['adx_deriv'], name='ADX Slope', 
                         marker_color=np.where(df['adx_deriv'] < 0, '#e74c3c', '#2ecc71')), row=3, col=1)

    fig.update_layout(height=900, template="plotly_white", showlegend=True, 
                      title=f"Comprehensive Strategy Analysis: {SYMBOL}")
    
    # Stats Summary
    stats_html = f"""
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:15px; margin-bottom:20px; font-family:sans-serif;">
        <div style="padding:20px; border-radius:10px; background:#ecf0f1; border-left:5px solid #27ae60;">
            <div style="color:#7f8c8d; font-size:12px;">Strategy Total Return</div>
            <div style="font-size:24px; font-weight:bold;">{total_ret:.2f}%</div>
        </div>
        <div style="padding:20px; border-radius:10px; background:#ecf0f1; border-left:5px solid #e74c3c;">
            <div style="color:#7f8c8d; font-size:12px;">Max Drawdown</div>
            <div style="font-size:24px; font-weight:bold;">{max_dd:.2f}%</div>
        </div>
        <div style="padding:20px; border-radius:10px; background:#ecf0f1; border-left:5px solid #95a5a6;">
            <div style="color:#7f8c8d; font-size:12px;">Market Return</div>
            <div style="font-size:24px; font-weight:bold;">{mkt_ret:.2f}%</div>
        </div>
    </div>
    """

    recent_df = df.tail(15)[['close', 'sma120', 'adx', 'adx_deriv', 'position']].copy()
    recent_df['position'] = recent_df['position'].replace({1: 'LONG', -1: 'SHORT', 0: 'FLAT'})
    table_html = f"<h3 style='font-family:sans-serif;'>Recent Data Log</h3>{recent_df.to_html(classes='table')}"

    full_page = f"""
    <html>
    <head>
        <title>Backtest Results</title>
        <style>
            body {{ padding: 30px; font-family: sans-serif; }}
            .table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            .table th, .table td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
            .table th {{ background: #f8f9fa; }}
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
        if "PORT" not in os.environ: # Local only
            webbrowser.open(f"http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    raw_df = fetch_binance_data(SYMBOL, INTERVAL, START_DATE)
    if not raw_df.empty:
        processed_df = apply_strategy(raw_df)
        generate_report(processed_df)
        run_server()
    else:
        print("Failed to fetch data.")
