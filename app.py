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

def run_backtest(df, adx_threshold):
    """Runs strategy for a specific ADX threshold and returns metrics."""
    df_bt = df.copy()
    df_bt['base_signal'] = np.where(df_bt['close'] > df_bt['sma120'], 1, -1)
    
    # Logic: Flat if ADX < threshold, else use base_signal
    df_bt['position'] = np.where(df_bt['adx'] < adx_threshold, 0, df_bt['base_signal'])
    df_bt['strategy_pos'] = df_bt['position'].shift(1).fillna(0)
    
    # Returns
    df_bt['mkt_ret'] = df_bt['close'].pct_change()
    df_bt['strat_ret'] = df_bt['mkt_ret'] * df_bt['strategy_pos']
    
    # Sharpe Ratio Calculation (Annualized, 365 days)
    mean_ret = df_bt['strat_ret'].mean()
    std_ret = df_bt['strat_ret'].std()
    
    if std_ret > 0 and not np.isnan(std_ret):
        sharpe = (mean_ret / std_ret) * np.sqrt(365)
    else:
        sharpe = 0
        
    total_ret = (1 + df_bt['strat_ret'].fillna(0)).prod() - 1
    return sharpe, total_ret, df_bt

def grid_search_adx(df):
    """Grid search for best ADX threshold from 1 to 60 based on Sharpe Ratio."""
    results = []
    print("Running Grid Search for ADX Threshold (1-60) based on Sharpe Ratio...")
    for a in range(1, 61):
        sharpe, ret, _ = run_backtest(df, a)
        results.append({'threshold': a, 'sharpe': sharpe, 'total_return': ret})
    
    res_df = pd.DataFrame(results)
    # Optimize based on Sharpe Ratio
    best_row = res_df.loc[res_df['sharpe'].idxmax()]
    return res_df, best_row['threshold']

def generate_report(df, grid_results, best_a):
    """Generates the HTML dashboard with Grid Search results."""
    print(f"Building report for best ADX threshold: {best_a}")
    
    # Run backtest with the winner
    sharpe_val, _, df_best = run_backtest(df, best_a)
    df_best['cum_mkt'] = (1 + df_best['mkt_ret'].fillna(0)).cumprod()
    df_best['cum_strat'] = (1 + df_best['strat_ret'].fillna(0)).cumprod()
    
    total_ret = (df_best['cum_strat'].iloc[-1] - 1) * 100
    mkt_ret = (df_best['cum_mkt'].iloc[-1] - 1) * 100
    max_dd = ((df_best['cum_strat'] / df_best['cum_strat'].cummax()) - 1).min() * 100

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                        row_heights=[0.3, 0.25, 0.25, 0.2],
                        subplot_titles=(f"Price & SMA120 (Best ADX: {best_a})", 
                                        "Grid Search: Sharpe Ratio vs ADX Threshold", 
                                        "Equity Curve", 
                                        "ADX (14)"))

    # Row 1: Price & SMA
    fig.add_trace(go.Scatter(x=df_best.index, y=df_best['close'], name='Price', line=dict(color='#2c3e50', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_best.index, y=df_best['sma120'], name='SMA 120', line=dict(color='#f39c12', width=2)), row=1, col=1)
    
    # Row 2: Grid Search Results (Sharpe Ratio)
    fig.add_trace(go.Scatter(x=grid_results['threshold'], y=grid_results['sharpe'], 
                             name='Sharpe Ratio', mode='lines+markers', line=dict(color='#3498db')), row=2, col=1)
    fig.add_vline(x=best_a, line_dash="dash", line_color="red", annotation_text=f"Best Sharpe: {best_a}", row=2, col=1)

    # Row 3: Equity
    fig.add_trace(go.Scatter(x=df_best.index, y=df_best['cum_strat'], name='Best Strategy', fill='tozeroy', line=dict(color='#27ae60')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_best.index, y=df_best['cum_mkt'], name='Buy & Hold', line=dict(color='#95a5a6', dash='dot')), row=3, col=1)
    
    # Row 4: ADX
    fig.add_trace(go.Scatter(x=df_best.index, y=df_best['adx'], name='ADX', line=dict(color='#8e44ad')), row=4, col=1)
    fig.add_hline(y=best_a, line_dash="dash", line_color="red", row=4, col=1)

    fig.update_layout(height=1200, template="plotly_white", title=f"Sharpe Optimized Grid Search: ADX Filter for {SYMBOL}")
    
    stats_html = f"""
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:15px; margin-bottom:20px; font-family:sans-serif;">
        <div style="padding:20px; border-radius:10px; background:#f8f9fa; border-top:5px solid #3498db; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color:#7f8c8d; font-size:12px; text-transform:uppercase;">Best Sharpe Ratio</div>
            <div style="font-size:24px; font-weight:bold; color:#3498db;">{sharpe_val:.3f}</div>
            <div style="font-size:12px; color:#95a5a6;">at ADX threshold {best_a}</div>
        </div>
        <div style="padding:20px; border-radius:10px; background:#f8f9fa; border-top:5px solid #27ae60; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color:#7f8c8d; font-size:12px; text-transform:uppercase;">Strategy Return</div>
            <div style="font-size:24px; font-weight:bold; color:#27ae60;">{total_ret:.2f}%</div>
        </div>
        <div style="padding:20px; border-radius:10px; background:#f8f9fa; border-top:5px solid #e74c3c; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color:#7f8c8d; font-size:12px; text-transform:uppercase;">Max Drawdown</div>
            <div style="font-size:24px; font-weight:bold; color:#e74c3c;">{max_dd:.2f}%</div>
        </div>
    </div>
    """

    recent_df = df_best.tail(15)[['close', 'sma120', 'adx', 'position']].copy()
    recent_df['position'] = recent_df['position'].replace({1: 'LONG', -1: 'SHORT', 0: 'FLAT'})
    table_html = f"<h3 style='font-family:sans-serif;'>Recent Activity (Sharpe Optimized a={best_a})</h3>{recent_df.to_html(classes='table')}"

    full_page = f"""
    <html>
    <head>
        <title>Sharpe Grid Search Results</title>
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
        # Pre-calculate Indicators once
        raw_df['sma120'] = raw_df['close'].rolling(window=SMA_PERIOD).mean()
        raw_df['adx'] = calculate_adx_wilder(raw_df, ADX_PERIOD)
        raw_df = raw_df.dropna()
        
        # Grid Search based on Sharpe
        grid_res, best_a = grid_search_adx(raw_df)
        
        # Generate Report for the best 'a'
        generate_report(raw_df, grid_res, int(best_a))
        
        # Server
        run_server()
