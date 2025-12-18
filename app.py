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
                                        "Equity

