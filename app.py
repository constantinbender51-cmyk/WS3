import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from urllib.parse import urlparse, parse_qs

# =============================================================================
# PARAMETERS
# =============================================================================
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LIMIT = 101                 
PORT = 8080                 
THRESHOLD_PCT = 0.10        
UPDATE_INTERVAL = 10        
MIN_WINDOW = 10             
MAX_WINDOW = 100            
CANDLE_WIDTH = 0.6          
WICK_WIDTH = 0.05           
# =============================================================================

# Global State
current_plot_data = None
trade_pnl_history = []
active_trades = [] 
current_unrealized_pnl = 0.0
exchange = ccxt.binance()

def fit_ols(x, y):
    if len(x) < 2: return None, None
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def fetch_data():
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
        return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    except:
        return pd.DataFrame()

def process_and_pnl(latest_price):
    global trade_pnl_history, active_trades, current_unrealized_pnl
    remaining = []
    upnl = 0.0
    for t in active_trades:
        closed = False
        p = (t['entry'] - latest_price) if t['type'] == 'short' else (latest_price - t['entry'])
        if t['type'] == 'short' and (latest_price >= t['stop'] or latest_price <= t['target']):
            closed = True
        elif t['type'] == 'long' and (latest_price <= t['stop'] or latest_price >= t['target']):
            closed = True
        
        if closed:
            trade_pnl_history.append(p)
        else:
            upnl += p
            remaining.append(t)
    active_trades = remaining
    current_unrealized_pnl = upnl

def generate_plot(df_closed):
    plt.figure(figsize=(15, 9))
    plt.style.use('dark_background')
    
    x_full = np.arange(len(df_closed))
    last_idx = x_full[-1]
    last_close = df_closed['close'].iloc[-1]
    
    # Track the largest window found for each side
    biggest_short_signal = None
    biggest_long_signal = None

    # Scan from largest window down to smallest
    for w in range(MAX_WINDOW, MIN_WINDOW - 1, -1):
        if len(df_closed) < w: continue
        x_win = x_full[-w:]
        y_win_close = df_closed['close'].values[-w:]
        y_win_high = df_closed['high'].values[-w:]
        y_win_low = df_closed['low'].values[-w:]
        
        m_mid, c_mid = fit_ols(x_win, y_win_close)
        if m_mid is None: continue
        y_trend = m_mid * x_win + c_mid
        
        m_u, c_u = fit_ols(x_win[y_win_high > y_trend], y_win_high[y_win_high > y_trend])
        m_l, c_l = fit_ols(x_win[y_win_low < y_trend], y_win_low[y_win_low < y_trend])
        
        if m_u is not None and m_l is not None:
            u_line = m_u * x_win + c_u
            l_line = m_l * x_win + c_l
            dist = u_line[-1] - l_line[-1]
            thresh_val = dist * THRESHOLD_PCT
            
            is_short = last_close < (l_line[-1] - thresh_val)
            is_long = last_close > (u_line[-1] + thresh_val)
            
            if is_short and biggest_short_signal is None:
                biggest_short_signal = {
                    'x': x_win, 'upper': u_line, 'lower': l_line, 
                    'thresh': l_line - thresh_val, 'type': 'short', 
                    'entry': last_close, 'stop': l_line[-1], 'target': l_line[-1] - dist, 'window': w
                }
            
            if is_long and biggest_long_signal is None:
                biggest_long_signal = {
                    'x': x_win, 'upper': u_line, 'lower': l_line, 
                    'thresh': u_line + thresh_val, 'type': 'long', 
                    'entry': last_close, 'stop': u_line[-1], 'target': u_line[-1] + dist, 'window': w
                }

    # Trading Logic (Execute only if side is free)
    for sig in [biggest_short_signal, biggest_long_signal]:
        if sig and not any(t['type'] == sig['type'] for t in active_trades):
            active_trades.append({
                'type': sig['type'], 'entry': sig['entry'], 'stop': sig['stop'], 
                'target': sig['target'], 'window': sig['window']
            })

    # Render Visuals for the biggest signals only
    for sig in [biggest_short_signal, biggest_long_signal]:
        if sig:
            color = 'cyan' if sig['type'] == 'long' else 'orange'
            plt.plot(sig['x'], sig['upper'], color=color, linewidth=1.2, alpha=0.8)
            plt.plot(sig['x'], sig['lower'], color=color, linewidth=1.2, alpha=0.8)
            plt.plot(sig['x'], sig['thresh'], color=color, linestyle=':', linewidth=1.5)

    # Position Markers
    for trade in active_trades:
        plt.axhline(trade['stop'], color='red', linestyle='--', alpha=0.4)
        plt.axhline(trade['target'], color='lime', linestyle='--', alpha=0.4)

    # Candles
    up, down = df_closed[df_closed.close >= df_closed.open], df_closed[df_closed.close < df_closed.open]
    for c, d in [('green', up), ('red', down)]:
        plt.bar(d.index, d.close - d.open, CANDLE_WIDTH, bottom=d.open, color=c, zorder=3)
        plt.bar(d.index, d.high - np.maximum(d.close, d.open), WICK_WIDTH, bottom=np.maximum(d.close, d.open), color=c, zorder=3)
        plt.bar(d.index, np.minimum(d.close, d.open) - d.low, WICK_WIDTH, bottom=d.low, color=c, zorder=3)

    buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
    return buf

# (DashboardHandler and Server initialization remain identical to the previous implementation)
