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
BACKTEST_HOURS = 365 * 24 + 1
# =============================================================================

# Global State
current_plot_data = None
trade_pnl_history = []
active_trades = [] 
current_unrealized_pnl = 0.0
backtest_results = {}
exchange = ccxt.binance()

def fit_ols(x, y):
    if len(x) < 2: return None, None
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def fetch_large_dataset(symbol, timeframe, total_limit):
    """Fetches historical data in chunks."""
    all_ohlcv = []
    # Estimated start time
    since = exchange.milliseconds() - (total_limit * 60 * 60 * 1000)
    while len(all_ohlcv) < total_limit:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv: break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df.iloc[-total_limit:]

def run_backtest():
    """Simulates the OLS breakout logic over the last year."""
    global backtest_results
    print("Initializing Backtest...")
    df = fetch_large_dataset(SYMBOL, TIMEFRAME, BACKTEST_HOURS + MAX_WINDOW)
    
    equity = [0]
    trades = []
    current_active = [] # [{'type', 'entry', 'stop', 'target', 'window'}]
    
    for i in range(MAX_WINDOW, len(df) - 1):
        df_win = df.iloc[i - MAX_WINDOW:i]
        price = df.iloc[i]['close']
        
        # 1. Exit Logic
        new_active = []
        for t in current_active:
            closed = False
            p = (t['entry'] - price) if t['type'] == 'short' else (price - t['entry'])
            if (t['type'] == 'short' and (price >= t['stop'] or price <= t['target'])) or \
               (t['type'] == 'long' and (price <= t['stop'] or price >= t['target'])):
                equity.append(equity[-1] + p)
                trades.append(p)
                closed = True
            if not closed: new_active.append(t)
        current_active = new_active
        
        # 2. Entry Logic
        x = np.arange(len(df_win))
        last_c = df_win['close'].iloc[-1]
        
        best_s, best_l = None, None
        for w in range(MAX_WINDOW, MIN_WINDOW - 1, -1):
            x_w = x[-w:]; yc = df_win['close'].values[-w:]
            yh = df_win['high'].values[-w:]; yl = df_win['low'].values[-w:]
            
            m_m, c_m = fit_ols(x_w, yc)
            if m_m is None: continue
            yt = m_m * x_w + c_m
            
            m_u, c_u = fit_ols(x_w[yh > yt], yh[yh > yt])
            m_l, c_l = fit_ols(x_w[yl < yt], yl[yl < yt])
            
            if m_u is not None and m_l is not None:
                u_v, l_v = m_u * x_w[-1] + c_u, m_l * x_w[-1] + c_l
                dist = u_v - l_v; th = dist * THRESHOLD_PCT
                
                if last_c < (l_v - th) and best_s is None:
                    best_s = {'type': 'short', 'entry': last_c, 'stop': l_v, 'target': l_v - dist}
                if last_c > (u_v + th) and best_l is None:
                    best_l = {'type': 'long', 'entry': last_c, 'stop': u_v, 'target': u_v + dist}
        
        if best_s and not any(t['type'] == 'short' for t in current_active): current_active.append(best_s)
        if best_l and not any(t['type'] == 'long' for t in current_active): current_active.append(best_l)

    wins = [p for p in trades if p > 0]
    backtest_results = {
        'equity': equity,
        'win_rate': len(wins) / len(trades) if trades else 0,
        'total_pnl': sum(trades),
        'count': len(trades)
    }
    print("Backtest Finished.")

def generate_plot(df_closed):
    plt.figure(figsize=(15, 12))
    plt.style.use('dark_background')
    
    # Bottom Subplot: Equity Curve
    ax2 = plt.subplot(2, 1, 2)
    if backtest_results:
        ax2.plot(backtest_results['equity'], color='lime', linewidth=1)
        ax2.set_title(f"Backtest Equity Curve | Win Rate: {backtest_results['win_rate']:.2%}")
    
    # Top Subplot: Main Chart
    ax1 = plt.subplot(2, 1, 1)
    # ... (Candle rendering logic from previous response) ...
    
    buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
    return buf

# (DashboardHandler and run_server from previous responses)

if __name__ == "__main__":
    # Start backtest in background
    threading.Thread(target=run_backtest, daemon=True).start()
    # (Existing Logic Loop and Server starting)
