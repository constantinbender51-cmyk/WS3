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
backtest_results = {'equity': [0], 'win_rate': 0, 'total_pnl': 0, 'count': 0}
backtest_progress = 0.0
exchange = ccxt.binance()

def fit_ols(x, y):
    if len(x) < 2: return None, None
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def fetch_large_dataset(symbol, timeframe, total_limit):
    all_ohlcv = []
    since = exchange.milliseconds() - (total_limit * 60 * 60 * 1000)
    while len(all_ohlcv) < total_limit:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv: break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df.iloc[-total_limit:]

def run_backtest():
    global backtest_results, backtest_progress
    df = fetch_large_dataset(SYMBOL, TIMEFRAME, BACKTEST_HOURS + MAX_WINDOW)
    
    equity = [0]
    trades = []
    current_active = []
    total_steps = len(df) - 1 - MAX_WINDOW
    
    for idx, i in enumerate(range(MAX_WINDOW, len(df) - 1)):
        backtest_progress = (idx / total_steps) * 100
        df_win = df.iloc[i - MAX_WINDOW:i]
        price = df.iloc[i]['close']
        
        # Exit Logic
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
        
        # Entry Logic (Biggest Window First)
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
    backtest_results = {'equity': equity, 'win_rate': len(wins)/len(trades) if trades else 0, 'total_pnl': sum(trades), 'count': len(trades)}
    backtest_progress = 100.0

def generate_plot(df_closed):
    plt.figure(figsize=(15, 12))
    plt.style.use('dark_background')
    
    # Bottom Subplot: Equity Curve
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(backtest_results['equity'], color='lime', linewidth=1)
    ax2.set_title(f"Backtest Equity Curve | Win Rate: {backtest_results['win_rate']:.2%}")
    
    # Top Subplot: Main Chart
    ax1 = plt.subplot(2, 1, 1)
    x_full = np.arange(len(df_closed))
    last_idx = x_full[-1]
    last_close = df_closed['close'].iloc[-1]
    
    # (Existing Chart Logic for breakout_visuals and active_trades horizontal lines)
    up_c, down_c = df_closed[df_closed.close >= df_closed.open], df_closed[df_closed.close < df_closed.open]
    for color, d in [('green', up_c), ('red', down_c)]:
        ax1.bar(d.index, d.close - d.open, 0.6, bottom=d.open, color=color, zorder=2)
        ax1.bar(d.index, d.high - np.maximum(d.close, d.open), 0.05, bottom=np.maximum(d.close, d.open), color=color, zorder=2)
        ax1.bar(d.index, np.minimum(d.close, d.open) - d.low, 0.05, bottom=d.low, color=color, zorder=2)

    buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
    return buf

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200); self.send_header('Content-type', 'text/html'); self.end_headers()
            p_bar = f"<div style='width:85%; background:#222; margin:10px auto;'><div style='width:{backtest_progress}%; background:lime; height:10px;'></div></div>"
            rows = "".join([f"<tr style='color:{'lime' if t['type']=='long' else 'orange'}'><td>{t['type'].upper()}</td><td>{t['entry']:.2f}</td><td>{t['stop']:.2f}</td><td>{t['target']:.2f}</td></tr>" for t in active_trades])
            html = f"<html><head><style>body{{background:#000;color:#fff;font-family:monospace;text-align:center}}table{{width:85%;margin:20px auto;border-collapse:collapse}}th,td{{padding:10px;border:1px solid #333}}.box{{padding:15px;background:#111;margin:10px auto;width:85%}}</style></head><body><img src='/chart.png?t={int(time.time())}'><br>{p_bar}<h3>Backtest Progress: {backtest_progress:.1f}%</h3><div class='box'><b>Net:</b> {sum(trade_pnl_history)+current_unrealized_pnl:.2f}</div><table><thead><tr><th>Side</th><th>Entry</th><th>Stop</th><th>Target</th></tr></thead><tbody>{rows}</tbody></table></body><script>setTimeout(()=>location.reload(),10000)</script></html>"
            self.wfile.write(html.encode())
        elif self.path.startswith('/chart.png'):
            if current_plot_data:
                self.send_response(200); self.send_header('Content-type', 'image/png'); self.end_headers()
                self.wfile.write(current_plot_data.getvalue())
    def do_POST(self): self.send_response(303); self.send_header('Location', '/'); self.end_headers()

def logic_loop():
    global current_plot_data
    while True:
        full_df = fetch_large_dataset(SYMBOL, TIMEFRAME, LIMIT)
        if not full_df.empty:
            df_closed, open_p = full_df.iloc[:-1].copy(), full_df.iloc[-1]['close']
            current_plot_data = generate_plot(df_closed)
        time.sleep(10)

if __name__ == "__main__":
    threading.Thread(target=run_backtest, daemon=True).start()
    threading.Thread(target=lambda: HTTPServer(('', PORT), DashboardHandler).serve_forever(), daemon=True).start()
    logic_loop()
