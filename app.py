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
LIMIT = 101                 # Candles for live chart
PORT = 8080                 
THRESHOLD_PCT = 0.10        # 10% threshold
UPDATE_INTERVAL = 10        # Live logic tick
MIN_WINDOW = 10             
MAX_WINDOW = 100            
BACKTEST_HOURS = 365 * 24 + 1
CANDLE_WIDTH = 0.6          
WICK_WIDTH = 0.05           
# =============================================================================

# Global State
current_plot_data = None
trade_pnl_history = []  # Live session history
active_trades = []      # Live active positions
current_unrealized_pnl = 0.0
backtest_results = {'equity': [0], 'win_rate': 0, 'total_pnl': 0, 'count': 0}
backtest_progress = 0.0
exchange = ccxt.binance()

def fit_ols(x, y):
    if len(x) < 2: return None, None
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def fetch_historical(total_limit):
    """Fetches large datasets in batches for the backtest."""
    all_ohlcv = []
    since = exchange.milliseconds() - (total_limit * 60 * 60 * 1000)
    while len(all_ohlcv) < total_limit:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
        except: break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df.iloc[-total_limit:]

def run_backtest():
    """Runs the OLS logic over the last 365 days."""
    global backtest_results, backtest_progress
    df = fetch_historical(BACKTEST_HOURS + MAX_WINDOW)
    
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
        
        # Entry Logic
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
    
    # Bottom: Equity Curve
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(backtest_results['equity'], color='cyan', linewidth=1)
    ax2.set_title(f"Yearly Backtest Equity Curve | Net: {backtest_results['total_pnl']:.2f}")
    ax2.set_ylabel("PnL (USDT)")

    # Top: Live Chart
    ax1 = plt.subplot(2, 1, 1)
    x_full = np.arange(len(df_closed))
    last_idx = x_full[-1]
    last_close = df_closed['close'].iloc[-1]
    
    biggest_signals = {'short': None, 'long': None}
    for w in range(MAX_WINDOW, MIN_WINDOW - 1, -1):
        if len(df_closed) < w: continue
        x_win = x_full[-w:]; yc = df_closed['close'].values[-w:]
        yh = df_closed['high'].values[-w:]; yl = df_closed['low'].values[-w:]
        m_m, c_m = fit_ols(x_win, yc)
        if m_m is None: continue
        yt = m_m * x_win + c_m
        m_u, c_u = fit_ols(x_win[yh > yt], yh[yh > yt])
        m_l, c_l = fit_ols(x_win[yl < yt], yl[yl < yt])
        if m_u and m_l:
            u_l, l_l = m_u * x_win + c_u, m_l * x_win + c_l
            dist = u_l[-1] - l_l[-1]; th = dist * THRESHOLD_PCT
            if last_close < (l_l[-1] - th) and not biggest_signals['short']:
                biggest_signals['short'] = {'x': x_win, 'u': u_l, 'l': l_l, 'th': l_l - th}
            if last_close > (u_l[-1] + th) and not biggest_signals['long']:
                biggest_signals['long'] = {'x': x_win, 'u': u_l, 'l': l_l, 'th': u_l + th}

    for s in biggest_signals.values():
        if s:
            ax1.plot(s['x'], s['u'], color='red', linewidth=1, alpha=0.7)
            ax1.plot(s['x'], s['l'], color='red', linewidth=1, alpha=0.7)
            ax1.plot(s['x'], s['th'], color='red', linestyle=':', linewidth=1.5)

    for t in active_trades:
        ax1.axhline(t['stop'], color='orange', linestyle='--', alpha=0.4)
        ax1.axhline(t['target'], color='lime', linestyle='--', alpha=0.4)

    up, down = df_closed[df_closed.close >= df_closed.open], df_closed[df_closed.close < df_closed.open]
    for color, d in [('green', up), ('red', down)]:
        ax1.bar(d.index, d.close - d.open, CANDLE_WIDTH, bottom=d.open, color=color, zorder=3)
        ax1.bar(d.index, d.high - np.maximum(d.close, d.open), WICK_WIDTH, bottom=np.maximum(d.close, d.open), color=color, zorder=3)
        ax1.bar(d.index, np.minimum(d.close, d.open) - d.low, WICK_WIDTH, bottom=d.low, color=color, zorder=3)

    buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
    return buf

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200); self.send_header('Content-type', 'text/html'); self.end_headers()
            p_bar = f"<div style='width:90%; background:#222; margin:10px auto;'><div style='width:{backtest_progress}%; background:cyan; height:12px;'></div></div>"
            rows = "".join([f"<tr style='color:{'lime' if t['type']=='long' else 'orange'}'><td>{t['type'].upper()}</td><td>{t['window']}</td><td>{t['entry']:.2f}</td><td>{t['stop']:.2f}</td><td>{t['target']:.2f}</td><td><form method='POST' action='/cancel?w={t['window']}&s={t['type']}'><input type='submit' value='Cancel'></form></td></tr>" for t in active_trades])
            html = f"<html><head><style>body{{background:#000;color:#fff;font-family:monospace;text-align:center}}table{{width:90%;margin:20px auto;border-collapse:collapse}}th,td{{padding:12px;border:1px solid #333}}.box{{padding:15px;background:#111;margin:10px auto;width:90%;border:1px solid #444}}button{{padding:12px 24px; font-weight:bold; cursor:pointer; background:#222; color:cyan; border:1px solid cyan;}}</style></head><body><img src='/chart.png?t={int(time.time())}'><br><button onclick='location.reload()'>REFRESH DASHBOARD</button>{p_bar}<h3>Backtest Progress: {backtest_progress:.1f}%</h3><div class='box'><b>Realized Session:</b> {sum(trade_pnl_history):.2f} | <b>Win Rate:</b> {backtest_results['win_rate']:.1%}</div><table><thead><tr><th>Side</th><th>Window</th><th>Entry</th><th>Stop</th><th>Target</th><th>Action</th></tr></thead><tbody>{rows}</tbody></table></body></html>"
            self.wfile.write(html.encode())
        elif self.path.startswith('/chart.png'):
            if current_plot_data:
                self.send_response(200); self.send_header('Content-type', 'image/png'); self.end_headers()
                self.wfile.write(current_plot_data.getvalue())
    def do_POST(self):
        global active_trades
        if self.path.startswith('/cancel'):
            q = parse_qs(urlparse(self.path).query)
            active_trades = [t for t in active_trades if not (t['window'] == int(q.get('w',[0])[0]) and t['type'] == q.get('s',[''])[0])]
        self.send_response(303); self.send_header('Location', '/'); self.end_headers()

def logic_loop():
    global current_plot_data
    while True:
        try:
            full_df = fetch_historical(LIMIT)
            if not full_df.empty:
                df_closed, price = full_df.iloc[:-1].copy(), full_df.iloc[-1]['close']
                # PnL Check
                remaining = []
                for t in active_trades:
                    p = (t['entry'] - price) if t['type'] == 'short' else (price - t['entry'])
                    if (t['type'] == 'short' and (price >= t['stop'] or price <= t['target'])) or (t['type'] == 'long' and (price <= t['stop'] or price >= t['target'])):
                        trade_pnl_history.append(p)
                    else: remaining.append(t)
                active_trades[:] = remaining
                current_plot_data = generate_plot(df_closed)
        except: pass
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    threading.Thread(target=run_backtest, daemon=True).start()
    threading.Thread(target=lambda: HTTPServer(('', PORT), DashboardHandler).serve_forever(), daemon=True).start()
    logic_loop()
