import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timedelta

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
# =============================================================================

# Global State
current_plot_data = None
trade_pnl_history = []  
active_trades = []      
current_unrealized_pnl = 0.0
backtest_progress = 0.0
exchange = ccxt.binance()

def fit_ols(x, y):
    if len(x) < 2: return None, None
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def get_time_to_close():
    """Calculates minutes/seconds until the next hour candle close."""
    now = datetime.utcnow()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    remaining = next_hour - now
    mins, secs = divmod(remaining.seconds, 60)
    return f"{mins:02d}m {secs:02d}s"

def generate_plot(df_closed, latest_price):
    plt.figure(figsize=(15, 9))
    plt.style.use('dark_background')
    
    ax1 = plt.gca()
    x_full = np.arange(len(df_closed))
    last_idx = x_full[-1]
    last_close = df_closed['close'].iloc[-1]
    
    best_visual = None
    
    # Scan for signals
    for w in range(MAX_WINDOW, MIN_WINDOW - 1, -1):
        if len(df_closed) < w: continue
        x_win = x_full[-w:]
        yc, yh, yl = df_closed['close'].values[-w:], df_closed['high'].values[-w:], df_closed['low'].values[-w:]
        
        m_m, c_m = fit_ols(x_win, yc)
        if m_m is None: continue
        yt = m_m * x_win + c_m
        m_u, c_u = fit_ols(x_win[yh > yt], yh[yh > yt])
        m_l, c_l = fit_ols(x_win[yl < yt], yl[yl < yt])
        
        if m_u and m_l:
            u_l, l_l = m_u * x_win + c_u, m_l * x_win + c_l
            dist = u_l[-1] - l_l[-1]; th = dist * THRESHOLD_PCT
            
            is_short = latest_price < (l_l[-1] - th) # Using current price for visual
            is_long = latest_price > (u_l[-1] + th)
            
            if (is_short or is_long) and best_visual is None:
                best_visual = {
                    'x': x_win, 'u': u_l, 'l': l_l, 
                    'th': (u_l + th) if is_long else (l_l - th),
                    'type': 'long' if is_long else 'short',
                    'window': w
                }

    if best_visual:
        # Check if this signal is actually an active trade already
        is_active = any(t['window'] == best_visual['window'] for t in active_trades)
        
        if is_active:
            color = 'lime' if best_visual['type'] == 'long' else 'orange'
            style = '-' # Solid line for confirmed trades
        else:
            color = 'red' # Red for "Pending / Candle not closed"
            style = '--' # Dashed line for pending
            
        ax1.plot(best_visual['x'], best_visual['u'], color=color, linestyle=style, linewidth=1.2)
        ax1.plot(best_visual['x'], best_visual['l'], color=color, linestyle=style, linewidth=1.2)
        ax1.plot(best_visual['x'], best_visual['th'], color=color, linestyle=':', linewidth=1.5)

    for t in active_trades:
        ax1.axhline(t['stop'], color='red', linestyle='--', alpha=0.3)
        ax1.axhline(t['target'], color='green', linestyle='--', alpha=0.3)

    # Candles
    up, down = df_closed[df_closed.close >= df_closed.open], df_closed[df_closed.close < df_closed.open]
    for color, d in [('green', up), ('red', down)]:
        ax1.bar(d.index, d.close - d.open, 0.6, bottom=d.open, color=color, zorder=3)
        ax1.bar(d.index, d.high - np.maximum(d.close, d.open), 0.05, bottom=np.maximum(d.close, d.open), color=color, zorder=3)
        ax1.bar(d.index, np.minimum(d.close, d.open) - d.low, 0.05, bottom=d.low, color=color, zorder=3)

    plt.title(f"Next Confirmation in: {get_time_to_close()}")
    buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
    return buf

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200); self.send_header('Content-type', 'text/html'); self.end_headers()
            rows = "".join([f"<tr style='color:{'lime' if t['type']=='long' else 'orange'}'><td>{t['type'].upper()}</td><td>{t['window']}</td><td>{t['entry']:.2f}</td><td>{t['stop']:.2f}</td><td>{t['target']:.2f}</td><td><form method='POST' action='/cancel?w={t['window']}&s={t['type']}'><input type='submit' value='Cancel'></form></td></tr>" for t in active_trades])
            
            html = f"""
            <html>
            <head>
                <style>
                    body{{background:#000;color:#fff;font-family:monospace;text-align:center}}
                    table{{width:90%;margin:20px auto;border-collapse:collapse}}
                    th,td{{padding:12px;border:1px solid #333}}
                    .box{{padding:15px;background:#111;margin:10px auto;width:90%; border: 1px solid #444;}}
                    .timer{{font-size: 24px; color: cyan; margin-bottom: 10px;}}
                    button{{padding:12px; cursor:pointer; background:#222; color:cyan; border:1px solid cyan; font-weight:bold;}}
                </style>
            </head>
            <body>
                <img src='/chart.png?t={int(time.time())}'>
                <div class="timer">Candle Close In: {get_time_to_close()}</div>
                <button onclick="location.reload()">REFRESH DASHBOARD</button>
                <div class="box">
                    <b>Legend:</b> <span style="color:red">Dashed Red = Pending Breakout</span> | 
                    <span style="color:lime">Solid Lime = Active Long</span> | 
                    <span style="color:orange">Solid Orange = Active Short</span>
                </div>
                <table>
                    <thead><tr><th>Side</th><th>Window</th><th>Entry</th><th>Stop (Confirmed)</th><th>Target</th><th>Action</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </body>
            </html>
            """
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
    global current_plot_data, active_trades
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
            full_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if not full_df.empty:
                df_closed, price = full_df.iloc[:-1].copy(), full_df.iloc[-1]['close']
                
                # Exit Logic
                active_trades = [t for t in active_trades if not ((t['type'] == 'short' and (price >= t['stop'] or price <= t['target'])) or (t['type'] == 'long' and (price <= t['stop'] or price >= t['target'])))]

                # Entry Logic (On CONFIRMED closed candles only)
                lc = df_closed['close'].iloc[-1]
                xf = np.arange(len(df_closed))
                for w in range(MAX_WINDOW, MIN_WINDOW -1, -1):
                    xw = xf[-w:]; yc, yh, yl = df_closed['close'].values[-w:], df_closed['high'].values[-w:], df_closed['low'].values[-w:]
                    mm, cm = fit_ols(xw, yc)
                    if mm is None: continue
                    yt = mm * xw + cm
                    mu, cu = fit_ols(xw[yh > yt], yh[yh > yt])
                    ml, cl = fit_ols(xw[yl < yt], yl[yl < yt])
                    if mu and ml:
                        uv, lv = mu * xw[-1] + cu, ml * xw[-1] + cl
                        di = uv - lv; th = di * THRESHOLD_PCT
                        if lc > (uv + th) and not any(t['type'] == 'long' for t in active_trades):
                            active_trades.append({'type': 'long', 'entry': lc, 'stop': uv, 'target': uv + di, 'window': w})
                            break
                        if lc < (lv - th) and not any(t['type'] == 'short' for t in active_trades):
                            active_trades.append({'type': 'short', 'entry': lc, 'stop': lv, 'target': lv - di, 'window': w})
                            break

                current_plot_data = generate_plot(df_closed, price)
        except: pass
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    threading.Thread(target=lambda: HTTPServer(('', PORT), DashboardHandler).serve_forever(), daemon=True).start()
    logic_loop()
