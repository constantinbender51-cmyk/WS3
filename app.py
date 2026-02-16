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
LIMIT = 200                 
PORT = 8080                 
THRESHOLD_PCT = 0.10        
UPDATE_INTERVAL = 10        
MIN_WINDOW = 10             
MAX_WINDOW = 100            
BACKTEST_HOURS = 365 * 24   
CANDLE_WIDTH = 0.6          
WICK_WIDTH = 0.05           
# =============================================================================

current_plot_data = None
trade_pnl_history = []  
active_trades = []      
backtest_results = {'equity': [0], 'win_rate': 0, 'total_pnl': 0, 'count': 0}
backtest_progress = 0.0
last_processed_timestamp = None
exchange = ccxt.binance()

def fit_ols(x, y):
    if len(x) < 2: return None, None
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def get_time_to_close():
    now = datetime.utcnow()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    rem = next_hour - now
    return f"{rem.seconds // 60:02d}m {rem.seconds % 60:02d}s"

def fetch_historical_data(total_limit):
    all_ohlcv = []
    since = exchange.milliseconds() - (total_limit * 60 * 60 * 1000)
    while len(all_ohlcv) < total_limit:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(0.1)
        except: break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df.iloc[-total_limit:] if not df.empty else pd.DataFrame()

def run_backtest():
    global backtest_results, backtest_progress
    df = fetch_historical_data(BACKTEST_HOURS + MAX_WINDOW)
    if df.empty: return
    equity = [0]; trades = []; current_active = []
    total_steps = len(df) - 1 - MAX_WINDOW
    x_indices = np.arange(len(df))

    for idx, i in enumerate(range(MAX_WINDOW, len(df) - 1)):
        if idx % 100 == 0: backtest_progress = (idx / total_steps) * 100
        price = df.iloc[i]['close']
        
        # 1. Backtest Exit
        new_active = []
        for t in current_active:
            w = t['window']
            fit_slice = slice(i - w - 1, i - 1)
            xw = x_indices[fit_slice]; yc = df['close'].values[fit_slice]; yh = df['high'].values[fit_slice]; yl = df['low'].values[fit_slice]
            
            mm, cm = fit_ols(xw, yc)
            closed = False; p = 0
            if mm is not None:
                yt = mm * xw + cm
                mu, cu = fit_ols(xw[yh > yt], yh[yh > yt]); ml, cl = fit_ols(xw[yl < yt], yl[yl < yt])
                if mu and ml:
                    # Project to current candle 'i'
                    stop_long = ml * x_indices[i] + cl
                    stop_short = mu * x_indices[i] + cu
                    if t['type'] == 'long':
                        if price <= stop_long or price >= t['target']: p = price - t['entry']; closed = True
                    elif t['type'] == 'short':
                        if price >= stop_short or price <= t['target']: p = t['entry'] - price; closed = True
            if closed: equity.append(equity[-1] + p); trades.append(p)
            else: new_active.append(t)
        current_active = new_active
        
        # 2. Backtest Entry
        has_long = any(t['type'] == 'long' for t in current_active)
        has_short = any(t['type'] == 'short' for t in current_active)
        if not has_long or not has_short:
            last_c = df.iloc[i-1]['close'] # Breakout candle
            
            for w in range(MAX_WINDOW, MIN_WINDOW - 1, -1):
                # Fit on i-w-1 to i-2 (Exclude breakout candle i-1)
                fit_slice = slice(i - w - 1, i - 1)
                xw = x_indices[fit_slice]; yc = df['close'].values[fit_slice]; yh = df['high'].values[fit_slice]; yl = df['low'].values[fit_slice]
                mm, cm = fit_ols(xw, yc)
                if mm is None: continue
                yt = mm * xw + cm
                mu, cu = fit_ols(xw[yh > yt], yh[yh > yt]); ml, cl = fit_ols(xw[yl < yt], yl[yl < yt])
                
                if mu and ml:
                    # Project to breakout candle i-1
                    proj_idx = x_indices[i-1]
                    proj_u = mu * proj_idx + cu
                    proj_l = ml * proj_idx + cl
                    dist = proj_u - proj_l; th = dist * THRESHOLD_PCT
                    
                    if not has_short and last_c < (proj_l - th):
                        current_active.append({'type': 'short', 'entry': last_c, 'target': proj_l - dist, 'window': w}); break
                    if not has_long and last_c > (proj_u + th):
                        current_active.append({'type': 'long', 'entry': last_c, 'target': proj_u + dist, 'window': w}); break

    wins = [p for p in trades if p > 0]
    backtest_results = {'equity': equity, 'win_rate': len(wins)/len(trades) if trades else 0, 'total_pnl': sum(trades), 'count': len(trades)}
    backtest_progress = 100.0

def generate_plot(df_closed, active_snapshot):
    plt.figure(figsize=(15, 12))
    plt.style.use('dark_background')
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(backtest_results['equity'], color='cyan', linewidth=1)
    ax2.set_title(f"Yearly Backtest Equity (Total PnL: {backtest_results['total_pnl']:.2f})")
    ax2.grid(color='#333', linestyle='--', linewidth=0.5)

    ax1 = plt.subplot(2, 1, 1)
    x_full = np.arange(len(df_closed))
    last_close = df_closed['close'].iloc[-1]
    
    # Draw all currently valid breakout lines (Faded)
    for w in range(MIN_WINDOW, MAX_WINDOW + 1):
        fit_slice = slice(-w - 1, -1)
        x_fit = x_full[fit_slice]
        if len(x_fit) < 2: continue
        yc = df_closed['close'].values[fit_slice]
        yh = df_closed['high'].values[fit_slice]
        yl = df_closed['low'].values[fit_slice]
        mm, cm = fit_ols(x_fit, yc)
        if mm is not None:
            yt = mm * x_fit + cm
            mu, cu = fit_ols(x_fit[yh > yt], yh[yh > yt])
            ml, cl = fit_ols(x_fit[yl < yt], yl[yl < yt])
            if mu and ml:
                idx_last = x_full[-1]
                proj_u = mu * idx_last + cu
                proj_l = ml * idx_last + cl
                dist = proj_u - proj_l; th = dist * THRESHOLD_PCT
                if last_close > (proj_u + th) or last_close < (proj_l - th):
                    # Valid Breakout
                    x_draw = x_full[-w:]
                    ax1.plot(x_draw, mu * x_draw + cu, color='#555', linestyle=':', linewidth=1)
                    ax1.plot(x_draw, ml * x_draw + cl, color='#555', linestyle=':', linewidth=1)

    # Draw Active Trades (Solid)
    for t in active_snapshot:
        w = t['window']
        fit_slice = slice(-w - 1, -1)
        x_fit = x_full[fit_slice]
        yc = df_closed['close'].values[fit_slice]; yh = df_closed['high'].values[fit_slice]; yl = df_closed['low'].values[fit_slice]
        mm, cm = fit_ols(x_fit, yc)
        if mm is not None:
            yt = mm * x_fit + cm
            mu, cu = fit_ols(x_fit[yh > yt], yh[yh > yt]); ml, cl = fit_ols(x_fit[yl < yt], yl[yl < yt])
            if mu and ml:
                x_draw = x_full[-w:]
                col = 'lime' if t['type']=='long' else 'orange'
                ax1.plot(x_draw, mu * x_draw + cu, color=col, linewidth=1.5)
                ax1.plot(x_draw, ml * x_draw + cl, color=col, linewidth=1.5)
        
        ax1.axhline(t['target'], color='lime', linestyle='--', label='Target')

    up = df_closed[df_closed.close >= df_closed.open]; down = df_closed[df_closed.close < df_closed.open]
    for col, d in [('green', up), ('red', down)]:
        ax1.bar(d.index, d.close - d.open, CANDLE_WIDTH, bottom=d.open, color=col, zorder=3)
        ax1.bar(d.index, d.high - np.maximum(d.close, d.open), WICK_WIDTH, bottom=np.maximum(d.close, d.open), color=col, zorder=3)
        ax1.bar(d.index, np.minimum(d.close, d.open) - d.low, WICK_WIDTH, bottom=d.low, color=col, zorder=3)

    buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
    return buf

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200); self.send_header('Content-type', 'text/html'); self.end_headers()
            p_bar = f"<div style='width:90%; background:#222; margin:10px auto; border:1px solid #444;'><div style='width:{backtest_progress}%; background:cyan; height:12px;'></div></div>"
            rows = ""
            for t in active_trades:
                rows += f"<tr style='color:{'lime' if t['type']=='long' else 'orange'}'><td>{t['type'].upper()}</td><td>{t['window']}</td><td>{t['entry']:.2f}</td><td>DYNAMIC</td><td>{t['target']:.2f}</td><td><form method='POST' action='/cancel?w={t['window']}&s={t['type']}'><input type='submit' value='CLOSE'></form></td></tr>"
            html = f"<html><head><title>OLS Bot</title><style>body{{background:#050505;color:#e0e0e0;font-family:'Courier New';text-align:center}}table{{width:90%;margin:20px auto;border-collapse:collapse;background:#111}}th,td{{padding:12px;border:1px solid #333}}button{{padding:12px;background:#222;color:cyan;border:1px solid cyan}}</style></head><body><img src='/chart.png?t={int(time.time())}' style='width:95%'><div style='color:cyan;font-size:24px;margin:10px'>Next: {get_time_to_close()}</div><button onclick='location.reload()'>REFRESH</button>{p_bar}<table><thead><tr><th>Side</th><th>Window</th><th>Entry</th><th>Stop</th><th>Target</th><th>Action</th></tr></thead><tbody>{rows}</tbody></table></body></html>"
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
    global current_plot_data, active_trades, trade_pnl_history, last_processed_timestamp
    while last_processed_timestamp is None:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=5)
            if ohlcv: last_processed_timestamp = ohlcv[-2][0]
        except: pass
        time.sleep(2)

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
            full_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if not full_df.empty:
                df_closed = full_df.iloc[:-1].copy()
                current_price = full_df.iloc[-1]['close']
                closed_ts = df_closed.iloc[-1]['timestamp']
                x_full = np.arange(len(df_closed))
                
                # 1. Exit Logic
                remaining = []
                for t in active_trades:
                    w = t['window']
                    fit_slice = slice(-w - 1, -1)
                    x_fit = x_full[fit_slice]
                    yc = df_closed['close'].values[fit_slice]; yh = df_closed['high'].values[fit_slice]; yl = df_closed['low'].values[fit_slice]
                    mm, cm = fit_ols(x_fit, yc)
                    closed = False; pnl = 0
                    if mm is not None:
                        yt = mm * x_fit + cm
                        mu, cu = fit_ols(x_fit[yh > yt], yh[yh > yt]); ml, cl = fit_ols(x_fit[yl < yt], yl[yl < yt])
                        if mu and ml:
                            # Project to live index (next candle index)
                            proj_idx = x_full[-1] + 1
                            curr_stop_long = ml * proj_idx + cl
                            curr_stop_short = mu * proj_idx + cu
                            
                            if t['type'] == 'long':
                                if current_price <= curr_stop_long or current_price >= t['target']:
                                    pnl = current_price - t['entry']; closed = True
                            elif t['type'] == 'short':
                                if current_price >= curr_stop_short or current_price <= t['target']:
                                    pnl = t['entry'] - current_price; closed = True
                    if closed: trade_pnl_history.append(pnl)
                    else: remaining.append(t)
                active_trades = remaining

                # 2. Entry Logic (New Candle Only)
                if closed_ts > last_processed_timestamp:
                    last_processed_timestamp = closed_ts
                    last_c = df_closed['close'].iloc[-1]
                    for w in range(MAX_WINDOW, MIN_WINDOW - 1, -1):
                        fit_slice = slice(-w - 1, -1)
                        x_fit = x_full[fit_slice]
                        yc = df_closed['close'].values[fit_slice]; yh = df_closed['high'].values[fit_slice]; yl = df_closed['low'].values[fit_slice]
                        mm, cm = fit_ols(x_fit, yc)
                        if mm is None: continue
                        yt = mm * x_fit + cm
                        mu, cu = fit_ols(x_fit[yh > yt], yh[yh > yt]); ml, cl = fit_ols(x_fit[yl < yt], yl[yl < yt])
                        if mu and ml:
                            proj_idx = x_full[-1]
                            proj_u = mu * proj_idx + cu
                            proj_l = ml * proj_idx + cl
                            dist = proj_u - proj_l; th = dist * THRESHOLD_PCT
                            
                            if not any(t['type']=='long' for t in active_trades) and last_c > (proj_u + th):
                                active_trades.append({'type': 'long', 'entry': last_c, 'target': proj_u + dist, 'window': w}); break
                            if not any(t['type']=='short' for t in active_trades) and last_c < (proj_l - th):
                                active_trades.append({'type': 'short', 'entry': last_c, 'target': proj_l - dist, 'window': w}); break

                current_plot_data = generate_plot(df_closed, active_trades)
        except Exception as e: print(f"Loop: {e}")
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    threading.Thread(target=run_backtest, daemon=True).start()
    threading.Thread(target=lambda: HTTPServer(('', PORT), DashboardHandler).serve_forever(), daemon=True).start()
    logic_loop()
