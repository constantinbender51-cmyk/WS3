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

# Global State
current_plot_data = None
trade_pnl_history = []  
active_trades = []      
backtest_results = {'equity': [0], 'win_rate': 0, 'total_pnl': 0, 'count': 0}
backtest_progress = 0.0
last_processed_timestamp = None  # <--- CRITICAL FIX STATE
exchange = ccxt.binance()

# =============================================================================
# MATH & HELPERS
# =============================================================================
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

# =============================================================================
# DATA FETCHING & BACKTEST
# =============================================================================
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
    print("Starting Backtest...")
    df = fetch_historical_data(BACKTEST_HOURS + MAX_WINDOW)
    if df.empty: return

    equity = [0]; trades = []; current_active = []
    total_steps = len(df) - 1 - MAX_WINDOW
    x_indices = np.arange(len(df))

    for idx, i in enumerate(range(MAX_WINDOW, len(df) - 1)):
        if idx % 100 == 0: backtest_progress = (idx / total_steps) * 100
        df_win = df.iloc[i - MAX_WINDOW:i]
        price = df.iloc[i]['close']
        
        # Backtest Exit Logic
        new_active = []
        for t in current_active:
            w = t['window']
            xw = x_indices[i-w:i]; yc = df['close'].values[i-w:i]; yh = df['high'].values[i-w:i]; yl = df['low'].values[i-w:i]
            mm, cm = fit_ols(xw, yc)
            closed = False; p = 0
            if mm is not None:
                yt = mm * xw + cm
                mu, cu = fit_ols(xw[yh > yt], yh[yh > yt]); ml, cl = fit_ols(xw[yl < yt], yl[yl < yt])
                if mu and ml:
                    # Trailing Stops in Backtest
                    stop_long = ml * xw[-1] + cl
                    stop_short = mu * xw[-1] + cu
                    
                    if t['type'] == 'long':
                        if price <= stop_long or price >= t['target']:
                            p = price - t['entry']; closed = True
                    elif t['type'] == 'short':
                        if price >= stop_short or price <= t['target']:
                            p = t['entry'] - price; closed = True
            
            if closed: equity.append(equity[-1] + p); trades.append(p)
            else: new_active.append(t)
        current_active = new_active
        
        # Backtest Entry Logic
        has_long = any(t['type'] == 'long' for t in current_active)
        has_short = any(t['type'] == 'short' for t in current_active)
        
        if not has_long or not has_short:
            x = x_indices[i - MAX_WINDOW:i]; last_c = df_win['close'].iloc[-1]
            best_s, best_l = None, None
            for w in range(MAX_WINDOW, MIN_WINDOW - 1, -1):
                xw = x[-w:]; yc = df_win['close'].values[-w:]; yh = df_win['high'].values[-w:]; yl = df_win['low'].values[-w:]
                mm, cm = fit_ols(xw, yc)
                if mm is None: continue
                yt = mm * xw + cm
                mu, cu = fit_ols(xw[yh > yt], yh[yh > yt]); ml, cl = fit_ols(xw[yl < yt], yl[yl < yt])
                if mu and ml:
                    uv, lv = mu * xw[-1] + cu, ml * xw[-1] + cl
                    di = uv - lv; th = di * THRESHOLD_PCT
                    if not has_short and best_s is None and last_c < (lv - th):
                        best_s = {'type': 'short', 'entry': last_c, 'target': lv - di, 'window': w}
                    if not has_long and best_l is None and last_c > (uv + th):
                        best_l = {'type': 'long', 'entry': last_c, 'target': uv + di, 'window': w}
                    if best_s and best_l: break
            if best_s: current_active.append(best_s)
            if best_l: current_active.append(best_l)

    wins = [p for p in trades if p > 0]
    backtest_results = {'equity': equity, 'win_rate': len(wins)/len(trades) if trades else 0, 'total_pnl': sum(trades), 'count': len(trades)}
    backtest_progress = 100.0
    print("Backtest Complete.")

# =============================================================================
# LIVE PLOTTING
# =============================================================================
def generate_plot(df_closed, latest_price):
    plt.figure(figsize=(15, 12))
    plt.style.use('dark_background')
    
    # Equity
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(backtest_results['equity'], color='cyan', linewidth=1)
    ax2.set_title(f"Yearly Backtest Equity (Total PnL: {backtest_results['total_pnl']:.2f})")
    ax2.grid(color='#333', linestyle='--', linewidth=0.5)

    # Chart
    ax1 = plt.subplot(2, 1, 1)
    x_full = np.arange(len(df_closed))
    
    visual_window = None; visual_type = None; is_active_visual = False
    
    # Priority: Show Active Trade
    if active_trades:
        visual_window = active_trades[0]['window']
        visual_type = active_trades[0]['type']
        is_active_visual = True
    else:
        # Scan for pending signals
        for w in range(MAX_WINDOW, MIN_WINDOW - 1, -1):
            if len(df_closed) < w: continue
            xw = x_full[-w:]; yc = df_closed['close'].values[-w:]; yh = df_closed['high'].values[-w:]; yl = df_closed['low'].values[-w:]
            mm, cm = fit_ols(xw, yc)
            if mm is None: continue
            yt = mm * xw + cm
            mu, cu = fit_ols(xw[yh > yt], yh[yh > yt]); ml, cl = fit_ols(xw[yl < yt], yl[yl < yt])
            if mu and ml:
                uv, lv = mu * xw[-1] + cu, ml * xw[-1] + cl
                di = uv - lv; th = di * THRESHOLD_PCT
                if latest_price < (lv - th): visual_window = w; visual_type = 'short'; break
                elif latest_price > (uv + th): visual_window = w; visual_type = 'long'; break

    if visual_window:
        xw = x_full[-visual_window:]; yc = df_closed['close'].values[-visual_window:]; yh = df_closed['high'].values[-visual_window:]; yl = df_closed['low'].values[-visual_window:]
        mm, cm = fit_ols(xw, yc)
        if mm:
            yt = mm * xw + cm; mu, cu = fit_ols(xw[yh > yt], yh[yh > yt]); ml, cl = fit_ols(xw[yl < yt], yl[yl < yt])
            if mu and ml:
                ul, ll = mu * xw + cu, ml * xw + cl
                di = ul[-1] - ll[-1]; th = di * THRESHOLD_PCT
                
                color = 'lime' if visual_type == 'long' else 'orange'
                style = '-' if is_active_visual else '--'
                if not is_active_visual: color = 'red'

                ax1.plot(xw, ul, color=color, linestyle=style, linewidth=1.2)
                ax1.plot(xw, ll, color=color, linestyle=style, linewidth=1.2)
                thresh_line = (ul + th) if visual_type == 'long' else (ll - th)
                ax1.plot(xw, thresh_line, color=color, linestyle=':', linewidth=1.5)
                
                if is_active_visual:
                    stop_val = ll[-1] if visual_type == 'long' else ul[-1]
                    ax1.axhline(stop_val, color='red', linestyle='-', linewidth=2, label="Dynamic Stop")

    for t in active_trades:
        ax1.axhline(t['target'], color='lime', linestyle='--', alpha=0.5, label='Target')

    up = df_closed[df_closed.close >= df_closed.open]; down = df_closed[df_closed.close < df_closed.open]
    for col, d in [('green', up), ('red', down)]:
        ax1.bar(d.index, d.close - d.open, CANDLE_WIDTH, bottom=d.open, color=col, zorder=3)
        ax1.bar(d.index, d.high - np.maximum(d.close, d.open), WICK_WIDTH, bottom=np.maximum(d.close, d.open), color=col, zorder=3)
        ax1.bar(d.index, np.minimum(d.close, d.open) - d.low, WICK_WIDTH, bottom=d.low, color=col, zorder=3)

    ax1.set_title(f"Live Chart | Active Window: {visual_window if visual_window else 'Scanning...'}")
    buf = BytesIO(); plt.savefig(buf, format='png'); plt.close(); buf.seek(0)
    return buf

# =============================================================================
# SERVER
# =============================================================================
class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200); self.send_header('Content-type', 'text/html'); self.end_headers()
            p_bar = f"<div style='width:90%; background:#222; margin:10px auto; border:1px solid #444;'><div style='width:{backtest_progress}%; background:cyan; height:12px;'></div></div>"
            rows = ""
            for t in active_trades:
                rows += f"<tr style='color:{'lime' if t['type']=='long' else 'orange'}'><td>{t['type'].upper()}</td><td>{t['window']}</td><td>{t['entry']:.2f}</td><td>DYNAMIC</td><td>{t['target']:.2f}</td><td><form method='POST' action='/cancel?w={t['window']}&s={t['type']}'><input type='submit' value='CLOSE'></form></td></tr>"
            
            html = f"""
            <html><head><title>OLS Bot</title>
            <style>body{{background:#050505;color:#e0e0e0;font-family:'Courier New',monospace;text-align:center;margin:0}}h3{{color:#888}}table{{width:90%;margin:20px auto;border-collapse:collapse;background:#111}}th,td{{padding:12px;border:1px solid #333}}th{{background:#222}}.box{{padding:15px;background:#111;margin:10px auto;width:90%;border:1px solid #444;display:flex;justify-content:space-around}}.timer{{color:cyan;font-size:24px;font-weight:bold;margin:10px 0}}button{{padding:12px 24px;cursor:pointer;background:#222;color:cyan;border:1px solid cyan;font-weight:bold;font-size:16px}}input[type=submit]{{background:#300;color:red;border:1px solid red;cursor:pointer}}</style>
            </head><body><img src='/chart.png?t={int(time.time())}' style='width:95%;margin-top:10px'><div class='timer'>Next Confirmation: {get_time_to_close()}</div><button onclick='location.reload()'>REFRESH DASHBOARD</button><h3>Backtest Progress: {backtest_progress:.1f}%</h3>{p_bar}<div class='box'><div><b>Live Realized:</b> {sum(trade_pnl_history):.2f}</div><div><b>Backtest Year PnL:</b> {backtest_results['total_pnl']:.2f}</div><div><b>Win Rate:</b> {backtest_results['win_rate']:.1%}</div></div><table><thead><tr><th>Side</th><th>Window</th><th>Entry</th><th>Stop (Trailing)</th><th>Target</th><th>Action</th></tr></thead><tbody>{rows}</tbody></table></body></html>"""
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

# =============================================================================
# MAIN LOGIC LOOP
# =============================================================================
def logic_loop():
    global current_plot_data, active_trades, trade_pnl_history, last_processed_timestamp
    
    # ----------------------------------------------------
    # INITIALIZATION PHASE: Set baseline to avoid startup entry
    # ----------------------------------------------------
    print("Initializing... Setting baseline timestamp.")
    while last_processed_timestamp is None:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=5)
            if ohlcv:
                # Store the timestamp of the LAST CLOSED candle
                # ohlcv[-1] is open/current. ohlcv[-2] is the last closed.
                last_processed_timestamp = ohlcv[-2][0] 
                print(f"Baseline Timestamp Set: {last_processed_timestamp}")
        except: pass
        time.sleep(2)

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
            full_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            if not full_df.empty:
                df_closed = full_df.iloc[:-1].copy()
                current_price = full_df.iloc[-1]['close'] # Live Price
                current_closed_ts = df_closed.iloc[-1]['timestamp'] # Latest closed candle TS
                x_idx = np.arange(len(df_closed))
                
                # ----------------------------------------
                # 1. DYNAMIC STOP / EXIT LOGIC (ALWAYS RUNS)
                # ----------------------------------------
                remaining = []
                for t in active_trades:
                    w = t['window']
                    xw = x_idx[-w:]; yc = df_closed['close'].values[-w:]; yh = df_closed['high'].values[-w:]; yl = df_closed['low'].values[-w:]
                    mm, cm = fit_ols(xw, yc)
                    closed = False; pnl = 0
                    
                    if mm is not None:
                        yt = mm * xw + cm
                        mu, cu = fit_ols(xw[yh > yt], yh[yh > yt]); ml, cl = fit_ols(xw[yl < yt], yl[yl < yt])
                        if mu and ml:
                            curr_stop_long = ml * xw[-1] + cl
                            curr_stop_short = mu * xw[-1] + cu
                            
                            if t['type'] == 'long':
                                if current_price <= curr_stop_long or current_price >= t['target']:
                                    pnl = current_price - t['entry']; closed = True
                                    print(f"CLOSED LONG: {current_price}")
                            elif t['type'] == 'short':
                                if current_price >= curr_stop_short or current_price <= t['target']:
                                    pnl = t['entry'] - current_price; closed = True
                                    print(f"CLOSED SHORT: {current_price}")

                    if closed: trade_pnl_history.append(pnl)
                    else: remaining.append(t)
                active_trades = remaining 

                # ----------------------------------------
                # 2. ENTRY LOGIC (RUNS ONLY ON NEW CANDLE)
                # ----------------------------------------
                if current_closed_ts > last_processed_timestamp:
                    print(f"NEW CANDLE DETECTED: {current_closed_ts}")
                    last_processed_timestamp = current_closed_ts # Update State
                    
                    has_long = any(t['type'] == 'long' for t in active_trades)
                    has_short = any(t['type'] == 'short' for t in active_trades)
                    last_c = df_closed['close'].iloc[-1]
                    
                    for w in range(MAX_WINDOW, MIN_WINDOW -1, -1):
                        if has_long and has_short: break
                        xw = x_idx[-w:]; yc = df_closed['close'].values[-w:]; yh = df_closed['high'].values[-w:]; yl = df_closed['low'].values[-w:]
                        mm, cm = fit_ols(xw, yc)
                        if mm is None: continue
                        yt = mm * xw + cm
                        mu, cu = fit_ols(xw[yh > yt], yh[yh > yt]); ml, cl = fit_ols(xw[yl < yt], yl[yl < yt])
                        
                        if mu and ml:
                            uv, lv = mu * xw[-1] + cu, ml * xw[-1] + cl
                            di = uv - lv; th = di * THRESHOLD_PCT
                            if not has_long and last_c > (uv + th):
                                active_trades.append({'type': 'long', 'entry': last_c, 'target': uv + di, 'window': w})
                                has_long = True
                                print(f"OPEN LONG: {last_c}")
                            if not has_short and last_c < (lv - th):
                                active_trades.append({'type': 'short', 'entry': last_c, 'target': lv - di, 'window': w})
                                has_short = True
                                print(f"OPEN SHORT: {last_c}")

                current_plot_data = generate_plot(df_closed, current_price)
        except Exception as e: print(f"Loop: {e}")
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    print(f"Starting OLS Bot on {SYMBOL}...")
    threading.Thread(target=run_backtest, daemon=True).start()
    threading.Thread(target=lambda: HTTPServer(('', PORT), DashboardHandler).serve_forever(), daemon=True).start()
    print(f"Dashboard: http://localhost:{PORT}")
    logic_loop()
