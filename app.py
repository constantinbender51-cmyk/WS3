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
LIMIT = 200                 # Live chart lookback
PORT = 8080                 # Web dashboard port
THRESHOLD_PCT = 0.10        # Breakout distance (10% of channel width)
UPDATE_INTERVAL = 10        # Seconds between logic ticks
MIN_WINDOW = 10             # Minimum OLS lookback
MAX_WINDOW = 100            # Maximum OLS lookback
BACKTEST_HOURS = 365 * 24   # 1 Year of backtest data
CANDLE_WIDTH = 0.6          
WICK_WIDTH = 0.05           
# =============================================================================

# Global State
current_plot_data = None
trade_pnl_history = []  
active_trades = []      
backtest_results = {'equity': [0], 'win_rate': 0, 'total_pnl': 0, 'count': 0}
backtest_progress = 0.0
exchange = ccxt.binance()

# =============================================================================
# MATHEMATICAL HELPER FUNCTIONS
# =============================================================================
def fit_ols(x, y):
    """Performs Ordinary Least Squares regression."""
    if len(x) < 2: return None, None
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def get_time_to_close():
    """Returns formatted string for time until next candle close."""
    now = datetime.utcnow()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    rem = next_hour - now
    return f"{rem.seconds // 60:02d}m {rem.seconds % 60:02d}s"

# =============================================================================
# DATA FETCHING
# =============================================================================
def fetch_historical_data(total_limit):
    """Fetches large dataset for backtesting in batches."""
    all_ohlcv = []
    since = exchange.milliseconds() - (total_limit * 60 * 60 * 1000)
    while len(all_ohlcv) < total_limit:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(0.1) # Rate limit kindness
        except: break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df.iloc[-total_limit:] if not df.empty else pd.DataFrame()

# =============================================================================
# BACKTEST LOGIC
# =============================================================================
def run_backtest():
    global backtest_results, backtest_progress
    print("Starting Backtest...")
    df = fetch_historical_data(BACKTEST_HOURS + MAX_WINDOW)
    
    if df.empty:
        print("Backtest failed: No data")
        return

    equity = [0]; trades = []; current_active = []
    total_steps = len(df) - 1 - MAX_WINDOW
    
    # Pre-calculate x arrays for speed
    x_indices = np.arange(len(df))

    for idx, i in enumerate(range(MAX_WINDOW, len(df) - 1)):
        # Progress Tracking
        if idx % 100 == 0: backtest_progress = (idx / total_steps) * 100
            
        df_win = df.iloc[i - MAX_WINDOW:i]
        price = df.iloc[i]['close'] # Entry/Exit price (Open of next candle approx)
        
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
        
        # 2. Entry Logic (Limit: 1 Long, 1 Short)
        # Optimization: Only calculate if we have room for a trade
        has_long = any(t['type'] == 'long' for t in current_active)
        has_short = any(t['type'] == 'short' for t in current_active)
        
        if not has_long or not has_short:
            x = x_indices[i - MAX_WINDOW:i]
            last_c = df_win['close'].iloc[-1]
            
            best_s, best_l = None, None
            # Scan largest window first
            for w in range(MAX_WINDOW, MIN_WINDOW - 1, -1):
                xw = x[-w:]; yc = df_win['close'].values[-w:]; yh = df_win['high'].values[-w:]; yl = df_win['low'].values[-w:]
                
                mm, cm = fit_ols(xw, yc)
                if mm is None: continue
                yt = mm * xw + cm
                
                mu, cu = fit_ols(xw[yh > yt], yh[yh > yt])
                ml, cl = fit_ols(xw[yl < yt], yl[yl < yt])
                
                if mu is not None and ml is not None:
                    uv, lv = mu * xw[-1] + cu, ml * xw[-1] + cl
                    di = uv - lv; th = di * THRESHOLD_PCT
                    
                    if not has_short and best_s is None and last_c < (lv - th):
                        best_s = {'type': 'short', 'entry': last_c, 'stop': lv, 'target': lv - di}
                    
                    if not has_long and best_l is None and last_c > (uv + th):
                        best_l = {'type': 'long', 'entry': last_c, 'stop': uv, 'target': uv + di}
                    
                    if best_s and best_l: break # Found both, stop scanning

            if best_s: current_active.append(best_s)
            if best_l: current_active.append(best_l)

    wins = [p for p in trades if p > 0]
    backtest_results = {
        'equity': equity, 
        'win_rate': len(wins)/len(trades) if trades else 0, 
        'total_pnl': sum(trades), 
        'count': len(trades)
    }
    backtest_progress = 100.0
    print("Backtest Complete.")

# =============================================================================
# LIVE VISUALIZATION
# =============================================================================
def generate_plot(df_closed, latest_price):
    plt.figure(figsize=(15, 12))
    plt.style.use('dark_background')
    
    # Subplot 2: Backtest Equity Curve
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(backtest_results['equity'], color='cyan', linewidth=1)
    ax2.set_title(f"Yearly Backtest Equity (Total PnL: {backtest_results['total_pnl']:.2f})")
    ax2.grid(color='#333', linestyle='--', linewidth=0.5)

    # Subplot 1: Live Candle Chart
    ax1 = plt.subplot(2, 1, 1)
    x_full = np.arange(len(df_closed))
    last_close = df_closed['close'].iloc[-1]
    
    # LOGIC: Determine which window to visualize
    # Priority 1: If trade is active, show THAT window.
    # Priority 2: If no trade, show the largest pending breakout window.
    
    visual_window = None
    visual_type = None
    is_active_visual = False
    
    if active_trades:
        # Show the window of the first active trade
        visual_window = active_trades[0]['window']
        visual_type = active_trades[0]['type']
        is_active_visual = True
    else:
        # Scan for potential breakouts to visualize pending signals
        for w in range(MAX_WINDOW, MIN_WINDOW - 1, -1):
            if len(df_closed) < w: continue
            xw = x_full[-w:]
            yc, yh, yl = df_closed['close'].values[-w:], df_closed['high'].values[-w:], df_closed['low'].values[-w:]
            
            mm, cm = fit_ols(xw, yc)
            if mm is None: continue
            yt = mm * xw + cm
            
            mu, cu = fit_ols(xw[yh > yt], yh[yh > yt])
            ml, cl = fit_ols(xw[yl < yt], yl[yl < yt])
            
            if mu is not None and ml is not None:
                uv, lv = mu * xw[-1] + cu, ml * xw[-1] + cl
                di = uv - lv; th = di * THRESHOLD_PCT
                
                # Check Pending Breakout (using latest price, not closed)
                if latest_price < (lv - th):
                    visual_window = w; visual_type = 'short'; break
                elif latest_price > (uv + th):
                    visual_window = w; visual_type = 'long'; break

    # Render Lines if a window was selected
    if visual_window:
        xw = x_full[-visual_window:]
        yc, yh, yl = df_closed['close'].values[-visual_window:], df_closed['high'].values[-visual_window:], df_closed['low'].values[-visual_window:]
        
        mm, cm = fit_ols(xw, yc)
        if mm is not None:
            yt = mm * xw + cm
            mu, cu = fit_ols(xw[yh > yt], yh[yh > yt])
            ml, cl = fit_ols(xw[yl < yt], yl[yl < yt])
            
            if mu and ml:
                ul, ll = mu * xw + cu, ml * xw + cl
                di = ul[-1] - ll[-1]; th = di * THRESHOLD_PCT
                
                # Colors: Lime (Active Long), Orange (Active Short), Red (Pending)
                if is_active_visual:
                    color = 'lime' if visual_type == 'long' else 'orange'
                    style = '-'
                else:
                    color = 'red'
                    style = '--' # Dashed for pending
                
                # Draw Channel
                ax1.plot(xw, ul, color=color, linestyle=style, linewidth=1.2, alpha=0.8)
                ax1.plot(xw, ll, color=color, linestyle=style, linewidth=1.2, alpha=0.8)
                
                # Draw Threshold Line
                thresh_line = (ul + th) if visual_type == 'long' else (ll - th)
                ax1.plot(xw, thresh_line, color=color, linestyle=':', linewidth=1.5)

    # Draw Stop/Target Lines for active trades
    for t in active_trades:
        ax1.axhline(t['stop'], color='red', linestyle='--', alpha=0.5, label='Stop')
        ax1.axhline(t['target'], color='lime', linestyle='--', alpha=0.5, label='Target')

    # Draw Candles
    up = df_closed[df_closed.close >= df_closed.open]
    down = df_closed[df_closed.close < df_closed.open]
    
    for col, d in [('green', up), ('red', down)]:
        ax1.bar(d.index, d.close - d.open, CANDLE_WIDTH, bottom=d.open, color=col, zorder=3)
        ax1.bar(d.index, d.high - np.maximum(d.close, d.open), WICK_WIDTH, bottom=np.maximum(d.close, d.open), color=col, zorder=3)
        ax1.bar(d.index, np.minimum(d.close, d.open) - d.low, WICK_WIDTH, bottom=d.low, color=col, zorder=3)

    ax1.set_title(f"Live Chart | Active Window: {visual_window if visual_window else 'Scanning...'}")
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

# =============================================================================
# WEB SERVER HANDLER
# =============================================================================
class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Progress Bar HTML
            p_bar = f"<div style='width:90%; background:#222; margin:10px auto; border:1px solid #444;'><div style='width:{backtest_progress}%; background:cyan; height:12px;'></div></div>"
            
            # Table Rows
            rows = ""
            for t in active_trades:
                rows += f"""
                <tr style='color:{'lime' if t['type']=='long' else 'orange'}'>
                    <td>{t['type'].upper()}</td>
                    <td>{t['window']}</td>
                    <td>{t['entry']:.2f}</td>
                    <td>{t['stop']:.2f}</td>
                    <td>{t['target']:.2f}</td>
                    <td>
                        <form method='POST' action='/cancel?w={t['window']}&s={t['type']}'>
                            <input type='submit' value='CLOSE'>
                        </form>
                    </td>
                </tr>"""

            html = f"""
            <html>
            <head>
                <title>OLS Breakout Bot</title>
                <style>
                    body {{ background:#050505; color:#e0e0e0; font-family:'Courier New', monospace; text-align:center; margin:0; }}
                    h3 {{ margin: 10px 0; color: #888; }}
                    table {{ width:90%; margin:20px auto; border-collapse:collapse; background:#111; }}
                    th, td {{ padding:12px; border:1px solid #333; }}
                    th {{ background:#222; color:#fff; }}
                    .box {{ padding:15px; background:#111; margin:10px auto; width:90%; border:1px solid #444; display:flex; justify-content:space-around; }}
                    .timer {{ color:cyan; font-size:24px; font-weight:bold; margin: 10px 0; }}
                    button {{ padding:12px 24px; cursor:pointer; background:#222; color:cyan; border:1px solid cyan; font-weight:bold; font-size:16px; }}
                    button:hover {{ background:cyan; color:#000; }}
                    input[type=submit] {{ background:#300; color:red; border:1px solid red; cursor:pointer; padding:5px 10px; }}
                </style>
            </head>
            <body>
                <img src='/chart.png?t={int(time.time())}' style='width:95%; margin-top:10px;'>
                
                <div class='timer'>Next Confirmation: {get_time_to_close()}</div>
                <button onclick='location.reload()'>REFRESH DASHBOARD</button>
                
                <h3>Backtest Progress: {backtest_progress:.1f}%</h3>
                {p_bar}
                
                <div class='box'>
                    <div><b>Live Realized:</b> {sum(trade_pnl_history):.2f}</div>
                    <div><b>Backtest Year PnL:</b> {backtest_results['total_pnl']:.2f}</div>
                    <div><b>Win Rate:</b> {backtest_results['win_rate']:.1%}</div>
                </div>
                
                <table>
                    <thead>
                        <tr><th>Side</th><th>Window</th><th>Entry</th><th>Stop (Confirmed)</th><th>Target</th><th>Action</th></tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
            
        elif self.path.startswith('/chart.png'):
            if current_plot_data:
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.end_headers()
                self.wfile.write(current_plot_data.getvalue())
            else:
                self.send_error(404)

    def do_POST(self):
        global active_trades
        if self.path.startswith('/cancel'):
            q = parse_qs(urlparse(self.path).query)
            w = int(q.get('w', [0])[0])
            s = q.get('s', [''])[0]
            # Manual close removes trade from active list
            active_trades = [t for t in active_trades if not (t['window'] == w and t['type'] == s)]
        self.send_response(303)
        self.send_header('Location', '/')
        self.end_headers()

# =============================================================================
# MAIN TRADING LOOP
# =============================================================================
def logic_loop():
    global current_plot_data, active_trades, trade_pnl_history
    while True:
        try:
            # 1. Fetch live data
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
            full_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            if not full_df.empty:
                df_closed = full_df.iloc[:-1].copy() # Completed candles
                current_price = full_df.iloc[-1]['close'] # Current live price
                
                # 2. EXIT LOGIC (Live Price Check)
                remaining_trades = []
                for t in active_trades:
                    closed = False
                    pnl = 0
                    
                    if t['type'] == 'short':
                        if current_price >= t['stop'] or current_price <= t['target']:
                            pnl = t['entry'] - current_price
                            closed = True
                    elif t['type'] == 'long':
                        if current_price <= t['stop'] or current_price >= t['target']:
                            pnl = current_price - t['entry']
                            closed = True
                    
                    if closed:
                        trade_pnl_history.append(pnl)
                    else:
                        remaining_trades.append(t)
                active_trades = remaining_trades

                # 3. ENTRY LOGIC (Confirmed Close Check)
                # Only check entry if we don't have a position for that side
                has_long = any(t['type'] == 'long' for t in active_trades)
                has_short = any(t['type'] == 'short' for t in active_trades)
                
                last_c = df_closed['close'].iloc[-1]
                x_idx = np.arange(len(df_closed))
                
                # Scan largest window first
                for w in range(MAX_WINDOW, MIN_WINDOW -1, -1):
                    if has_long and has_short: break # Max positions reached
                    
                    xw = x_idx[-w:]
                    yc = df_closed['close'].values[-w:]
                    yh = df_closed['high'].values[-w:]
                    yl = df_closed['low'].values[-w:]
                    
                    mm, cm = fit_ols(xw, yc)
                    if mm is None: continue
                    yt = mm * xw + cm
                    
                    mu, cu = fit_ols(xw[yh > yt], yh[yh > yt])
                    ml, cl = fit_ols(xw[yl < yt], yl[yl < yt])
                    
                    if mu is not None and ml is not None:
                        uv, lv = mu * xw[-1] + cu, ml * xw[-1] + cl
                        di = uv - lv
                        th = di * THRESHOLD_PCT
                        
                        # Check Long Entry
                        if not has_long and last_c > (uv + th):
                            active_trades.append({
                                'type': 'long', 'entry': last_c, 
                                'stop': uv, 'target': uv + di, 'window': w
                            })
                            has_long = True # Prevent duplicate longs same tick
                        
                        # Check Short Entry
                        if not has_short and last_c < (lv - th):
                            active_trades.append({
                                'type': 'short', 'entry': last_c, 
                                'stop': lv, 'target': lv - di, 'window': w
                            })
                            has_short = True

                # 4. UPDATE PLOT
                current_plot_data = generate_plot(df_closed, current_price)
                
        except Exception as e:
            print(f"Loop Error: {e}")
        
        time.sleep(UPDATE_INTERVAL)

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    print(f"Starting OLS Bot on {SYMBOL}...")
    
    # 1. Start Backtest Thread
    threading.Thread(target=run_backtest, daemon=True).start()
    
    # 2. Start Web Server Thread
    server_thread = threading.Thread(target=lambda: HTTPServer(('', PORT), DashboardHandler).serve_forever(), daemon=True)
    server_thread.start()
    print(f"Dashboard available at http://localhost:{PORT}")
    
    # 3. Start Main Logic Loop (Main Thread)
    logic_loop()
