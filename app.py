import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
MIN_WINDOW = 10
MAX_WINDOW = 100
THRESHOLD_PCT = 0.10
BACKTEST_HOURS = 365 * 24  # 1 Year
TRADES_TO_SHOW = 3         # Charts to generate
PORT = 8080

exchange = ccxt.binance()

# Global State for Dashboard
backtest_progress = 0.0
backtest_status = "Initializing..."
completed_trades = []
forensic_charts = []  # Stores base64 images of the first 3 trades
equity_curve = [0]
total_pnl = 0.0

# =============================================================================
# MATH & HELPERS
# =============================================================================
def fit_ols(x, y):
    if len(x) < 2: return None, None
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def plot_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#111')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_forensic_chart(df, trade, trade_id):
    """Generates a specific chart for a trade showing the channel at entry."""
    entry_idx = trade['entry_idx']
    exit_idx = trade['exit_idx']
    window = trade['window']
    
    # Zoom: 20 candles before entry, 20 after exit
    start_plot = max(0, entry_idx - window - 20)
    end_plot = min(len(df), exit_idx + 20)
    df_zoom = df.iloc[start_plot:end_plot].reset_index(drop=True)
    
    rel_entry = entry_idx - start_plot
    rel_exit = exit_idx - start_plot
    
    plt.figure(figsize=(10, 5))
    plt.style.use('dark_background')
    
    # 1. Candles
    up = df_zoom[df_zoom.close >= df_zoom.open]
    down = df_zoom[df_zoom.close < df_zoom.open]
    plt.bar(up.index, up.close - up.open, 0.6, bottom=up.open, color='green', alpha=0.6)
    plt.bar(down.index, down.close - down.open, 0.6, bottom=down.open, color='red', alpha=0.6)
    plt.bar(up.index, up.high - up.low, 0.05, bottom=up.low, color='green', alpha=0.6)
    plt.bar(down.index, down.high - down.low, 0.05, bottom=down.low, color='red', alpha=0.6)

    # 2. Channel (Lagged Fit)
    # Fit on [entry - window - 1] to [entry - 1]
    fit_slice = slice(entry_idx - window - 1, entry_idx - 1)
    yc = df['close'].iloc[fit_slice].values
    yh = df['high'].iloc[fit_slice].values
    yl = df['low'].iloc[fit_slice].values
    
    # X relative to zoom
    x_fit_rel = np.arange(rel_entry - window - 1, rel_entry - 1)
    
    mm, cm = fit_ols(x_fit_rel, yc)
    
    if mm is not None:
        yt = mm * x_fit_rel + cm
        mu, cu = fit_ols(x_fit_rel[yh > yt], yh[yh > yt])
        ml, cl = fit_ols(x_fit_rel[yl < yt], yl[yl < yt])
        
        if mu and ml:
            x_draw = np.arange(rel_entry - window - 1, rel_entry + 5)
            ul = mu * x_draw + cu
            ll = ml * x_draw + cl
            
            plt.plot(x_draw, ul, color='white', linestyle='--', linewidth=1, label='Upper Channel')
            plt.plot(x_draw, ll, color='white', linestyle='--', linewidth=1, label='Lower Channel')
            
            # Threshold
            dist = ul - ll
            if trade['type'] == 'long':
                thresh = ul + (dist * THRESHOLD_PCT)
                plt.plot(x_draw, thresh, color='yellow', linestyle=':', linewidth=2, label='Breakout')
            else:
                thresh = ll - (dist * THRESHOLD_PCT)
                plt.plot(x_draw, thresh, color='yellow', linestyle=':', linewidth=2, label='Breakout')

    # 3. Markers
    plt.plot(rel_entry, trade['entry_price'], marker='^' if trade['type']=='long' else 'v', 
             color='cyan', markersize=12, label='ENTRY', zorder=10)
    plt.plot(rel_exit, trade['exit_price'], marker='X', color='orange', markersize=12, label='EXIT', zorder=10)
    
    plt.title(f"Trade #{trade_id} | {trade['type'].upper()} | Window: {window} | PnL: {trade['pnl']:.2f}")
    plt.legend()
    return plot_to_base64()

def fetch_data():
    print("Fetching data...")
    # Fetch batches
    all_ohlcv = []
    since = exchange.milliseconds() - (BACKTEST_HOURS * 60 * 60 * 1000)
    while len(all_ohlcv) < BACKTEST_HOURS:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(0.1)
        except: break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

# =============================================================================
# SIMULATION LOOP
# =============================================================================
def run_simulation():
    global backtest_progress, backtest_status, total_pnl, equity_curve
    
    backtest_status = "Fetching Data..."
    df = fetch_data()
    
    backtest_status = "Running Simulation..."
    active_trade = None
    
    total_steps = len(df) - (MAX_WINDOW + 2)
    
    for idx, i in enumerate(range(MAX_WINDOW + 2, len(df))):
        # Update Progress
        if idx % 50 == 0:
            backtest_progress = (idx / total_steps) * 100
            
        price = df.iloc[i]['close']
        
        # --- EXIT LOGIC ---
        if active_trade:
            t = active_trade
            closed = False
            
            # Dynamic Stop
            w = t['window']
            # Lagged Fit [i-w : i]
            fit_slice = slice(i - w, i)
            x_fit = np.arange(i - w, i)
            yc = df['close'].values[fit_slice]; yh = df['high'].values[fit_slice]; yl = df['low'].values[fit_slice]
            
            mm, cm = fit_ols(x_fit, yc)
            if mm is not None:
                yt = mm * x_fit + cm
                mu, cu = fit_ols(x_fit[yh > yt], yh[yh > yt]); ml, cl = fit_ols(x_fit[yl < yt], yl[yl < yt])
                if mu and ml:
                    stop_long = ml * i + cl
                    stop_short = mu * i + cu
                    
                    if t['type'] == 'long':
                        if price <= stop_long or price >= t['target']:
                            t['pnl'] = price - t['entry_price']; closed = True
                    else:
                        if price >= stop_short or price <= t['target']:
                            t['pnl'] = t['entry_price'] - price; closed = True
            
            if closed:
                t['exit_idx'] = i; t['exit_price'] = price
                completed_trades.append(t)
                total_pnl += t['pnl']
                equity_curve.append(total_pnl)
                
                # Generate Forensic Chart for first 3 trades
                if len(completed_trades) <= TRADES_TO_SHOW:
                    img = generate_forensic_chart(df, t, len(completed_trades))
                    forensic_charts.append(img)
                
                active_trade = None
            continue

        # --- ENTRY LOGIC ---
        last_c = df.iloc[i-1]['close'] # Breakout candle
        for w in range(MAX_WINDOW, MIN_WINDOW - 1, -1):
            # Lagged Fit [i-w-1 : i-1]
            fit_slice = slice(i - w - 1, i - 1)
            x_fit = np.arange(i - w - 1, i - 1)
            yc = df['close'].values[fit_slice]; yh = df['high'].values[fit_slice]; yl = df['low'].values[fit_slice]
            
            mm, cm = fit_ols(x_fit, yc)
            if mm is None: continue
            yt = mm * x_fit + cm
            mu, cu = fit_ols(x_fit[yh > yt], yh[yh > yt]); ml, cl = fit_ols(x_fit[yl < yt], yl[yl < yt])
            
            if mu and ml:
                # Project to breakout candle i-1
                proj_idx = i - 1
                proj_u = mu * proj_idx + cu
                proj_l = ml * proj_idx + cl
                dist = proj_u - proj_l; th = dist * THRESHOLD_PCT
                
                if last_c > (proj_u + th):
                    active_trade = {'type': 'long', 'entry_price': df.iloc[i]['open'], 'entry_idx': i, 'window': w, 'target': proj_u + dist}
                    break
                elif last_c < (proj_l - th):
                    active_trade = {'type': 'short', 'entry_price': df.iloc[i]['open'], 'entry_idx': i, 'window': w, 'target': proj_l - dist}
                    break

    backtest_progress = 100.0
    backtest_status = "Simulation Complete."

# =============================================================================
# DASHBOARD SERVER
# =============================================================================
class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header('Content-type', 'text/html'); self.end_headers()
        
        # Build Table Rows
        rows = ""
        # Show last 5 trades in table
        for t in reversed(completed_trades[-5:]):
            rows += f"<tr style='color:{'lime' if t['pnl']>0 else 'red'}'><td>{t['type'].upper()}</td><td>{t['window']}</td><td>{t['entry_price']:.2f}</td><td>{t['exit_price']:.2f}</td><td>{t['pnl']:.2f}</td></tr>"
            
        # Build Forensic Images
        images_html = ""
        for i, img in enumerate(forensic_charts):
            images_html += f"<div style='margin:20px; border:1px solid #444; padding:10px; background:#000;'><h3>Forensic Trade #{i+1}</h3><img src='data:image/png;base64,{img}' style='width:100%; max-width:800px;'></div>"

        # Progress Bar Color
        bar_color = "cyan" if backtest_progress < 100 else "lime"
        
        html = f"""
        <html>
        <head>
            <title>Forensic Backtest Dashboard</title>
            <meta http-equiv="refresh" content="55445663">
            <style>
                body {{ background:#050505; color:#e0e0e0; font-family:'Courier New', monospace; text-align:center; margin:0; padding:20px; }}
                .progress-container {{ width:80%; background:#222; margin:20px auto; border:1px solid #444; height:20px; }}
                .progress-bar {{ width:{backtest_progress}%; background:{bar_color}; height:100%; transition: width 0.5s; }}
                table {{ width:80%; margin:20px auto; border-collapse:collapse; background:#111; }}
                th, td {{ padding:12px; border:1px solid #333; }}
                th {{ background:#222; }}
                .stats-box {{ display:flex; justify-content:space-around; width:80%; margin:20px auto; padding:15px; background:#111; border:1px solid #444; }}
                h1 {{ color: {bar_color}; }}
            </style>
        </head>
        <body>
            <h1>{backtest_status}</h1>
            
            <div class="progress-container">
                <div class="progress-bar"></div>
            </div>
            
            <div class="stats-box">
                <div><b>Total Trades:</b> {len(completed_trades)}</div>
                <div><b>Total PnL:</b> {total_pnl:.2f}</div>
                <div><b>Equity:</b> {equity_curve[-1]:.2f}</div>
            </div>
            
            <h2>Forensic Analysis (First 3 Trades)</h2>
            <div style="display:flex; flex-wrap:wrap; justify-content:center;">
                {images_html}
            </div>
            
            <h2>Recent Trade Log</h2>
            <table>
                <thead>
                    <tr><th>Type</th><th>Window</th><th>Entry</th><th>Exit</th><th>PnL</th></tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
            
        </body>
        </html>
        """
        self.wfile.write(html.encode())

if __name__ == "__main__":
    # Start Simulation in Background
    threading.Thread(target=run_simulation, daemon=True).start()
    
    print(f"Server started on port {PORT}...")
    try:
        HTTPServer(('', PORT), DashboardHandler).serve_forever()
    except KeyboardInterrupt:
        pass
