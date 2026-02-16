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
TRADES_TO_SHOW = 5         # Increased to 5
PORT = 8080

exchange = ccxt.binance()
report_html = "<h1>Initializing... Refresh in 30s.</h1>"

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
    entry_idx = trade['entry_idx']
    exit_idx = trade['exit_idx']
    window = trade['window']
    
    # Zoom: 20 candles before entry, 20 after exit
    start_plot = max(0, entry_idx - window - 20)
    end_plot = min(len(df), exit_idx + 20)
    df_zoom = df.iloc[start_plot:end_plot].reset_index(drop=True)
    
    rel_entry = entry_idx - start_plot
    rel_exit = exit_idx - start_plot
    
    plt.figure(figsize=(12, 6))
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
    
    # X relative to zoom
    x_fit_rel = np.arange(rel_entry - window - 1, rel_entry - 1)
    
    mm, cm = fit_ols(x_fit_rel, yc)
    
    if mm is not None:
        # Reconstruct High/Low bands
        yh = df['high'].iloc[fit_slice].values
        yl = df['low'].iloc[fit_slice].values
        yt = mm * x_fit_rel + cm
        mu, cu = fit_ols(x_fit_rel[yh > yt], yh[yh > yt])
        ml, cl = fit_ols(x_fit_rel[yl < yt], yl[yl < yt])
        
        if mu and ml:
            x_draw = np.arange(rel_entry - window - 1, rel_entry + 5)
            ul = mu * x_draw + cu
            ll = ml * x_draw + cl
            
            plt.plot(x_draw, ul, color='white', linestyle='--', linewidth=1, label='Channel Top')
            plt.plot(x_draw, ll, color='white', linestyle='--', linewidth=1, label='Channel Bottom')

    # 3. Fixed Stop Loss Line (The Breakout Line)
    # Draw a horizontal red line from Entry to Exit at the Stop Price
    plt.hlines(y=trade['stop_price'], xmin=rel_entry, xmax=rel_exit, colors='red', linewidth=2, label='Fixed Stop (Breakout Line)')
    
    # 4. Target Line
    plt.hlines(y=trade['target'], xmin=rel_entry, xmax=rel_exit, colors='lime', linestyle='--', linewidth=1.5, label='Target')

    # 5. Markers
    plt.plot(rel_entry, trade['entry_price'], marker='^' if trade['type']=='long' else 'v', 
             color='cyan', markersize=12, label='ENTRY', zorder=10)
    plt.plot(rel_exit, trade['exit_price'], marker='X', color='orange', markersize=12, label='EXIT', zorder=10)
    
    plt.title(f"Trade #{trade_id} | {trade['type'].upper()} | Window: {window} | PnL: {trade['pnl']:.2f}")
    plt.legend()
    return plot_to_base64()

def fetch_data():
    print("Fetching data...")
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
    global report_html
    
    df = fetch_data()
    print(f"Data Loaded: {len(df)} candles")
    
    active_trade = None
    completed_trades = []
    trade_images = []
    equity = [0]
    total_pnl = 0.0
    
    print("Starting Simulation...")
    
    for i in range(MAX_WINDOW + 2, len(df)):
        price = df.iloc[i]['close']
        
        # --- EXIT LOGIC (FIXED STOP) ---
        if active_trade:
            t = active_trade
            closed = False
            
            # Stop is now FIXED at t['stop_price']
            # Target is FIXED at t['target']
            
            if t['type'] == 'long':
                # Stop hit (Price dropped below breakout line)
                if price <= t['stop_price']:
                    t['pnl'] = t['stop_price'] - t['entry_price'] # Loss is limited to difference
                    # Slippage Reality: actually we exit at 'price' which might be lower
                    t['pnl'] = price - t['entry_price'] 
                    closed = True
                # Target hit
                elif price >= t['target']:
                    t['pnl'] = t['target'] - t['entry_price']
                    # Slippage Reality: exit at price
                    t['pnl'] = price - t['entry_price']
                    closed = True
            
            else: # Short
                # Stop hit (Price rose above breakout line)
                if price >= t['stop_price']:
                    t['pnl'] = t['entry_price'] - price
                    closed = True
                # Target hit
                elif price <= t['target']:
                    t['pnl'] = t['entry_price'] - price
                    closed = True
            
            if closed:
                t['exit_idx'] = i
                t['exit_price'] = price
                completed_trades.append(t)
                total_pnl += t['pnl']
                equity.append(total_pnl)
                
                print(f"Trade Closed: {t['pnl']:.2f}")
                img = generate_forensic_chart(df, t, len(completed_trades))
                trade_images.append(img)
                
                active_trade = None
                if len(completed_trades) >= TRADES_TO_SHOW: break
            continue

        # --- ENTRY LOGIC ---
        last_c = df.iloc[i-1]['close'] # Breakout candle
        
        for w in range(MAX_WINDOW, MIN_WINDOW - 1, -1):
            fit_slice = slice(i - w - 1, i - 1)
            x_fit = np.arange(i - w - 1, i - 1)
            yc = df['close'].values[fit_slice]; yh = df['high'].values[fit_slice]; yl = df['low'].values[fit_slice]
            
            mm, cm = fit_ols(x_fit, yc)
            if mm is None: continue
            yt = mm * x_fit + cm
            mu, cu = fit_ols(x_fit[yh > yt], yh[yh > yt]); ml, cl = fit_ols(x_fit[yl < yt], yl[yl < yt])
            
            if mu and ml:
                # Project to breakout candle (i-1)
                proj_idx = i - 1
                proj_u = mu * proj_idx + cu
                proj_l = ml * proj_idx + cl
                dist = proj_u - proj_l; th = dist * THRESHOLD_PCT
                
                if last_c > (proj_u + th):
                    # Long Entry: Stop is the UPPER BAND (proj_u)
                    active_trade = {
                        'type': 'long', 
                        'entry_price': df.iloc[i]['open'], 
                        'entry_idx': i, 
                        'window': w, 
                        'target': proj_u + dist,
                        'stop_price': proj_u  # FIXED STOP at Breakout Line
                    }
                    break
                
                elif last_c < (proj_l - th):
                    # Short Entry: Stop is the LOWER BAND (proj_l)
                    active_trade = {
                        'type': 'short', 
                        'entry_price': df.iloc[i]['open'], 
                        'entry_idx': i, 
                        'window': w, 
                        'target': proj_l - dist,
                        'stop_price': proj_l # FIXED STOP at Breakout Line
                    }
                    break

    # Build Report
    html = """<html><head><style>
        body { background:#050505; color:#eee; font-family:'Courier New'; text-align:center; }
        .box { border:1px solid #444; margin:20px auto; padding:10px; width:90%; background:#111; }
        </style></head><body><h1>Forensic Analysis (Fixed Stop)</h1>"""
    
    # Equity Curve
    plt.figure(figsize=(10, 4))
    plt.style.use('dark_background')
    plt.plot(equity, color='cyan')
    plt.title(f"Equity Curve ({len(completed_trades)} Trades)")
    html += f"<div class='box'><img src='data:image/png;base64,{plot_to_base64()}'></div>"
    
    # Trade Charts
    for i, img in enumerate(trade_images):
        html += f"<div class='box'><h3>Trade #{i+1}</h3><img src='data:image/png;base64,{img}'></div>"
        
    html += "</body></html>"
    report_html = html
    print("Report Ready.")

class ReportHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(report_html.encode())

if __name__ == "__main__":
    threading.Thread(target=run_simulation, daemon=True).start()
    print(f"Server started on port {PORT}...")
    HTTPServer(('', PORT), ReportHandler).serve_forever()
