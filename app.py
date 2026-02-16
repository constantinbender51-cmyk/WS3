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
TRADES_TO_SHOW = 3         # Stop after finding this many trades
PORT = 8080

exchange = ccxt.binance()
report_html = "<h1>Running 1-Year Backtest... Please Refresh in 30 Seconds.</h1>"

# =============================================================================
# MATH
# =============================================================================
def fit_ols(x, y):
    if len(x) < 2: return None, None
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def fetch_data():
    print(f"Fetching 1 Year of Data for {SYMBOL}...")
    # Fetch in batches because 8760 > 1000 limit
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
    # Convert timestamp for potential debugging, though we use index for speed
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def plot_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_forensic_chart(df, trade, trade_id):
    entry_idx = trade['entry_idx']
    exit_idx = trade['exit_idx']
    window = trade['window']
    
    # Define zoom: Start 20 candles before entry, End 20 candles after exit
    start_plot = max(0, entry_idx - window - 20)
    end_plot = min(len(df), exit_idx + 20)
    df_zoom = df.iloc[start_plot:end_plot].reset_index(drop=True)
    
    # Calculate relative indices for plotting
    rel_entry = entry_idx - start_plot
    rel_exit = exit_idx - start_plot
    
    plt.figure(figsize=(14, 7))
    plt.style.use('dark_background')
    
    # 1. Candles
    up = df_zoom[df_zoom.close >= df_zoom.open]
    down = df_zoom[df_zoom.close < df_zoom.open]
    plt.bar(up.index, up.close - up.open, 0.6, bottom=up.open, color='green', alpha=0.6)
    plt.bar(down.index, down.close - down.open, 0.6, bottom=down.open, color='red', alpha=0.6)
    plt.bar(up.index, up.high - up.low, 0.05, bottom=up.low, color='green', alpha=0.6)
    plt.bar(down.index, down.high - down.low, 0.05, bottom=down.low, color='red', alpha=0.6)

    # 2. Re-Draw the Signal Channel (Lagged Fit)
    # We fit on indices [entry - window - 1] to [entry - 1]
    # This proves we ignored the breakout candle for the math
    fit_slice = slice(entry_idx - window - 1, entry_idx - 1)
    
    # Get the raw values used for the signal
    yc = df['close'].iloc[fit_slice].values
    yh = df['high'].iloc[fit_slice].values
    yl = df['low'].iloc[fit_slice].values
    
    # Create X array relative to the Zoomed Plot
    # The fit data starts at (rel_entry - window - 1)
    x_fit_rel = np.arange(rel_entry - window - 1, rel_entry - 1)
    
    mm, cm = fit_ols(x_fit_rel, yc)
    
    if mm is not None:
        yt = mm * x_fit_rel + cm
        mu, cu = fit_ols(x_fit_rel[yh > yt], yh[yh > yt])
        ml, cl = fit_ols(x_fit_rel[yl < yt], yl[yl < yt])
        
        if mu and ml:
            # Draw lines PAST the entry to show the breakout clearly
            x_draw = np.arange(rel_entry - window - 1, rel_entry + 5)
            ul = mu * x_draw + cu
            ll = ml * x_draw + cl
            
            plt.plot(x_draw, ul, color='white', linestyle='--', linewidth=1, label='Upper Band (At Signal)')
            plt.plot(x_draw, ll, color='white', linestyle='--', linewidth=1, label='Lower Band (At Signal)')
            
            # Draw Threshold
            dist = ul - ll
            if trade['type'] == 'long':
                thresh = ul + (dist * THRESHOLD_PCT)
                plt.plot(x_draw, thresh, color='yellow', linestyle=':', linewidth=2, label='Breakout Threshold')
            else:
                thresh = ll - (dist * THRESHOLD_PCT)
                plt.plot(x_draw, thresh, color='yellow', linestyle=':', linewidth=2, label='Breakout Threshold')

    # 3. Markers
    plt.plot(rel_entry, trade['entry_price'], marker='^' if trade['type']=='long' else 'v', 
             color='cyan', markersize=12, label='ENTRY', zorder=10)
    plt.plot(rel_exit, trade['exit_price'], marker='X', color='orange', markersize=12, label='EXIT', zorder=10)
    
    plt.title(f"Trade #{trade_id} | Type: {trade['type'].upper()} | Window: {window} | PnL: {trade['pnl']:.2f}")
    plt.legend()
    return plot_to_base64()

def run_simulation():
    global report_html
    df = fetch_data()
    print(f"Data Loaded: {len(df)} candles.")
    
    active_trade = None
    completed_trades = []
    trade_images = []
    equity = [0]
    
    # Simulation Loop
    for i in range(MAX_WINDOW + 2, len(df)):
        price = df.iloc[i]['close']
        
        # --- EXIT LOGIC ---
        if active_trade:
            t = active_trade
            closed = False
            
            # Recalculate Dynamic Stop for THIS candle 'i'
            w = t['window']
            fit_slice = slice(i - w, i) # Lagged fit [i-w : i]
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
                t['exit_idx'] = i
                t['exit_price'] = price
                completed_trades.append(t)
                equity.append(equity[-1] + t['pnl'])
                
                # Generate forensic chart for this trade
                print(f"Generating Chart for Trade #{len(completed_trades)}...")
                img = generate_forensic_chart(df, t, len(completed_trades))
                trade_images.append(img)
                
                active_trade = None
                if len(completed_trades) >= TRADES_TO_SHOW:
                    break
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
                proj_idx = i - 1
                proj_u = mu * proj_idx + cu
                proj_l = ml * proj_idx + cl
                dist = proj_u - proj_l; th = dist * THRESHOLD_PCT
                
                if last_c > (proj_u + th):
                    active_trade = {
                        'type': 'long', 'entry_price': df.iloc[i]['open'], 
                        'entry_idx': i, 'window': w, 'target': proj_u + dist
                    }
                    break
                elif last_c < (proj_l - th):
                    active_trade = {
                        'type': 'short', 'entry_price': df.iloc[i]['open'], 
                        'entry_idx': i, 'window': w, 'target': proj_l - dist
                    }
                    break

    # Final Report Generation
    print("Generating Final HTML Report...")
    html = """
    <html><head><style>
        body { background-color: #1e1e1e; color: #ddd; font-family: sans-serif; text-align: center; }
        .chart-box { background: #000; padding: 10px; margin: 20px auto; width: 90%; border: 1px solid #444; }
        h3 { margin-top: 0; color: cyan; }
    </style></head><body>
    <h1>Forensic Backtest Analysis (1 Year)</h1>
    """
    
    # Equity Curve
    plt.figure(figsize=(10, 4))
    plt.style.use('dark_background')
    plt.plot(equity, color='cyan', linewidth=2)
    plt.title("Equity Curve (First 3 Trades)")
    eq_img = plot_to_base64()
    
    html += f"<div class='chart-box'><h3>Equity Curve</h3><img src='data:image/png;base64,{eq_img}'></div>"
    
    for idx, img in enumerate(trade_images):
        html += f"<div class='chart-box'><h3>Trade #{idx+1}</h3><img src='data:image/png;base64,{img}'></div>"
    
    html += "</body></html>"
    report_html = html
    print("Done.")

class ReportHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(report_html.encode())

if __name__ == "__main__":
    threading.Thread(target=run_simulation, daemon=True).start()
    print(f"Server started on port {PORT}. Waiting for results...")
    HTTPServer(('', PORT), ReportHandler).serve_forever()
