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
               # Candles to fetch from exchange
PORT = 8080                 # Web server port
THRESHOLD_PCT = 0.10        # Breakout distance (10% of channel width)
UPDATE_INTERVAL = 10        # Seconds between data refreshes
MIN_WINDOW = 10             # Minimum lookback for OLS
MAX_WINDOW = 200            # Maximum lookback for OLS
CHART_SIZE = (15, 9)        # Matplotlib figure size
CANDLE_WIDTH = 0.6          # Body width
WICK_WIDTH = 0.05           # Wick width
LIMIT = MAX_WINDOW+1
# =============================================================================

# Global State
current_plot_data = None
trade_pnl_history = []
active_trades = [] 
current_unrealized_pnl = 0.0
exchange = ccxt.binance()

def get_full_data():
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
        return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def fit_ols(x, y):
    if len(x) < 2: return None, None
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def process_and_pnl(latest_price):
    global trade_pnl_history, active_trades, current_unrealized_pnl
    remaining = []
    upnl = 0.0
    for t in active_trades:
        closed = False
        p = (t['entry'] - latest_price) if t['type'] == 'short' else (latest_price - t['entry'])
        
        # Exit Logic: Price hits Stop (breakout line) or Target (channel distance)
        if t['type'] == 'short' and (latest_price >= t['stop'] or latest_price <= t['target']):
            closed = True
        elif t['type'] == 'long' and (latest_price <= t['stop'] or latest_price >= t['target']):
            closed = True
        
        if closed:
            trade_pnl_history.append(p)
        else:
            upnl += p
            remaining.append(t)
    active_trades = remaining
    current_unrealized_pnl = upnl

def generate_plot(df_closed):
    plt.figure(figsize=CHART_SIZE)
    plt.style.use('dark_background')
    
    x_full = np.arange(len(df_closed))
    last_idx = x_full[-1]
    last_close = df_closed['close'].iloc[-1]
    
    potential_short = None
    potential_long = None
    breakout_visuals = []

    # Iterate from largest window down to smallest
    for w in range(MAX_WINDOW, MIN_WINDOW - 1, -1):
        if len(df_closed) < w: continue
        x_win = x_full[-w:]
        y_win_close = df_closed['close'].values[-w:]
        y_win_high = df_closed['high'].values[-w:]
        y_win_low = df_closed['low'].values[-w:]
        
        m_mid, c_mid = fit_ols(x_win, y_win_close)
        if m_mid is None: continue
        y_trend = m_mid * x_win + c_mid
        
        # Fit boundary lines to points above/below trend
        m_u, c_u = fit_ols(x_win[y_win_high > y_trend], y_win_high[y_win_high > y_trend])
        m_l, c_l = fit_ols(x_win[y_win_low < y_trend], y_win_low[y_win_low < y_trend])
        
        if m_u is not None and m_l is not None:
            u_line = m_u * x_win + c_u
            l_line = m_l * x_win + c_l
            dist = u_line[-1] - l_line[-1]
            thresh_val = dist * THRESHOLD_PCT
            
            is_short = last_close < (l_line[-1] - thresh_val)
            is_long = last_close > (u_line[-1] + thresh_val)
            
            if is_short and potential_short is None:
                potential_short = {
                    'type': 'short', 'entry': last_close, 'stop': l_line[-1], 
                    'target': l_line[-1] - dist, 'window': w
                }
                breakout_visuals.append({'x': x_win, 'line': l_line, 'thresh': l_line - thresh_val})
            
            if is_long and potential_long is None:
                potential_long = {
                    'type': 'long', 'entry': last_close, 'stop': u_line[-1], 
                    'target': u_line[-1] + dist, 'window': w
                }
                breakout_visuals.append({'x': x_win, 'line': u_line, 'thresh': u_line + thresh_val})

    # Trade Execution
    if potential_short and not any(t['type'] == 'short' for t in active_trades):
        active_trades.append(potential_short)
    if potential_long and not any(t['type'] == 'long' for t in active_trades):
        active_trades.append(potential_long)

    # Plot Breakout Indicators
    for vis in breakout_visuals:
        plt.plot(vis['x'], vis['line'], color='red', linewidth=1.2, zorder=1)
        plt.plot(vis['x'], vis['thresh'], color='red', linestyle=':', linewidth=1.5, zorder=1)

    # Plot Active Trade Levels
    for trade in active_trades:
        plt.axhline(trade['stop'], color='orange', linestyle='--', alpha=0.6)
        plt.axhline(trade['target'], color='lime', linestyle='--', alpha=0.6)

    # Render Candles
    up_c = df_closed[df_closed.close >= df_closed.open]
    down_c = df_closed[df_closed.close < df_closed.open]
    
    for color, d in [('green', up_c), ('red', down_c)]:
        plt.bar(d.index, d.close - d.open, CANDLE_WIDTH, bottom=d.open, color=color, zorder=2)
        plt.bar(d.index, d.high - np.maximum(d.close, d.open), WICK_WIDTH, bottom=np.maximum(d.close, d.open), color=color, zorder=2)
        plt.bar(d.index, np.minimum(d.close, d.open) - d.low, WICK_WIDTH, bottom=d.low, color=color, zorder=2)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            rows = "".join([
                f"<tr style='color:{'lime' if t['type']=='long' else 'orange'}'>"
                f"<td>{t['type'].upper()}</td><td>{t['window']}</td><td>{t['entry']:.2f}</td>"
                f"<td>{t['stop']:.2f}</td><td>{t['target']:.2f}</td>"
                f"<td><form style='margin:0' method='POST' action='/cancel?w={t['window']}&s={t['type']}'>"
                f"<input type='submit' value='Cancel'></form></td></tr>" 
                for t in active_trades
            ])
            
            html = f"""
            <html>
            <head>
                <style>
                    body {{ background:#000; color:#fff; font-family:monospace; text-align:center; }}
                    table {{ width:85%; margin:20px auto; border-collapse:collapse; background:#111; }}
                    th, td {{ padding:12px; border:1px solid #333; }}
                    img {{ width:95%; margin:10px auto; }}
                    .box {{ margin:20px; padding:15px; border:1px solid #444; background:#080808; }}
                    input {{ background:#222; color:red; border:1px solid red; cursor:pointer; padding:5px 10px; }}
                </style>
            </head>
            <body>
                <img src='/chart.png?t={int(time.time())}'>
                <br><button onclick='location.reload()'>Refresh Data</button>
                <div class='box'>
                    <b>Realized:</b> {sum(trade_pnl_history):.2f} | 
                    <b>Unrealized:</b> {current_unrealized_pnl:.2f} | 
                    <b>Net:</b> {sum(trade_pnl_history)+current_unrealized_pnl:.2f}
                </div>
                <table>
                    <thead><tr><th>Side</th><th>Window</th><th>Entry</th><th>Stop (Orange)</th><th>Target (Lime)</th><th>Action</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
                <div style='margin-top:20px;'>History: {[round(p, 2) for p in trade_pnl_history]}</div>
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
            else: self.send_error(404)

    def do_POST(self):
        global active_trades
        if self.path.startswith('/cancel'):
            q = parse_qs(urlparse(self.path).query)
            w = int(q.get('w', [0])[0])
            s = q.get('s', [''])[0]
            active_trades = [t for t in active_trades if not (t['window'] == w and t['type'] == s)]
        self.send_response(303)
        self.send_header('Location', '/')
        self.end_headers()

def logic_loop():
    global current_plot_data
    while True:
        full_df = get_full_data()
        if not full_df.empty:
            df_closed = full_df.iloc[:-1].copy()
            open_price = full_df.iloc[-1]['close']
            
            # Update exits/pnl using open candle price
            process_and_pnl(open_price)
            # Entry logic and charting using closed candles
            current_plot_data = generate_plot(df_closed)
            
            print(f"Price: {open_price} | Net: {sum(trade_pnl_history)+current_unrealized_pnl:.2f}")
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    threading.Thread(target=lambda: HTTPServer(('', PORT), DashboardHandler).serve_forever(), daemon=True).start()
    logic_loop()
