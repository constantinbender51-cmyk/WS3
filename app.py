import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from urllib.parse import urlparse, parse_qs

# Configuration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LIMIT = 200
PORT = 8080

# Global State
current_plot_data = None
trade_pnl_history = []
active_trades = [] # Limit: 1 long, 1 short
current_unrealized_pnl = 0.0
exchange = ccxt.binance()

def get_full_data():
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
        return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    except:
        return pd.DataFrame()

def fit_ols(x, y):
    if len(x) < 2: return None, None
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def process_and_pnl(latest_price):
    """
    Updates Unrealized PnL every 10s based on the open candle (latest_price).
    Also checks for trade exit conditions (Stop/Target) based on the same price.
    """
    global trade_pnl_history, active_trades, current_unrealized_pnl
    remaining = []
    upnl = 0.0
    for t in active_trades:
        closed = False
        p = 0
        if t['type'] == 'short':
            p = t['entry'] - latest_price
            if latest_price >= t['stop'] or latest_price <= t['target']:
                closed = True
        else: # long
            p = latest_price - t['entry']
            if latest_price <= t['stop'] or latest_price >= t['target']:
                closed = True
        
        if closed:
            trade_pnl_history.append(p)
        else:
            upnl += p
            remaining.append(t)
    active_trades = remaining
    current_unrealized_pnl = upnl

def generate_plot(df_closed):
    plt.figure(figsize=(15, 9))
    plt.style.use('dark_background')
    
    x_full = np.arange(len(df_closed))
    last_idx = x_full[-1]
    last_close = df_closed['close'].iloc[-1]
    
    has_long = any(t['type'] == 'long' for t in active_trades)
    has_short = any(t['type'] == 'short' for t in active_trades)
    
    for w in range(10, 101):
        if len(df_closed) < w: continue
        x_win = x_full[-w:]
        y_win_close = df_closed['close'].values[-w:]
        y_win_high = df_closed['high'].values[-w:]
        y_win_low = df_closed['low'].values[-w:]
        
        m_mid, c_mid = fit_ols(x_win, y_win_close)
        if m_mid is None: continue
        y_trend = m_mid * x_win + c_mid
        
        u_mask = y_win_high > y_trend
        l_mask = y_win_low < y_trend
        
        m_u, c_u = fit_ols(x_win[u_mask], y_win_high[u_mask])
        m_l, c_l = fit_ols(x_win[l_mask], y_win_low[l_mask])
        
        if m_u is not None and m_l is not None:
            u_val = m_u * last_idx + c_u
            l_val = m_l * last_idx + c_l
            dist = u_val - l_val
            threshold = dist * 0.10 # 10% threshold
            
            # Short Entry (only if no active short)
            if last_close < (l_val - threshold):
                plt.plot(x_win, y_trend, color='red', linewidth=0.5, zorder=1)
                plt.plot(x_win, m_u * x_win + c_u, color='red', linewidth=0.5, zorder=1)
                plt.plot(x_win, m_l * x_win + c_l, color='red', linewidth=0.5, zorder=1)
                if not has_short:
                    active_trades.append({
                        'type': 'short', 'entry': last_close, 'stop': l_val, 
                        'target': l_val - dist, 'window': w
                    })
                    has_short = True
            
            # Long Entry (only if no active long)
            elif last_close > (u_val + threshold):
                plt.plot(x_win, y_trend, color='red', linewidth=0.5, zorder=1)
                plt.plot(x_win, m_u * x_win + c_u, color='red', linewidth=0.5, zorder=1)
                plt.plot(x_win, m_l * x_win + c_l, color='red', linewidth=0.5, zorder=1)
                if not has_long:
                    active_trades.append({
                        'type': 'long', 'entry': last_close, 'stop': u_val, 
                        'target': u_val + dist, 'window': w
                    })
                    has_long = True

    # Render Candles
    width = .6
    up, down = df_closed[df_closed.close >= df_closed.open], df_closed[df_closed.close < df_closed.open]
    for c, d in [('green', up), ('red', down)]:
        plt.bar(d.index, d.close - d.open, width, bottom=d.open, color=c, zorder=2)
        plt.bar(d.index, d.high - np.maximum(d.close, d.open), 0.05, bottom=np.maximum(d.close, d.open), color=c, zorder=2)
        plt.bar(d.index, np.minimum(d.close, d.open) - d.low, 0.05, bottom=d.low, color=c, zorder=2)

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
                f"<td><form style='display:inline' method='POST' action='/cancel?w={t['window']}&s={t['type']}'><input type='submit' value='Cancel'></form></td></tr>" 
                for t in active_trades
            ])
            
            html = f"""
            <html>
            <head>
                <style>
                    body {{ background: #000; color: #fff; font-family: monospace; margin: 0; display: flex; flex-direction: column; align-items: center; }}
                    table {{ border-collapse: collapse; width: 80%; margin-top: 20px; text-align: left; }}
                    th, td {{ padding: 10px; border-bottom: 1px solid #333; }}
                    img {{ width: 95%; max-width: 1200px; margin-top: 10px; }}
                    input {{ background: #333; color: red; border: 1px solid red; cursor: pointer; }}
                    .pnl-box {{ margin: 20px; padding: 15px; border: 1px solid #444; width: 80%; background: #111; }}
                </style>
            </head>
            <body>
                <img src="/chart.png?t={int(time.time())}">
                <button onclick="location.reload()" style="padding:10px; margin:10px;">Refresh Data</button>
                <div class="pnl-box">
                    <b>Realized PnL:</b> {sum(trade_pnl_history):.2f} <br>
                    <b>Unrealized PnL (Open Candle):</b> {current_unrealized_pnl:.2f} <br>
                    <b>Net PnL:</b> {sum(trade_pnl_history) + current_unrealized_pnl:.2f}
                </div>
                <table>
                    <thead><tr><th>Side</th><th>Window</th><th>Entry</th><th>Stop</th><th>Target</th><th>Action</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
                <div style="margin-top:20px;">PnL History: {[round(p, 2) for p in trade_pnl_history]}</div>
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
            query = parse_qs(urlparse(self.path).query)
            window = int(query.get('w', [0])[0])
            side = query.get('s', [''])[0]
            active_trades = [t for t in active_trades if not (t['window'] == window and t['type'] == side)]
            self.send_response(303)
            self.send_header('Location', '/')
            self.end_headers()

def run_server():
    HTTPServer(('', PORT), DashboardHandler).serve_forever()

def logic_loop():
    global current_plot_data
    while True:
        full_df = get_full_data()
        if not full_df.empty:
            # Entry logic uses confirmed data (all rows except the last open one)
            df_closed = full_df.iloc[:-1].copy()
            # PnL and Exit tracking uses the latest price from the open candle
            open_candle_price = full_df.iloc[-1]['close']
            
            process_and_pnl(open_candle_price)
            current_plot_data = generate_plot(df_closed)
            print(f"Update: {open_candle_price} | Unrealized: {current_unrealized_pnl:.2f}")
        time.sleep(10)

if __name__ == "__main__":
    threading.Thread(target=run_server, daemon=True).start()
    logic_loop()
