import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

# Configuration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LIMIT = 200
PORT = 8080

# Global State
current_plot_data = None
trade_pnl_history = []
active_trades = [] # List of dicts: {'type': 'low_cross', 'entry': price, 'stop': price, 'target': price}
exchange = ccxt.binance()

def get_data():
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
        # Drop the last row (open candle)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df.iloc[:-1].copy()
    except:
        return pd.DataFrame()

def fit_ols(x, y):
    if len(x) < 2:
        return None, None
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def process_trades(current_close):
    global trade_pnl_history, active_trades
    remaining_trades = []
    for trade in active_trades:
        # Exit condition 1: Price touches/re-enters the breakout line (Stop)
        # Exit condition 2: Price moves the full channel distance (Target)
        if current_close >= trade['stop']:
            pnl = trade['entry'] - current_close # Short PnL logic for "low cross"
            trade_pnl_history.append(pnl)
        elif current_close <= trade['target']:
            pnl = trade['entry'] - current_close
            trade_pnl_history.append(pnl)
        else:
            remaining_trades.append(trade)
    active_trades = remaining_trades

def generate_plot(df):
    plt.figure(figsize=(15, 9))
    plt.style.use('dark_background')
    
    x_full = np.arange(len(df))
    last_idx = x_full[-1]
    last_close = df['close'].iloc[-1]
    
    # Update trade tracking with the confirmed closed candle
    process_trades(last_close)
    
    for w in range(10, 101):
        if len(df) < w:
            continue
            
        x_win = x_full[-w:]
        y_win_close = df['close'].values[-w:]
        y_win_high = df['high'].values[-w:]
        y_win_low = df['low'].values[-w:]
        
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
            threshold = dist * 0.05
            
            # Low cross detection: Close is below (lower line - 5% of channel)
            if last_close < (l_val - threshold):
                plt.plot(x_win, y_trend, color='red', linewidth=0.5, zorder=1)
                plt.plot(x_win, m_u * x_win + c_u, color='red', linewidth=0.5, zorder=1)
                plt.plot(x_win, m_l * x_win + c_l, color='red', linewidth=0.5, zorder=1)
                
                # Initiate trade if not already tracking this specific window breakout
                # Note: Simplified tracking here; in production, use unique window IDs
                active_trades.append({
                    'entry': last_close,
                    'stop': l_val,
                    'target': l_val - dist
                })
            
            # High cross detection (as per previous logic, but now with 5% threshold)
            elif last_close > (u_val + threshold):
                plt.plot(x_win, y_trend, color='red', linewidth=0.5, zorder=1)
                plt.plot(x_win, m_u * x_win + c_u, color='red', linewidth=0.5, zorder=1)
                plt.plot(x_win, m_l * x_win + c_l, color='red', linewidth=0.5, zorder=1)

    # Render Candles
    width = .6
    up, down = df[df.close >= df.open], df[df.close < df.open]
    for c, d in [('green', up), ('red', down)]:
        plt.bar(d.index, d.close - d.open, width, bottom=d.open, color=c, zorder=2)
        plt.bar(d.index, d.high - np.maximum(d.close, d.open), 0.05, bottom=np.maximum(d.close, d.open), color=c, zorder=2)
        plt.bar(d.index, np.minimum(d.close, d.open) - d.low, 0.05, bottom=d.low, color=c, zorder=2)

    plt.title(f"Total PnL Recorded: {sum(trade_pnl_history):.2f} | Active: {len(active_trades)}")
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
            html = """<html><head><script>setInterval(()=>document.getElementById('c').src='/chart.png?t='+Date.now(),10000);</script></head>
            <body style="background:black;margin:0;overflow:hidden"><img id="c" src="/chart.png" style="width:100%"></body></html>"""
            self.wfile.write(html.encode())
        elif self.path.startswith('/chart.png'):
            if current_plot_data:
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.end_headers()
                self.wfile.write(current_plot_data.getvalue())
            else: self.send_error(404)

def run_server():
    HTTPServer(('', PORT), DashboardHandler).serve_forever()

def logic_loop():
    global current_plot_data
    while True:
        df = get_data()
        if not df.empty:
            current_plot_data = generate_plot(df)
            print(f"Update: {df.iloc[-1]['close']} | PnL History: {len(trade_pnl_history)}")
        time.sleep(10)

if __name__ == "__main__":
    threading.Thread(target=run_server, daemon=True).start()
    logic_loop()
