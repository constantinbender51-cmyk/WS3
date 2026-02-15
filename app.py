import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from datetime import datetime
from scipy.optimize import minimize

# Configuration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LIMIT = 150 # Increased buffer
WINDOWS = range(100, 9, -10)
PORT = 8080
PENALTY_WEIGHT = 1e6  # Weight for crossing the line

class Trade:
    def __init__(self, entry_price, direction, stop_price, window_size, entry_time):
        self.entry_price = entry_price
        self.direction = direction
        self.stop_price = stop_price
        self.window_size = window_size
        self.entry_time = entry_time
        self.is_active = True
        self.take_profit = entry_price + (abs(entry_price - stop_price) * 2 * direction)
        self.pnl = 0.0

    def update(self, current_price, current_time):
        if not self.is_active: return

        if (self.direction == 1 and current_price >= self.take_profit) or \
           (self.direction == -1 and current_price <= self.take_profit):
            self.close(current_price, "TP")
            return

        if (self.direction == 1 and current_price < self.stop_price) or \
           (self.direction == -1 and current_price > self.stop_price):
            self.close(current_price, "SL")
            return

        self.pnl = (current_price - self.entry_price) / self.entry_price * self.direction

    def close(self, price, reason):
        self.pnl = (price - self.entry_price) / self.entry_price * self.direction
        self.is_active = False
        self.exit_price = price
        self.exit_reason = reason

trade_history = []
active_trades = []
current_plot_data = None
exchange = ccxt.binance()

def get_candles():
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT + 5)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Fetch Error: {e}")
        return pd.DataFrame()

def fit_trendline_constrained(prices, direction):
    """
    Fits a line y = mx + c using Asymmetric Least Squares.
    top: minimizes distance to Highs, heavily penalizes High > Line.
    bottom: minimizes distance to Lows, heavily penalizes Low < Line.
    """
    n = len(prices)
    if n < 2: return None, None
    
    x = np.arange(n)
    y = prices.values
    
    # Initial Guess: Simple Least Squares
    A = np.vstack([x, np.ones(n)]).T
    m_init, c_init = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Optimization
    def loss_function(params):
        m, c = params
        y_line = m * x + c
        diff = y - y_line
        
        if direction == 'top':
            # Violation: Price > Line (diff > 0) -> Huge Penalty
            # Compliance: Price <= Line (diff <= 0) -> Min Squared Error
            # We want line to hug the top (minimize distance) but not cross.
            residuals = np.where(diff > 0, diff**2 * PENALTY_WEIGHT, diff**2)
        else:
            # Violation: Price < Line (diff < 0) -> Huge Penalty
            # Compliance: Price >= Line (diff >= 0) -> Min Squared Error
            residuals = np.where(diff < 0, diff**2 * PENALTY_WEIGHT, diff**2)
            
        return np.sum(residuals)

    # L-BFGS-B or Nelder-Mead generally robust for this
    res = minimize(loss_function, [m_init, c_init], method='Nelder-Mead', tol=1e-5)
    return res.x[0], res.x[1]

def generate_dashboard(df, lines_data):
    plt.figure(figsize=(14, 8))
    plt.style.use('dark_background')
    
    # Candles
    width = .6
    width2 = .1
    up = df[df.close >= df.open]
    down = df[df.close < df.open]
    
    plt.bar(up.index, up.close - up.open, width, bottom=up.open, color='#00ff00')
    plt.bar(up.index, up.high - up.close, width2, bottom=up.close, color='#00ff00')
    plt.bar(up.index, up.low - up.open, width2, bottom=up.open, color='#00ff00')
    
    plt.bar(down.index, down.close - down.open, width, bottom=down.open, color='#ff0000')
    plt.bar(down.index, down.high - down.open, width2, bottom=down.open, color='#ff0000')
    plt.bar(down.index, down.low - down.close, width2, bottom=down.close, color='#ff0000')
    
    # Lines
    for line in lines_data:
        # line: (start_offset_index, y_values, color)
        # map local x (0..w) to global df index
        end_idx = len(df) - 1 # Current candle index
        start_idx = end_idx - (len(line[1]) - 1)
        x_vals = range(start_idx, end_idx + 1)
        
        # Ensure x_vals matches y_vals length
        if len(x_vals) == len(line[1]):
            plt.plot(x_vals, line[1], color=line[2], linewidth=1, alpha=0.6)

    plt.title(f'BTC/USDT Constrained Regression Channels - {datetime.now()}')
    plt.grid(True, alpha=0.15)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            rows = ""
            for t in active_trades:
                c = "#0f0" if t.pnl > 0 else "#f00"
                rows += f"<tr><td>{t.entry_time.strftime('%H:%M')}</td><td>{t.direction}</td><td>{t.entry_price:.1f}</td><td>{t.stop_price:.1f}</td><td>{t.take_profit:.1f}</td><td style='color:{c}'>{t.pnl*100:.2f}%</td></tr>"
            
            hist_rows = ""
            for t in trade_history[-10:]:
                 c = "#0f0" if t.pnl > 0 else "#f00"
                 hist_rows += f"<tr><td>{t.exit_reason}</td><td>{t.exit_price:.1f}</td><td style='color:{c}'>{t.pnl*100:.2f}%</td></tr>"

            html = f"""
            <html><head><meta http-equiv="refresh" content="60">
            <style>body{{background:#0d1117;color:#c9d1d9;font-family:monospace}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #30363d;padding:8px;text-align:left}} th{{background:#161b22}}</style>
            </head><body>
            <h2>Collective Ownership Protocol - Signal Dashboard</h2>
            <div style="width:100%;overflow-x:auto"><img src="/chart.png" style="max-width:100%"></div>
            <h3>Active</h3><table><tr><th>Time</th><th>Dir</th><th>Entry</th><th>Stop</th><th>Target</th><th>PnL</th></tr>{rows}</table>
            <h3>History</h3><table><tr><th>Reason</th><th>Exit</th><th>PnL</th></tr>{hist_rows}</table>
            </body></html>
            """
            self.wfile.write(html.encode())
        elif self.path == '/chart.png':
            global current_plot_data
            if current_plot_data:
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.end_headers()
                self.wfile.write(current_plot_data.getvalue())
            else:
                self.send_error(404)

def run_server():
    server = HTTPServer(('', PORT), DashboardHandler)
    print(f"Dashboard active at port {PORT}")
    server.serve_forever()

def logic_loop():
    global current_plot_data
    while True:
        df = get_candles()
        if df.empty:
            time.sleep(60)
            continue
            
        current_close = df.iloc[-1]['close']
        current_time = df.iloc[-1]['timestamp']
        plot_lines = []
        
        # Update Trades
        for trade in active_trades[:]:
            trade.update(current_close, current_time)
            if not trade.is_active:
                trade_history.append(trade)
                active_trades.remove(trade)
        
        # Analyze Windows
        # Fit on history: indices [-(w+1) : -1] relative to end
        # Predict on current: index -1
        
        for w in WINDOWS:
            if w + 2 > len(df): continue
            
            # Data for fitting: Exclude current candle
            fit_df = df.iloc[-(w+1):-1]
            if len(fit_df) < 5: continue
            
            # --- TOP LINE ---
            m_top, c_top = fit_trendline_constrained(fit_df['high'], 'top')
            
            # Project to current candle (x = w)
            # fit_df has indices 0 to w-1. Current candle is next step (w).
            proj_price_top = m_top * w + c_top
            
            # Plot vector: 0 to w (includes current)
            full_line_top = m_top * np.arange(w + 1) + c_top
            
            if current_close > proj_price_top:
                # Breakout Long
                plot_lines.append((w, full_line_top, 'cyan'))
                
                # Simple dedup: don't open if we have a very similar trade
                if not any(t.direction == 1 and abs(t.entry_price - current_close)/current_close < 0.005 for t in active_trades):
                     active_trades.append(Trade(current_close, 1, proj_price_top, w, current_time))
            else:
                plot_lines.append((w, full_line_top, '#444')) # Dark Gray

            # --- BOTTOM LINE ---
            m_bot, c_bot = fit_trendline_constrained(fit_df['low'], 'bottom')
            proj_price_bot = m_bot * w + c_bot
            full_line_bot = m_bot * np.arange(w + 1) + c_bot
            
            if current_close < proj_price_bot:
                # Breakout Short
                plot_lines.append((w, full_line_bot, 'magenta'))
                if not any(t.direction == -1 and abs(t.entry_price - current_close)/current_close < 0.005 for t in active_trades):
                    active_trades.append(Trade(current_close, -1, proj_price_bot, w, current_time))
            else:
                 plot_lines.append((w, full_line_bot, '#444'))

        current_plot_data = generate_dashboard(df, plot_lines)
        time.sleep(3600)

if __name__ == "__main__":
    t = threading.Thread(target=run_server)
    t.daemon = True
    t.start()
    try:
        logic_loop()
    except KeyboardInterrupt:
        pass
