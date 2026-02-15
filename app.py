import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from datetime import datetime

# Configuration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LIMIT = 100
WINDOWS = range(100, 9, -10)  # 100, 90, ... 10
PORT = 8080

class Trade:
    def __init__(self, entry_price, direction, stop_price, window_size, entry_time):
        self.entry_price = entry_price
        self.direction = direction  # 1 for Long, -1 for Short
        self.stop_price = stop_price
        self.window_size = window_size
        self.entry_time = entry_time
        self.is_active = True
        self.take_profit = entry_price + (abs(entry_price - stop_price) * 2 * direction)
        self.pnl = 0.0

    def update(self, current_price, current_time):
        if not self.is_active:
            return

        # Check Take Profit
        if (self.direction == 1 and current_price >= self.take_profit) or \
           (self.direction == -1 and current_price <= self.take_profit):
            self.close(current_price, "TP")
            return

        # Check Stop Loss (Cross back into channel)
        # Assuming the line is fixed at entry. If strictly dynamic, this logic requires keeping the line eq.
        if (self.direction == 1 and current_price < self.stop_price) or \
           (self.direction == -1 and current_price > self.stop_price):
            self.close(current_price, "SL/Re-entry")
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
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT + 5) # Buffer
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Fetch Error: {e}")
        return pd.DataFrame()

def find_trendline(prices, direction):
    """
    Finds a line that touches but does not cross the candles (except potentially the last).
    direction: 'top' (resistance) or 'bottom' (support)
    Returns: slope (m), intercept (c), line_values
    """
    n = len(prices)
    if n < 2: return None, None, None
    
    x = np.arange(n)
    y = prices.values

    # Convex Hull approach simplified for upper/lower chain
    # For Top Line: We want a line connecting two indices (i, j) such that all k in (i, j) are below it.
    # To maximize the "tightness", we usually look for the line that creates the upper boundary of the convex hull.
    # Simplified brute force for robustness over strict hull algorithms on small N:
    
    best_m, best_c = None, None
    
    # We want the line defined by the Pivot points that envelopes the data.
    # Logic: Iterate through all pairs of peaks/troughs. Check validity. 
    # Valid line: y_k <= m*x_k + c for all k (Top)
    
    # Optimization: Only consider local extrema as pivots to reduce compute
    # Pivot logic
    pivots = []
    if direction == 'top':
        for i in range(1, n-1):
            if y[i-1] <= y[i] and y[i] >= y[i+1]: pivots.append(i)
    else:
        for i in range(1, n-1):
            if y[i-1] >= y[i] and y[i] <= y[i+1]: pivots.append(i)
            
    # Add endpoints as potential pivots if they are extrema
    if direction == 'top':
        if y[0] > y[1]: pivots.insert(0, 0)
        if y[-1] > y[-2]: pivots.append(n-1)
    else:
        if y[0] < y[1]: pivots.insert(0, 0)
        if y[-1] < y[-2]: pivots.append(n-1)

    if len(pivots) < 2: 
        # Fallback to absolute max/min
        pivots = [0, n-1] 

    candidates = []
    
    for i in range(len(pivots)):
        for j in range(i + 1, len(pivots)):
            p1, p2 = pivots[i], pivots[j]
            if p2 == p1: continue
            
            m = (y[p2] - y[p1]) / (p2 - p1)
            c = y[p1] - m * p1
            
            # Validation: Does this line cross any candle? 
            # (Strictly: Highs for Top, Lows for Bottom)
            line_y = m * x + c
            
            if direction == 'top':
                if np.all(y <= line_y + 1e-9): # Tolerance for float math
                    candidates.append((m, c, p2)) # Prefer lines extending to recent
            else:
                if np.all(y >= line_y - 1e-9):
                    candidates.append((m, c, p2))

    if not candidates:
        return None, None, None
        
    # Heuristic: Choose the line that is "closest" to the recent price action (last pivot is latest)
    # or the one with the shallowest slope?
    # Standard TA: The most recent valid trendline.
    candidates.sort(key=lambda x: x[2], reverse=True) # Sort by second pivot index descending
    best_m, best_c, _ = candidates[0]
    
    return best_m, best_c, best_m * x + best_c

def generate_dashboard(df, lines_data):
    plt.figure(figsize=(12, 8))
    plt.style.use('dark_background')
    
    # Plot Candles (Simplified)
    width = .6
    width2 = .1
    up = df[df.close >= df.open]
    down = df[df.close < df.open]
    
    col1 = 'green'
    col2 = 'red'
    
    plt.bar(up.index, up.close - up.open, width, bottom=up.open, color=col1)
    plt.bar(up.index, up.high - up.close, width2, bottom=up.close, color=col1)
    plt.bar(up.index, up.low - up.open, width2, bottom=up.open, color=col1)
    
    plt.bar(down.index, down.close - down.open, width, bottom=down.open, color=col2)
    plt.bar(down.index, down.high - down.open, width2, bottom=down.open, color=col2)
    plt.bar(down.index, down.low - down.close, width2, bottom=down.close, color=col2)
    
    # Plot Lines
    for line in lines_data:
        # line = (start_idx, end_idx, y_values, color)
        # Adjust index to match global DF index
        offset = len(df) - len(line[2])
        x_vals = range(offset, len(df))
        plt.plot(x_vals, line[2], color=line[3], linewidth=1, alpha=0.7)

    plt.title(f'BTC/USDT Iterative Trendlines - {datetime.now()}')
    plt.grid(True, alpha=0.2)
    
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
            
            # Stats Table
            table_html = "<h3>Active Trades</h3><table border='1'><tr><th>Time</th><th>Dir</th><th>Entry</th><th>Stop</th><th>Target</th><th>PnL</th></tr>"
            for t in active_trades:
                color = "green" if t.pnl > 0 else "red"
                table_html += f"<tr><td>{t.entry_time}</td><td>{t.direction}</td><td>{t.entry_price:.2f}</td><td>{t.stop_price:.2f}</td><td>{t.take_profit:.2f}</td><td style='color:{color}'>{t.pnl*100:.2f}%</td></tr>"
            table_html += "</table>"
            
            history_html = "<h3>History</h3><table border='1'><tr><th>Reason</th><th>Exit Price</th><th>PnL</th></tr>"
            for t in trade_history[-5:]:
                 color = "green" if t.pnl > 0 else "red"
                 history_html += f"<tr><td>{t.exit_reason}</td><td>{t.exit_price:.2f}</td><td style='color:{color}'>{t.pnl*100:.2f}%</td></tr>"
            history_html += "</table>"

            html = f"""
            <html>
                <head><meta http-equiv="refresh" content="60"></head>
                <body style="background:#111; color:#eee; font-family:monospace;">
                    <h2>Collective Ownership Protocol - Signal Dashboard</h2>
                    <img src="/chart.png" width="100%">
                    {table_html}
                    {history_html}
                </body>
            </html>
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
        
        # Plotting container
        plot_lines = []
        
        # 1. Update Active Trades
        for trade in active_trades[:]:
            trade.update(current_close, current_time)
            if not trade.is_active:
                trade_history.append(trade)
                active_trades.remove(trade)
        
        # 2. Analyze Windows
        # We process windows 100 to 10.
        # Data slice for calculation excludes the current candle for line fitting?
        # "touch but not cross any... except the most recent candle"
        # So we fit on candles [Start : -1], then project to [-1].
        
        breakout_detected = False
        
        for w in WINDOWS:
            if w >= len(df): continue
            
            # Slice the data: last w+1 candles, excluding the very last one for fitting
            # actually we need the last w candles ending at index -2.
            # fit_data = df.iloc[-(w+1):-1] 
            
            # Correction: "100 candles" implies looking back 100 bars.
            # Fit on df.iloc[-w-1 : -1] (The historical window)
            # Check crossover on df.iloc[-1] (The current candle)
            
            slice_df = df.iloc[-(w+1):-1]
            if len(slice_df) < 2: continue
            
            # Top Line
            m_top, c_top, line_vals_top = find_trendline(slice_df['high'], 'top')
            if m_top is not None:
                # Project to current candle
                # slice indices are 0..w-1. Current candle is at index w.
                proj_price_top = m_top * w + c_top
                
                # Check Breakout
                if current_close > proj_price_top:
                    # Check if we already have a similar position? 
                    # Prompt: "Assume multiple positions can be held"
                    # But we don't want to spam 10 trades for one breakout.
                    # Simple filter: limit 1 trade per window per hour? 
                    # Or just fire. Prompt says "Indicate a breakout... Record return".
                    
                    # Store line for plotting
                    full_line_y = m_top * np.arange(w+1) + c_top
                    plot_lines.append((-(w+1), -1, full_line_y, 'cyan'))
                    
                    # Entry Logic: Trigger if not already long on this window?
                    # Simplified: Just trigger.
                    t = Trade(current_close, 1, proj_price_top, w, current_time)
                    active_trades.append(t)
                    breakout_detected = True
                else:
                    full_line_y = m_top * np.arange(w+1) + c_top
                    plot_lines.append((-(w+1), -1, full_line_y, 'gray'))

            # Bottom Line
            m_bot, c_bot, line_vals_bot = find_trendline(slice_df['low'], 'bottom')
            if m_bot is not None:
                proj_price_bot = m_bot * w + c_bot
                
                if current_close < proj_price_bot:
                    full_line_y = m_bot * np.arange(w+1) + c_bot
                    plot_lines.append((-(w+1), -1, full_line_y, 'magenta'))
                    
                    t = Trade(current_close, -1, proj_price_bot, w, current_time)
                    active_trades.append(t)
                    breakout_detected = True
                else:
                    full_line_y = m_bot * np.arange(w+1) + c_bot
                    plot_lines.append((-(w+1), -1, full_line_y, 'gray'))

        # Generate Plot
        current_plot_data = generate_dashboard(df, plot_lines)
        
        # Sleep
        time.sleep(3600) # Check every hour

# Execution
if __name__ == "__main__":
    t = threading.Thread(target=run_server)
    t.daemon = True
    t.start()
    
    try:
        logic_loop()
    except KeyboardInterrupt:
        pass
