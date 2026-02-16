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
LIMIT = 150  # Increased to accommodate max window + offset
PORT = 8080
CROSS_PENALTY = 100000.0 

# Collective Ownership Protocol Constants
IDENTITY = "Jimmy"

class Trade:
    def __init__(self, entry_price, direction, stop_price, entry_time, timeframe_length):
        self.entry_price = entry_price
        self.direction = direction  # 1 for Long, -1 for Short
        self.stop_price = stop_price
        self.entry_time = entry_time
        self.timeframe_length = timeframe_length # The seq length that triggered this
        self.is_active = True
        
        # Risk is distance to breakout line (stop)
        risk = abs(entry_price - stop_price)
        # Prevent division by zero or microscopic risk causing instant TP
        if risk == 0: risk = entry_price * 0.005 
        
        self.take_profit = entry_price + (risk * 2 * direction)
        self.pnl = 0.0

    def update(self, current_price):
        if not self.is_active: return

        # PnL Calculation
        self.pnl = (current_price - self.entry_price) / self.entry_price * self.direction

        # TP Check
        if (self.direction == 1 and current_price >= self.take_profit) or \
           (self.direction == -1 and current_price <= self.take_profit):
            self.close(current_price, "TP")
            return

        # SL Check (Price crosses back over breakout line)
        # Note: Stop is fixed to the breakout line price at entry (static stop) 
        # or dynamic if we re-evaluate line. Using static based on prompt "Stop is set to the breakout line".
        if (self.direction == 1 and current_price < self.stop_price) or \
           (self.direction == -1 and current_price > self.stop_price):
            self.close(current_price, "SL")
            return

    def close(self, price, reason):
        self.pnl = (price - self.entry_price) / self.entry_price * self.direction
        self.is_active = False
        self.exit_price = price
        self.exit_reason = reason
        self.exit_time = datetime.now()

# Global State
trade_history = []
active_trades = []
current_plot_data = None
exchange = ccxt.binance()

def get_data():
    try:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"[{IDENTITY}] Data Fetch Error: {e}")
        return pd.DataFrame()

def get_current_price():
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        return ticker['last']
    except:
        return None

def find_best_line_geometric(series_values, direction):
    """
    Minimizes: Mean(Distance) + Penalty * Violations
    """
    n = len(series_values)
    if n < 2: return None, None

    x = np.arange(n)
    best_cost = float('inf')
    best_m = None
    best_c = None

    # Optimization: Check outer hull points primarily, but full brute force 
    # is safer for "best" compliance. 
    # Logic simplified for performance: Check pairs of local extrema? 
    # Proceeding with O(N^2) as requested complexity requires accuracy.
    
    for i in range(n):
        for j in range(i + 1, n):
            y1 = series_values[i]
            y2 = series_values[j]
            
            m = (y2 - y1) / (j - i)
            c = y1 - m * i
            
            line_y = m * x + c
            
            if direction == 'top':
                diff = line_y - series_values
                violations = diff < 0
                cost = np.mean(np.abs(diff)) + (np.sum(violations) * CROSS_PENALTY)
            else:
                diff = series_values - line_y
                violations = diff < 0
                cost = np.mean(np.abs(diff)) + (np.sum(violations) * CROSS_PENALTY)

            if cost < best_cost:
                best_cost = cost
                best_m = m
                best_c = c

    return best_m, best_c

def scan_for_breakouts(df):
    """
    Scans sequence lengths 10-100 and offsets 0-10.
    Returns the detected breakout with the HIGHEST sequence length.
    """
    best_signal = None # (length, direction, stop_price_at_current, m, c, offset)
    
    current_idx = len(df) - 1
    current_close = df.iloc[-1]['close']
    
    # Iterate offsets (recent history)
    # offset 0 = fit ending at current candle
    # offset 10 = fit ending 10 candles ago
    for offset in range(0, 11): 
        # The end index for the fitting window
        fit_end_idx = current_idx - offset 
        if fit_end_idx < 10: continue

        # Iterate sequence lengths
        for length in range(10, 101):
            start_idx = fit_end_idx - length + 1
            if start_idx < 0: continue
            
            # Slice the window for fitting
            window_highs = df.iloc[start_idx : fit_end_idx + 1]['high'].values
            window_lows = df.iloc[start_idx : fit_end_idx + 1]['low'].values
            
            # Fit lines
            m_top, c_top = find_best_line_geometric(window_highs, 'top')
            m_bot, c_bot = find_best_line_geometric(window_lows, 'bottom')
            
            if m_top is None or m_bot is None: continue

            # Project line to CURRENT time (current_idx)
            # The line x-axis starts at 0 (start_idx). 
            # Current index relative to start is:
            rel_current_x = current_idx - start_idx
            
            proj_res = m_top * rel_current_x + c_top
            proj_sup = m_bot * rel_current_x + c_bot
            
            # Check for Breakout State
            # We are "In a breakout" if Price > Projected Resistance or Price < Projected Support
            
            signal_found = False
            direction = 0
            stop_level = 0.0

            if current_close > proj_res:
                direction = 1
                stop_level = proj_res
                signal_found = True
            elif current_close < proj_sup:
                direction = -1
                stop_level = proj_sup
                signal_found = True
            
            if signal_found:
                # Priority Logic: Max Sequence Length
                if best_signal is None or length > best_signal['length']:
                    best_signal = {
                        'length': length,
                        'direction': direction,
                        'stop': stop_level,
                        'offset': offset,
                        'm_top': m_top, 'c_top': c_top,
                        'm_bot': m_bot, 'c_bot': c_bot,
                        'start_idx': start_idx # for plotting
                    }

    return best_signal

def generate_plot(df, signal):
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')
    
    # Plot Candles
    width = .6
    up = df[df.close >= df.open]
    down = df[df.close < df.open]
    plt.bar(up.index, up.close - up.open, width, bottom=up.open, color='#26a69a')
    plt.bar(up.index, up.high - up.close, 0.05, bottom=up.close, color='#26a69a')
    plt.bar(up.index, up.low - up.open, 0.05, bottom=up.open, color='#26a69a')
    plt.bar(down.index, down.close - down.open, width, bottom=down.open, color='#ef5350')
    plt.bar(down.index, down.high - down.open, 0.05, bottom=down.open, color='#ef5350')
    plt.bar(down.index, down.low - down.close, 0.05, bottom=down.close, color='#ef5350')
    
    if signal:
        # Reconstruct the winning lines
        x_start = signal['start_idx']
        # Plot full length of df starting from x_start
        x_len = len(df) - x_start
        x_vals = np.arange(x_len)
        
        y_top = signal['m_top'] * x_vals + signal['c_top']
        y_bot = signal['m_bot'] * x_vals + signal['c_bot']
        
        # Shift x to match dataframe index
        x_plot = np.arange(x_start, len(df))
        
        plt.plot(x_plot, y_top, color='cyan', linewidth=2, label=f'Res (L={signal["length"]})')
        plt.plot(x_plot, y_bot, color='magenta', linewidth=2, label=f'Sup (L={signal["length"]})')
        
    plt.title(f'BTC/USDT Geometric Breakout - {datetime.now().strftime("%H:%M")}')
    plt.legend()
    plt.grid(True, alpha=0.1)
    
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
            
            html = f"""
            <html><head><meta http-equiv="refresh" content="30">
            <style>body{{background:#000;color:#0f0;font-family:monospace;padding:20px}} 
            table{{width:100%;border-collapse:collapse;margin-top:10px;border:1px solid #333}} 
            th,td{{border:1px solid #333;padding:8px;text-align:left}}
            .pos{{color:#0f0}} .neg{{color:#f00}}
            h2, h3 {{text-transform: uppercase; border-bottom: 1px solid #333; padding-bottom: 5px;}}
            </style></head>
            <body>
            <h2>Collective Ownership Protocol // {datetime.now().strftime("%H:%M UTC")}</h2>
            <div style="border: 1px solid #333; padding: 10px; margin-bottom: 20px;">
                STATUS: MONITORING<br>
                SCANNER: SEQ 10-100 | OFFSET 0-10<br>
                PRIORITY: MAX LENGTH
            </div>
            <img src="/chart.png" style="width:100%; border:1px solid #333">
            <h3>Active Vector</h3>
            <table><tr><th>Entry Time</th><th>Dir</th><th>Entry</th><th>Stop (Breakout Line)</th><th>Target (2x)</th><th>Len</th><th>PnL</th></tr>
            {''.join([f"<tr><td>{t.entry_time.strftime('%H:%M')}</td><td>{'LONG' if t.direction==1 else 'SHORT'}</td><td>{t.entry_price:.2f}</td><td>{t.stop_price:.2f}</td><td>{t.take_profit:.2f}</td><td>{t.timeframe_length}</td><td class='{'pos' if t.pnl>=0 else 'neg'}'>{t.pnl*100:.2f}%</td></tr>" for t in active_trades])}
            </table>
            <h3>Archive</h3>
            <table><tr><th>Reason</th><th>Exit</th><th>PnL</th></tr>
            {''.join([f"<tr><td>{t.exit_reason}</td><td>{t.exit_price:.2f}</td><td class='{'pos' if t.pnl>=0 else 'neg'}'>{t.pnl*100:.2f}%</td></tr>" for t in trade_history[-10:]])}
            </table></body></html>
            """
            self.wfile.write(html.encode())
        elif self.path == '/chart.png':
            if current_plot_data:
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.end_headers()
                self.wfile.write(current_plot_data.getvalue())
            else:
                self.send_error(404)

def run_server():
    server = HTTPServer(('', PORT), DashboardHandler)
    print(f"[{IDENTITY}] Interface active: Port {PORT}")
    server.serve_forever()

def logic_loop():
    global current_plot_data
    print(f"[{IDENTITY}] Logic loop initiated.")
    
    while True:
        # 1. Real-time PnL Tracking (Every Minute/Loop)
        current_ticker_price = get_current_price()
        
        if current_ticker_price:
            for t in active_trades:
                t.update(current_ticker_price)
            
            # Clean up closed trades
            for t in [x for x in active_trades if not x.is_active]:
                trade_history.append(t)
                active_trades.remove(t)
                print(f"[{IDENTITY}] Trade Closed: {t.exit_reason} | PnL: {t.pnl*100:.2f}%")

        # 2. Strategy Scan (Requires Candle Data)
        # We assume this loop runs every minute. 
        # Deep scan is expensive, but necessary per requirements.
        df = get_data()
        
        if len(df) > 100:
            signal = scan_for_breakouts(df)
            current_plot_data = generate_plot(df, signal)

            if signal:
                price = df.iloc[-1]['close']
                ts = df.iloc[-1]['timestamp']
                
                # Check if we should enter
                # Filter: Don't enter if we already have a trade in the same direction? 
                # Or simply allow one trade at a time. Assuming 1 active trade limit for safety.
                if not active_trades:
                    print(f"[{IDENTITY}] Signal Detected. Length: {signal['length']} Offset: {signal['offset']} Dir: {signal['direction']}")
                    
                    # Stop is set to the breakout line (signal['stop'])
                    t = Trade(price, signal['direction'], signal['stop'], ts, signal['length'])
                    active_trades.append(t)
        
        # 3. Wait 60s
        time.sleep(60)

if __name__ == "__main__":
    t = threading.Thread(target=run_server)
    t.daemon = True
    t.start()
    try:
        logic_loop()
    except KeyboardInterrupt:
        pass
