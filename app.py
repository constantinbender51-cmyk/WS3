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
LIMIT = 105  # Fetch slightly more to ensure valid 100 window
WINDOW = 100 # Fixed lookback
PORT = 8080
CROSS_PENALTY = 100000.0 # Massive penalty for crossing the line (simulating hard constraint)

class Trade:
    def __init__(self, entry_price, direction, stop_price, entry_time):
        self.entry_price = entry_price
        self.direction = direction  # 1 for Long, -1 for Short
        self.stop_price = stop_price
        self.entry_time = entry_time
        self.is_active = True
        # Target 2x risk
        dist = abs(entry_price - stop_price)
        self.take_profit = entry_price + (dist * 2 * direction)
        self.pnl = 0.0

    def update(self, current_price):
        if not self.is_active: return

        # TP Check
        if (self.direction == 1 and current_price >= self.take_profit) or \
           (self.direction == -1 and current_price <= self.take_profit):
            self.close(current_price, "TP")
            return

        # SL Check (Price crosses back into channel / hits stop)
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

# Global State
trade_history = []
active_trades = []
current_plot_data = None
exchange = ccxt.binance()

def get_candles():
    try:
        # Fetch enough data
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Fetch Error: {e}")
        return pd.DataFrame()

def find_best_line_geometric(series, direction):
    """
    Finds the line connecting two points (i, j) in the series that minimizes:
    Mean(Distance to all points) + Penalty * (Sum of violations)
    """
    values = series.values
    n = len(values)
    if n < 2: return None, None

    x = np.arange(n)
    best_cost = float('inf')
    best_m = 0
    best_c = 0
    found = False

    # Brute force all pairs (approx 5000 iterations for n=100, very fast)
    # Optimization: Only connect local pivots? 
    # For strict compliance with prompt "connect two highs", we check all.
    
    for i in range(n):
        for j in range(i + 1, n):
            # Define Line between i and j
            y1 = values[i]
            y2 = values[j]
            
            # Prevent division by zero (impossible since j > i)
            m = (y2 - y1) / (j - i)
            c = y1 - m * i
            
            # Calculate Line for entire window
            line_y = m * x + c
            
            # Calculate Cost
            if direction == 'top':
                # Distance: We want line > price. Dist = line_y - price
                diff = line_y - values 
                # Violation: Price > Line (diff < 0)
                violations = diff < 0
                # Cost = Average positive distance (fit) + Penalty for violations
                cost = np.mean(np.abs(diff)) + (np.sum(violations) * CROSS_PENALTY)
                
            else: # bottom
                # Distance: We want line < price. Dist = price - line_y
                diff = values - line_y
                # Violation: Price < Line (diff < 0)
                violations = diff < 0
                cost = np.mean(np.abs(diff)) + (np.sum(violations) * CROSS_PENALTY)

            if cost < best_cost:
                best_cost = cost
                best_m = m
                best_c = c
                found = True

    if not found: return None, None
    return best_m, best_c

def generate_plot(df, top_line_y, bot_line_y):
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')
    
    # Plot Candles
    width = .6
    width2 = .05
    up = df[df.close >= df.open]
    down = df[df.close < df.open]
    
    plt.bar(up.index, up.close - up.open, width, bottom=up.open, color='#26a69a')
    plt.bar(up.index, up.high - up.close, width2, bottom=up.close, color='#26a69a')
    plt.bar(up.index, up.low - up.open, width2, bottom=up.open, color='#26a69a')
    
    plt.bar(down.index, down.close - down.open, width, bottom=down.open, color='#ef5350')
    plt.bar(down.index, down.high - down.open, width2, bottom=down.open, color='#ef5350')
    plt.bar(down.index, down.low - down.close, width2, bottom=down.close, color='#ef5350')
    
    # Plot Lines
    x = np.arange(len(df))
    plt.plot(x, top_line_y, color='cyan', linewidth=1.5, label='Resistance')
    plt.plot(x, bot_line_y, color='magenta', linewidth=1.5, label='Support')
    
    plt.title(f'BTC/USDT Geometric Channels - {datetime.now().strftime("%H:%M")}')
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
            <html><head><meta http-equiv="refresh" content="60">
            <style>body{{background:#111;color:#eee;font-family:monospace;padding:20px}} 
            table{{width:100%;border-collapse:collapse;margin-top:10px}} 
            th,td{{border:1px solid #444;padding:8px;text-align:left}}
            .pos{{color:#0f0}} .neg{{color:#f00}}</style></head>
            <body>
            <h2>Collective Ownership Protocol - {datetime.now().strftime("%H:%M UTC")}</h2>
            <img src="/chart.png" style="width:100%; border:1px solid #333">
            <h3>Active Positions</h3>
            <table><tr><th>Time</th><th>Type</th><th>Entry</th><th>Stop</th><th>Target</th><th>PnL</th></tr>
            {''.join([f"<tr><td>{t.entry_time.strftime('%H:%M')}</td><td>{'LONG' if t.direction==1 else 'SHORT'}</td><td>{t.entry_price:.2f}</td><td>{t.stop_price:.2f}</td><td>{t.take_profit:.2f}</td><td class='{'pos' if t.pnl>=0 else 'neg'}'>{t.pnl*100:.2f}%</td></tr>" for t in active_trades])}
            </table>
            <h3>History</h3>
            <table><tr><th>Reason</th><th>Exit</th><th>PnL</th></tr>
            {''.join([f"<tr><td>{t.exit_reason}</td><td>{t.exit_price:.2f}</td><td class='{'pos' if t.pnl>=0 else 'neg'}'>{t.pnl*100:.2f}%</td></tr>" for t in trade_history[-5:]])}
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
    print(f"Dashboard active at port {PORT}")
    server.serve_forever()

def logic_loop():
    global current_plot_data
    while True:
        # 1. Fetch Data
        df = get_candles()
        if len(df) < WINDOW + 1:
            print("Waiting for more data...")
            time.sleep(60)
            continue
            
        # We work with the last 'WINDOW' candles for fitting, EXCLUDING the current active candle if possible
        # to prevent repainting, or including it if closed.
        # Assuming last row is the just-closed candle or currently forming. 
        # Safety: Look at the last 100 closed candles.
        
        # Slicing: -101 to -1 (100 candles)
        history_df = df.iloc[-(WINDOW+1):-1].reset_index(drop=True)
        current_candle = df.iloc[-1]
        current_price = current_candle['close']
        
        # 2. Update Trades
        for t in active_trades:
            t.update(current_price)
        for t in [x for x in active_trades if not x.is_active]:
            trade_history.append(t)
            active_trades.remove(t)

        # 3. Fit Lines
        # Top: Highs, Bottom: Lows
        m_top, c_top = find_best_line_geometric(history_df['high'], 'top')
        m_bot, c_bot = find_best_line_geometric(history_df['low'], 'bottom')
        
        if m_top is not None and m_bot is not None:
            # Generate Y values for plotting (history + current projection)
            # history indices 0..99. Current index 100.
            x_range = np.arange(WINDOW + 1)
            line_top_vals = m_top * x_range + c_top
            line_bot_vals = m_bot * x_range + c_bot
            
            current_resistance = line_top_vals[-1]
            current_support = line_bot_vals[-1]

            # 4. Check Breakouts (on current candle)
            # Long
            if current_price > current_resistance:
                # Filter spam
                if not any(t.direction == 1 for t in active_trades):
                    print(f"Long Signal: {current_price} > {current_resistance}")
                    active_trades.append(Trade(current_price, 1, current_support, current_candle['timestamp']))
            
            # Short
            elif current_price < current_support:
                if not any(t.direction == -1 for t in active_trades):
                    print(f"Short Signal: {current_price} < {current_support}")
                    active_trades.append(Trade(current_price, -1, current_resistance, current_candle['timestamp']))

            # Update Plot Data
            # Note: We need to pass the FULL df to plot, but the lines correspond to the last 101 points
            # Padding line arrays to match full df length
            pad = len(df) - len(line_top_vals)
            full_top = np.pad(line_top_vals, (pad, 0), constant_values=np.nan)
            full_bot = np.pad(line_bot_vals, (pad, 0), constant_values=np.nan)
            
            current_plot_data = generate_plot(df, full_top, full_bot)
        
        # 5. Wait for next hour
        # Calculate time to sleep until next hour start + buffer
        # For simplicity in this script, just sleep 60s and check timestamp or sleep 1h
        print(f"Cycle complete. Price: {current_price}. Sleeping...")
        time.sleep(3600) 

if __name__ == "__main__":
    t = threading.Thread(target=run_server)
    t.daemon = True
    t.start()
    try:
        logic_loop()
    except KeyboardInterrupt:
        pass
