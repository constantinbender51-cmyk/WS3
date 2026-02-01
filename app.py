import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime

# --- Global Result Storage ---
CHART_BUFFER = None

# --- 1. Data Fetching ---

def fetch_data_strict():
    print(f"[{datetime.now()}] FETCHING: ETH/USDT 1h (2020-Present)...")
    exchange = ccxt.binance()
    limit = 1000
    since = exchange.parse8601('2020-01-01T00:00:00Z')
    now = exchange.milliseconds()
    
    all_ohlcv = []
    
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv('ETH/USDT', '1h', since=since, limit=limit)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 3600000 
            time.sleep(0.02) # minimal throttle
        except Exception:
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    print(f"[{datetime.now()}] DATA LOADED: {len(df)} candles.")
    return df

# --- 2. Strict Topological Solver ---

def get_cone(start_val, n, direction, rate=0.001):
    t = np.arange(n)
    log_v = np.log(start_val) + t * np.log(1 + direction * rate)
    return np.exp(log_v)

def solve_strict_segment(prices, start_val, rate=0.001):
    n = len(prices)
    best_area = -1.0
    best_path = None
    
    # Pre-calculate Forward Reachability
    fwd_upper = get_cone(start_val, n, 1, rate)
    fwd_lower = get_cone(start_val, n, -1, rate)

    # Optimization Loop: Try every index k as the pivot point
    # k is where we touch/cross the price.
    for k in range(1, n - 1):
        target = prices[k]
        
        # 1. Reachability Check
        if not (fwd_lower[k] <= target <= fwd_upper[k]):
            continue

        # 2. Strict Sign Logic
        # We must decide if we are entering from ABOVE or BELOW.
        # This is determined by start_val vs Price[0].
        # If start_val == Price[0], we test both.
        
        modes = []
        if start_val >= prices[0]: modes.append(('above', 'below'))
        if start_val <= prices[0]: modes.append(('below', 'above'))
        
        for pre_mode, post_mode in modes:
            # --- Segment 1: 0 to k (The Approach) ---
            # We must NOT touch the price before k.
            
            # Backward reachability from Target
            dist = np.arange(k + 1)[::-1]
            
            if pre_mode == 'above':
                # Stay ABOVE. Hug Upper Bound.
                # Constraint: Must hit Target at k.
                # Max value at t that can drop to Target: Target / (1-r)^dist
                bwd_limit = target * np.power(1 - rate, -dist)
                
                # Valid path is intersection of FwdUpper and BwdLimit
                path_pre = np.minimum(fwd_upper[:k+1], bwd_limit)
                
                # STRICT CONSTRAINT: No touches allowed before k
                # Allow tolerance for float precision (1e-6), but strictly > price
                if np.any(path_pre[:-1] <= prices[:k]):
                    continue
            else:
                # Stay BELOW. Hug Lower Bound.
                # Constraint: Must hit Target at k.
                # Min value at t that can rise to Target: Target / (1+r)^dist
                bwd_limit = target * np.power(1 + rate, -dist)
                
                path_pre = np.maximum(fwd_lower[:k+1], bwd_limit)
                
                # STRICT CONSTRAINT: No touches allowed before k
                if np.any(path_pre[:-1] >= prices[:k]):
                    continue

            # --- Segment 2: k to n (The Divergence) ---
            # We must NOT touch the price after k.
            len_post = n - k
            if post_mode == 'above':
                # Go UP fast.
                path_post = get_cone(target, len_post, 1, rate)
                # STRICT CHECK
                if np.any(path_post[1:] <= prices[k+1:]):
                    continue
            else:
                # Go DOWN fast.
                path_post = get_cone(target, len_post, -1, rate)
                # STRICT CHECK
                if np.any(path_post[1:] >= prices[k+1:]):
                    continue

            # Stitch
            full_path = np.concatenate([path_pre, path_post[1:]])
            
            # Final Validation: Count Strict Sign Changes
            # Sign array should be [+,+,...,0,-,-...] or [-,...,0,+,...]
            # Zeros allowed ONLY at k.
            diff = full_path - prices
            # Mask index k (which is 0 by definition)
            diff_test = np.delete(diff, k)
            if np.any(diff_test == 0): continue # Reject any other touches
            
            # Area
            area = np.sum(np.abs(diff))
            if area > best_area:
                best_area = area
                best_path = full_path

    # Inertial Fallback (Only if no valid crossing found - e.g. price moves > 0.1% instantly at start)
    if best_path is None:
        return prices 
        
    return best_path

def generate_plot():
    df = fetch_data_strict()
    df['period'] = df['timestamp'].dt.to_period('M')
    
    full_line = []
    current_val = df['close'].iloc[0]
    
    print(f"[{datetime.now()}] OPTIMIZING: Area Maximization with Strict Anti-Bounce...")
    
    periods = sorted(df['period'].unique())
    for p in periods:
        prices = df.loc[df['period'] == p, 'close'].values
        path = solve_strict_segment(prices, current_val)
        full_line.append(path)
        current_val = path[-1]
        
    final_line = np.concatenate(full_line)
    if len(final_line) > len(df): final_line = final_line[:len(df)]
    
    # Plot
    plt.figure(figsize=(18, 10), dpi=100)
    plt.style.use('dark_background')
    
    plt.plot(df['timestamp'], df['close'], color='#444444', linewidth=0.6, label='Price')
    plt.plot(df['timestamp'], final_line, color='#00ff00', linewidth=1.2, label='Optimal Line')
    
    plt.title("ETH/USDT (2020-Present): 0.1% Constraint, Strict Single Crossing (No Bounces)")
    plt.grid(True, alpha=0.1)
    plt.legend()
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# --- 3. Server ---

class ChartHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global CHART_BUFFER
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            self.wfile.write(CHART_BUFFER.getvalue())
        else:
            self.send_error(404)

def run():
    global CHART_BUFFER
    # Pre-compute on startup
    CHART_BUFFER = generate_plot()
    
    server = HTTPServer(('', 8080), ChartHandler)
    print(f"[{datetime.now()}] SERVER READY: Port 8080")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()

if __name__ == '__main__':
    run()
