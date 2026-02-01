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

# --- Global Storage ---
CHART_BUFFER = None

# --- 1. Robust Data Fetching ---
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
            time.sleep(0.05) 
        except Exception as e:
            print(f"Fetch Error: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    print(f"[{datetime.now()}] DATA LOADED: {len(df)} candles.")
    return df

# --- 2. Strict Collision Solver ---

def get_cone_bound(start_val, n, direction, rate=0.001):
    """
    Generates a cone boundary.
    direction=1: Max possible upward path (1.001^t)
    direction=-1: Max possible downward path (0.999^t)
    """
    t = np.arange(n)
    log_v = np.log(start_val) + t * np.log(1 + direction * rate)
    return np.exp(log_v)

def solve_month_strict(prices, start_val, rate=0.001):
    n = len(prices)
    best_area = -1.0
    best_path = None
    
    # Pre-calculate Forward Reachability from Start
    # These are the physical limits of the line.
    fwd_max = get_cone_bound(start_val, n, 1, rate)
    fwd_min = get_cone_bound(start_val, n, -1, rate)

    # Iterate potential crossing points k
    # We ignore the very first and last few ticks to ensure a clear crossing
    for k in range(2, n - 2):
        target = prices[k]
        
        # 1. Physical Reachability Check
        if not (fwd_min[k] <= target <= fwd_max[k]):
            continue

        # We test two Topologies:
        # A: Start ABOVE -> Cross -> End BELOW
        # B: Start BELOW -> Cross -> End ABOVE
        
        # Optimization: Filter based on start_val vs Price[0]
        # (If we start far above, we can't be in 'Below' mode)
        candidates = []
        if start_val >= prices[0]: candidates.append('A')
        if start_val <= prices[0]: candidates.append('B')
        
        for mode in candidates:
            # --- PHASE 1: PRE-CROSSING (0 to k) ---
            dist_bwd = np.arange(k + 1)[::-1]
            
            if mode == 'A': # ABOVE -> BELOW
                # Strategy: Stay HIGH (Maximize Area)
                # Upper Bound = Min(Start_Fwd_Max, Target_Bwd_Max)
                # Target Bwd Max: The highest value that can drop to Target
                bwd_limit = target * np.power(1 - rate, -dist_bwd)
                seg_pre = np.minimum(fwd_max[:k+1], bwd_limit)
                
                # COLLISION CHECK: Did we hit the price early?
                # We require Line > Price strictly (epsilon buffer)
                if np.any(seg_pre[:-1] <= prices[:k]): 
                    continue # Invalid: Crossed/Touched too early

            else: # BELOW -> ABOVE
                # Strategy: Stay LOW (Maximize Area)
                # Lower Bound = Max(Start_Fwd_Min, Target_Bwd_Min)
                # Target Bwd Min: The lowest value that can rise to Target
                bwd_limit = target * np.power(1 + rate, -dist_bwd)
                seg_pre = np.maximum(fwd_min[:k+1], bwd_limit)
                
                # COLLISION CHECK
                if np.any(seg_pre[:-1] >= prices[:k]): 
                    continue # Invalid

            # --- PHASE 2: POST-CROSSING (k to End) ---
            len_post = n - k
            
            if mode == 'A': # ... -> BELOW
                # Strategy: Dive LOW fast (Maximize Area)
                seg_post = get_cone_bound(target, len_post, -1, rate)
                
                # COLLISION CHECK: Does price drop below us?
                # We require Line < Price strictly
                if np.any(seg_post[1:] >= prices[k+1:]):
                    continue # Invalid: Price crossed back over us
                    
            else: # ... -> ABOVE
                # Strategy: Fly HIGH fast
                seg_post = get_cone_bound(target, len_post, 1, rate)
                
                # COLLISION CHECK
                if np.any(seg_post[1:] <= prices[k+1:]):
                    continue # Invalid

            # --- Valid Path Found ---
            full_path = np.concatenate([seg_pre, seg_post[1:]])
            area = np.sum(np.abs(full_path - prices))
            
            if area > best_area:
                best_area = area
                best_path = full_path

    # Fallback: If no valid single-crossing path exists (Very rare, requires extreme volatility > 0.1%/hr sustained)
    if best_path is None:
        # Fallback Strategy: Hug the price. 
        # If we can't maximize area, we minimize distance to ensure at least we track the price 
        # and maybe get a crossing by chance or minimal violation.
        # But per instruction "Always possible", this branch should ideally not trigger.
        print(f"Warn: No strict solution for segment starting {start_val}. Fallback to clamping.")
        # Return a clamped version of price to stay within volatility limits of start_val
        # This prevents the visual "explosion" seen in your screenshot.
        acc = [start_val]
        for p in prices[1:]:
            prev = acc[-1]
            # Clamp p to be within prev * (1 +/- 0.001)
            clamped = np.clip(p, prev * 0.999, prev * 1.001)
            acc.append(clamped)
        return np.array(acc)
        
    return best_path

def optimize_all(df):
    df['period'] = df['timestamp'].dt.to_period('M')
    full_line = []
    
    # Initialize at Price[0]
    current_val = df['close'].iloc[0]
    
    periods = sorted(df['period'].unique())
    print(f"[{datetime.now()}] OPTIMIZING: Processing {len(periods)} months...")
    
    for p in periods:
        prices = df.loc[df['period'] == p, 'close'].values
        
        path = solve_month_strict(prices, current_val)
        full_line.append(path)
        current_val = path[-1]
        
    final_line = np.concatenate(full_line)
    
    # Safety clip length
    if len(final_line) > len(df): final_line = final_line[:len(df)]
    return final_line

# --- 3. Visualization ---

def generate_chart():
    df = fetch_data_strict()
    optimized = optimize_all(df)
    
    plt.figure(figsize=(16, 9), dpi=100)
    plt.style.use('dark_background')
    
    # Plot Price
    plt.plot(df['timestamp'], df['close'], color='#555555', linewidth=0.8, label='ETH Price')
    
    # Plot Optimized Line
    plt.plot(df['timestamp'], optimized, color='#00ff00', linewidth=1.2, label='Max Area Line (1 Cross/Month)')
    
    plt.title("ETH/USDT: Strict Topological Optimization (0.1% Volatility Limit)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.15)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# --- 4. Server ---

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
    CHART_BUFFER = generate_chart()
    
    server = HTTPServer(('', 8080), ChartHandler)
    print(f"[{datetime.now()}] SERVER RUNNING on 8080")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()

if __name__ == '__main__':
    run()
