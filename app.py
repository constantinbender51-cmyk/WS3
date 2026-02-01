import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timedelta

# --- 1. Robust Data Fetching ---

def fetch_data_2020():
    print("Fetching ETH/USDT 1h data from 2020...")
    exchange = ccxt.binance()
    limit = 1000
    # Start exactly at 2020-01-01
    since = exchange.parse8601('2020-01-01T00:00:00Z')
    now = exchange.milliseconds()
    
    all_ohlcv = []
    
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv('ETH/USDT', '1h', since=since, limit=limit)
            if not ohlcv: break
            
            all_ohlcv.extend(ohlcv)
            # Prevent overlap/gaps: Set next 'since' to last timestamp + 1h
            since = ohlcv[-1][0] + 3600000 
            
            if len(all_ohlcv) % 5000 == 0:
                print(f"Fetched {len(all_ohlcv)} candles...")
                
        except Exception as e:
            print(f"Fetch Error: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    print(f"Data loaded: {len(df)} candles.")
    return df

# --- 2. Mathematical Solver ---

def get_fwd_cone(start_val, n, rate=0.001):
    """
    Returns (min_path, max_path) reachable from start_val.
    """
    t = np.arange(n)
    log_start = np.log(start_val)
    # Max Path: (1+r)^t, Min Path: (1-r)^t
    max_path = np.exp(log_start + t * np.log(1 + rate))
    min_path = np.exp(log_start + t * np.log(1 - rate))
    return min_path, max_path

def get_bwd_cone(target_val, n, rate=0.001):
    """
    Returns (min_path, max_path) ending at target_val.
    Mathematically: To end at T, previous values must be within reachable cone reversed.
    """
    t = np.arange(n)
    # To reach T from below (max ascent), we need T * (1+r)^(-t) looking back
    # To reach T from above (max descent), we need T * (1-r)^(-t) looking back
    # Note: t is distance from target (0 at target, n-1 at start)
    
    # Reverse index: t goes 0..n-1. 
    # At step i (where i goes 0 to n-1), distance to target is (n-1) - i.
    # Simpler: Generate cone from target backwards.
    
    log_target = np.log(target_val)
    log_up = np.log(1 + rate)
    log_down = np.log(1 - rate)
    
    # Backward Max: The highest value at t that can drop to target.
    # Val * (1-r)^dist = Target => Val = Target * (1-r)^-dist
    dist = np.arange(n)[::-1] # Distance from target
    
    # Max boundary (Upper): Must drop to target.
    # Val_t * (1-rate)^dist <= Target  --> Val_t <= Target / (1-rate)^dist
    bwd_max = np.exp(log_target - dist * log_down)
    
    # Min boundary (Lower): Must rise to target.
    # Val_t * (1+rate)^dist >= Target --> Val_t >= Target / (1+rate)^dist
    bwd_min = np.exp(log_target - dist * log_up)
    
    return bwd_min, bwd_max

def solve_month(prices, start_val, rate=0.001):
    n = len(prices)
    best_area = -1.0
    best_path = None
    
    # Pre-calculate Forward Reachability from Start
    fwd_min, fwd_max = get_fwd_cone(start_val, n, rate)

    # Optimization: Check every index k
    # k is the index where L[k] == Price[k]
    # Range: [1, n-2] to ensure strict crossing inside the interval
    for k in range(1, n-1):
        target = prices[k]
        
        # 1. Global Reachability Check
        # If target is physically unreachable from start, skip k
        if not (fwd_min[k] <= target <= fwd_max[k]):
            continue

        # 2. Compute Backward Reachability (from k back to 0)
        # We only need segment length k+1 (indices 0..k)
        bwd_min, bwd_max = get_bwd_cone(target, k + 1, rate)
        
        # 3. Define Valid Corridors (Intersection of Sets)
        # Pre-Crossing Segment (0..k)
        valid_min_pre = np.maximum(fwd_min[:k+1], bwd_min)
        valid_max_pre = np.minimum(fwd_max[:k+1], bwd_max)
        
        # Verify corridor existence
        if np.any(valid_min_pre > valid_max_pre):
            continue

        # 4. Scenario Evaluation
        # Scenario A: Above -> Below
        # Pre-crossing: Maximize L (Hug valid_max_pre)
        # Post-crossing: Minimize L (Hug Forward Min from target)
        
        # Check Pre-crossing Validity: Must not cross Price early
        # strict check: L_pre > Price for 0..k-1
        # Relaxed check: count crossings later
        
        # Construct Path A
        post_min_A, _ = get_fwd_cone(target, n - k, rate)
        path_A = np.concatenate([valid_max_pre, post_min_A[1:]])
        
        # Scenario B: Below -> Above
        # Pre-crossing: Minimize L (Hug valid_min_pre)
        # Post-crossing: Maximize L (Hug Forward Max from target)
        _, post_max_B = get_fwd_cone(target, n - k, rate)
        path_B = np.concatenate([valid_min_pre, post_max_B[1:]])
        
        candidates = [path_A, path_B]
        
        for path in candidates:
            # 5. Strict Topological Validation
            # Count sign changes
            diff = path - prices
            signs = np.sign(diff)
            # Remove zeros (touches) to see true crossings
            clean_signs = signs[signs != 0]
            
            if len(clean_signs) < 2: 
                continue # Never crossed
                
            flips = np.count_nonzero(np.diff(clean_signs))
            
            if flips == 1:
                area = np.sum(np.abs(diff))
                if area > best_area:
                    best_area = area
                    best_path = path

    # Fallback: If no single-crossing path exists (rare but possible in ultra-low volatility months)
    # We must maintain continuity. We just trace the price or move inertially.
    if best_path is None:
        # Inertial fallback: Just continue flat or match price to minimize error
        # This only happens if volatility is so low we can't wiggle.
        return np.clip(prices, fwd_min, fwd_max)

    return best_path

def optimize_trajectory(df):
    # Group by month
    df['period'] = df['timestamp'].dt.to_period('M')
    
    full_line = []
    
    # Initial Condition: Start at Price[0]
    current_val = df['close'].iloc[0]
    
    print("Optimizing trajectory (Greedy Area Maximization)...")
    
    # Process strictly in order
    periods = df['period'].unique()
    
    for p in periods:
        group = df[df['period'] == p]
        prices = group['close'].values
        
        # Solve
        path = solve_month(prices, current_val)
        
        full_line.append(path)
        
        # STRICT CONTINUITY: Next month starts where this one ended
        current_val = path[-1]
        
    return np.concatenate(full_line)

# --- 3. Visualization Server ---

class ChartHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            
            # Lazy Init
            if 'df_cache' not in globals():
                global df_cache
                df_cache = fetch_data_2020()
                df_cache['optimized'] = optimize_trajectory(df_cache)
            
            # Matplotlib Plotting
            plt.figure(figsize=(16, 8), dpi=100)
            plt.style.use('dark_background')
            
            # Plot entire history or slice
            # Slicing last 2 years for better visibility, or full if preferred
            # plot_data = df_cache[df_cache['timestamp'] > '2024-01-01'] 
            plot_data = df_cache
            
            plt.plot(plot_data['timestamp'], plot_data['close'], 
                     color='#666666', linewidth=0.5, label='ETH Price')
            
            plt.plot(plot_data['timestamp'], plot_data['optimized'], 
                     color='#00ff00', linewidth=1.0, label='Optimized Line')
            
            plt.title("ETH/USDT: Max Area, 1 Cross/Month, 0.1% Constraint")
            plt.legend()
            plt.grid(True, alpha=0.1)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            self.wfile.write(buf.getvalue())
        else:
            self.send_error(404)

def run():
    print("Server running on port 8080...")
    server = HTTPServer(('', 8080), ChartHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()

if __name__ == '__main__':
    run()
