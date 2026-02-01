import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
import io
from http.server import BaseHTTPRequestHandler, HTTPServer

# --- 1. Data Fetching ---
def fetch_data():
    exchange = ccxt.binance()
    # Fetch approx 3 months of data (24 * 90 = 2160)
    ohlcv = exchange.fetch_ohlcv('ETH/USDT', '1h', limit=1000) 
    # Note: Binance API limit is 1000 per call, using 1000 for speed/demo consistency
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# --- 2. Optimization Logic ---
def get_cone(start_val, steps, direction, rate=0.001):
    """
    Generates a volatility cone. 
    direction=1 for max boundary (1.001^t), -1 for min boundary (0.999^t).
    """
    t = np.arange(steps)
    # Log-space calculation for stability: P_t = P_0 * (1 +/- rate)^t
    log_v = np.log(start_val) + t * np.log(1 + (direction * rate))
    return np.exp(log_v)

def solve_segment(prices, start_val, rate=0.001):
    """
    Greedy optimizer for a single month.
    Returns the path that maximizes area with exactly 1 crossing.
    """
    n = len(prices)
    best_area = -1
    best_path = None
    
    # Pre-calculate forward reachability from start_val
    # We only need the relevant boundary for the chosen polarity, but we calc both.
    fwd_max = get_cone(start_val, n, 1, rate)
    fwd_min = get_cone(start_val, n, -1, rate)

    # Iterate over every possible crossing index k
    # Stride=1 ensures accuracy; increase stride for performance if N is large.
    for k in range(1, n-1): 
        target = prices[k]
        
        # Check if target is reachable from start
        if not (fwd_min[k] <= target <= fwd_max[k]):
            continue

        # Two topological modes:
        # Mode A: Start ABOVE price -> Cross at k -> End BELOW price
        # Mode B: Start BELOW price -> Cross at k -> End ABOVE price
        
        modes = [('above', 'below'), ('below', 'above')]
        
        for pre_mode, post_mode in modes:
            # 1. Backward Reachability from Crossing (k to 0)
            # To maximize area before k:
            # If 'above', we want highest possible path (Min of FwdMax and BwdMax)
            # If 'below', we want lowest possible path (Max of FwdMin and BwdMin)
            
            # Generate backward cone from target
            bwd_len = k + 1
            if pre_mode == 'above':
                # We need upper bound. 
                # Backward from target, max possible value at t < k is target * (1.001)^(k-t)
                # This is equivalent to generating a "max" cone from target and reversing it.
                bwd_cone = get_cone(target, bwd_len, 1, rate)[::-1] # High values back to 0
                # The actual valid upper bound is min(fwd constraint, bwd constraint)
                segment_pre = np.minimum(fwd_max[:bwd_len], bwd_cone)
            else:
                # We need lower bound.
                bwd_cone = get_cone(target, bwd_len, -1, rate)[::-1] # Low values back to 0
                segment_pre = np.maximum(fwd_min[:bwd_len], bwd_cone)
                
            # 2. Forward Reachability from Crossing (k to end)
            # To maximize area after k:
            # If 'below', we want lowest possible path (FwdMin from target)
            # If 'above', we want highest possible path (FwdMax from target)
            post_len = n - k
            if post_mode == 'below':
                segment_post = get_cone(target, post_len, -1, rate)
            else:
                segment_post = get_cone(target, post_len, 1, rate)
                
            # Stitch
            # segment_pre ends at k, segment_post starts at k. 
            # Slice post to avoid duplicate k
            full_path = np.concatenate([segment_pre, segment_post[1:]])
            
            # 3. Validate Constraints
            # Constraint: Strictly 1 crossing
            # We check sign changes. 
            diff = full_path - prices
            signs = np.sign(diff)
            # Filter zeros (touching line is not a cross until it passes)
            clean_signs = signs[signs != 0]
            if len(clean_signs) == 0: continue # unlikely
            crossings = np.count_nonzero(np.diff(clean_signs))
            
            if crossings != 1:
                continue
                
            # Constraint: Deviation <= 0.1% (Implicit in cone construction, but stitching point needs check)
            # The join at k is continuous (both are `target`), so step change is valid if cones are valid.
            
            # Calculate Area
            area = np.sum(np.abs(diff))
            
            if area > best_area:
                best_area = area
                best_path = full_path

    return best_path if best_path is not None else prices # Fallback

def calculate_full_trajectory():
    df = fetch_data()
    df['month'] = df['timestamp'].dt.to_period('M')
    
    full_line_segments = []
    
    # Initialize start value at the first price
    current_val = df['close'].iloc[0]
    
    grouped = df.groupby('month')
    for name, group in grouped:
        prices = group['close'].values
        
        # Solve
        path = solve_segment(prices, current_val)
        
        full_line_segments.append(path)
        current_val = path[-1]
        
    final_line = np.concatenate(full_line_segments)
    
    # Handle length mismatch if grouping dropped indices (rare/edge case)
    if len(final_line) != len(df):
        # Truncate or pad. For simplicity, we re-slice df
        df = df.iloc[:len(final_line)]
        
    df['optimized'] = final_line
    return df

# --- 3. Visualization & Server ---

class ChartHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            
            # Generate Plot
            df = calculate_full_trajectory()
            
            plt.figure(figsize=(12, 6), dpi=100)
            plt.style.use('dark_background')
            
            plt.plot(df['timestamp'], df['close'], label='ETH Price', color='#555555', linewidth=1)
            plt.plot(df['timestamp'], df['optimized'], label='Max Area Line', color='#00ff00', linewidth=1.5)
            
            plt.title("ETH/USDT: Area Maximization (1 Cross/Month, 0.1% Vol Limit)")
            plt.legend()
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            self.wfile.write(buf.getvalue())
        else:
            self.send_error(404)

def run_server():
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, ChartHandler)
    print("Serving Matplotlib chart on port 8080...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()

if __name__ == '__main__':
    run_server()
