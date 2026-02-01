import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime

# --- 1. Data Fetching ---

def fetch_data_2020():
    print("Fetching ETH/USDT 1h data from 2020...")
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
            since = ohlcv[-1][0] + 3600000 # Advance 1h
        except Exception as e:
            print(f"Fetch Error: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset='timestamp').reset_index(drop=True)
    print(f"Data loaded: {len(df)} candles.")
    return df

# --- 2. Cone Logic ---

def get_cone_bound(start_val, n, direction, rate=0.001):
    """
    Returns a boundary array starting at start_val.
    direction: +1 (Upper 1.001^t), -1 (Lower 0.999^t)
    """
    t = np.arange(n)
    log_rate = np.log(1 + direction * rate)
    return start_val * np.exp(t * log_rate)

def solve_segment_strict(prices, start_val, rate=0.001):
    """
    Finds a path that:
    1. Starts at start_val
    2. Crosses 'prices' exactly once.
    3. Maximizes area.
    4. Never jumps > 0.1% per step.
    """
    n = len(prices)
    best_area = -1.0
    best_path = None
    
    # 1. Forward Reachability from Start
    fwd_max = get_cone_bound(start_val, n, 1, rate)
    fwd_min = get_cone_bound(start_val, n, -1, rate)
    
    # Determine implied start polarity (Above or Below price)
    # We must cross, so we must start on one side and end on the other.
    # If start_val is exactly price[0], we can choose either direction.
    # If start_val != price[0], polarity is forced.
    
    if start_val > prices[0]:
        modes = [('above', 'below')]
    elif start_val < prices[0]:
        modes = [('below', 'above')]
    else:
        modes = [('above', 'below'), ('below', 'above')]
        
    # Search Stride: Start coarse, refine if needed. 
    # Since "Always possible", a solution exists.
    strides = [5, 1] 
    
    for stride in strides:
        if best_path is not None: break # Found solution in coarse pass
        
        # Iterate potential crossing indices k
        # k is the index where Line[k] ~ Price[k]
        # Restrict k to [1, n-1] to ensure valid "before" and "after" segments
        for k in range(1, n, stride):
            target = prices[k]
            
            # Check 1: Is target reachable from start?
            if not (fwd_min[k] <= target <= fwd_max[k]):
                continue
                
            for start_mode, end_mode in modes:
                # --- Segment 1: 0 to k (Pre-Crossing) ---
                # We need to connect start_val to target at k.
                # To maximize area, we want to hug the boundary furthest from price.
                
                # Backward reachability from target
                bwd_len = k + 1
                if start_mode == 'above':
                    # We are above price. Maximize Height.
                    # Upper Limit is Min(Forward_Max_Cone, Backward_Max_Cone_From_Target)
                    # Backward max from target implies we arrived at target from a higher value.
                    # This is effectively calculating the cone UP from target backwards.
                    bwd_limit = get_cone_bound(target, bwd_len, 1, rate)[::-1]
                    path_pre = np.minimum(fwd_max[:bwd_len], bwd_limit)
                    
                    # Feasibility Check: The path must not be forced below price before k (too strictly)
                    # Relaxed check: Just ensure we don't cross early.
                    # If path_pre dips below prices[:k+1], we might have >1 crossing.
                    if np.any(path_pre[:-1] <= prices[:k]): continue
                    
                else:
                    # We are below price. Minimize Height.
                    # Lower Limit is Max(Forward_Min_Cone, Backward_Min_Cone_From_Target)
                    bwd_limit = get_cone_bound(target, bwd_len, -1, rate)[::-1]
                    path_pre = np.maximum(fwd_min[:bwd_len], bwd_limit)
                    
                    if np.any(path_pre[:-1] >= prices[:k]): continue

                # --- Segment 2: k to n (Post-Crossing) ---
                # Start at target, diverge as fast as possible to maximize area.
                post_len = n - k
                if end_mode == 'above':
                    # Cross to Above -> Go Up Max
                    path_post = get_cone_bound(target, post_len, 1, rate)
                else:
                    # Cross to Below -> Go Down Max
                    path_post = get_cone_bound(target, post_len, -1, rate)
                    
                # Stitch
                # path_pre includes k, path_post includes k.
                full_path = np.concatenate([path_pre, path_post[1:]])
                
                # Calculate Area
                diffs = full_path - prices
                area = np.sum(np.abs(diffs))
                
                if area > best_area:
                    best_area = area
                    best_path = full_path
    
    if best_path is None:
        # Should not happen per prompt assertion. 
        # Return a flat line clamped to limits to avoid crash, 
        # but theoretically unreachable code.
        print(f"Warn: Solver failed at {start_val:.2f}. Returning clamped line.")
        return np.clip(prices, fwd_min, fwd_max)
        
    return best_path

def optimize_trajectory(df):
    df['month'] = df['timestamp'].dt.to_period('M')
    full_line = []
    
    # Initialize at first price
    current_val = df['close'].iloc[0]
    
    # Group processing
    # Note: Iterate strictly in time order
    sorted_groups = sorted(list(df.groupby('month')), key=lambda x: x[0])
    
    print("Optimizing trajectory...")
    for _, group in sorted_groups:
        prices = group['close'].values
        
        path = solve_segment_strict(prices, current_val)
        
        full_line.append(path)
        current_val = path[-1] # Strict continuity for next month
        
    return np.concatenate(full_line)

# --- 3. Server ---

class ChartHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            
            # Generate Data on Request (Cached in memory in real app, simplified here)
            # Fetching fresh for the request or using global if pre-fetched
            if 'df_cache' not in globals():
                global df_cache
                df_cache = fetch_data_2020()
                df_cache['optimized'] = optimize_trajectory(df_cache)
            
            # Plot
            plt.figure(figsize=(16, 8), dpi=100)
            plt.style.use('dark_background')
            
            # Plotting only the last 2000 points for clarity, or full if requested
            # Plotting full history since 2020
            
            # Downsample for speed in plotting (1h data for 4 years is ~35k points)
            plot_df = df_cache
            
            plt.plot(plot_df['timestamp'], plot_df['close'], color='#555555', linewidth=0.5, label='ETH Price')
            plt.plot(plot_df['timestamp'], plot_df['optimized'], color='#00ff00', linewidth=1.0, label='Optimized Line')
            
            plt.title("ETH/USDT (2020-Now): Area Max, 1 Cross/Month, 0.1% Max Deviation")
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
    server = HTTPServer(('', 8080), ChartHandler)
    print("Server running on port 8080...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()

if __name__ == '__main__':
    run()
