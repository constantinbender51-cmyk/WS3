import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server
import matplotlib.pyplot as plt
import io
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime

# --- Global Storage ---
DF_RESULT = None

# --- 1. Data Fetching (Startup) ---
def fetch_data_startup():
    print(f"[{datetime.now()}] INITIALIZING: Fetching ETH/USDT 1h data (2020 - Present)...")
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
            # Update 'since' to last timestamp + 1h to avoid duplicates/gaps
            since = ohlcv[-1][0] + 3600000 
            
            # Rate limit handling (binance is generous, but safe practice)
            time.sleep(0.05)
            
            if len(all_ohlcv) % 10000 == 0:
                print(f"   Fetched {len(all_ohlcv)} candles...")
                
        except Exception as e:
            print(f"   Fetch Error: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Deduplicate and sort
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    print(f"[{datetime.now()}] DATA LOADED: {len(df)} total rows.")
    return df

# --- 2. Cone & Optimization Logic ---

def get_cone_bound(start_val, n, direction, rate=0.001):
    """
    Returns trajectory of max deviation.
    direction: +1 for upper bound (1.001^t), -1 for lower bound (0.999^t).
    """
    t = np.arange(n)
    # Log-linear calculation avoids floating point overflow issues on long sequences
    log_start = np.log(start_val)
    log_rate = np.log(1 + direction * rate)
    return np.exp(log_start + t * log_rate)

def solve_month_trajectory(prices, start_val, rate=0.001):
    """
    Finds the optimal path L for a single month segment.
    Constraints:
    1. L[0] == start_val
    2. |L[t] - L[t-1]| <= rate * L[t-1]
    3. L crosses prices exactly once.
    Objective: Maximize Sum(|L - P|)
    """
    n = len(prices)
    best_area = -1.0
    best_path = None
    
    # Pre-calculate Forward Reachability from Start
    # These define the absolute physical limits of the line given the starting point.
    fwd_upper = get_cone_bound(start_val, n, 1, rate)
    fwd_lower = get_cone_bound(start_val, n, -1, rate)

    # Optimization: Iterate every possible crossing index k
    # We stride by 1 to ensure we find the valid solution (Always Possible).
    # k is the index where L[k] intersects Price[k]
    
    # We restrict k slightly (1 to n-1) to ensure crossing happens "during" the month
    # and not instantaneously at the boundaries (which complicates counting).
    for k in range(1, n):
        target = prices[k]
        
        # 1. Pruning: Is target physically reachable from start?
        if not (fwd_lower[k] <= target <= fwd_upper[k]):
            continue

        # 2. Determine Pre-Crossing Strategy (0 to k)
        # We need a path from start_val to target.
        # To maximize area, we should trace the boundary FURTHEST from price.
        # BUT the path must remain valid.
        # Valid Corridor = Intersection(Forward Cone from Start, Backward Cone from Target)
        
        # Calculate Backward Cone from Target (k back to 0)
        # We perform this locally for the segment length k+1
        dist_bwd = np.arange(k + 1)[::-1] # [k, k-1, ... 0]
        
        # If we approach target from above, we use the "Max Descent" cone backwards
        # If we approach target from below, we use the "Max Ascent" cone backwards
        
        # We check both topological modes:
        # Mode A: Start Above Price -> Cross Down at k
        # Mode B: Start Below Price -> Cross Up at k
        
        # Implied mode by start position relative to price is usually fixed, 
        # but if start == price, both are possible.
        
        modes = []
        if start_val >= prices[0]: modes.append(('above', 'below'))
        if start_val <= prices[0]: modes.append(('below', 'above'))
        
        for pre_mode, post_mode in modes:
            # --- Segment 1: Pre-Crossing (0..k) ---
            if pre_mode == 'above':
                # We want to stay HIGH.
                # Bound is Min(Fwd_Upper, Bwd_Upper_From_Target)
                # Bwd_Upper_From_Target: Maximum value at t that can drop to Target.
                # V_t * (1-rate)^dist = Target => V_t = Target / (1-rate)^dist
                log_target = np.log(target)
                log_down = np.log(1 - rate)
                bwd_limit = np.exp(log_target - dist_bwd * log_down)
                
                path_pre = np.minimum(fwd_upper[:k+1], bwd_limit)
                
                # Check: Does this forced path cross price early?
                # If path_pre dips below price before k, it violates "single crossing".
                # We allow touches (<=), but not deep crossing.
                if np.any(path_pre[:-1] < prices[:k]): 
                    continue
                    
            else: # pre_mode == 'below'
                # We want to stay LOW.
                # Bound is Max(Fwd_Lower, Bwd_Lower_From_Target)
                # Bwd_Lower_From_Target: Minimum value at t that can rise to Target.
                # V_t * (1+rate)^dist = Target => V_t = Target / (1+rate)^dist
                log_target = np.log(target)
                log_up = np.log(1 + rate)
                bwd_limit = np.exp(log_target - dist_bwd * log_up)
                
                path_pre = np.maximum(fwd_lower[:k+1], bwd_limit)
                
                if np.any(path_pre[:-1] > prices[:k]):
                    continue

            # --- Segment 2: Post-Crossing (k..n) ---
            # We are at target. We want to diverge maximally.
            len_post = n - k
            if post_mode == 'above':
                # We crossed UP. Now go UP fast.
                path_post = get_cone_bound(target, len_post, 1, rate)
            else:
                # We crossed DOWN. Now go DOWN fast.
                path_post = get_cone_bound(target, len_post, -1, rate)
                
            # Stitch: path_pre ends at k, path_post starts at k.
            full_path = np.concatenate([path_pre, path_post[1:]])
            
            # --- Score It ---
            area = np.sum(np.abs(full_path - prices))
            
            if area > best_area:
                best_area = area
                best_path = full_path

    # Fallback: If strict 0.1% makes crossing impossible (rare), 
    # we default to following price to maintain continuity constraints.
    if best_path is None:
        print(f"Warning: No valid crossing found for segment starting {start_val}. Cloning price.")
        return prices 
        
    return best_path

def run_algorithm(df):
    print(f"[{datetime.now()}] ALGORITHM STARTING: Optimizing area...")
    
    df['period'] = df['timestamp'].dt.to_period('M')
    full_line = []
    
    # Initialize L[0] = Price[0]
    current_val = df['close'].iloc[0]
    
    # Sort periods to ensure chronological continuity
    periods = sorted(df['period'].unique())
    
    count = 0
    total_periods = len(periods)
    
    for p in periods:
        # Extract data for this month
        mask = df['period'] == p
        prices = df.loc[mask, 'close'].values
        
        # Solve for optimal path
        path = solve_month_trajectory(prices, current_val)
        
        full_line.append(path)
        
        # Set start of next month to end of this month
        current_val = path[-1]
        
        count += 1
        if count % 12 == 0:
            print(f"   Processed {count}/{total_periods} months...")

    # Combine all segments
    final_line = np.concatenate(full_line)
    
    # Ensure length match (pandas groupby might split weirdly in rare cases)
    if len(final_line) != len(df):
        print(f"   Re-indexing result ({len(final_line)} vs {len(df)})")
        # Truncate or pad logic if necessary, usually exact match if sort is correct
        final_line = final_line[:len(df)]
        
    df['optimized'] = final_line
    print(f"[{datetime.now()}] ALGORITHM COMPLETE.")
    return df

# --- 3. Server Logic ---

class ChartHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            
            # Generate Plot from Global Result
            plt.figure(figsize=(16, 9), dpi=100)
            plt.style.use('dark_background')
            
            # Plotting
            # To keep file size manageable, we plot the whole history
            plt.plot(DF_RESULT['timestamp'], DF_RESULT['close'], 
                     color='#444444', linewidth=0.6, label='ETH Price')
            
            plt.plot(DF_RESULT['timestamp'], DF_RESULT['optimized'], 
                     color='#00ff00', linewidth=1.0, label='Optimized Line')
            
            plt.title("ETH/USDT (2020-Present): Max Area, 1 Cross/Month, 0.1% Constraint")
            plt.xlabel("Date")
            plt.ylabel("Price (USDT)")
            plt.legend()
            plt.grid(True, alpha=0.15)
            plt.tight_layout()
            
            # Write to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            self.wfile.write(buf.getvalue())
        else:
            self.send_error(404)

def main():
    global DF_RESULT
    
    # 1. Fetch
    df = fetch_data_startup()
    
    # 2. Compute
    DF_RESULT = run_algorithm(df)
    
    # 3. Serve
    port = 8080
    server = HTTPServer(('', port), ChartHandler)
    print(f"[{datetime.now()}] SERVER READY: Listening on port {port}...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()

if __name__ == "__main__":
    main()
