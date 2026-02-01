import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timedelta

# --- 1. Data Fetching (Pagination) ---

def fetch_history_from_2020(symbol='ETH/USDT', timeframe='1h'):
    exchange = ccxt.binance()
    # Binance limit per request
    limit = 1000 
    since = exchange.parse8601('2020-01-01T00:00:00Z')
    now = exchange.milliseconds()
    
    all_ohlcv = []
    
    print(f"Fetching data from 2020 for {symbol}...")
    
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Update 'since' to the last timestamp + 1 timeframe duration
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 60 * 60 * 1000 # +1 hour in ms
            
            # Rate limit mitigation
            # time.sleep(exchange.rateLimit / 1000) 
            
            # Progress indicator (optional, keeps logs dense)
            if len(all_ohlcv) % 5000 == 0:
                print(f"Fetched {len(all_ohlcv)} candles...")
                
        except Exception as e:
            print(f"Fetch error: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Drop duplicates if any overlap occurred
    df = df.drop_duplicates(subset='timestamp').reset_index(drop=True)
    print(f"Total rows fetched: {len(df)}")
    return df

# --- 2. Optimization Logic ---

def get_cone(start_val, steps, direction, rate=0.001):
    t = np.arange(steps)
    # P_t = P_0 * (1 +/- rate)^t
    log_v = np.log(start_val) + t * np.log(1 + (direction * rate))
    return np.exp(log_v)

def generate_inertial_path(start_val, prices, rate=0.001):
    """
    Fallback generator.
    If exact crossing is impossible, this path adheres strictly to the 
    volatility constraint (0.1%) while attempting to converge with price.
    It eliminates the vertical jump artifact.
    """
    n = len(prices)
    path = np.zeros(n)
    current = start_val
    path[0] = current
    
    # Pre-compute log factors
    log_up = np.log(1 + rate)
    log_down = np.log(1 - rate)
    
    for t in range(1, n):
        target = prices[t]
        # Determine direction towards target
        if target > current:
            # Move up max allowed amount
            # Check if we can reach target in one step (unlikely but possible)
            next_max = current * np.exp(log_up)
            current = min(next_max, target)
        else:
            # Move down max allowed amount
            next_min = current * np.exp(log_down)
            current = max(next_min, target)
        path[t] = current
        
    return path

def solve_segment(prices, start_val, rate=0.001):
    n = len(prices)
    best_area = -1
    best_path = None
    
    # Optimization: If N is large, stride the search.
    # For 1h candles over 1 month (approx 720), stride=1 is fine.
    # If partial month is very short, ensure stride doesn't skip.
    stride = 1 if n < 1000 else 5
    
    fwd_max = get_cone(start_val, n, 1, rate)
    fwd_min = get_cone(start_val, n, -1, rate)

    # Search for crossing point k
    # We restrict k to be at least 1 step in and 1 step before end
    for k in range(1, n-1, stride): 
        target = prices[k]
        
        # Pruning: Is target reachable from start?
        if not (fwd_min[k] <= target <= fwd_max[k]):
            continue

        scenarios = [('above', 'below'), ('below', 'above')]
        
        for pre_mode, post_mode in scenarios:
            # --- Backward Reachability (0 to k) ---
            bwd_len = k + 1
            if pre_mode == 'above':
                # Upper bound: Min(FwdMax, BwdMaxFromTarget)
                bwd_cone = get_cone(target, bwd_len, 1, rate)[::-1]
                seg_pre = np.minimum(fwd_max[:bwd_len], bwd_cone)
            else:
                # Lower bound: Max(FwdMin, BwdMinFromTarget)
                bwd_cone = get_cone(target, bwd_len, -1, rate)[::-1]
                seg_pre = np.maximum(fwd_min[:bwd_len], bwd_cone)
            
            # check validity of pre-segment (cannot cross bounds)
            if pre_mode == 'above' and np.any(seg_pre < fwd_min[:bwd_len]): continue
            if pre_mode == 'below' and np.any(seg_pre > fwd_max[:bwd_len]): continue

            # --- Forward Reachability (k to end) ---
            post_len = n - k
            if post_mode == 'below':
                seg_post = get_cone(target, post_len, -1, rate)
            else:
                seg_post = get_cone(target, post_len, 1, rate)

            # Stitch
            full_path = np.concatenate([seg_pre, seg_post[1:]])
            
            # --- Verify Crossing Count ---
            diff = full_path - prices
            signs = np.sign(diff)
            # Forward propagate zeros to avoid false crossing counts on exact touches
            for i in range(1, n):
                if signs[i] == 0: signs[i] = signs[i-1]
                
            crossings = np.count_nonzero(np.diff(signs))
            
            if crossings != 1:
                continue
            
            area = np.sum(np.abs(diff))
            if area > best_area:
                best_area = area
                best_path = full_path

    # FALLBACK
    if best_path is None:
        # User reported a jump. 
        # The jump happens because we returned 'prices' previously.
        # Now we return a volatility-compliant path that drifts.
        return generate_inertial_path(start_val, prices, rate)
        
    return best_path

def calculate_full_trajectory():
    df = fetch_history_from_2020()
    df['month'] = df['timestamp'].dt.to_period('M')
    
    full_line_segments = []
    
    # Initialization
    # We must ensure the very first point matches price[0]
    current_val = df['close'].iloc[0]
    
    # Process Group by Group
    # Note: groupby preserves order of keys, but we double check sort
    groups = df.groupby('month', sort=False)
    
    for name, group in groups:
        prices = group['close'].values
        
        # Constraint: L[0] must equal current_val (from previous month's end)
        # Note: If this is the very first month, current_val == prices[0]
        # If this is subsequent month, prices[0] might be different from current_val
        # The solver calculates path relative to current_val.
        
        path = solve_segment(prices, current_val)
        
        full_line_segments.append(path)
        current_val = path[-1]
        
    final_line = np.concatenate(full_line_segments)
    
    # Length alignment check
    if len(final_line) != len(df):
        print(f"Warning: Length mismatch. DF: {len(df)}, Line: {len(final_line)}")
        # Truncate to match shortest
        min_len = min(len(df), len(final_line))
        df = df.iloc[:min_len]
        final_line = final_line[:min_len]
        
    df['optimized'] = final_line
    return df

# --- 3. Visualization Server ---

class ChartHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            
            df = calculate_full_trajectory()
            
            # Plotting
            plt.figure(figsize=(14, 7), dpi=100)
            plt.style.use('dark_background')
            
            # Sub-sample for performance if needed (e.g. every 4th point)
            # df_plot = df.iloc[::4] 
            df_plot = df
            
            plt.plot(df_plot['timestamp'], df_plot['close'], label='ETH Price', color='#444444', linewidth=0.8)
            plt.plot(df_plot['timestamp'], df_plot['optimized'], label='Constrained Line', color='#00ff00', linewidth=1.2)
            
            plt.title("ETH/USDT 2020-Present: Area Maximization (No Discontinuities)")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True, alpha=0.15)
            plt.tight_layout()
            
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
    print("Serving chart on port 8080...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()

if __name__ == '__main__':
    run_server()
