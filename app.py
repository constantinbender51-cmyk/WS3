import ccxt
import pandas as pd
import numpy as np
from datetime import datetime

def fetch_data(symbol='ETH/USDT', timeframe='1h', limit=2000):
    """Fetches OHLC data from Binance."""
    exchange = ccxt.binance()
    # Fetch in batches if necessary, here we grab the latest snapshot
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def get_cone_boundaries(start_val, n_steps, max_change=0.001):
    """
    Calculates the max expansion cone from a starting value.
    Returns (min_path, max_path) arrays.
    """
    # expansion factors: (1-delta)^t and (1+delta)^t
    # using log space for numerical stability: log(v_t) = log(v_0) + t*log(1+delta)
    log_start = np.log(start_val)
    log_up = np.log(1 + max_change)
    log_down = np.log(1 - max_change)
    
    steps = np.arange(n_steps)
    
    upper = np.exp(log_start + steps * log_up)
    lower = np.exp(log_start + steps * log_down)
    return lower, upper

def solve_month_trajectory(prices, start_val, max_change=0.001):
    """
    Finds the optimal path L that:
    1. Starts at start_val
    2. Crosses prices exactly once at optimal index k
    3. Maximizes sum(|Price - L|)
    4. Adheres to step constraints.
    """
    n = len(prices)
    best_area = -1.0
    best_path = None
    
    # Pre-calculate Forward Reachability from Start
    fwd_min, fwd_max = get_cone_boundaries(start_val, n, max_change)

    # Iterating every possible crossing point k
    # k is the index where L[k] == Prices[k] (approximately)
    # The line must switch sides at k.
    
    # Optimization: stride to save time if N is large, else stride=1
    for k in range(0, n, 1):
        target_price = prices[k]
        
        # 1. Check Forward Reachability
        if not (fwd_min[k] <= target_price <= fwd_max[k]):
            continue

        # 2. Backward Reachability from Crossing Point (k back to 0)
        # We need to hit target_price at k coming from start_val
        # The valid corridor at t < k is intersection of Fwd(Start) and Bwd(Target)
        bwd_min_pre, bwd_max_pre = get_cone_boundaries(target_price, k + 1, max_change)
        bwd_min_pre = bwd_min_pre[::-1] # Reverse to match time 0..k
        bwd_max_pre = bwd_max_pre[::-1]
        
        # Valid corridor 0..k
        valid_min_pre = np.maximum(fwd_min[:k+1], bwd_min_pre)
        valid_max_pre = np.minimum(fwd_max[:k+1], bwd_max_pre)
        
        if np.any(valid_min_pre > valid_max_pre):
            continue # No valid path exists to hit target
            
        # 3. Forward Reachability from Crossing Point (k to End)
        fwd_min_post, fwd_max_post = get_cone_boundaries(target_price, n - k, max_change)
        
        # Construct the line
        # Logic: To maximize area, we hug the boundary furthest from price.
        # But we must ensure we are on ONE side of the price before k and OTHER side after k.
        
        # Segment 1: 0 to k
        # Determine polarity based on start_val vs Price[0]. 
        # If start_val > Price[0], we should stay ABOVE price.
        # However, we can choose the side. The constraint is "cross ONCE".
        # So we check both scenarios: (Above -> Below) and (Below -> Above)
        
        scenarios = [('above', 'below'), ('below', 'above')]
        
        for pre_mode, post_mode in scenarios:
            current_path = np.zeros(n)
            
            # --- Build Pre-Crossing Segment (0..k) ---
            # If we want to be ABOVE price, we hug valid_max_pre.
            # If valid_min_pre is already > price, we are forced above.
            # If valid_max_pre < price, this mode is invalid.
            
            seg1_valid = True
            
            if pre_mode == 'above':
                # Constraint: L > Price (relaxed to >= due to crossing)
                # To maximize area: Maximize L. So pick valid_max_pre.
                # Check feasibility: valid_max_pre must be >= Price
                # Actually, we just take valid_max_pre. If it dips below price where it shouldn't, 
                # we technically violate "cross once", but we treat k as the SOLE crossing.
                # Strictly: We should take max(valid_max_pre, but limited by ?).
                # Simplified: Just trace valid_max_pre.
                current_path[:k+1] = valid_max_pre
            else:
                # Mode Below: Minimize L. Pick valid_min_pre.
                current_path[:k+1] = valid_min_pre
                
            # --- Build Post-Crossing Segment (k..n) ---
            if post_mode == 'above':
                # Maximize L -> fwd_max_post
                current_path[k:] = fwd_max_post
            else:
                # Minimize L -> fwd_min_post
                current_path[k:] = fwd_min_post
            
            # 4. Verify Crossing Count strictly
            # We calculate diffs. Sign changes should be exactly 1 (at k).
            diffs = current_path - prices
            # Handle exact zero matches by forwarding previous sign
            signs = np.sign(diffs)
            # Replace 0 with previous sign to avoid double counting touch as cross
            for i in range(1, n):
                if signs[i] == 0: signs[i] = signs[i-1]
                
            # Count sign flips
            flips = np.count_nonzero(np.diff(signs))
            
            # Refine logic: The optimization might force the line to cross violently if we just pick bounds.
            # However, picking the bound IS the only way to maximize area.
            # If picking the bound causes >1 crossing, this specific k with this specific mode is invalid.
            if flips != 1:
                continue
                
            # Calculate Area
            area = np.sum(np.abs(diffs))
            
            if area > best_area:
                best_area = area
                best_path = current_path

    return best_path

def main():
    # 1. Fetch
    print("Fetching data...")
    df = fetch_data(limit=24 * 30 * 3) # Approx 3 months
    prices = df['close'].values
    times = df['timestamp']
    
    # 2. Group by Month
    # We need indices for each month to process segments
    df['month_group'] = times.dt.to_period('M')
    groups = df.groupby('month_group')
    
    full_line = []
    
    # Initial Start Value: Match price strictly or allow deviation?
    # Prompt implies "the line" exists. We start at price to be neutral.
    current_val = prices[0]
    
    print("Optimizing trajectory...")
    for period, group in groups:
        month_prices = group['close'].values
        
        # Ensure continuity: Start of month = End of prev month
        # First month starts at current_val (Price[0])
        
        # Solve for this month
        path = solve_month_trajectory(month_prices, current_val)
        
        if path is None:
            # Fallback if solver fails (e.g., extreme volatility makes 0.1% constraint impossible to hit ANY point)
            # In a real engine, we'd backtrack. Here, we reset to price.
            print(f"Warn: No valid path for {period}. Resetting to price.")
            path = month_prices 
            
        full_line.append(path)
        current_val = path[-1] # Set start for next month
        
    # Concatenate
    final_line = np.concatenate(full_line)
    
    # Attach to DF
    # Note: If grouped logic split indices, we assume strict ordering preserved.
    df['optimized_line'] = final_line
    df['diff_pct'] = (df['optimized_line'] - df['close']) / df['close'] * 100
    
    # Output Stats
    total_area = np.sum(np.abs(df['optimized_line'] - df['close']))
    print(f"Optimization Complete.")
    print(f"Total Area (Sum of abs diff): {total_area:.2f}")
    print(f"Last Value: {final_line[-1]:.2f}")
    
    # Validation of Deviation Constraint
    # We check step-by-step pct change of the line
    line_pct_change = np.abs(np.diff(final_line) / final_line[:-1])
    max_dev = np.max(line_pct_change)
    print(f"Max Step Deviation: {max_dev:.6f} (Constraint <= 0.001)")
    
    # Validation of Crossing Constraint
    # Calculated per month in solver, but globally:
    crossings = 0
    signs = np.sign(final_line - prices)
    # clean zeros
    for i in range(1, len(signs)):
        if signs[i] == 0: signs[i] = signs[i-1]
    crossings = np.count_nonzero(np.diff(signs))
    print(f"Total Crossings: {crossings} (Should be approx 1 per month)")

if __name__ == "__main__":
    main()
