import itertools
import numpy as np
import time

# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================
# Example prices (can be any length)
PRICES = [
    100.0, 102.5, 101.2, 105.0, 107.3, 104.5, 103.0, 108.2, 
    110.5, 109.0, 112.4, 115.8, 114.2, 113.5, 118.0, 119.5,
    121.0, 120.5, 122.3, 125.0, 124.5, 126.0, 128.5, 127.0, 130.0
]

# Size of the optimization window (number of prices)
WINDOW_SIZE = 12 

# Penalty for switching from one action to another
SWITCHING_PENALTY_WEIGHT = 10

# Actions available at each step
ACTIONS = ['Long', 'Hold', 'Short']

# ==========================================

def optimize_segment(segment_prices, last_action=None):
    """
    Optimizes a single segment of prices.
    If last_action is provided, it influences the switching penalty of the first move.
    """
    n_intervals = len(segment_prices) - 1
    price_diffs = np.diff(segment_prices)
    
    best_score = -float('inf')
    best_seq = None
    
    # Iterate through all combinations for this window
    for sequence in itertools.product(ACTIONS, repeat=n_intervals):
        multipliers = np.array([1 if a == 'Long' else (-1 if a == 'Short' else 0) for a in sequence])
        strategy_returns = price_diffs * multipliers
        
        # 1. Return/Risk
        total_return = np.sum(strategy_returns)
        std_dev = np.std(strategy_returns)
        risk_adj_return = total_return / (std_dev + 1e-9)
        
        # 2. Switching Penalty
        switches = 0
        # Check transition from the previous window's last action
        if last_action and sequence[0] != last_action:
            switches += 1
            
        for i in range(1, n_intervals):
            if sequence[i] != sequence[i-1]:
                switches += 1
        
        penalty = (switches / n_intervals) * SWITCHING_PENALTY_WEIGHT
        current_score = risk_adj_return - penalty
        
        if current_score > best_score:
            best_score = current_score
            best_seq = sequence
            
    return best_seq

def solve_windowed_trading():
    """
    Processes the full price list using overlapping windows to maintain continuity.
    """
    full_sequence = []
    last_action = None
    start_time = time.time()
    
    # Calculate how many intervals we need to cover
    # We move through prices in chunks of (WINDOW_SIZE - 1)
    step = WINDOW_SIZE - 1
    
    print(f"Starting windowed optimization (Window Size: {WINDOW_SIZE})")
    
    for i in range(0, len(PRICES) - 1, step):
        # Define the window boundaries
        end_idx = min(i + WINDOW_SIZE, len(PRICES))
        segment = PRICES[i:end_idx]
        
        # If segment is too small (e.g. only 1 price left), break
        if len(segment) < 2:
            break
            
        print(f"  Optimizing window: Price index {i} to {end_idx-1}...")
        
        # Solve this window
        window_best_seq = optimize_segment(segment, last_action)
        
        # Store results and update last_action for the next window
        full_sequence.extend(window_best_seq)
        last_action = window_best_seq[-1]

    end_time = time.time()
    
    # Final Statistics Calculation
    price_diffs = np.diff(PRICES)
    multipliers = np.array([1 if a == 'Long' else (-1 if a == 'Short' else 0) for a in full_sequence])
    strategy_returns = price_diffs * multipliers
    
    total_switches = sum(1 for i in range(1, len(full_sequence)) if full_sequence[i] != full_sequence[i-1])
    
    print("\n" + "="*40)
    print("WINDOWED OPTIMIZATION COMPLETE")
    print("="*40)
    print(f"Total Prices:     {len(PRICES)}")
    print(f"Total Return:     {np.sum(strategy_returns):.2f}")
    print(f"Volatility (Std): {np.std(strategy_returns):.2f}")
    print(f"Total Switches:   {total_switches}")
    print(f"Execution Time:   {end_time - start_time:.2f}s")
    print("-" * 40)
    
    print("Final Strategy Sequence:")
    for i, action in enumerate(full_sequence):
        print(f"t({i}->{i+1}): {action}")

if __name__ == "__main__":
    solve_windowed_trading()