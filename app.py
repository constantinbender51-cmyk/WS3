import itertools
import numpy as np
import time
import urllib.request
import json

# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================
# Asset pair for Kraken (e.g., 'XXBTZUSD' for BTC/USD, 'XETHZUSD' for ETH/USD)
PAIR = 'XXBTZUSD'

# Size of the optimization window (number of prices)
# Note: 12 is manageable (3^11 = 177k combinations). 
# Increasing this significantly will slow down the script.
WINDOW_SIZE = 10 

# Penalty for switching from one action to another (Increased to 25 as requested)
SWITCHING_PENALTY_WEIGHT = 25.0 

# Actions available at each step
ACTIONS = ['Long', 'Hold', 'Short']

# ==========================================

def get_kraken_monthly_close(pair):
    """
    Fetches monthly OHLC data from Kraken's public API and returns closing prices.
    Monthly interval code is 43200 (minutes in a 30-day month approx).
    """
    print(f"Fetching monthly data for {pair} from Kraken...")
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=43200"
    
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            
            if data.get('error'):
                print(f"Error from Kraken: {data['error']}")
                return None
            
            # Kraken returns a dictionary where the key is the pair name
            # The result list contains: [time, open, high, low, close, vwap, volume, count]
            result_key = list(data['result'].keys())[0]
            ohlc_data = data['result'][result_key]
            
            # Extract close prices (index 4) and convert to float
            closes = [float(day[4]) for day in ohlc_data]
            return closes
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return None

def optimize_segment(segment_prices, last_action=None):
    """
    Optimizes a single segment of prices using brute force.
    """
    n_intervals = len(segment_prices) - 1
    price_diffs = np.diff(segment_prices)
    
    best_score = -float('inf')
    best_seq = None
    
    # Iterate through all combinations for this window (3^(n_intervals))
    for sequence in itertools.product(ACTIONS, repeat=n_intervals):
        # Map actions to multipliers
        multipliers = np.array([1 if a == 'Long' else (-1 if a == 'Short' else 0) for a in sequence])
        strategy_returns = price_diffs * multipliers
        
        # 1. Return/Risk (Sharpe-like ratio)
        total_return = np.sum(strategy_returns)
        std_dev = np.std(strategy_returns)
        # Handle zero volatility cases
        risk_adj_return = total_return / (std_dev + 1e-9)
        
        # 2. Switching Penalty
        switches = 0
        if last_action and sequence[0] != last_action:
            switches += 1
            
        for i in range(1, n_intervals):
            if sequence[i] != sequence[i-1]:
                switches += 1
        
        # Calculate penalty: (Switches * Weight) / intervals
        # High weight (25) will significantly penalize any change in action
        penalty = (switches * SWITCHING_PENALTY_WEIGHT) / n_intervals
        current_score = risk_adj_return - penalty
        
        if current_score > best_score:
            best_score = current_score
            best_seq = sequence
            
    return best_seq

def solve_windowed_trading():
    """
    Processes the full price list using overlapping windows.
    """
    # Fetch real data
    prices = get_kraken_monthly_close(PAIR)
    
    if not prices or len(prices) < 2:
        print("Insufficient data to run optimization.")
        return

    print(f"Data loaded: {len(prices)} monthly periods.")
    
    full_sequence = []
    last_action = None
    start_time = time.time()
    
    # Step size to overlap windows (keeping the last price of window N as the first of N+1)
    step = WINDOW_SIZE - 1
    
    print(f"Starting windowed optimization (Window Size: {WINDOW_SIZE}, Penalty: {SWITCHING_PENALTY_WEIGHT})")
    
    for i in range(0, len(prices) - 1, step):
        end_idx = min(i + WINDOW_SIZE, len(prices))
        segment = prices[i:end_idx]
        
        if len(segment) < 2:
            break
            
        print(f"  Optimizing window: Index {i} to {end_idx-1}...")
        window_best_seq = optimize_segment(segment, last_action)
        
        full_sequence.extend(window_best_seq)
        last_action = window_best_seq[-1]

    end_time = time.time()
    
    # Final Statistics
    price_diffs = np.diff(prices[:len(full_sequence)+1])
    multipliers = np.array([1 if a == 'Long' else (-1 if a == 'Short' else 0) for a in full_sequence])
    strategy_returns = price_diffs * multipliers
    
    total_switches = sum(1 for i in range(1, len(full_sequence)) if full_sequence[i] != full_sequence[i-1])
    
    print("\n" + "="*40)
    print("OPTIMAL TRADING SEQUENCE FOUND (KRAKEN DATA)")
    print("="*40)
    print(f"Asset Pair:       {PAIR}")
    print(f"Total Periods:    {len(full_sequence)}")
    print(f"Total Return:     {np.sum(strategy_returns):.2f}")
    print(f"Volatility (Std): {np.std(strategy_returns):.2f}")
    print(f"Total Switches:   {total_switches}")
    print(f"Execution Time:   {end_time - start_time:.2f}s")
    print("-" * 40)
    
    # Print the last 12 months of the strategy for brevity
    print("Recent Strategy Action Plan (Last 12 months):")
    recent = full_sequence[-12:]
    for i, action in enumerate(recent):
        idx = len(full_sequence) - len(recent) + i
        print(f"Month Index {idx}: {action}")

if __name__ == "__main__":
    solve_windowed_trading()