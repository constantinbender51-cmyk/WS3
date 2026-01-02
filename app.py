import itertools
import numpy as np
import time
import urllib.request
import json

# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================
# The asset pair name for Binance (e.g., BTCUSDT, ETHUSDT)
# Note: Binance does not use slashes in the symbol.
PAIR = 'BTCUSDT' 

# Size of the optimization window (number of prices)
# 10 provides a good balance between depth and performance (3^9 combinations).
WINDOW_SIZE = 10 

# Penalty for switching actions (Set to 25 as requested)
# This strongly favors long-term positions over frequent trading.
SWITCHING_PENALTY_WEIGHT = 35

# Actions available at each step
ACTIONS = ['Long', 'Hold', 'Short']

# ==========================================

def get_binance_monthly_close(symbol):
    """
    Fetches monthly kline (OHLC) data from Binance's public API.
    Uses interval '1M' for monthly data.
    """
    print(f"Fetching monthly data for {symbol} from Binance...")
    # Binance endpoint: /api/v3/klines
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1M&limit=1000"
    
    try:
        # Standard headers to prevent rejection
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            
            # Binance returns a list of lists.
            # Format: [ [OpenTime, Open, High, Low, Close, Vol, CloseTime, ...], ... ]
            if not isinstance(data, list) or len(data) == 0:
                print("No kline data found in the response.")
                return None
            
            # Index 4 is the Close price in Binance's kline array
            closes = [float(entry[4]) for entry in data]
            return closes

    except Exception as e:
        print(f"Connection error or invalid symbol: {e}")
        return None

def optimize_segment(segment_prices, last_action=None):
    """
    Optimizes a single segment of prices to maximize risk-adjusted returns
    while minimizing switches based on the penalty weight.
    """
    n_intervals = len(segment_prices) - 1
    price_diffs = np.diff(segment_prices)
    
    best_score = -float('inf')
    best_seq = None
    
    # Iterate through all combinations for this window (3^n)
    for sequence in itertools.product(ACTIONS, repeat=n_intervals):
        # Numeric mapping: Long=1, Short=-1, Hold=0
        multipliers = np.array([1 if a == 'Long' else (-1 if a == 'Short' else 0) for a in sequence])
        strategy_returns = price_diffs * multipliers
        
        # Metric 1: Risk-Adjusted Return (Sharpe-like ratio)
        total_return = np.sum(strategy_returns)
        std_dev = np.std(strategy_returns)
        # Add epsilon to avoid division by zero
        risk_adj_return = total_return / (std_dev + 1e-9)
        
        # Metric 2: Switching Penalty
        switches = 0
        # Continuity check: Transition from the previous window's last action
        if last_action and sequence[0] != last_action:
            switches += 1
            
        for i in range(1, n_intervals):
            if sequence[i] != sequence[i-1]:
                switches += 1
        
        # Calculate penalty score
        penalty_score = (switches * SWITCHING_PENALTY_WEIGHT) / n_intervals
        
        # Combine metrics
        current_score = risk_adj_return - penalty_score
        
        if current_score > best_score:
            best_score = current_score
            best_seq = sequence
            
    return best_seq

def run_backtest():
    """
    Main execution loop: Fetch data -> Run Windowed Optimization -> Report Results
    """
    prices = get_binance_monthly_close(PAIR)
    
    if not prices or len(prices) < 2:
        print("Aborting: Not enough price data available for the symbol provided.")
        return

    print(f"Data received. Analyzing {len(prices)} months of closing prices.")
    
    full_sequence = []
    last_action = None
    start_time = time.time()
    
    # Process overlapping windows (step is WINDOW_SIZE - 1 to share the boundary price)
    step_size = WINDOW_SIZE - 1
    
    for i in range(0, len(prices) - 1, step_size):
        end_idx = min(i + WINDOW_SIZE, len(prices))
        segment = prices[i:end_idx]
        
        if len(segment) < 2:
            break
            
        window_best_seq = optimize_segment(segment, last_action)
        full_sequence.extend(window_best_seq)
        last_action = window_best_seq[-1]

    execution_time = time.time() - start_time
    
    # Final Statistics Calculation
    # Slice prices to match the length of actions generated
    price_slice = prices[:len(full_sequence)+1]
    diffs = np.diff(price_slice)
    multipliers = np.array([1 if a == 'Long' else (-1 if a == 'Short' else 0) for a in full_sequence])
    returns = diffs * multipliers
    
    total_switches = sum(1 for i in range(1, len(full_sequence)) if full_sequence[i] != full_sequence[i-1])
    
    print("\n" + "="*50)
    print(f"BINANCE TRADING STRATEGY REPORT: {PAIR}")
    print("="*50)
    print(f"Total Periods:    {len(full_sequence)} months")
    print(f"Total Net Return: {np.sum(returns):.2f}")
    print(f"Strategy StdDev:  {np.std(returns):.2f}")
    print(f"Total Switches:   {total_switches}")
    print(f"Penalty Weight:   {SWITCHING_PENALTY_WEIGHT}")
    print(f"Execution Time:   {execution_time:.2f}s")
    print("-" * 50)
    
    print("Action Log (Last 12 Months):")
    recent_actions = full_sequence[-12:]
    for i, action in enumerate(recent_actions):
        month_idx = len(full_sequence) - len(recent_actions) + i + 1
        print(f"Month {month_idx:03}: {action}")

if __name__ == "__main__":
    run_backtest()