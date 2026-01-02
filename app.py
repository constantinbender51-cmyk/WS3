import itertools
import numpy as np
import time

# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================
# 15 sample prices (p1 to p15)
PRICES = [100.0, 102.5, 101.2, 105.0, 107.3, 104.5, 103.0, 108.2, 
          110.5, 109.0, 112.4, 115.8, 114.2, 113.5, 118.0]

# Penalty for switching from one action to another (higher = more stability)
SWITCHING_PENALTY_WEIGHT = 1 

# Actions available at each step
ACTIONS = ['Long', 'Hold', 'Short']

# ==========================================

def solve_trading_problem():
    """
    Finds the optimal sequence of actions to maximize:
    (Mean Return / Std Dev) - (Switching Penalty * Switches)
    """
    n_prices = len(PRICES)
    n_intervals = n_prices - 1
    
    # Pre-calculate price changes
    price_diffs = np.diff(PRICES)
    
    best_score = -float('inf')
    best_sequence = None
    
    print(f"Analyzing {3**n_intervals:,} possible sequences...")
    start_time = time.time()

    # Iterate through all 3^(n-1) action sequences
    for sequence in itertools.product(ACTIONS, repeat=n_intervals):
        # Map actions to return multipliers
        # Long = 1, Hold = 0, Short = -1
        multipliers = np.array([1 if a == 'Long' else (-1 if a == 'Short' else 0) for a in sequence])
        
        # Calculate resulting returns for this path
        strategy_returns = price_diffs * multipliers
        
        # 1. Return/Risk Metric
        total_return = np.sum(strategy_returns)
        std_dev = np.std(strategy_returns)
        
        # Avoid division by zero
        risk_adj_return = total_return / (std_dev + 1e-9)
        
        # 2. Switching Penalty
        # Count how many times the action at t differs from t-1
        switches = 0
        for i in range(1, n_intervals):
            if sequence[i] != sequence[i-1]:
                switches += 1
        
        # Normalize switching penalty by number of possible switch points
        penalty = (switches / (n_intervals - 1)) * SWITCHING_PENALTY_WEIGHT
        
        # Final Objective: Maximize Risk Adjusted Return minus Penalty
        current_score = risk_adj_return - penalty
        
        if current_score > best_score:
            best_score = current_score
            best_sequence = sequence
            best_metrics = {
                "Return": total_return,
                "StdDev": std_dev,
                "Switches": switches,
                "Score": current_score
            }

    end_time = time.time()
    
    # Display Results
    print("\n" + "="*40)
    print("OPTIMAL TRADING SEQUENCE FOUND")
    print("="*40)
    print(f"Computation Time: {end_time - start_time:.2f} seconds")
    print(f"Final Score:      {best_metrics['Score']:.4f}")
    print(f"Total Return:     {best_metrics['Return']:.2f}")
    print(f"Volatility (Std): {best_metrics['StdDev']:.2f}")
    print(f"Total Switches:   {best_metrics['Switches']}")
    print("-" * 40)
    
    print("Step-by-Step Action Plan:")
    for i, action in enumerate(best_sequence):
        print(f"Interval {i+1} (p{i+1}->p{i+2}): {action}")

if __name__ == "__main__":
    solve_trading_problem()