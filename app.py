import itertools
import numpy as np

def solve_trading_sequence(prices, switching_penalty_weight=0.1):
    """
    Finds the optimal sequence of Long (L), Hold (H), and Short (S) actions
    to maximize (Mean Return / Std Dev) - (Switching Penalty).
    """
    actions = ['Long', 'Hold', 'Short']
    n = len(prices)
    
    # Calculate period-over-period returns
    # returns[i] is the gain from holding 'Long' between price[i] and price[i+1]
    price_returns = np.diff(prices)
    num_intervals = len(price_returns)
    
    best_score = -float('inf')
    best_sequence = None
    results_summary = {}

    # Iterate through all 3^n possible action sequences
    # Note: Action at index i determines the return from prices[i] to prices[i+1]
    for sequence in itertools.product(actions, repeat=num_intervals):
        current_returns = []
        switches = 0
        
        for i in range(num_intervals):
            action = sequence[i]
            
            # Define return multiplier based on action
            if action == 'Long':
                current_returns.append(price_returns[i])
            elif action == 'Short':
                current_returns.append(-price_returns[i])
            else: # Hold
                current_returns.append(0)
            
            # Count switches (penalty for changing strategy)
            if i > 0 and sequence[i] != sequence[i-1]:
                switches += 1
        
        # Calculate Metrics
        total_return = sum(current_returns)
        std_dev = np.std(current_returns) if len(current_returns) > 1 else 1.0
        
        # Avoid division by zero for standard deviation
        risk_adjusted_return = total_return / (std_dev + 1e-9)
        
        # Final Objective Function: Maximize (Return/Risk) - (Penalty * Switches)
        # We normalize switches by the max possible switches (n-1)
        normalized_switch_penalty = (switches / (num_intervals - 1)) if num_intervals > 1 else 0
        score = risk_adjusted_return - (switching_penalty_weight * normalized_switch_penalty)
        
        if score > best_score:
            best_score = score
            best_sequence = sequence
            results_summary = {
                "Score": score,
                "Total Return": total_return,
                "Std Dev": std_dev,
                "Switches": switches,
                "Sequence": sequence
            }

    return results_summary

# --- Execution ---
if __name__ == "__main__":
    # Example: 10 sample prices (p1 to p10)
    sample_prices = [100, 102, 101, 105, 107, 104, 103, 108, 110, 109]
    
    # Set the penalty weight (higher = less switching)
    penalty_weight = 0.5 
    
    result = solve_trading_sequence(sample_prices, switching_penalty_weight=penalty_weight)
    
    print("--- Optimal Trading Strategy ---")
    print(f"Prices: {sample_prices}")
    print(f"Optimal Sequence: {result['Sequence']}")
    print(f"Total Return: {result['Total Return']:.2f}")
    print(f"Volatility (Std Dev): {result['Std Dev']:.2f}")
    print(f"Number of Switches: {result['Switches']}")
    print(f"Final Combined Score: {result['Score']:.4f}")