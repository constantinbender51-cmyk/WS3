import random
import time
import sys
from collections import Counter, defaultdict

def delayed_print(text, delay=0.01):
    """Prints text with a specific delay to simulate a real-time feed."""
    print(text)
    sys.stdout.flush()
    time.sleep(delay)

def get_percentile(price):
    """Converts price to percentile buckets of 100."""
    if price >= 0:
        return (int(price) // 100) + 1
    else:
        return (int(price + 1) // 100) - 1

def run_analysis():
    delayed_print("--- Starting Consensus-Based Sequence Analysis ---")
    
    # 1. Generate Mock Data (Random Walk)
    prices = [5000]
    for _ in range(9999):
        prices.append(prices[-1] + random.randint(-150, 150))
    
    percentiles = [get_percentile(p) for p in prices]
    
    split_idx = int(len(percentiles) * 0.7)
    train_perc = percentiles[:split_idx]
    test_perc = percentiles[split_idx:]
    
    # 2. Training: Build Global Frequency Maps
    # abs_map: seq(5) -> Counter(successor)
    # der_map: seq(5) -> Counter(successor_change) 
    # Note: We use length 5 for both to match your requirement.
    abs_map = defaultdict(Counter)
    der_map = defaultdict(Counter)

    for i in range(split_idx - 5):
        # Absolute sequence of 5 and its successor
        a_seq = tuple(percentiles[i:i+5])
        a_succ = percentiles[i+5]
        abs_map[a_seq][a_succ] += 1
        
        # Derivative sequence of 5 (using true history) and its successor
        # To get a deriv seq of 5 ending at i+4, we need indices (i-1 to i+4)
        if i > 0:
            d_seq = tuple(percentiles[j] - percentiles[j-1] for j in range(i, i+5))
            d_succ = percentiles[i+5] - percentiles[i+4]
            der_map[d_seq][d_succ] += 1
    
    delayed_print("Training complete. Models populated.")

    # 3. Testing with Combined Scoring
    correct_abs = 0
    correct_der = 0
    correct_comb = 0
    total_samples = len(test_perc) - 5
    
    for i in range(total_samples):
        curr_idx = split_idx + i
        
        # 5-value Absolute Sequence
        a_seq = tuple(percentiles[curr_idx : curr_idx+5])
        # 5-value Derivative Sequence
        d_seq = tuple(percentiles[j] - percentiles[j-1] for j in range(curr_idx, curr_idx+5))
        
        last_val = a_seq[-1]
        actual_val = percentiles[curr_idx+5]
        
        # --- Model 1: Absolute Only (Baseline) ---
        if a_seq in abs_map:
            if abs_map[a_seq].most_common(1)[0][0] == actual_val:
                correct_abs += 1

        # --- Model 2: Derivative Only (Baseline) ---
        if d_seq in der_map:
            pred_change = der_map[d_seq].most_common(1)[0][0]
            if last_val + pred_change == actual_val:
                correct_der += 1

        # --- Model 3: Combined Consensus Model ---
        # 1. Get all candidate successors from the absolute map
        candidates = abs_map[a_seq] # Counter of {val: count}
        
        if candidates:
            best_val = None
            max_combined_score = -1
            
            for val, abs_count in candidates.items():
                # What change does this prediction imply?
                implied_change = val - last_val
                
                # How many times has the derivative expert seen this change after this d_seq?
                der_count = der_map[d_seq][implied_change]
                
                # Combined Score: Sum of counts
                combined_score = abs_count + der_count
                
                if combined_score > max_combined_score:
                    max_combined_score = combined_score
                    best_val = val
            
            if best_val == actual_val:
                correct_comb += 1

    # 4. Reporting
    delayed_print("\n--- FINAL RESULTS ---")
    delayed_print(f"Absolute Model Accuracy:   {correct_abs/total_samples*100:.2f}%")
    delayed_print(f"Derivative Model Accuracy: {correct_der/total_samples*100:.2f}%")
    delayed_print(f"Combined Consensus Acc:    {correct_comb/total_samples*100:.2f}%")
    
    delayed_print("\nLogic Explanation:")
    delayed_print("The Combined model checks absolute candidates and weights them")
    delayed_print("by how often the resulting momentum change has occurred in history.")

if __name__ == "__main__":
    run_analysis()