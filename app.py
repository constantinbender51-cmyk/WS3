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
    abs_map = defaultdict(Counter)
    der_map = defaultdict(Counter)

    for i in range(split_idx - 5):
        # Absolute sequence of 5 and its successor
        a_seq = tuple(percentiles[i:i+5])
        a_succ = percentiles[i+5]
        abs_map[a_seq][a_succ] += 1
        
        # Derivative sequence of 5 (using true history) and its successor
        if i > 0:
            d_seq = tuple(percentiles[j] - percentiles[j-1] for j in range(i, i+5))
            d_succ = percentiles[i+5] - percentiles[i+4]
            der_map[d_seq][d_succ] += 1
    
    delayed_print("Training complete. Models populated.")

    # 3. Testing with Combined Scoring
    correct_abs = 0
    correct_der = 0
    correct_comb = 0
    
    # Tracking counts where predictions were actually possible
    total_abs_possible = 0
    total_der_possible = 0
    total_comb_possible = 0
    
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
            results = abs_map[a_seq].most_common(1)
            if results: # Safety check for empty Counter
                total_abs_possible += 1
                if results[0][0] == actual_val:
                    correct_abs += 1

        # --- Model 2: Derivative Only (Baseline) ---
        if d_seq in der_map:
            results = der_map[d_seq].most_common(1)
            if results:
                total_der_possible += 1
                pred_change = results[0][0]
                if last_val + pred_change == actual_val:
                    correct_der += 1

        # --- Model 3: Combined Consensus Model ---
        candidates = abs_map.get(a_seq)
        
        if candidates:
            best_val = None
            max_combined_score = -1
            
            for val, abs_count in candidates.items():
                implied_change = val - last_val
                
                # Check how often this change occurred in the derivative expert's history
                der_count = der_map[d_seq][implied_change]
                
                # Scoring: Absolute freq + Derivative freq
                combined_score = abs_count + der_count
                
                if combined_score > max_combined_score:
                    max_combined_score = combined_score
                    best_val = val
            
            if best_val is not None:
                total_comb_possible += 1
                if best_val == actual_val:
                    correct_comb += 1

    # 4. Reporting
    delayed_print("\n--- FINAL RESULTS ---")
    
    def calc_acc(correct, possible):
        return (correct / possible * 100) if possible > 0 else 0

    delayed_print(f"Absolute Model Accuracy:   {calc_acc(correct_abs, total_abs_possible):.2f}% (Coverage: {total_abs_possible}/{total_samples})")
    delayed_print(f"Derivative Model Accuracy: {calc_acc(correct_der, total_der_possible):.2f}% (Coverage: {total_der_possible}/{total_samples})")
    delayed_print(f"Combined Consensus Acc:    {calc_acc(correct_comb, total_comb_possible):.2f}% (Coverage: {total_comb_possible}/{total_samples})")
    
    delayed_print("\nAnalysis Summary:")
    delayed_print("The combined model weights absolute candidates by their historical momentum frequency.")
    delayed_print("The IndexError was fixed by ensuring 'most_common' is not called on unknown sequences.")

if __name__ == "__main__":
    run_analysis()