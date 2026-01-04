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
    delayed_print("--- Starting Full-Coverage Consensus Analysis with Benchmarks ---")
    
    # 1. Generate Mock Data (Random Walk)
    prices = [5000]
    for _ in range(9999):
        prices.append(prices[-1] + random.randint(-150, 150))
    
    percentiles = [get_percentile(p) for p in prices]
    
    split_idx = int(len(percentiles) * 0.7)
    train_perc = percentiles[:split_idx]
    test_perc = percentiles[split_idx:]
    
    # Unique values from training to allow predictions and benchmarks
    all_train_values = list(set(train_perc))
    all_train_changes = list(set(train_perc[j] - train_perc[j-1] for j in range(1, len(train_perc))))
    
    # 2. Training: Build Global Frequency Maps
    abs_map = defaultdict(Counter)
    der_map = defaultdict(Counter)

    for i in range(split_idx - 5):
        a_seq = tuple(percentiles[i:i+5])
        a_succ = percentiles[i+5]
        abs_map[a_seq][a_succ] += 1
        
        if i > 0:
            d_seq = tuple(percentiles[j] - percentiles[j-1] for j in range(i, i+5))
            d_succ = percentiles[i+5] - percentiles[i+4]
            der_map[d_seq][d_succ] += 1
    
    delayed_print("Training complete. Models populated.")

    # 3. Testing with Combined Scoring and Random Benchmark
    correct_abs = 0
    correct_der = 0
    correct_comb = 0
    correct_rand = 0 # Random Benchmark counter
    
    total_samples = len(test_perc) - 5
    
    for i in range(total_samples):
        curr_idx = split_idx + i
        
        a_seq = tuple(percentiles[curr_idx : curr_idx+5])
        d_seq = tuple(percentiles[j] - percentiles[j-1] for j in range(curr_idx, curr_idx+5))
        
        last_val = a_seq[-1]
        actual_val = percentiles[curr_idx+5]
        
        # --- Random Benchmark ---
        # Predict any random percentile that existed in training
        if random.choice(all_train_values) == actual_val:
            correct_rand += 1

        # --- Model 1: Absolute Only ---
        if a_seq in abs_map:
            pred_abs = abs_map[a_seq].most_common(1)[0][0]
        else:
            pred_abs = random.choice(all_train_values)
            
        if pred_abs == actual_val:
            correct_abs += 1

        # --- Model 2: Derivative Only ---
        if d_seq in der_map:
            pred_change = der_map[d_seq].most_common(1)[0][0]
        else:
            pred_change = random.choice(all_train_changes)
            
        if last_val + pred_change == actual_val:
            correct_der += 1

        # --- Model 3: Combined Consensus Model ---
        best_val = None
        max_combined_score = -1
        
        abs_candidates = abs_map.get(a_seq, Counter())
        der_candidates = der_map.get(d_seq, Counter())
        
        possible_next_vals = set(abs_candidates.keys())
        for change in der_candidates.keys():
            possible_next_vals.add(last_val + change)
            
        if not possible_next_vals:
            best_val = random.choice(all_train_values)
        else:
            for val in possible_next_vals:
                implied_change = val - last_val
                a_count = abs_candidates[val]
                d_count = der_candidates[implied_change]
                
                score = a_count + d_count
                if score > max_combined_score:
                    max_combined_score = score
                    best_val = val
        
        if best_val == actual_val:
            correct_comb += 1

    # 4. Reporting
    delayed_print("\n--- FINAL RESULTS ---")
    delayed_print(f"Random Benchmark:          {correct_rand/total_samples*100:.2f}%")
    delayed_print(f"Absolute Model Accuracy:   {correct_abs/total_samples*100:.2f}%")
    delayed_print(f"Derivative Model Accuracy: {correct_der/total_samples*100:.2f}%")
    delayed_print(f"Combined Consensus Acc:    {correct_comb/total_samples*100:.2f}%")
    
    delayed_print("\nAnalysis Summary:")
    delayed_print("The Random Benchmark shows the expected accuracy of guessing based purely on historical distribution.")
    delayed_print("The Combined Consensus model weights potential outcomes by their frequency in both price level and momentum history.")

if __name__ == "__main__":
    run_analysis()