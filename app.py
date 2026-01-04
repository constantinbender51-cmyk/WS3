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
    """
    Converts price to percentile buckets of 100.
    Logic: 0-99 -> 1, 100-199 -> 2, etc.
    """
    if price >= 0:
        return (int(price) // 100) + 1
    else:
        return (int(price + 1) // 100) - 1

def get_aligned_derivatives(data, start_idx, length):
    """
    Computes a derivative sequence of 'length' by looking at 
    data[i] - data[i-1].
    Handles the very first index of the dataset (index 0) by assuming 0.
    """
    derivs = []
    for i in range(start_idx, start_idx + length):
        if i == 0:
            derivs.append(0)
        else:
            derivs.append(data[i] - data[i-1])
    return tuple(derivs)

def run_analysis():
    delayed_print("--- Starting Corrected Derivative Alignment Analysis ---")
    
    # 1. Create mock prices (Random Walk)
    prices = [5000]
    for _ in range(9999):
        change = random.randint(-150, 150)
        prices.append(prices[-1] + change)
    
    # 2. Convert to percentiles
    percentiles = [get_percentile(p) for p in prices]
    
    # Global derivative list for the whole dataset
    global_derivatives = [0]
    for i in range(1, len(percentiles)):
        global_derivatives.append(percentiles[i] - percentiles[i-1])

    # Split data 70/30
    split_idx = int(len(percentiles) * 0.7)
    train_perc = percentiles[:split_idx]
    test_perc = percentiles[split_idx:]
    
    unique_train_perc = list(set(train_perc))
    
    delayed_print(f"Data generated. Train size: {len(train_perc)}, Test size: {len(test_perc)}")

    # 3. Training: Build Sequence Counts
    abs_counts = defaultdict(Counter)
    deriv_counts = defaultdict(Counter)
    combined_counts = defaultdict(Counter)

    # We iterate through the training set to build the model
    # Note: We need to use 'percentiles' (the full list) to get the correct 
    # derivatives for the start of the window if i > 0.
    for i in range(split_idx - 5):
        abs_seq = tuple(percentiles[i:i+5])
        # Look up true derivatives using history
        der_seq = get_aligned_derivatives(percentiles, i, 5)
        
        successor_perc = percentiles[i+5]
        successor_der = global_derivatives[i+5]
        
        abs_counts[abs_seq][successor_perc] += 1
        deriv_counts[der_seq][successor_der] += 1
        combined_counts[(abs_seq, der_seq)][successor_perc] += 1
    
    delayed_print("Training complete: Captured historical derivatives for all sequences.")

    # 4. Prediction Loop (Test Set)
    correct_abs = 0
    correct_deriv = 0
    correct_combined = 0
    correct_rand = 0
    
    totals = {"abs": 0, "deriv": 0, "combined": 0}
    test_len = len(test_perc) - 5
    
    for i in range(test_len):
        # Current index in the context of the full 'percentiles' list
        current_full_idx = split_idx + i
        
        abs_seq = tuple(percentiles[current_full_idx : current_full_idx + 5])
        der_seq = get_aligned_derivatives(percentiles, current_full_idx, 5)
        
        actual_perc = percentiles[current_full_idx + 5]
        actual_der = global_derivatives[current_full_idx + 5]

        # Random Benchmark
        if random.choice(unique_train_perc) == actual_perc:
            correct_rand += 1

        # Absolute Prediction
        if abs_seq in abs_counts:
            if abs_counts[abs_seq].most_common(1)[0][0] == actual_perc:
                correct_abs += 1
            totals["abs"] += 1

        # Derivative Prediction (Predicting the next change)
        if der_seq in deriv_counts:
            if deriv_counts[der_seq].most_common(1)[0][0] == actual_der:
                correct_deriv += 1
            totals["deriv"] += 1

        # Combined Prediction (Highest count of the joint state)
        if (abs_seq, der_seq) in combined_counts:
            if combined_counts[(abs_seq, der_seq)].most_common(1)[0][0] == actual_perc:
                correct_combined += 1
            totals["combined"] += 1
            
    # 5. Final Calculations
    acc_abs = (correct_abs / totals["abs"] * 100) if totals["abs"] > 0 else 0
    acc_deriv = (correct_deriv / totals["deriv"] * 100) if totals["deriv"] > 0 else 0
    acc_comb = (correct_combined / totals["combined"] * 100) if totals["combined"] > 0 else 0
    acc_rand = (correct_rand / test_len * 100)
    
    delayed_print("\n--- FINAL RESULTS ---")
    delayed_print(f"Random Benchmark:      {acc_rand:.2f}%")
    delayed_print(f"Absolute Model:        {acc_abs:.2f}% (Coverage: {totals['abs']}/{test_len})")
    delayed_print(f"Derivative Model:      {acc_deriv:.2f}% (Coverage: {totals['deriv']}/{test_len})")
    delayed_print(f"Combined Model:        {acc_comb:.2f}% (Coverage: {totals['combined']}/{test_len})")
    
    delayed_print("\n--- Summary ---")
    delayed_print(f"The combined model captures both the price level (absolute) and the momentum (derivative).")
    if acc_comb > acc_abs:
        delayed_print(f"Result: The joint model improved accuracy by {acc_comb - acc_abs:.2f}% over the absolute model.")
    else:
        delayed_print("Result: The absolute state alone was as predictive as the joint state.")

if __name__ == "__main__":
    run_analysis()