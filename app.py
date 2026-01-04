import random
import time
import sys
from collections import Counter, defaultdict

def delayed_print(text, delay=0.1):
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

def run_analysis():
    delayed_print("--- Starting Mock Price Analysis ---")
    
    # 1. Create 10,000 mock prices
    prices = [5000]
    for _ in range(9999):
        # Using a slightly volatile random walk
        change = random.randint(-150, 150)
        prices.append(prices[-1] + change)
    
    # 2. Convert to percentiles
    percentiles = [get_percentile(p) for p in prices]
    
    # Split data 70/30 (7000 train, 3000 test)
    split_idx = int(len(percentiles) * 0.7)
    train_data = percentiles[:split_idx]
    test_data = percentiles[split_idx:]
    
    unique_train_percentiles = list(set(train_data))
    num_test_sequences = len(test_data) - 5
    
    delayed_print(f"Data generated. Train size: {len(train_data)}")
    delayed_print(f"Test size: {len(test_data)} (Contains {num_test_sequences} sequences of 5)")

    # 3 & 4. Sequence counting (Percentiles)
    seq_counts = defaultdict(Counter)
    for i in range(len(train_data) - 5):
        sequence = tuple(train_data[i:i+5])
        successor = train_data[i+5]
        seq_counts[sequence][successor] += 1
    
    delayed_print("Sequence mapping for percentiles completed.")

    # 8. Derivative dataset (Assume first derivative is 0)
    derivatives = [0]
    for i in range(1, len(percentiles)):
        derivatives.append(percentiles[i] - percentiles[i-1])
    
    train_deriv = derivatives[:split_idx]
    test_deriv = derivatives[split_idx:]
    unique_train_deriv = list(set(train_deriv))
    
    # 9. Sequence counting (Derivatives)
    deriv_seq_counts = defaultdict(Counter)
    for i in range(len(train_deriv) - 5):
        sequence = tuple(train_deriv[i:i+5])
        successor = train_deriv[i+5]
        deriv_seq_counts[sequence][successor] += 1
        
    delayed_print("Sequence mapping for derivatives completed.")

    # Prediction Loops
    p_correct = 0
    d_correct = 0
    r_correct = 0
    c_correct = 0
    
    for i in range(num_test_sequences):
        # Actual values
        seq_p = tuple(test_data[i:i+5])
        act_p = test_data[i+5]
        
        seq_d = tuple(test_deriv[i:i+5])
        act_d = test_deriv[i+5]
        
        last_p = seq_p[-1]

        # 1. Random Benchmark
        if random.choice(unique_train_percentiles) == act_p:
            r_correct += 1

        # 2. Percentile Prediction
        if seq_p in seq_counts:
            if seq_counts[seq_p].most_common(1)[0][0] == act_p:
                p_correct += 1
        
        # 3. Derivative Prediction
        if seq_d in deriv_seq_counts:
            if deriv_seq_counts[seq_d].most_common(1)[0][0] == act_d:
                d_correct += 1
        
        # 4. Combined Prediction
        # We combine counts by translating derivative outcomes into percentile outcomes
        combined_votes = Counter()
        
        # Add absolute counts
        if seq_p in seq_counts:
            for val, count in seq_counts[seq_p].items():
                combined_votes[val] += count
        
        # Add derivative counts (mapped to percentiles)
        if seq_d in deriv_seq_counts:
            for d_val, count in deriv_seq_counts[seq_d].items():
                translated_p = last_p + d_val
                combined_votes[translated_p] += count
        
        if combined_votes:
            best_combined = combined_votes.most_common(1)[0][0]
            if best_combined == act_p:
                c_correct += 1
            
    # Calculate Accuracies
    acc_p = (p_correct / num_test_sequences) * 100
    acc_d = (d_correct / num_test_sequences) * 100
    acc_r = (r_correct / num_test_sequences) * 100
    acc_c = (c_correct / num_test_sequences) * 100
    
    # 12. Final Results
    delayed_print("\n--- FINAL RESULTS ---")
    delayed_print(f"Total Test Sequences: {num_test_sequences}")
    delayed_print(f"Random Benchmark Accuracy: {acc_r:.2f}%")
    delayed_print(f"Percentile Model Accuracy: {acc_p:.2f}%")
    delayed_print(f"Derivative Model Accuracy: {acc_d:.2f}%")
    delayed_print(f"Combined Model Accuracy:   {acc_c:.2f}%")
    
    if acc_c > max(acc_p, acc_d):
        delayed_print("Observation: The combined model improved prediction accuracy.")
    else:
        delayed_print("Observation: Combining models did not significantly outperform individual models.")

if __name__ == "__main__":
    run_analysis()