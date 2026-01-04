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
    Original logic from step 2: 
    0-99 -> 1, 100-199 -> 2, etc.
    10099 -> 101
    -1 to -99 -> -1, -100 to -199 -> -2
    """
    if price >= 0:
        return (int(price) // 100) + 1
    else:
        # Reverting to exact requested underflow logic: -99 is -1, -100 is -2
        return (int(price) // 100)

def run_analysis():
    delayed_print("--- Starting Mock Price Analysis ---")
    
    # 1. Create 10,000 mock prices
    prices = [5000]
    random.seed(42) # Seeded for consistency in debugging
    for _ in range(9999):
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
    delayed_print(f"Test size: {len(test_data)} (Sequences: {num_test_sequences})")

    # 3 & 4. Sequence counting (Percentiles)
    seq_counts = defaultdict(Counter)
    for i in range(len(train_data) - 5):
        sequence = tuple(train_data[i:i+5])
        successor = train_data[i+5]
        seq_counts[sequence][successor] += 1
    
    # 8. Derivative dataset (derivative of 0 for the first element)
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

    # Scoring
    p_correct = 0
    d_correct = 0
    r_correct = 0
    c_correct = 0
    
    for i in range(num_test_sequences):
        # 5. Take first 5 steps and predict successor
        # Extract sequences
        target_seq_p = tuple(test_data[i:i+5])
        actual_p = test_data[i+5]
        
        target_seq_d = tuple(test_deriv[i:i+5])
        actual_d = test_deriv[i+5]
        
        # Benchmark
        if random.choice(unique_train_percentiles) == actual_p:
            r_correct += 1

        # Percentile Prediction (Most counts)
        if target_seq_p in seq_counts:
            prediction_p = seq_counts[target_seq_p].most_common(1)[0][0]
            if prediction_p == actual_p:
                p_correct += 1
        
        # Derivative Prediction
        if target_seq_d in deriv_seq_counts:
            prediction_d = deriv_seq_counts[target_seq_d].most_common(1)[0][0]
            if prediction_d == actual_d:
                d_correct += 1
        
        # Combined Prediction (Sum of counts)
        combined_votes = Counter()
        # Add absolute model counts
        if target_seq_p in seq_counts:
            for val, count in seq_counts[target_seq_p].items():
                combined_votes[val] += count
        
        # Add derivative model counts (mapped to absolute)
        # last_p + predicted_derivative
        last_p_in_seq = target_seq_p[-1]
        if target_seq_d in deriv_seq_counts:
            for deriv_val, count in deriv_seq_counts[target_seq_d].items():
                combined_votes[last_p_in_seq + deriv_val] += count
        
        if combined_votes:
            prediction_c = combined_votes.most_common(1)[0][0]
            if prediction_c == actual_p:
                c_correct += 1
            
    # Calculate Final Accuracies
    acc_p = (p_correct / num_test_sequences) * 100
    acc_d = (d_correct / num_test_sequences) * 100
    acc_r = (r_correct / num_test_sequences) * 100
    acc_c = (c_correct / num_test_sequences) * 100
    
    delayed_print("\n--- ACCURACY REPORT ---")
    delayed_print(f"1. Random Benchmark:     {acc_r:.2f}%")
    delayed_print(f"2. Percentile Model:     {acc_p:.2f}%")
    delayed_print(f"3. Derivative Model:     {acc_d:.2f}%")
    delayed_print(f"4. Combined Model:       {acc_c:.2f}%")

if __name__ == "__main__":
    run_analysis()