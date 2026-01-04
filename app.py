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
    Handles overflow: 10099 -> 101
    Handles underflow: -99 -> -1, -100 -> -2
    """
    if price >= 0:
        return (int(price) // 100) + 1
    else:
        # For negative prices: -1 to -100 is bucket -1
        return (int(price + 1) // 100) - 1

def run_analysis():
    delayed_print("--- Starting Mock Price Analysis ---")
    
    # 1. Create 10,000 mock prices
    # Simulating a random walk to make sequences somewhat meaningful
    prices = [5000]
    for _ in range(9999):
        change = random.randint(-150, 150)
        prices.append(prices[-1] + change)
    
    # 2. Convert to percentiles
    percentiles = [get_percentile(p) for p in prices]
    
    # Split data 70/30
    split_idx = int(len(percentiles) * 0.7)
    train_data = percentiles[:split_idx]
    test_data = percentiles[split_idx:]
    
    delayed_print(f"Data generated. Train size: {len(train_data)}, Test size: {len(test_data)}")

    # 3 & 4. Sequence counting (Percentiles)
    # Map sequence (tuple of 5) -> Counter of successors
    seq_counts = defaultdict(Counter)
    for i in range(len(train_data) - 5):
        sequence = tuple(train_data[i:i+5])
        successor = train_data[i+5]
        seq_counts[sequence][successor] += 1
    
    delayed_print("Sequence mapping for percentiles completed.")

    # 5, 6 & 7. Predict and Compute Accuracy (Percentiles)
    correct_preds = 0
    total_preds = 0
    
    for i in range(len(test_data) - 5):
        sequence = tuple(test_data[i:i+5])
        actual = test_data[i+5]
        
        if sequence in seq_counts:
            # Predict the successor with the most counts
            prediction = seq_counts[sequence].most_common(1)[0][0]
            if prediction == actual:
                correct_preds += 1
            total_preds += 1
            
    perc_accuracy = (correct_preds / total_preds * 100) if total_preds > 0 else 0
    delayed_print(f"Percentile Prediction Accuracy: {perc_accuracy:.2f}% ({correct_preds}/{total_preds})")

    # 8. Derivative dataset
    # 1 2 3 4 3 -> 0 1 1 1 -1
    derivatives = [0]
    for i in range(1, len(percentiles)):
        derivatives.append(percentiles[i] - percentiles[i-1])
    
    train_deriv = derivatives[:split_idx]
    test_deriv = derivatives[split_idx:]
    
    delayed_print("Derivative dataset created.")

    # 9. Sequence counting (Derivatives)
    deriv_seq_counts = defaultdict(Counter)
    for i in range(len(train_deriv) - 5):
        sequence = tuple(train_deriv[i:i+5])
        successor = train_deriv[i+5]
        deriv_seq_counts[sequence][successor] += 1
        
    delayed_print("Sequence mapping for derivatives completed.")

    # 10 & 11. Predict and Compute Accuracy (Derivatives)
    d_correct_preds = 0
    d_total_preds = 0
    
    for i in range(len(test_deriv) - 5):
        sequence = tuple(test_deriv[i:i+5])
        actual = test_deriv[i+5]
        
        if sequence in deriv_seq_counts:
            prediction = deriv_seq_counts[sequence].most_common(1)[0][0]
            if prediction == actual:
                d_correct_preds += 1
            d_total_preds += 1
            
    deriv_accuracy = (d_correct_preds / d_total_preds * 100) if d_total_preds > 0 else 0
    
    # 12. Final Results
    delayed_print("\n--- FINAL RESULTS ---")
    delayed_print(f"Absolute Percentile Accuracy: {perc_accuracy:.2f}%")
    delayed_print(f"Derivative (Return) Accuracy: {deriv_accuracy:.2f}%")
    
    if deriv_accuracy > perc_accuracy:
        delayed_print("Conclusion: Derivative model performed better.")
    else:
        delayed_print("Conclusion: Absolute percentile model performed better.")

if __name__ == "__main__":
    run_analysis()