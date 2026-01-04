import random
import time
import sys
import json
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime

def delayed_print(text, delay=0.1):
    """Prints text with a specific delay to simulate a real-time feed."""
    print(text)
    sys.stdout.flush()
    time.sleep(delay)

def get_binance_data(symbol="ETHUSDT", interval="1h", start_str="2018-01-01"):
    """Fetches historical kline data from Binance public API."""
    delayed_print(f"--- Fetching {symbol} {interval} data from Binance ---")
    
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    
    all_prices = []
    current_start = start_ts
    
    while current_start < end_ts:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={current_start}&limit=1000"
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                if not data:
                    break
                
                batch_prices = [float(candle[4]) for candle in data]
                all_prices.extend(batch_prices)
                current_start = data[-1][6] + 1
                
                date_str = datetime.fromtimestamp(current_start / 1000).strftime('%Y-%m')
                sys.stdout.write(f"\rCollected {len(all_prices)} candles. Currently at {date_str}...")
                sys.stdout.flush()
                time.sleep(0.05) 
        except Exception as e:
            delayed_print(f"\nError fetching data: {e}")
            break
            
    delayed_print(f"\nTotal data points fetched: {len(all_prices)}")
    return all_prices

def get_percentile(price):
    """Converts price to percentile buckets of 100."""
    if price >= 0:
        return (int(price) // 100) + 1
    else:
        return (int(price + 1) // 100) - 1

def run_analysis():
    # 1. Get Real Data
    prices = get_binance_data()
    if len(prices) < 100:
        delayed_print("Not enough data to run analysis.")
        return

    percentiles = [get_percentile(p) for p in prices]
    
    split_idx = int(len(percentiles) * 0.7)
    train_perc = percentiles[:split_idx]
    test_perc = percentiles[split_idx:]
    
    all_train_values = list(set(train_perc))
    all_train_changes = list(set(train_perc[j] - train_perc[j-1] for j in range(1, len(train_perc))))
    
    # 2. Training
    abs_map = defaultdict(Counter)
    der_map = defaultdict(Counter)

    delayed_print("Training models on historical patterns...")
    for i in range(split_idx - 5):
        a_seq = tuple(percentiles[i:i+5])
        a_succ = percentiles[i+5]
        abs_map[a_seq][a_succ] += 1
        
        if i > 0:
            d_seq = tuple(percentiles[j] - percentiles[j-1] for j in range(i, i+5))
            d_succ = percentiles[i+5] - percentiles[i+4]
            der_map[d_seq][d_succ] += 1
    
    delayed_print("Training complete.")

    # 3. Testing with "Actionable" Filter
    # "Standard" metrics (counting everything)
    std_correct = {"abs": 0, "der": 0, "comb": 0, "rand": 0, "coin": 0}
    
    # "Actionable" metrics (Prediction != Last Val, AND Exact Match)
    act_correct = {"abs": 0, "der": 0, "comb": 0, "rand": 0, "coin": 0}
    act_total   = {"abs": 0, "der": 0, "comb": 0, "rand": 0, "coin": 0}

    # "Directional" metrics (Prediction != Last Val, AND Sign Match)
    dir_correct = {"abs": 0, "der": 0, "comb": 0, "rand": 0, "coin": 0}
    dir_total   = {"abs": 0, "der": 0, "comb": 0, "rand": 0, "coin": 0}
    
    total_samples = len(test_perc) - 5
    delayed_print(f"Running analysis on {total_samples} test sequences...")

    for i in range(total_samples):
        curr_idx = split_idx + i
        
        a_seq = tuple(percentiles[curr_idx : curr_idx+5])
        d_seq = tuple(percentiles[j] - percentiles[j-1] for j in range(curr_idx, curr_idx+5))
        
        last_val = a_seq[-1]
        actual_val = percentiles[curr_idx+5]
        
        # Helper to process prediction stats
        def process_pred(model_name, prediction):
            # 1. Standard Accuracy
            if prediction == actual_val:
                std_correct[model_name] += 1
            
            # Prediction Difference vs Actual Difference
            pred_diff = prediction - last_val
            actual_diff = actual_val - last_val

            # Only process "Actionable/Directional" if the model PREDICTS a move
            if pred_diff != 0:
                act_total[model_name] += 1
                dir_total[model_name] += 1
                
                # 2. Actionable Precision (Exact bucket match)
                if prediction == actual_val:
                    act_correct[model_name] += 1
                
                # 3. Directional Accuracy (Sign match)
                # Correct if both > 0 OR both < 0
                if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
                    dir_correct[model_name] += 1

        # --- Benchmark 1: Global Random ---
        rand_pred = random.choice(all_train_values)
        process_pred("rand", rand_pred)

        # --- Benchmark 2: 3-Sided Coin Flip (Local Random) ---
        # Randomly choose -1 (Down), 0 (Same), or +1 (Up)
        coin_move = random.choice([-1, 0, 1])
        coin_pred = last_val + coin_move
        process_pred("coin", coin_pred)

        # --- Model 1: Absolute Only ---
        if a_seq in abs_map:
            pred_abs = abs_map[a_seq].most_common(1)[0][0]
        else:
            pred_abs = random.choice(all_train_values)
        process_pred("abs", pred_abs)

        # --- Model 2: Derivative Only ---
        if d_seq in der_map:
            pred_change = der_map[d_seq].most_common(1)[0][0]
        else:
            pred_change = random.choice(all_train_changes)
        process_pred("der", last_val + pred_change)

        # --- Model 3: Combined Consensus ---
        abs_candidates = abs_map.get(a_seq, Counter())
        der_candidates = der_map.get(d_seq, Counter())
        
        possible_next_vals = set(abs_candidates.keys())
        for change in der_candidates.keys():
            possible_next_vals.add(last_val + change)
            
        if not possible_next_vals:
            pred_comb = random.choice(all_train_values)
        else:
            best_val = None
            max_combined_score = -1
            for val in possible_next_vals:
                implied_change = val - last_val
                score = abs_candidates[val] + der_candidates[implied_change]
                if score > max_combined_score:
                    max_combined_score = score
                    best_val = val
            pred_comb = best_val

        process_pred("comb", pred_comb)

    # 4. Reporting
    def calc_perc(correct, total):
        return (correct / total * 100) if total > 0 else 0

    delayed_print("\n--- STANDARD ACCURACY (Includes Inertia) ---")
    delayed_print(f"Global Random: {calc_perc(std_correct['rand'], total_samples):.2f}%")
    delayed_print(f"3-Sided Coin:  {calc_perc(std_correct['coin'], total_samples):.2f}%")
    delayed_print(f"Absolute:      {calc_perc(std_correct['abs'], total_samples):.2f}%")
    delayed_print(f"Derivative:    {calc_perc(std_correct['der'], total_samples):.2f}%")
    delayed_print(f"Combined:      {calc_perc(std_correct['comb'], total_samples):.2f}%")

    delayed_print("\n--- MOVEMENT PRECISION (Exact Bucket Match) ---")
    delayed_print(f"Global Random: {calc_perc(act_correct['rand'], act_total['rand']):.2f}%  (Attempts: {act_total['rand']})")
    delayed_print(f"3-Sided Coin:  {calc_perc(act_correct['coin'], act_total['coin']):.2f}%  (Attempts: {act_total['coin']})")
    delayed_print(f"Absolute:      {calc_perc(act_correct['abs'], act_total['abs']):.2f}%  (Attempts: {act_total['abs']})")
    delayed_print(f"Derivative:    {calc_perc(act_correct['der'], act_total['der']):.2f}%  (Attempts: {act_total['der']})")
    delayed_print(f"Combined:      {calc_perc(act_correct['comb'], act_total['comb']):.2f}%  (Attempts: {act_total['comb']})")

    delayed_print("\n--- DIRECTIONAL ACCURACY (Up/Down Correctness) ---")
    delayed_print("Accuracy = Correct Direction / Total Direction Predictions")
    delayed_print(f"Global Random: {calc_perc(dir_correct['rand'], dir_total['rand']):.2f}%")
    delayed_print(f"3-Sided Coin:  {calc_perc(dir_correct['coin'], dir_total['coin']):.2f}%")
    delayed_print(f"Absolute:      {calc_perc(dir_correct['abs'], dir_total['abs']):.2f}%")
    delayed_print(f"Derivative:    {calc_perc(dir_correct['der'], dir_total['der']):.2f}%")
    delayed_print(f"Combined:      {calc_perc(dir_correct['comb'], dir_total['comb']):.2f}%")

if __name__ == "__main__":
    run_analysis()