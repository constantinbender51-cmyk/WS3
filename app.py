import random
import time
import sys
import json
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime

# --- CONFIGURATION ---
SYMBOL = "ETHUSDT"
INTERVAL = "1h"
START_DATE = "2020-01-01" # Shortened slightly for faster grid search, adjust as needed

def delayed_print(text, delay=0.0):
    """Prints text. Delay removed for grid search speed, but kept signature."""
    print(text)
    sys.stdout.flush()

def get_binance_data(symbol=SYMBOL, interval=INTERVAL, start_str=START_DATE):
    """Fetches historical kline data from Binance public API."""
    print(f"\n--- Fetching {symbol} {interval} data from Binance (Once) ---")
    
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    
    all_prices = []
    current_start = start_ts
    
    # Simple loading bar vars
    total_time = end_ts - start_ts
    
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
                
                # Progress indicator
                progress = (current_start - start_ts) / total_time
                sys.stdout.write(f"\rDownload Progress: [{int(progress*20)*'#'}{(20-int(progress*20))*'-'}] {len(all_prices)} candles")
                sys.stdout.flush()
                
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
            
    print(f"\nTotal data points fetched: {len(all_prices)}")
    return all_prices

def get_bucket(price, bucket_size):
    """Converts price to variable bucket size."""
    if price >= 0:
        return (int(price) // bucket_size) + 1
    else:
        return (int(price + 1) // bucket_size) - 1

def evaluate_parameters(prices, bucket_size, seq_len):
    """
    Runs the analysis for specific parameters.
    Returns the Directional Accuracy of the 'Combined' model.
    """
    
    # 1. Preprocess Data based on Bucket Size
    buckets = [get_bucket(p, bucket_size) for p in prices]
    
    split_idx = int(len(buckets) * 0.7)
    train_buckets = buckets[:split_idx]
    test_buckets = buckets[split_idx:]
    
    # 2. Training
    abs_map = defaultdict(Counter)
    der_map = defaultdict(Counter)
    
    # Train on history
    for i in range(len(train_buckets) - seq_len):
        # Absolute Pattern
        a_seq = tuple(train_buckets[i : i + seq_len])
        a_succ = train_buckets[i + seq_len]
        abs_map[a_seq][a_succ] += 1
        
        # Derivative Pattern (requires at least 1 previous point relative to start of seq)
        if i > 0:
            d_seq = tuple(train_buckets[j] - train_buckets[j-1] for j in range(i, i + seq_len))
            d_succ = train_buckets[i + seq_len] - train_buckets[i + seq_len - 1]
            der_map[d_seq][d_succ] += 1

    # 3. Testing (Focusing on Combined Model Directional Accuracy)
    dir_correct = 0
    dir_total = 0 # Valid cases where market actually moved
    
    total_samples = len(test_buckets) - seq_len
    
    for i in range(total_samples):
        curr_idx = split_idx + i
        
        # Build sequences from test data
        a_seq = tuple(buckets[curr_idx : curr_idx + seq_len])
        d_seq = tuple(buckets[j] - buckets[j-1] for j in range(curr_idx, curr_idx + seq_len))
        
        last_val = a_seq[-1]
        actual_val = buckets[curr_idx + seq_len]
        
        # --- Combined Consensus Logic ---
        abs_candidates = abs_map.get(a_seq, Counter())
        der_candidates = der_map.get(d_seq, Counter())
        
        possible_next_vals = set(abs_candidates.keys())
        for change in der_candidates.keys():
            possible_next_vals.add(last_val + change)
            
        pred_comb = last_val # Default to no move
        
        if possible_next_vals:
            best_val = None
            max_combined_score = -1
            for val in possible_next_vals:
                implied_change = val - last_val
                # Simple weighting: 1:1
                score = abs_candidates[val] + der_candidates[implied_change]
                if score > max_combined_score:
                    max_combined_score = score
                    best_val = val
            pred_comb = best_val

        # --- Calculate Directional Accuracy ---
        pred_diff = pred_comb - last_val
        actual_diff = actual_val - last_val
        
        # Only count if Model predicted a move AND Market actually moved
        if pred_diff != 0 and actual_diff != 0:
            dir_total += 1
            if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
                dir_correct += 1
                
    if dir_total == 0:
        return 0.0, 0
    
    return (dir_correct / dir_total * 100), dir_total

def run_grid_search():
    # 1. Fetch Data Once
    prices = get_binance_data()
    if len(prices) < 500:
        print("Not enough data.")
        return

    # 2. Define Grid
    bucket_sizes = list(range(100, 9, -10)) # 100, 90, ... 10
    seq_lengths = [3, 4, 5, 6]
    
    results = []
    
    print(f"\n--- Starting Grid Search ({len(bucket_sizes) * len(seq_lengths)} combinations) ---")
    print(f"{'Bucket':<8} | {'SeqLen':<8} | {'Dir Acc %':<10} | {'Valid Trades':<12}")
    print("-" * 45)

    start_time = time.time()

    for b_size in bucket_sizes:
        for s_len in seq_lengths:
            accuracy, sample_size = evaluate_parameters(prices, b_size, s_len)
            
            # Filter out insignificant sample sizes (optional, prevents 100% acc on 1 trade)
            if sample_size > 50: 
                results.append({
                    "bucket": b_size,
                    "seq_len": s_len,
                    "accuracy": accuracy,
                    "trades": sample_size
                })
                print(f"{b_size:<8} | {s_len:<8} | {accuracy:<10.2f} | {sample_size:<12}")
            else:
                 print(f"{b_size:<8} | {s_len:<8} | {'Low Data':<10} | {sample_size:<12}")

    # 3. Sort and Display Best
    print("\n" + "="*30)
    print(" TOP 5 CONFIGURATIONS ")
    print("="*30)
    
    # Sort by Accuracy Descending
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    for rank, res in enumerate(results[:5], 1):
        print(f"#{rank}: Bucket {res['bucket']}, Len {res['seq_len']} -> {res['accuracy']:.2f}% ({res['trades']} trades)")

    print(f"\nGrid search completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    try:
        run_grid_search()
    except KeyboardInterrupt:
        print("\nSearch interrupted.")