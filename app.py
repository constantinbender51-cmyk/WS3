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
START_DATE = "2020-01-01" 

def get_binance_data(symbol=SYMBOL, interval=INTERVAL, start_str=START_DATE):
    """Fetches historical kline data from Binance public API."""
    print(f"\n--- Fetching {symbol} {interval} data from Binance (Once) ---")
    
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    
    all_prices = []
    current_start = start_ts
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
    Returns the BEST Directional Accuracy among (Abs, Der, Comb).
    """
    
    # 1. Preprocess Data based on Bucket Size
    buckets = [get_bucket(p, bucket_size) for p in prices]
    
    split_idx = int(len(buckets) * 0.7)
    train_buckets = buckets[:split_idx]
    test_buckets = buckets[split_idx:]
    
    # Need sets for fallback random choices if pattern unseen
    all_train_values = list(set(train_buckets))
    all_train_changes = list(set(train_buckets[j] - train_buckets[j-1] for j in range(1, len(train_buckets))))

    # 2. Training
    abs_map = defaultdict(Counter)
    der_map = defaultdict(Counter)
    
    for i in range(len(train_buckets) - seq_len):
        # Absolute Pattern
        a_seq = tuple(train_buckets[i : i + seq_len])
        a_succ = train_buckets[i + seq_len]
        abs_map[a_seq][a_succ] += 1
        
        # Derivative Pattern
        if i > 0:
            d_seq = tuple(train_buckets[j] - train_buckets[j-1] for j in range(i, i + seq_len))
            d_succ = train_buckets[i + seq_len] - train_buckets[i + seq_len - 1]
            der_map[d_seq][d_succ] += 1

    # 3. Testing
    # Stats: [correct_count, valid_total_count]
    stats = {
        "Absolute": [0, 0],
        "Derivative": [0, 0],
        "Combined": [0, 0]
    }
    
    total_samples = len(test_buckets) - seq_len
    
    for i in range(total_samples):
        curr_idx = split_idx + i
        
        # Current Sequences
        a_seq = tuple(buckets[curr_idx : curr_idx + seq_len])
        d_seq = tuple(buckets[j] - buckets[j-1] for j in range(curr_idx, curr_idx + seq_len))
        
        last_val = a_seq[-1]
        actual_val = buckets[curr_idx + seq_len]
        actual_diff = actual_val - last_val

        # --- Predictions ---

        # 1. Absolute Model
        if a_seq in abs_map:
            pred_abs = abs_map[a_seq].most_common(1)[0][0]
        else:
            pred_abs = random.choice(all_train_values)

        # 2. Derivative Model
        if d_seq in der_map:
            pred_change = der_map[d_seq].most_common(1)[0][0]
        else:
            pred_change = random.choice(all_train_changes)
        pred_der = last_val + pred_change

        # 3. Combined Model
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

        # --- Evaluate Direction for all 3 ---
        # Helper to update stats
        def update_stat(name, prediction):
            pred_diff = prediction - last_val
            # Condition: Model predicts move AND Market actually moves
            if pred_diff != 0 and actual_diff != 0:
                stats[name][1] += 1 # Total Valid
                if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
                    stats[name][0] += 1 # Correct

        update_stat("Absolute", pred_abs)
        update_stat("Derivative", pred_der)
        update_stat("Combined", pred_comb)

    # 4. Find Winner
    best_acc = -1
    best_model = "None"
    best_trades = 0

    for name, (correct, total) in stats.items():
        if total > 0:
            acc = (correct / total) * 100
            if acc > best_acc:
                best_acc = acc
                best_model = name
                best_trades = total
    
    if best_acc == -1: return 0.0, "None", 0

    return best_acc, best_model, best_trades

def run_grid_search():
    # 1. Fetch Data
    prices = get_binance_data()
    if len(prices) < 500:
        print("Not enough data.")
        return

    # 2. Define Grid
    bucket_sizes = list(range(100, 9, -10)) # 100, 90 ... 10
    seq_lengths = [3, 4, 5, 6]
    
    results = []
    
    print(f"\n--- Starting Grid Search ({len(bucket_sizes) * len(seq_lengths)} combinations) ---")
    print(f"{'Bucket':<8} | {'SeqLen':<8} | {'Best Model':<12} | {'Dir Acc %':<10} | {'Trades':<8}")
    print("-" * 60)

    start_time = time.time()

    for b_size in bucket_sizes:
        for s_len in seq_lengths:
            accuracy, model_name, trades = evaluate_parameters(prices, b_size, s_len)
            
            # Filter for statistical significance (e.g., >50 valid trades)
            if trades > 50: 
                results.append({
                    "bucket": b_size,
                    "seq_len": s_len,
                    "model": model_name,
                    "accuracy": accuracy,
                    "trades": trades
                })
                print(f"{b_size:<8} | {s_len:<8} | {model_name:<12} | {accuracy:<10.2f} | {trades:<8}")
            else:
                 print(f"{b_size:<8} | {s_len:<8} | {'Too Few Data':<12} | {'N/A':<10} | {trades:<8}")

    # 3. Sort and Display Best
    print("\n" + "="*40)
    print(" TOP 10 CONFIGURATIONS (By Directional Acc) ")
    print("="*40)
    
    # Sort by Accuracy Descending
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    for rank, res in enumerate(results[:10], 1):
        print(f"#{rank}: Bucket {res['bucket']}, Len {res['seq_len']} -> {res['model']} Model: {res['accuracy']:.2f}% ({res['trades']} trades)")

    print(f"\nGrid search completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    try:
        run_grid_search()
    except KeyboardInterrupt:
        print("\nSearch interrupted.")