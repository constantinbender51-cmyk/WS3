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
    
    # Convert start date to milliseconds
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    
    all_prices = []
    current_start = start_ts
    
    # Binance limit is 1000 per request
    while current_start < end_ts:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={current_start}&limit=1000"
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                if not data:
                    break
                
                # Close price is at index 4
                batch_prices = [float(candle[4]) for candle in data]
                all_prices.extend(batch_prices)
                
                # Update start time to the open time of the next candle
                current_start = data[-1][6] + 1
                
                # Simple progress log
                date_str = datetime.fromtimestamp(current_start / 1000).strftime('%Y-%m')
                sys.stdout.write(f"\rCollected {len(all_prices)} candles. Currently at {date_str}...")
                sys.stdout.flush()
                
                # Small sleep to avoid rate limiting
                time.sleep(0.1)
        except Exception as e:
            delayed_print(f"\nError fetching data: {e}")
            break
            
    delayed_print(f"\nTotal data points fetched: {len(all_prices)}")
    return all_prices

def get_percentile(price):
    """Converts price to percentile buckets of 100 (e.g., 2450 -> 25)."""
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
    
    # Unique values from training for benchmarks and fallbacks
    all_train_values = list(set(train_perc))
    all_train_changes = list(set(train_perc[j] - train_perc[j-1] for j in range(1, len(train_perc))))
    
    # 2. Training: Build Global Frequency Maps
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

    # 3. Testing
    correct_abs = 0
    correct_der = 0
    correct_comb = 0
    correct_rand = 0 
    
    total_samples = len(test_perc) - 5
    delayed_print(f"Running analysis on {total_samples} test sequences...")

    for i in range(total_samples):
        curr_idx = split_idx + i
        
        a_seq = tuple(percentiles[curr_idx : curr_idx+5])
        d_seq = tuple(percentiles[j] - percentiles[j-1] for j in range(curr_idx, curr_idx+5))
        
        last_val = a_seq[-1]
        actual_val = percentiles[curr_idx+5]
        
        # --- Random Benchmark ---
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
        abs_candidates = abs_map.get(a_seq, Counter())
        der_candidates = der_map.get(d_seq, Counter())
        
        possible_next_vals = set(abs_candidates.keys())
        for change in der_candidates.keys():
            possible_next_vals.add(last_val + change)
            
        if not possible_next_vals:
            best_val = random.choice(all_train_values)
        else:
            best_val = None
            max_combined_score = -1
            for val in possible_next_vals:
                implied_change = val - last_val
                score = abs_candidates[val] + der_candidates[implied_change]
                if score > max_combined_score:
                    max_combined_score = score
                    best_val = val
        
        if best_val == actual_val:
            correct_comb += 1

    # 4. Reporting
    delayed_print("\n--- FINAL RESULTS (ETH/USDT Hourly) ---")
    delayed_print(f"Random Benchmark:          {correct_rand/total_samples*100:.2f}%")
    delayed_print(f"Absolute Model Accuracy:   {correct_abs/total_samples*100:.2f}%")
    delayed_print(f"Derivative Model Accuracy: {correct_der/total_samples*100:.2f}%")
    delayed_print(f"Combined Consensus Acc:    {correct_comb/total_samples*100:.2f}%")
    
    delayed_print("\nSummary:")
    delayed_print("Real-world crypto data is significantly noisier than a pure random walk.")
    delayed_print("The Combined Consensus model attempts to filter noise by finding overlap between price level and trend.")

if __name__ == "__main__":
    run_analysis()