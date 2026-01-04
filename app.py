import urllib.request
import json
import time
import sys
import random
from collections import Counter, defaultdict
from datetime import datetime

# --- Configuration ---
ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", 
    "ADAUSDT", "DOGEUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT"
]
BUCKET_DIVISORS = list(range(10, 110, 10)) # [10, 20, ..., 100]
SEQ_LENGTHS = [3, 4, 5, 6]
START_DATE = "2021-01-01" # Using 2021 to ensure faster processing while keeping enough data

def delayed_print(text, delay=0.0):
    print(text)
    sys.stdout.flush()
    if delay > 0:
        time.sleep(delay)

def get_binance_data(symbol, interval="1h", start_str=START_DATE):
    """Fetches historical kline data from Binance."""
    delayed_print(f"[{symbol}] Fetching data...", delay=0.1)
    
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
                
                # Tiny sleep to avoid aggressive rate limiting
                time.sleep(0.05)
        except Exception as e:
            delayed_print(f"Error fetching {symbol}: {e}")
            return []
            
    return all_prices

def get_percentile(price, divisor):
    """Converts price to bucket based on dynamic divisor."""
    if price >= 0:
        return (int(price) // divisor) + 1
    else:
        return (int(price + 1) // divisor) - 1

def evaluate_absolute_model(prices, divisor, seq_len):
    """
    Runs the Absolute Model analysis for a specific configuration.
    Returns: (Directional Accuracy %, Valid Case Count)
    """
    # 1. Convert to buckets
    buckets = [get_percentile(p, divisor) for p in prices]
    
    # 2. Split Data
    split_idx = int(len(buckets) * 0.7)
    train_data = buckets[:split_idx]
    test_data = buckets[split_idx:]
    
    # 3. Train
    abs_map = defaultdict(Counter)
    # Fallback list for random choice if needed (though we only care about valid signals here)
    train_values = list(set(train_data))
    
    for i in range(len(train_data) - seq_len):
        seq = tuple(train_data[i : i+seq_len])
        successor = train_data[i+seq_len]
        abs_map[seq][successor] += 1
        
    # 4. Test (Directional Accuracy Only)
    dir_correct = 0
    dir_total = 0
    
    # Pre-calculate prices to avoid index lookups in loop
    # We need the original prices corresponding to the test buckets to determine actual direction
    # This is an approximation since we dropped the exact price mapping, 
    # but for "bucket direction" we can use the bucket values themselves.
    
    for i in range(len(test_data) - seq_len):
        seq = tuple(test_data[i : i+seq_len])
        last_val = seq[-1]
        actual_val = test_data[i+seq_len]
        
        # Prediction
        if seq in abs_map:
            prediction = abs_map[seq].most_common(1)[0][0]
            
            # Prediction Difference
            pred_diff = prediction - last_val
            actual_diff = actual_val - last_val
            
            # Check Actionable Signal (Prediction is not Flat)
            if pred_diff != 0:
                # Check Volatility (Market is not Flat)
                if actual_diff != 0:
                    dir_total += 1
                    # Check Direction Match
                    if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
                        dir_correct += 1
                        
    accuracy = (dir_correct / dir_total * 100) if dir_total > 0 else 0.0
    return accuracy, dir_total

def run_grid_search():
    delayed_print("--- STARTING CRYPTO GRID SEARCH ---")
    delayed_print(f"Assets: {len(ASSETS)}")
    delayed_print(f"Configs per Asset: {len(BUCKET_DIVISORS) * len(SEQ_LENGTHS)}")
    delayed_print("-----------------------------------")
    
    results = [] # List of tuples: (Asset, Divisor, Length, Accuracy, Count)
    
    for asset in ASSETS:
        prices = get_binance_data(asset)
        if not prices:
            continue
            
        delayed_print(f"[{asset}] Testing configurations...")
        
        for div in BUCKET_DIVISORS:
            for length in SEQ_LENGTHS:
                acc, count = evaluate_absolute_model(prices, div, length)
                
                # Filter out statistically insignificant results (e.g., less than 10 signals)
                if count >= 30: 
                    results.append({
                        "Asset": asset,
                        "Divisor": div,
                        "Length": length,
                        "Accuracy": acc,
                        "Signals": count
                    })
    
    # Sort by Accuracy Descending
    results.sort(key=lambda x: x["Accuracy"], reverse=True)
    
    delayed_print("\n\n--- TOP 20 CONFIGURATIONS (Ranked by Directional Accuracy) ---")
    delayed_print(f"{'RANK':<5} {'ASSET':<10} {'DIVISOR':<10} {'LENGTH':<8} {'ACCURACY':<10} {'SIGNALS':<10}")
    delayed_print("-" * 60)
    
    for i, res in enumerate(results[:20]):
        delayed_print(f"{i+1:<5} {res['Asset']:<10} {res['Divisor']:<10} {res['Length']:<8} {res['Accuracy']:.2f}%    {res['Signals']:<10}")

    # Also find the best generic settings (Average across all assets)
    delayed_print("\n--- BEST GENERIC SETTINGS (Avg Accuracy across assets) ---")
    
    config_scores = defaultdict(list)
    for res in results:
        key = (res['Divisor'], res['Length'])
        config_scores[key].append(res['Accuracy'])
        
    avg_results = []
    for (div, length), scores in config_scores.items():
        if len(scores) > 3: # Only consider configs that worked for multiple assets
            avg_acc = sum(scores) / len(scores)
            avg_results.append((div, length, avg_acc, len(scores)))
            
    avg_results.sort(key=lambda x: x[2], reverse=True)
    
    delayed_print(f"{'DIVISOR':<10} {'LENGTH':<8} {'AVG ACC':<10} {'ASSETS COVERED'}")
    for i, (div, length, acc, count) in enumerate(avg_results[:5]):
        delayed_print(f"{div:<10} {length:<8} {acc:.2f}%     {count}")

if __name__ == "__main__":
    run_grid_search()