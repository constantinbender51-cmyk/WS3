import os
import sys
import json
import time
import base64
import requests
import urllib.request
from datetime import datetime
from collections import Counter, defaultdict
import pandas as pd

# --- CONFIGURATION ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GITHUB_PAT = os.getenv("PAT")
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/"

# --- UPDATED ASSETS LIST (Kraken-Safe) ---
ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", 
    "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "TRXUSDT",
    "BCHUSDT", "XLMUSDT", "LTCUSDT", "SUIUSDT", "HBARUSDT",
    "SHIBUSDT", "TONUSDT", "UNIUSDT", "ZECUSDT"
]

BASE_INTERVAL = "15m" 
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"
DATA_DIR = "data"

TIMEFRAMES = {
    "15m": None, 
    "30m": "30min",
    "60m": "1h",
    "240m": "4h",
    "1d": "1D"
}

# --- 1. DATA FETCHING ---

def get_binance_data(symbol, start_str=START_DATE, end_str=END_DATE):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    filename = f"{DATA_DIR}/{symbol}_{BASE_INTERVAL}_{start_str}_{end_str}.json"

    if os.path.exists(filename):
        print(f"\n[{symbol}] Found cached data at {filename}. Loading...")
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}. Redownloading...")

    print(f"\n[{symbol}] Fetching raw {BASE_INTERVAL} data...")
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_str, "%Y-%m-%d").timestamp() * 1000)
    
    all_candles = []
    current_start = start_ts
    
    while current_start < end_ts:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={BASE_INTERVAL}&startTime={current_start}&endTime={end_ts}&limit=1000"
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                if not data: break
                batch = [(int(c[6]), float(c[4])) for c in data]
                all_candles.extend(batch)
                last_time = data[-1][6]
                if last_time >= end_ts - 1: break
                current_start = last_time + 1
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    if all_candles:
        with open(filename, 'w') as f:
            json.dump(all_candles, f)
    return all_candles

def resample_prices(raw_data, target_freq):
    if target_freq is None:
        return [x[1] for x in raw_data]
    df = pd.DataFrame(raw_data, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    resampled = df['price'].resample(target_freq).last().dropna()
    return resampled.tolist()

# --- 2. CORE STRATEGY ---

def get_bucket(price, bucket_size):
    # Math safety only: prevents division by zero. Not a strategy fallback.
    if bucket_size <= 0: bucket_size = 1e-9 
    return int(price // bucket_size)

def calculate_bucket_size(prices, bucket_count):
    min_p, max_p = min(prices), max(prices)
    price_range = max_p - min_p
    if bucket_count <= 0: return 1.0
    size = price_range / bucket_count
    # Math safety: ensure size is never absolute zero to prevent crash
    return size if size > 0 else 1e-9

def train_models(train_buckets, seq_len):
    abs_map = defaultdict(Counter)
    der_map = defaultdict(Counter)
    
    for i in range(len(train_buckets) - seq_len):
        a_seq = tuple(train_buckets[i : i + seq_len])
        a_succ = train_buckets[i + seq_len]
        abs_map[a_seq][a_succ] += 1
        
        if seq_len > 1:
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
            d_succ = train_buckets[i + seq_len] - train_buckets[i + seq_len - 1]
            der_map[d_seq][d_succ] += 1
            
    return abs_map, der_map

def serialize_map(model_map):
    serialized = {}
    for seq_tuple, counter_obj in model_map.items():
        key_str = "|".join(map(str, seq_tuple))
        serialized[key_str] = dict(counter_obj)
    return serialized

def prepare_deployment_models(prices, top_configs):
    serialized_models = []
    for config in top_configs:
        b_count = config['bucket_count']
        b_size = calculate_bucket_size(prices, b_count)
        s_len = config['seq_len']
        buckets = [get_bucket(p, b_size) for p in prices]
        abs_map, der_map = train_models(buckets, s_len)
        
        model_entry = {
            "config": config,
            "trained_parameters": {
                "bucket_size": b_size, 
                "seq_len": s_len,
                "abs_map": serialize_map(abs_map),
                "der_map": serialize_map(der_map),
                # We can remove 'all_vals' and 'all_changes' since we don't fallback anymore
                "all_vals": [], 
                "all_changes": []
            }
        }
        serialized_models.append(model_entry)
    return serialized_models

# --- PREDICTION LOGIC (NO FALLBACKS) ---

def get_prediction(model_type, abs_map, der_map, a_seq, d_seq, last_val):
    """
    Returns the predicted next VALUE.
    Returns None if the pattern is unknown (ABSTAIN).
    """
    if model_type == "Absolute":
        # If sequence unseen, return None
        if a_seq not in abs_map: 
            return None 
        return abs_map[a_seq].most_common(1)[0][0]
    
    elif model_type == "Derivative":
        # If sequence unseen, return None
        if d_seq not in der_map: 
            return None
        pred_change = der_map[d_seq].most_common(1)[0][0]
        return last_val + pred_change
    
    elif model_type == "Combined":
        # If either map misses the sequence, we might still have partial info,
        # BUT for "Combined" we usually want strong signals. 
        # Strategy: Use whatever is available, but if overlap is empty, return None.
        
        abs_cand = abs_map.get(a_seq, Counter())
        der_cand = der_map.get(d_seq, Counter())
        
        # If both are empty (unseen in both), abstain
        if not abs_cand and not der_cand:
            return None
            
        poss = set(abs_cand.keys())
        # Convert derivative keys to absolute values relative to last_val
        der_poss = {last_val + c for c in der_cand.keys()}
        
        # We only consider values supported by BOTH if possible
        # Or you can take the union. Here we take Union but score them.
        all_candidates = poss.union(der_poss)
        
        if not all_candidates:
            return None
        
        best, max_s = None, -1
        for v in all_candidates:
            # Score = freq in Absolute + freq in Derivative (if applicable)
            s = abs_cand[v] + der_cand[v - last_val]
            if s > max_s: max_s, best = s, v
            
        return best
        
    return None

def evaluate_config(prices, bucket_size, seq_len):
    buckets = [get_bucket(p, bucket_size) for p in prices]
    if len(buckets) < seq_len + 10: return -1, "None", 0
    
    abs_map, der_map = train_models(buckets, seq_len)
    
    stats = {"Absolute": [0,0], "Derivative": [0,0], "Combined": [0,0]}
    
    total_samples = len(buckets) - seq_len
    
    for i in range(total_samples):
        curr = i 
        a_seq = tuple(buckets[curr : curr+seq_len])
        d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq))) if seq_len > 1 else ()
            
        last = a_seq[-1]
        act = buckets[curr+seq_len]
        act_diff = act - last
        
        # Get predictions (can be None)
        p_abs = get_prediction("Absolute", abs_map, der_map, a_seq, d_seq, last)
        p_der = get_prediction("Derivative", abs_map, der_map, a_seq, d_seq, last)
        p_comb = get_prediction("Combined", abs_map, der_map, a_seq, d_seq, last)
        
        for name, pred in [("Absolute", p_abs), ("Derivative", p_der), ("Combined", p_comb)]:
            if pred is None: 
                continue # ABSTAIN: Pattern never seen, skip trade
                
            p_diff = pred - last
            
            if p_diff != 0 and act_diff != 0:
                stats[name][1] += 1
                if (p_diff > 0 and act_diff > 0) or (p_diff < 0 and act_diff < 0):
                    stats[name][0] += 1
                    
    best_acc, best_name, best_trades = -1, "None", 0
    for name, (corr, tot) in stats.items():
        if tot > 0:
            acc = (corr/tot)*100
            if acc > best_acc: best_acc, best_name, best_trades = acc, name, tot
            
    return best_acc, best_name, best_trades

def run_portfolio_analysis(prices, top_configs):
    models = []
    for config in top_configs:
        b_count = config['bucket_count']
        b_size = calculate_bucket_size(prices, b_count)
        s_len = config['seq_len']
        buckets = [get_bucket(p, b_size) for p in prices]
        abs_map, der_map = train_models(buckets, s_len)
        models.append({
            "config": config, "buckets": buckets, "seq_len": s_len,
            "abs_map": abs_map, "der_map": der_map
        })

    if not models: return 0, 0
    
    max_seq_len = max(m['seq_len'] for m in models)
    total_test_len = len(models[0]['buckets']) - max_seq_len
    
    unique_correct = 0
    unique_total = 0
    
    for i in range(total_test_len):
        curr_raw_idx = i 
        active_directions = [] 
        
        for model in models:
            target_idx = curr_raw_idx + max_seq_len
            start_seq_idx = target_idx - model['seq_len']
            
            a_seq = tuple(model['buckets'][start_seq_idx : target_idx])
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq))) if model['seq_len'] > 1 else ()
                
            last_val = a_seq[-1]
            
            # Predict
            pred_val = get_prediction(model['config']['model_type'], model['abs_map'], model['der_map'], 
                                      a_seq, d_seq, last_val)
            
            if pred_val is None:
                continue # Model abstains
            
            pred_diff = pred_val - last_val
            
            # Check correctness against reality
            actual_val = model['buckets'][target_idx]
            diff = actual_val - last_val
            model_actual_dir = 1 if diff > 0 else (-1 if diff < 0 else 0)

            if pred_diff != 0:
                direction = 1 if pred_diff > 0 else -1
                active_directions.append({
                    "dir": direction,
                    "is_correct": (direction == model_actual_dir),
                    "is_flat": (model_actual_dir == 0)
                })

        if not active_directions: continue
            
        dirs = [x['dir'] for x in active_directions]
        up_votes = dirs.count(1)
        down_votes = dirs.count(-1)
        
        final_dir = 0
        if up_votes > down_votes: final_dir = 1
        elif down_votes > up_votes: final_dir = -1
        else: continue
        
        winning_voters = [x for x in active_directions if x['dir'] == final_dir]
        if all(x['is_flat'] for x in winning_voters): continue 
            
        unique_total += 1
        if any(x['is_correct'] for x in winning_voters):
            unique_correct += 1

    return unique_correct, unique_total

def find_optimal_strategy(prices):
    bucket_counts = list(range(10, 251, 10))
    seq_lengths = [3, 4, 5, 6, 8]
    results = []
    
    for b_count in bucket_counts:
        b_size = calculate_bucket_size(prices, b_count)
        for s in seq_lengths:
            acc, mod, tr = evaluate_config(prices, b_size, s)
            if tr > 20: 
                results.append({
                    "bucket_count": b_count, "bucket_size": b_size,   
                    "seq_len": s, "model_type": mod,
                    "accuracy": acc, "trades": tr
                })
    
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    return results[:5]

def upload_to_github(file_path, content):
    if not GITHUB_PAT:
        print("Error: No PAT found in .env")
        return

    url = GITHUB_API_URL + file_path
    headers = {"Authorization": f"Bearer {GITHUB_PAT}", "Accept": "application/vnd.github.v3+json"}
    
    json_content = json.dumps(content, indent=2)
    b64_content = base64.b64encode(json_content.encode("utf-8")).decode("utf-8")
    
    data = {"message": f"Update optimized model for {file_path}", "content": b64_content}
    
    check_resp = requests.get(url, headers=headers)
    if check_resp.status_code == 200:
        data["sha"] = check_resp.json()["sha"]
        
    resp = requests.put(url, headers=headers, data=json.dumps(data))
    if resp.status_code in [200, 201]:
        print(f"Success: Uploaded {file_path}")
    else:
        print(f"Failed to upload {file_path}: {resp.text}")

def main():
    if not GITHUB_PAT:
        print("WARNING: 'PAT' not found. GitHub upload will fail.")

    for asset in ASSETS:
        raw_data = get_binance_data(asset)
        if not raw_data: continue
        
        for tf_name, tf_pandas in TIMEFRAMES.items():
            print(f"Processing {asset} [{tf_name}]...")
            prices = resample_prices(raw_data, tf_pandas)
            if len(prices) < 200: continue
                
            top_configs = find_optimal_strategy(prices)
            if not top_configs:
                print(f"No valid strategy found for {asset} {tf_name}")
                continue
            
            u_correct, u_total = run_portfolio_analysis(prices, top_configs)
            u_acc = (u_correct / u_total * 100) if u_total > 0 else 0
            
            print(f"--> {asset} {tf_name} In-Sample Accuracy: {u_acc:.2f}% ({u_total} trades)")

            serialized_strategies = prepare_deployment_models(prices, top_configs)
            final_payload = {
                "asset": asset, "timeframe": tf_name,
                "timestamp": datetime.now().isoformat(),
                "data_points": len(prices), "optimization_type": "full_dataset_fit",
                "combined_accuracy": u_acc, "combined_trades": u_total,
                "strategy_union": serialized_strategies 
            }
            
            filename = f"{asset}_{tf_name}.json"
            upload_to_github(filename, final_payload)
            
    print("\n--- FULL DATASET OPTIMIZATION & UPLOAD COMPLETE ---")

if __name__ == "__main__":
    main()
