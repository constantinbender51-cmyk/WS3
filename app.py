import os
import sys
import json
import base64
import time
import requests
import urllib.request
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict

# =========================================
# 1. CONFIGURATION
# =========================================

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GITHUB_PAT = os.getenv("PAT")
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "model-2"
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/"

# --- Data Settings ---
DATA_DIR = "data"
BASE_INTERVAL = "1m"  # Changed to 1m to support lower timeframes
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"

# --- Asset List ---
ASSET_COUNT = 3  # Limit to 3 assets as requested
ALL_ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", 
    "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "TRXUSDT",
    "BCHUSDT", "XLMUSDT", "LTCUSDT", "SUIUSDT", "HBARUSDT",
    "SHIBUSDT", "TONUSDT", "UNIUSDT", "ZECUSDT"
]
ASSETS = ALL_ASSETS[:ASSET_COUNT]

# --- Timeframes (Short Term Focus) ---
TIMEFRAMES = {
    "1m": "1min",   # Pandas alias for 1 minute
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1H"
}

# --- Grid Search (Match 100 exactly where possible) ---
BUCKET_COUNTS = range(25, 201, 25) 
SEQ_LENGTHS = [5, 8, 12] 
MIN_TRADES = 15
SCORE_THRESHOLD = 0.70

# =========================================
# 2. DATA UTILITIES (Infrastructure)
# =========================================

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: 
        sys.stdout.write('\n')

def get_binance_data(symbol, start_str=START_DATE, end_str=END_DATE):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    filename = f"{DATA_DIR}/{symbol}_{BASE_INTERVAL}_{start_str}_{end_str}.json"

    # 1. Check Cache
    if os.path.exists(filename):
        print(f"[{symbol}] Loading cached {BASE_INTERVAL} data...")
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], list) and len(data[0]) > 4:
                        return [[x[0], float(x[4])] for x in data]
                    elif len(data[0]) == 2:
                        return data
        except Exception as e:
            print(f"Error loading cache: {e}")

    # 2. Fetch
    print(f"[{symbol}] Downloading {BASE_INTERVAL} data from Binance (This may take a while)...")
    
    start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_str, "%Y-%m-%d").timestamp() * 1000)
    total_duration = end_ts - start_ts
    
    all_candles = []
    current_start = start_ts
    
    while current_start < end_ts:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={BASE_INTERVAL}&startTime={current_start}&endTime={end_ts}&limit=1000"
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                
                if not data: 
                    # If no data returned, break to avoid infinite loop
                    print_progress_bar(total_duration, total_duration, prefix='Progress:', suffix='Done', length=40)
                    break
                
                batch = [[int(c[0]), float(c[4])] for c in data]
                all_candles.extend(batch)
                
                # Update progress
                current_start = data[-1][0] + 1
                progress = min(current_start - start_ts, total_duration)
                print_progress_bar(progress, total_duration, prefix='Progress:', suffix=f'({len(all_candles)} candles)', length=40)
                
        except Exception as e:
            print(f"\nError fetching batch: {e}")
            time.sleep(1) # Backoff slightly on error
            continue
            
    print(f"\n[{symbol}] Download complete. Total candles: {len(all_candles)}")
    
    if all_candles:
        with open(filename, 'w') as f:
            json.dump(all_candles, f)
            
    return all_candles

def resample_prices(raw_data, target_freq):
    """
    Resample 1m data to target frequency (e.g., '5min', '1H')
    """
    if not raw_data: return []
    
    # 1m to 1m (No resampling needed, just extract prices)
    if target_freq == "1min" or target_freq == "1m":
         return [x[1] for x in raw_data]

    if target_freq is None: return [x[1] for x in raw_data]

    df = pd.DataFrame(raw_data, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Pandas resample
    resampled = df['price'].resample(target_freq).last().dropna()
    return resampled.tolist()

def split_data(prices):
    total = len(prices)
    if total < 500: return None, None, None
    train_end = int(total * 0.80)
    preval_end = int(total * 0.90)
    return prices[:train_end], prices[train_end:preval_end], prices[preval_end:]

# =========================================
# 3. STRATEGY LOGIC (EXACT 100 COPY)
# =========================================

def get_bucket(price, bucket_size):
    if bucket_size <= 0: bucket_size = 1e-9
    return int(price // bucket_size)

def calculate_bucket_size(prices, bucket_count):
    if not prices: return 1.0
    min_p, max_p = min(prices), max(prices)
    price_range = max_p - min_p
    size = price_range / bucket_count
    return size if size > 0 else 0.01

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

def get_single_prediction(mode, abs_map, der_map, a_seq, d_seq, last_val):
    if mode == "Absolute":
        if a_seq in abs_map:
            return abs_map[a_seq].most_common(1)[0][0]
    elif mode == "Derivative":
        if d_seq in der_map:
            pred_change = der_map[d_seq].most_common(1)[0][0]
            return last_val + pred_change
    return None

def get_prediction(model_type, abs_map, der_map, a_seq, d_seq, last_val):
    if model_type == "Absolute":
        return get_single_prediction("Absolute", abs_map, der_map, a_seq, d_seq, last_val)
        
    elif model_type == "Derivative":
        return get_single_prediction("Derivative", abs_map, der_map, a_seq, d_seq, last_val)
        
    elif model_type == "Combined":
        pred_abs = get_single_prediction("Absolute", abs_map, der_map, a_seq, d_seq, last_val)
        pred_der = get_single_prediction("Derivative", abs_map, der_map, a_seq, d_seq, last_val)
        
        dir_abs = 0
        if pred_abs is not None:
            dir_abs = 1 if pred_abs > last_val else -1 if pred_abs < last_val else 0
            
        dir_der = 0
        if pred_der is not None:
            dir_der = 1 if pred_der > last_val else -1 if pred_der < last_val else 0
            
        if dir_abs == 0 and dir_der == 0: return None
        if dir_abs != 0 and dir_der != 0 and dir_abs != dir_der: return None 
        
        if dir_abs != 0: return pred_abs
        if dir_der != 0: return pred_der
            
    return None

# =========================================
# 4. OPTIMIZATION & ENSEMBLE
# =========================================

def backtest_segment(train_prices, test_prices, b_count, s_len, model_type):
    b_size = calculate_bucket_size(train_prices, b_count)
    t_buckets = [get_bucket(p, b_size) for p in train_prices]
    v_buckets = [get_bucket(p, b_size) for p in test_prices]
    
    abs_map, der_map = train_models(t_buckets, s_len)
    
    correct = 0
    trades = 0
    loop_range = len(v_buckets) - s_len
    
    if loop_range <= 0: return 0, 0
    
    for i in range(loop_range):
        slice_bkts = v_buckets[i : i + s_len + 1]
        a_seq = tuple(slice_bkts[:-1])
        actual_val = slice_bkts[-1]
        last_val = a_seq[-1]
        
        d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq))) if s_len > 1 else ()
        
        pred_val = get_prediction(model_type, abs_map, der_map, a_seq, d_seq, last_val)
        
        if pred_val is not None:
            pred_diff = pred_val - last_val
            actual_diff = actual_val - last_val
            
            if pred_diff != 0 and actual_diff != 0:
                trades += 1
                if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
                    correct += 1
                    
    acc = (correct / trades * 100) if trades > 0 else 0
    return acc, trades

def run_final_ensemble_logic(train_preval_prices, holdout_prices, top_configs):
    models = []
    for cfg in top_configs:
        b_size = calculate_bucket_size(train_preval_prices, cfg['b_count'])
        t_buckets = [get_bucket(p, b_size) for p in train_preval_prices]
        h_buckets = [get_bucket(p, b_size) for p in holdout_prices]
        
        abs_map, der_map = train_models(t_buckets, cfg['s_len'])
        
        models.append({
            "cfg": cfg,
            "val_buckets": h_buckets,
            "abs_map": abs_map,
            "der_map": der_map
        })
        
    max_seq = max(m['cfg']['s_len'] for m in models)
    loop_len = len(holdout_prices) - max_seq
    
    correct = 0
    total_trades = 0
    abstains = 0
    conflicts = 0
    
    for i in range(loop_len):
        active_signals = []
        
        for model in models:
            s_len = model['cfg']['s_len']
            v_bkts = model['val_buckets']
            
            end_idx = i + max_seq
            start_idx = end_idx - s_len
            
            if start_idx < 0: continue

            curr_slice = v_bkts[start_idx : end_idx + 1]
            a_seq = tuple(curr_slice[:-1])
            actual_val = curr_slice[-1]
            last_val = a_seq[-1]
            
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq))) if s_len > 1 else ()
            
            pred_val = get_prediction(model['cfg']['model'], model['abs_map'], model['der_map'], 
                                      a_seq, d_seq, last_val)
            
            if pred_val is not None:
                p_diff = pred_val - last_val
                if p_diff != 0:
                    direction = 1 if p_diff > 0 else -1
                    active_signals.append({
                        "dir": direction,
                        "b_count": model['cfg']['b_count'], 
                        "pred_val": pred_val,
                        "last_val": last_val,
                        "actual_val": actual_val
                    })

        if not active_signals:
            abstains += 1
            continue
            
        directions = {x['dir'] for x in active_signals}
        
        if len(directions) > 1:
            conflicts += 1
            continue 
            
        active_signals.sort(key=lambda x: x['b_count'])
        winner = active_signals[0]
        
        w_pred_diff = winner['pred_val'] - winner['last_val']
        w_act_diff = winner['actual_val'] - winner['last_val']
        
        if w_act_diff != 0: 
            total_trades += 1
            if (w_pred_diff > 0 and w_act_diff > 0) or (w_pred_diff < 0 and w_act_diff < 0):
                correct += 1

    acc = (correct / total_trades * 100) if total_trades > 0 else 0
    return {"accuracy": acc, "trades": total_trades}

def serialize_map(m):
    """
    MODIFIED: Model Pruning enabled.
    Instead of saving the full frequency dictionary (which creates 20MB+ files),
    we only save the single most likely outcome for each sequence.
    
    Old: { "seq": { "val1": 50, "val2": 2 } }
    New: { "seq": val1 }
    """
    compressed = {}
    for k, v in m.items():
        if v:
            # Take the single most common value (the winner)
            # This discards the counts and losing alternatives
            best_val = v.most_common(1)[0][0]
            compressed["|".join(map(str, k))] = best_val
    return compressed

def upload_to_github(filename, content):
    if not GITHUB_PAT: return
    
    # Check payload size before upload
    content_json = json.dumps(content, indent=2)
    size_mb = len(content_json.encode('utf-8')) / (1024 * 1024)
    print(f"Prepared payload for {filename}: {size_mb:.2f} MB")
    
    url = GITHUB_API_URL + filename
    headers = {"Authorization": f"Bearer {GITHUB_PAT}", "Accept": "application/vnd.github.v3+json"}
    content_b64 = base64.b64encode(content_json.encode("utf-8")).decode("utf-8")
    
    data = {"message": f"Update {filename}", "content": content_b64}
    r = requests.get(url, headers=headers)
    if r.status_code == 200: data["sha"] = r.json()["sha"]
        
    requests.put(url, headers=headers, json=data)
    print(f"Uploaded {filename}")

# =========================================
# 5. MAIN EXECUTION
# =========================================

def main():
    if not GITHUB_PAT: print("WARNING: No GITHUB_PAT found.")
    
    print(f"Starting run for {ASSET_COUNT} assets on short timeframes (1m-1h)...")

    for asset in ASSETS:
        # Get 1m data (Base for all short timeframes)
        raw_1m = get_binance_data(asset)
        
        if not raw_1m or len(raw_1m) < 10000: # Higher threshold for 1m data
            print(f"Skipping {asset} (Insufficient Data)")
            continue

        for tf_name, tf_alias in TIMEFRAMES.items():
            print(f"\n--- Processing {asset} {tf_name} ---")
            
            # Resample 1m -> target
            prices = resample_prices(raw_1m, tf_alias)
            
            if len(prices) < 1000: 
                print(f"Not enough data after resampling to {tf_name}")
                continue

            train, preval, holdout = split_data(prices)
            results = []
            
            # Grid Search
            for b in BUCKET_COUNTS:
                for s in SEQ_LENGTHS:
                    for m in ["Absolute", "Derivative", "Combined"]:
                        acc, tr = backtest_segment(train, preval, b, s, m)
                        if tr >= MIN_TRADES:
                            score = ((acc / 100.0) - SCORE_THRESHOLD) * tr
                            results.append({"b_count": b, "s_len": s, "model": m, "score": score, "acc": acc})
            
            if not results: 
                print("No configurations met minimum criteria.")
                continue

            results.sort(key=lambda x: x['score'], reverse=True)
            top_5 = results[:5]
            
            ensemble_res = run_final_ensemble_logic(train + preval, holdout, top_5)
            print(f"Holdout Result: {ensemble_res['accuracy']:.2f}% ({ensemble_res['trades']} trades)")

            # Prepare for Deployment
            full_data = prices
            deployment_models = []
            for cfg in top_5:
                b_size = calculate_bucket_size(full_data, cfg['b_count'])
                bkts = [get_bucket(p, b_size) for p in full_data]
                abs_map, der_map = train_models(bkts, cfg['s_len'])
                
                deployment_models.append({
                    "config": cfg,
                    "params": {
                        "bucket_size": b_size,
                        "abs_map": serialize_map(abs_map),
                        "der_map": serialize_map(der_map)
                    }
                })

            final_payload = {
                "asset": asset,
                "interval": tf_name,
                "timestamp": datetime.now().isoformat(),
                "holdout_stats": ensemble_res,
                "strategies": deployment_models
            }
            
            upload_to_github(f"{asset}_{tf_name}.json", final_payload)

if __name__ == "__main__":
    main()
