import os
import sys
import json
import base64
import time
import requests
import random
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

# =========================================
# 1. CONFIGURATION
# =========================================

# --- Github Settings ---
# Ensure you have a .env file or set PAT in environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GITHUB_PAT = os.getenv("PAT")
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "model-2"  # Saving to model-2 as requested
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/"

# --- Data Settings ---
DATA_DIR = "volume data"
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"

# --- Asset List (19 Assets) ---
ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", 
    "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "TRXUSDT",
    "BCHUSDT", "XLMUSDT", "LTCUSDT", "SUIUSDT", "HBARUSDT",
    "SHIBUSDT", "TONUSDT", "UNIUSDT", "ZECUSDT"
]

# --- Timeframes ---
# We will look for these intervals in the volume data folder
TIMEFRAMES = ["15m", "1h", "4h", "1d"]

# --- Optimization Grid ---
BUCKET_COUNTS = range(10, 201, 10)  # 10, 20 ... 200
SEQ_LENGTHS = [4, 5, 6, 8, 10, 12]
MIN_TRADES = 15
SCORE_THRESHOLD = 0.55  # 55% accuracy minimum to contribute to score

# =========================================
# 2. DATA LOADING & SPLITTING
# =========================================

def get_data_from_file(symbol, interval):
    """
    Loads data from 'volume data/{SYMBOL}_{INTERVAL}_2020-01-01_2026-01-01.json'.
    Returns a list of float prices.
    """
    file_name = f"{symbol}_{interval}_{START_DATE}_{END_DATE}.json"
    file_path = os.path.join(DATA_DIR, file_name)
    
    if not os.path.exists(file_path):
        # Fallback: Try to find any file matching symbol + interval if exact date match fails
        if os.path.exists(DATA_DIR):
            for f in os.listdir(DATA_DIR):
                if f.startswith(f"{symbol}_{interval}") and f.endswith(".json"):
                    file_path = os.path.join(DATA_DIR, f)
                    break
        else:
            return None

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Handle Binance Kline format [[t, o, h, l, c, v...], ...] or simple list
                if isinstance(data, list):
                    if isinstance(data[0], list) and len(data[0]) > 4:
                        return [float(x[4]) for x in data] # Close price
                    elif isinstance(data[0], (int, float)):
                        return [float(x) for x in data]
                    elif isinstance(data[0], dict) and 'close' in data[0]:
                         return [float(x['close']) for x in data]
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path}: {e}")
            return None
            
    return None

def split_data(prices):
    """
    Splits data into 80% Train, 10% Pre-Validation, 10% Holdout.
    """
    total = len(prices)
    if total < 500: return None, None, None
    
    train_end = int(total * 0.80)
    preval_end = int(total * 0.90)
    
    train = prices[:train_end]
    preval = prices[train_end:preval_end]
    holdout = prices[preval_end:]
    
    return train, preval, holdout

# =========================================
# 3. CORE STRATEGY LOGIC
# =========================================

def get_bucket(price, bucket_size):
    if bucket_size <= 0: bucket_size = 1e-9
    return int(price // bucket_size)

def calculate_bucket_size(prices, bucket_count):
    if not prices: return 1.0
    min_p, max_p = min(prices), max(prices)
    price_range = max_p - min_p
    size = price_range / bucket_count
    return size if size > 0 else 0.000001

def train_models(train_buckets, seq_len):
    """Builds probability maps."""
    abs_map = defaultdict(Counter)
    der_map = defaultdict(Counter)
    
    # We use a sliding window
    for i in range(len(train_buckets) - seq_len):
        a_seq = tuple(train_buckets[i : i + seq_len])
        a_succ = train_buckets[i + seq_len]
        abs_map[a_seq][a_succ] += 1
        
        if seq_len > 1:
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
            d_succ = train_buckets[i + seq_len] - train_buckets[i + seq_len - 1]
            der_map[d_seq][d_succ] += 1
            
    return abs_map, der_map

def get_prediction(model_type, abs_map, der_map, a_seq, d_seq, last_val):
    """
    Returns predicted NEXT VALUE or None.
    """
    pred_abs = None
    pred_der = None
    
    # 1. Absolute Prediction
    if model_type in ["Absolute", "Combined"]:
        if a_seq in abs_map:
            pred_abs = abs_map[a_seq].most_common(1)[0][0]

    # 2. Derivative Prediction
    if model_type in ["Derivative", "Combined"]:
        if d_seq in der_map:
            change = der_map[d_seq].most_common(1)[0][0]
            pred_der = last_val + change

    # 3. Decision Logic
    if model_type == "Absolute":
        return pred_abs
    elif model_type == "Derivative":
        return pred_der
    elif model_type == "Combined":
        # Consensus logic
        if pred_abs is None and pred_der is None: return None
        if pred_abs is not None and pred_der is None: return pred_abs
        if pred_abs is None and pred_der is not None: return pred_der
        
        # Conflict check
        diff_abs = pred_abs - last_val
        diff_der = pred_der - last_val
        
        dir_abs = 1 if diff_abs > 0 else -1 if diff_abs < 0 else 0
        dir_der = 1 if diff_der > 0 else -1 if diff_der < 0 else 0
        
        if dir_abs == dir_der and dir_abs != 0:
            # If they agree on direction, prefer the one with higher frequency support? 
            # For simplicity in Combined, we average or pick one. 
            # Here: Prioritize Derivative as it captures momentum better.
            return pred_der
            
    return None

def backtest_segment(train_prices, test_prices, b_count, s_len, model_type):
    """
    Train on train_prices, Test on test_prices.
    IMPORTANT: bucket_size is calculated ONLY on train_prices to prevent lookahead bias.
    """
    b_size = calculate_bucket_size(train_prices, b_count)
    
    t_buckets = [get_bucket(p, b_size) for p in train_prices]
    v_buckets = [get_bucket(p, b_size) for p in test_prices]
    
    abs_map, der_map = train_models(t_buckets, s_len)
    
    correct = 0
    trades = 0
    
    # Iterate through test set
    # We need 's_len' previous candles to make a prediction
    # If test set is a continuation of train, we can use the end of train as context?
    # Simpler: Treat test set as standalone sequence for evaluation loop, 
    # but practically we start at index = s_len
    
    loop_range = len(v_buckets) - s_len
    if loop_range <= 0: return 0, 0
    
    for i in range(loop_range):
        slice_bkts = v_buckets[i : i + s_len + 1]
        a_seq = tuple(slice_bkts[:-1])
        actual_next = slice_bkts[-1]
        last_val = a_seq[-1]
        
        d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq))) if s_len > 1 else ()
        
        pred = get_prediction(model_type, abs_map, der_map, a_seq, d_seq, last_val)
        
        if pred is not None:
            pred_diff = pred - last_val
            act_diff = actual_next - last_val
            
            if pred_diff != 0 and act_diff != 0:
                trades += 1
                if (pred_diff > 0 and act_diff > 0) or (pred_diff < 0 and act_diff < 0):
                    correct += 1
                    
    acc = (correct / trades * 100) if trades > 0 else 0
    return acc, trades

# =========================================
# 4. OPTIMIZATION LOOP
# =========================================

def optimize_asset_timeframe(asset, interval, prices):
    train, preval, holdout = split_data(prices)
    if train is None: return None
    
    results = []
    
    # GRID SEARCH
    # We iterate configs, train on TRAIN, test on PREVAL
    for b in BUCKET_COUNTS:
        for s in SEQ_LENGTHS:
            for m in ["Absolute", "Derivative", "Combined"]:
                # Test on Pre-Validation
                acc, tr = backtest_segment(train, preval, b, s, m)
                
                if tr >= MIN_TRADES:
                    # Score Function: Reward accuracy > 55% heavily, weighted by volume
                    score = ((acc / 100.0) - SCORE_THRESHOLD) * tr
                    
                    results.append({
                        "b_count": b,
                        "s_len": s,
                        "model": m,
                        "preval_acc": acc,
                        "preval_tr": tr,
                        "score": score
                    })
                    
    if not results: return None
    
    # Sort by Score
    results.sort(key=lambda x: x['score'], reverse=True)
    top_5 = results[:5]
    
    # FINAL HOLDOUT TEST (ENSEMBLE)
    # Validate the Top 5 on the Holdout set (The "100" logic)
    ensemble_results = run_ensemble_holdout(train + preval, holdout, top_5)
    
    # PREPARE FOR DEPLOYMENT
    # Retrain maps on FULL DATA (Train + PreVal + Holdout) for the JSON
    full_data = prices
    deployment_strategies = []
    
    for cfg in top_5:
        b_size = calculate_bucket_size(full_data, cfg['b_count'])
        bkts = [get_bucket(p, b_size) for p in full_data]
        abs_map, der_map = train_models(bkts, cfg['s_len'])
        
        deployment_strategies.append({
            "config": {
                "bucket_count": cfg['b_count'],
                "seq_len": cfg['s_len'],
                "model_type": cfg['model']
            },
            "metrics": {
                "preval_score": cfg['score'],
                "preval_acc": cfg['preval_acc']
            },
            "params": {
                "bucket_size": b_size,
                "abs_map": serialize_map(abs_map),
                "der_map": serialize_map(der_map)
            }
        })

    return {
        "asset": asset,
        "interval": interval,
        "timestamp": datetime.now().isoformat(),
        "holdout_performance": ensemble_results,
        "strategies": deployment_strategies
    }

def run_ensemble_holdout(train_preval_prices, holdout_prices, top_configs):
    """
    Simulates the ensemble trading on the Holdout set.
    """
    models = []
    for cfg in top_configs:
        # Train on known history (Train + PreVal)
        b_size = calculate_bucket_size(train_preval_prices, cfg['b_count'])
        t_buckets = [get_bucket(p, b_size) for p in train_preval_prices]
        h_buckets = [get_bucket(p, b_size) for p in holdout_prices]
        
        abs_map, der_map = train_models(t_buckets, cfg['s_len'])
        
        models.append({
            "cfg": cfg, "h_buckets": h_buckets,
            "abs": abs_map, "der": der_map
        })
        
    max_s = max(c['s_len'] for c in top_configs)
    loop_len = len(holdout_prices) - max_s
    
    correct = 0
    trades = 0
    
    for i in range(loop_len):
        votes = []
        
        for m in models:
            s = m['cfg']['s_len']
            # Align indices so we look at the same "real time" moment
            # We want prediction for holdout[i + max_s]
            # using context ending at holdout[i + max_s - 1]
            # The slice needs to be length s+1 ending at i+max_s
            
            end_idx = i + max_s
            start_idx = end_idx - s
            
            if start_idx < 0: continue
            
            slice_b = m['h_buckets'][start_idx : end_idx + 1]
            a_seq = tuple(slice_b[:-1])
            actual = slice_b[-1]
            last = a_seq[-1]
            
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq))) if s > 1 else ()
            
            pred = get_prediction(m['cfg']['model'], m['abs'], m['der'], a_seq, d_seq, last)
            
            if pred is not None:
                diff = pred - last
                if diff != 0:
                    votes.append(1 if diff > 0 else -1)
                    
        if not votes: continue
        
        # Vote Aggregation
        vote_sum = sum(votes)
        final_dir = 0
        if vote_sum > 0: final_dir = 1
        elif vote_sum < 0: final_dir = -1
        
        if final_dir != 0:
            # Check Reality (using price at i + max_s)
            real_p = holdout_prices[i + max_s]
            real_last = holdout_prices[i + max_s - 1] # Roughly
            # More accurately:
            # We use the buckets of the first model (or any model) to determine "Direction"
            # But simpler: compare holdout prices directly
            
            act_diff = real_p - real_last
            if act_diff != 0:
                trades += 1
                if (final_dir == 1 and act_diff > 0) or (final_dir == -1 and act_diff < 0):
                    correct += 1
                    
    acc = (correct / trades * 100) if trades > 0 else 0
    return {"accuracy": acc, "trades": trades}

def serialize_map(m):
    return { "|".join(map(str, k)): dict(v) for k, v in m.items() }

# =========================================
# 5. GITHUB UPLOAD
# =========================================

def upload_to_github(filename, content):
    if not GITHUB_PAT:
        print("Skipping Upload (No PAT)")
        return
        
    url = GITHUB_API_URL + filename
    headers = {
        "Authorization": f"Bearer {GITHUB_PAT}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    json_str = json.dumps(content, indent=2)
    b64_content = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")
    
    payload = {
        "message": f"Optimization Update {filename}",
        "content": b64_content
    }
    
    # Check for existing file to get SHA
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        payload["sha"] = r.json()["sha"]
        
    r = requests.put(url, headers=headers, json=payload)
    if r.status_code in [200, 201]:
        print(f"Uploaded: {filename}")
    else:
        print(f"Upload Failed {filename}: {r.text}")

# =========================================
# 6. MAIN ORCHESTRATOR
# =========================================

def process_single_task(task):
    asset, interval = task
    print(f"Processing {asset} {interval}...")
    
    prices = get_data_from_file(asset, interval)
    if not prices or len(prices) < 2000:
        return f"Skipped {asset} {interval} (No Data)"
        
    result_data = optimize_asset_timeframe(asset, interval, prices)
    
    if result_data:
        fname = f"{asset}_{interval}.json"
        
        # Console Report
        perf = result_data['holdout_performance']
        print(f"--> {asset} {interval} Optimized. Holdout Acc: {perf['accuracy']:.2f}% ({perf['trades']} trds)")
        
        upload_to_github(fname, result_data)
        return f"Success {asset} {interval}"
    else:
        return f"Failed {asset} {interval} (Low Score)"

def main():
    if not os.path.exists(DATA_DIR):
        print(f"WARNING: Directory '{DATA_DIR}' not found. Ensure data is present.")
    
    tasks = []
    for asset in ASSETS:
        for tf in TIMEFRAMES:
            tasks.append((asset, tf))
            
    print(f"Starting Optimization for {len(tasks)} pairs...")
    print(f"Split: 80% Train | 10% Pre-Val | 10% Holdout")
    print(f"Grid: Buckets 10-200, Seqs 4-12")
    
    # Sequential execution to avoid API rate limits on Upload if concurrent
    # Or parallelize calculation, synchronize upload.
    # For safety/simplicity in this environment:
    
    for t in tasks:
        msg = process_single_task(t)
        print(msg)

if __name__ == "__main__":
    main()
