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
DATA_DIR = "volume data"
BASE_INTERVAL = "15m"
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"

# --- Asset List ---
ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", 
    "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "TRXUSDT",
    "BCHUSDT", "XLMUSDT", "LTCUSDT", "SUIUSDT", "HBARUSDT",
    "SHIBUSDT", "TONUSDT", "UNIUSDT", "ZECUSDT"
]

# --- Timeframes ---
TIMEFRAMES = {
    "15m": None,
    "1h": "1h",
    "4h": "4h",
    "1d": "1D"
}

# --- Grid Search ---
BUCKET_COUNTS = range(10, 201, 10)
SEQ_LENGTHS = [4, 5, 6, 8, 10, 12]
MIN_TRADES = 15
SCORE_THRESHOLD = 0.55

# =========================================
# 2. DATA UTILITIES
# =========================================

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
    print(f"[{symbol}] Downloading {BASE_INTERVAL} data from Binance...")
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
                batch = [[int(c[0]), float(c[4])] for c in data]
                all_candles.extend(batch)
                current_start = data[-1][0] + 1
                sys.stdout.write(f"\rFetched {len(all_candles)} candles...")
                sys.stdout.flush()
        except Exception as e:
            print(f"\nError: {e}")
            break
            
    if all_candles:
        with open(filename, 'w') as f:
            json.dump(all_candles, f)
            
    return all_candles

def resample_prices(raw_data, target_freq):
    if not raw_data: return []
    if target_freq is None: return [x[1] for x in raw_data]

    df = pd.DataFrame(raw_data, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    resampled = df['price'].resample(target_freq).last().dropna()
    return resampled.tolist()

def split_data(prices):
    total = len(prices)
    if total < 500: return None, None, None
    train_end = int(total * 0.80)
    preval_end = int(total * 0.90)
    return prices[:train_end], prices[train_end:preval_end], prices[preval_end:]

# =========================================
# 3. CORE STRATEGY (100 LOGIC)
# =========================================

def get_bucket(price, bucket_size):
    if bucket_size <= 0: bucket_size = 1e-9
    return int(price // bucket_size)

def calculate_bucket_size(prices, bucket_count):
    if not prices: return 1.0
    min_p, max_p = min(prices), max(prices)
    price_range = max_p - min_p
    size = price_range / bucket_count
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
    """
    STRICT 100 LOGIC:
    - If Combined: Calculate directions.
    - If conflict -> None.
    - If Absolute exists -> Return Absolute.
    - If Absolute missing -> Return Derivative.
    """
    
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
        
        # Conflict check
        if dir_abs != 0 and dir_der != 0 and dir_abs != dir_der: return None 
        
        # PRIORITIZE ABSOLUTE (The "100" Preference)
        if dir_abs != 0: return pred_abs
        if dir_der != 0: return pred_der
            
    return None

# =========================================
# 4. OPTIMIZATION & BACKTEST
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

def run_ensemble_holdout(train_preval_prices, holdout_prices, top_configs):
    models = []
    for cfg in top_configs:
        b_size = calculate_bucket_size(train_preval_prices, cfg['b_count'])
        t_buckets = [get_bucket(p, b_size) for p in train_preval_prices]
        h_buckets = [get_bucket(p, b_size) for p in holdout_prices]
        abs_map, der_map = train_models(t_buckets, cfg['s_len'])
        models.append({"cfg": cfg, "h_buckets": h_buckets, "abs": abs_map, "der": der_map})
        
    max_s = max(c['s_len'] for c in top_configs)
    loop_len = len(holdout_prices) - max_s
    correct, trades = 0, 0
    
    for i in range(loop_len):
        votes = []
        for m in models:
            s = m['cfg']['s_len']
            end_idx = i + max_s
            start_idx = end_idx - s
            if start_idx < 0: continue
            
            slice_b = m['h_buckets'][start_idx : end_idx + 1]
            a_seq = tuple(slice_b[:-1])
            last = a_seq[-1]
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq))) if s > 1 else ()
            
            pred = get_prediction(m['cfg']['model'], m['abs'], m['der'], a_seq, d_seq, last)
            if pred is not None:
                diff = pred - last
                if diff != 0: votes.append(1 if diff > 0 else -1)
        
        if not votes: continue
        
        final_dir = 0
        if sum(votes) > 0: final_dir = 1
        elif sum(votes) < 0: final_dir = -1
        
        if final_dir != 0:
            act_diff = holdout_prices[i + max_s] - holdout_prices[i + max_s - 1]
            if act_diff != 0:
                trades += 1
                if (final_dir == 1 and act_diff > 0) or (final_dir == -1 and act_diff < 0):
                    correct += 1
                    
    acc = (correct / trades * 100) if trades > 0 else 0
    return {"accuracy": acc, "trades": trades}

def serialize_map(m):
    return { "|".join(map(str, k)): dict(v) for k, v in m.items() }

def upload_to_github(filename, content):
    if not GITHUB_PAT: return
    url = GITHUB_API_URL + filename
    headers = {"Authorization": f"Bearer {GITHUB_PAT}", "Accept": "application/vnd.github.v3+json"}
    content_b64 = base64.b64encode(json.dumps(content, indent=2).encode("utf-8")).decode("utf-8")
    
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

    for asset in ASSETS:
        raw_15m = get_binance_data(asset)
        if not raw_15m or len(raw_15m) < 2000:
            print(f"Skipping {asset} (Insufficient Data)")
            continue

        for tf_name, tf_alias in TIMEFRAMES.items():
            print(f"\n--- Processing {asset} {tf_name} ---")
            prices = resample_prices(raw_15m, tf_alias)
            if len(prices) < 1000: continue

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
            
            if not results: continue

            results.sort(key=lambda x: x['score'], reverse=True)
            top_5 = results[:5]
            
            ensemble_res = run_ensemble_holdout(train + preval, holdout, top_5)
            print(f"Holdout Result: {ensemble_res['accuracy']:.2f}% ({ensemble_res['trades']} trades)")

            # Retrain on Full Data for JSON
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
