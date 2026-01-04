import os
import time
import json
import base64
import requests
import random
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from io import BytesIO
from collections import defaultdict
from typing import List, Tuple

# ============================================================================
# PARAMETERS & ENV CONFIG
# ============================================================================
GITHUB_REPO = "constantinbender51-cmyk/Models"
GITHUB_PAT = os.environ.get('PAT', '')

# Trading & Logic Params
SYMBOL = "ETHUSDT"
INTERVAL = "1h"
START_YEAR = 2018
N_CATEGORIES = 1000
TRAIN_SPLIT = 0.8
SEQUENCE_LENGTH = 4  # Using last 3 hours to predict the 4th
CATEGORY_STEP = 1    # Range of "neighboring" sequences to check
STARTING_EQUITY = 10000
# Multiplier to make price changes meaningful in dollar terms for equity curve
DOLLAR_MULTIPLIER = 1 

# ============================================================================
# BINANCE DATA FETCHING
# ============================================================================

def fetch_binance_klines(symbol: str, interval: str, start_time_ms: int):
    """Fetches up to 1000 klines from Binance API."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time_ms,
        "limit": 1000
    }
    
    # Simple exponential backoff for API reliability
    for attempt in range(5):
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            time.sleep(2**attempt)
        except Exception:
            time.sleep(2**attempt)
    return []

def get_historical_data(symbol: str, interval: str, start_year: int):
    """Iteratively fetches all hourly data from start_year to present."""
    start_dt = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    curr_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    all_prices = []
    print(f"Fetching {symbol} {interval} data since {start_year}...")
    
    while curr_ms < end_ms:
        data = fetch_binance_klines(symbol, interval, curr_ms)
        if not data:
            break
            
        # Extract Close Prices (Index 4)
        batch_prices = [float(candle[4]) for candle in data]
        all_prices.extend(batch_prices)
        
        # Last candle's open time + 1 to move to next batch
        last_time = data[-1][0]
        if last_time <= curr_ms: break # Prevent infinite loop
        curr_ms = last_time + 1
        
        print(f"Collected {len(all_prices)} data points... (Latest: {datetime.fromtimestamp(last_time/1000).strftime('%Y-%m-%d')})", end="\r")
        time.sleep(0.1) # Respect API limits
        
    print(f"\nTotal Data Points: {len(all_prices)}")
    return all_prices

# ============================================================================
# CORE LOGIC: CATEGORIZATION & PREDICTION
# ============================================================================

def categorize_data(prices: List[float], n_cats: int):
    min_p, max_p = min(prices), max(prices)
    step = (max_p - min_p) / n_cats
    cats = [int(max(0, min(n_cats - 1, (p - min_p) / step))) for p in prices]
    return cats, min_p, max_p

def compute_probs(categories: List[int], seq_len: int):
    cat_counts = defaultdict(int)
    dir_counts = defaultdict(int)
    
    # Category sequences
    for i in range(len(categories) - seq_len + 1):
        seq = tuple(categories[i:i+seq_len])
        cat_counts[seq] += 1
    
    # Directional deltas
    dirs = [categories[i+1] - categories[i] for i in range(len(categories) - 1)]
    for i in range(len(dirs) - seq_len + 2):
        seq = tuple(dirs[i:i+seq_len-1])
        dir_counts[seq] += 1
        
    c_total = sum(cat_counts.values())
    d_total = sum(dir_counts.values())
    
    cp = {k: v/c_total for k, v in cat_counts.items()} if c_total > 0 else {}
    dp = {k: v/d_total for k, v in dir_counts.items()} if d_total > 0 else {}
    return cp, dp

def predict_next(last_n: List[int], cat_probs: dict, dir_probs: dict, step: int, seq_len: int):
    best_prob = -1
    best_pred = last_n[-1]
    
    # Simplified search: look for historical frequency of following current cat
    # Given the last_n, what is the most likely next?
    current_val = last_n[-1]
    for possible_next in range(max(0, current_val - step), min(N_CATEGORIES, current_val + step + 1)):
        test_seq = tuple(list(last_n) + [possible_next])
        p = cat_probs.get(test_seq, 0)
        
        # Also look at direction
        delta_seq = tuple([last_n[i+1] - last_n[i] for i in range(len(last_n)-1)] + [possible_next - last_n[-1]])
        p += dir_probs.get(delta_seq, 0)
        
        if p > best_prob:
            best_prob = p
            best_pred = possible_next
            
    action = "hold"
    if best_pred > current_val: action = "buy"
    elif best_pred < current_val: action = "sell"
    return best_pred, action

# ============================================================================
# GITHUB UPLOAD
# ============================================================================

def upload_to_github(filename: str, content_bytes: bytes, message: str):
    if not GITHUB_PAT:
        print("Warning: GITHUB_PAT not set. Skipping upload.")
        return

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/visuals/{filename}"
    headers = {"Authorization": f"token {GITHUB_PAT}"}
    
    resp = requests.get(url, headers=headers)
    sha = resp.json().get('sha') if resp.status_code == 200 else None
    
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode('utf-8')
    }
    if sha: payload["sha"] = sha
        
    requests.put(url, headers=headers, json=payload)

# ============================================================================
# MAIN
# ============================================================================

def main():
    # 1. Get Real Data
    prices = get_historical_data(SYMBOL, INTERVAL, START_YEAR)
    if not prices:
        print("Failed to fetch data.")
        return

    # 2. Pre-process
    cats, min_p, max_p = categorize_data(prices, N_CATEGORIES)
    split_idx = int(len(cats) * TRAIN_SPLIT)
    train_cats = cats[:split_idx]
    test_cats = cats[split_idx:]
    test_prices = prices[split_idx:]

    # 3. Training
    cp, dp = compute_probs(train_cats, SEQUENCE_LENGTH)

    # 4. Simulation
    equity = STARTING_EQUITY
    history = []
    position = 0 # 1=Long, -1=Short, 0=Flat
    entry_price = 0
    
    print(f"Simulating trades on {len(test_cats)} data points...")
    for i in range(SEQUENCE_LENGTH - 1, len(test_cats)):
        # Look back context
        lookback = test_cats[i - (SEQUENCE_LENGTH - 1) : i]
        pred_cat, action = predict_next(lookback, cp, dp, CATEGORY_STEP, SEQUENCE_LENGTH)
        
        actual_price = test_prices[i]
        actual_cat = test_cats[i]
        
        # Simple Logic: Trade the Close Price
        if action == "buy" and position <= 0:
            if position == -1: # Close short
                equity += (entry_price - actual_price) * DOLLAR_MULTIPLIER
            position, entry_price = 1, actual_price
        elif action == "sell" and position >= 0:
            if position == 1: # Close long
                equity += (actual_price - entry_price) * DOLLAR_MULTIPLIER
            position, entry_price = -1, actual_price
            
        history.append({'equity': equity, 'actual': actual_cat, 'pred': pred_cat})

    # 5. Visuals
    print(f"Final Equity: ${equity:,.2f}")
    
    # Plot Equity
    plt.figure(figsize=(10, 5))
    plt.plot([h['equity'] for h in history], color='#10b981')
    plt.title(f"{SYMBOL} Equity Curve (Since 2018 Training)")
    plt.ylabel("Equity ($)")
    buf = BytesIO()
    plt.savefig(buf, format='png')
    upload_to_github("eth_equity_curve.png", buf.getvalue(), "Update ETH equity curve")
    plt.close()

    # Plot Comparison (Snapshot)
    plt.figure(figsize=(10, 5))
    subset = history[-100:]
    plt.plot([h['actual'] for h in subset], label="Actual", alpha=0.6)
    plt.plot([h['pred'] for h in subset], label="Predicted", linestyle='--')
    plt.legend()
    plt.title("Actual vs Predicted Categories (Last 100 Hours)")
    buf = BytesIO()
    plt.savefig(buf, format='png')
    upload_to_github("eth_accuracy.png", buf.getvalue(), "Update ETH accuracy plot")
    plt.close()

if __name__ == "__main__":
    main()