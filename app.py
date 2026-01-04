import random
import time
import os
import json
import base64
import requests
from collections import defaultdict
from typing import List, Tuple
import matplotlib.pyplot as plt
from io import BytesIO

# ============================================================================
# PARAMETERS & ENV CONFIG
# ============================================================================
GITHUB_REPO = "constantinbender51-cmyk/Models"
# Railway provides the PAT via environment variables
GITHUB_PAT = os.environ.get('PAT', '')

N_PRICES = 5000  # Reduced for faster deployment execution
MIN_PRICE = 100
MAX_PRICE = 600
N_CATEGORIES = 100
TRAIN_SPLIT = 0.7
CATEGORY_STEP = 1
STARTING_EQUITY = 10000
CATEGORY_TO_DOLLAR = 10

# ============================================================================
# PREDICTION LOGIC
# ============================================================================

def generate_prices(n: int, min_price: float, max_price: float) -> List[float]:
    prices = []
    curr = (min_price + max_price) / 2
    for _ in range(n):
        curr += random.uniform(-5, 5)
        curr = max(min_price, min(max_price, curr))
        prices.append(curr)
    return prices

def categorize_price(price: float, min_price: float, max_price: float, n_categories: int) -> int:
    step = (max_price - min_price) / n_categories
    return int(max(0, min(n_categories - 1, (price - min_price) / step)))

def compute_probs(categories: List[int]):
    cat_counts = defaultdict(int)
    dir_counts = defaultdict(int)
    
    # Category sequences
    for i in range(len(categories) - 2):
        seq = tuple(categories[i:i+3])
        cat_counts[seq] += 1
    
    # Directional sequences
    dirs = [categories[i+1] - categories[i] for i in range(len(categories) - 1)]
    for i in range(len(dirs) - 2):
        seq = tuple(dirs[i:i+3])
        dir_counts[seq] += 1
        
    c_total = sum(cat_counts.values())
    d_total = sum(dir_counts.values())
    
    cat_probs = {k: v/c_total for k, v in cat_counts.items()}
    dir_probs = {k: v/d_total for k, v in dir_counts.items()}
    return cat_probs, dir_probs

def predict_next(last_two: List[int], cat_probs: dict, dir_probs: dict, step: int):
    c1, c2 = last_two
    best_prob = -1
    best_pred = c2
    
    for d1 in range(-step, step + 1):
        for d2 in range(-step, step + 1):
            for d3 in range(-step, step + 1):
                v = [c1 + d1, c2 + d2, c2 + d3]
                cp = cat_probs.get(tuple(v), 0)
                dp = dir_probs.get((v[1]-v[0], v[2]-v[1]), 0)
                if (cp + dp) > best_prob:
                    best_prob = cp + dp
                    best_pred = v[2]
    
    action = "hold"
    if best_pred > c2: action = "buy"
    elif best_pred < c2: action = "sell"
    return best_pred, action

# ============================================================================
# GITHUB UPLOAD LOGIC
# ============================================================================

def upload_to_github(filename: str, content_bytes: bytes, message: str):
    """Uploads a file to GitHub repository using REST API"""
    if not GITHUB_PAT:
        print("Error: GITHUB_PAT not found in environment.")
        return

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/visuals/{filename}"
    
    # Check if file exists to get SHA for update
    headers = {"Authorization": f"token {GITHUB_PAT}"}
    resp = requests.get(url, headers=headers)
    sha = resp.json().get('sha') if resp.status_code == 200 else None
    
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode('utf-8')
    }
    if sha:
        payload["sha"] = sha
        
    put_resp = requests.put(url, headers=headers, json=payload)
    if put_resp.status_code in [200, 201]:
        print(f"Successfully uploaded {filename} to GitHub.")
    else:
        print(f"Failed to upload {filename}: {put_resp.text}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("Starting Prediction Engine...")
    prices = generate_prices(N_PRICES, MIN_PRICE, MAX_PRICE)
    categories = [categorize_price(p, MIN_PRICE, MAX_PRICE, N_CATEGORIES) for p in prices]
    
    split = int(N_PRICES * TRAIN_SPLIT)
    train_cats = categories[:split]
    test_cats = categories[split:]
    
    cat_probs, dir_probs = compute_probs(train_cats)
    
    equity = STARTING_EQUITY
    history = []
    position = 0
    entry_cat = 0
    
    print("Generating predictions and simulating trades...")
    for i in range(2, len(test_cats)):
        last_two = test_cats[i-2:i]
        pred_cat, action = predict_next(last_two, cat_probs, dir_probs, CATEGORY_STEP)
        actual = test_cats[i]
        
        # Simple Trade Simulation
        if action == "buy" and position == 0:
            position, entry_cat = 1, actual
        elif action == "sell" and position == 1:
            equity += (actual - entry_cat) * CATEGORY_TO_DOLLAR
            position = 0
        elif action == "sell" and position == 0:
            position, entry_cat = -1, actual
        elif action == "buy" and position == -1:
            equity += (entry_cat - actual) * CATEGORY_TO_DOLLAR
            position = 0
            
        history.append({
            'equity': equity,
            'actual': actual,
            'pred': pred_cat
        })

    # --- Generate Plots ---
    print("Generating Visualizations...")
    
    # Plot 1: Equity Curve
    plt.figure(figsize=(10, 5))
    plt.plot([h['equity'] for h in history], color='#10b981', linewidth=2)
    plt.title("Equity Curve (Category-Based Trading)")
    plt.xlabel("Predictions")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    upload_to_github("equity_curve.png", buf.getvalue(), "Update equity curve plot")
    plt.close()

    # Plot 2: Actual vs Predicted (Last 50)
    plt.figure(figsize=(10, 5))
    subset = history[-50:]
    plt.bar(range(50), [h['actual'] for h in subset], alpha=0.5, label='Actual Category', color='#3b82f6')
    plt.step(range(50), [h['pred'] for h in subset], where='mid', label='Predicted', color='#f59e0b', linewidth=2)
    plt.title("Actual vs Predicted Categories (Last 50 Samples)")
    plt.legend()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    upload_to_github("prediction_accuracy.png", buf.getvalue(), "Update prediction accuracy plot")
    plt.close()

    print("Execution complete. Check your GitHub repository for the 'visuals' folder.")

if __name__ == "__main__":
    main()