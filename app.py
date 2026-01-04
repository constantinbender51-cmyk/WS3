import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import requests
import base64
import json

# ==========================================
# 0. Helper: Load .env manually
# ==========================================
def load_env():
    """Loads environment variables from .env file if present."""
    if os.path.exists('.env'):
        with open('.env') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value.strip()

# ==========================================
# 1. Data Generation
# ==========================================
def generate_price_data(n=2000, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, 100, n)
    trend = t * 0.1 
    cycle = 5 * np.sin(t) + 3 * np.sin(t * 3)
    noise = np.random.normal(0, 1, n)
    random_walk = np.cumsum(np.random.normal(0, 0.5, n))
    price = 100 + trend + cycle + noise + random_walk
    price = np.maximum(price, 10.0) 
    return pd.DataFrame({'price': price})

# ==========================================
# 2. Sequence Model
# ==========================================
class SequenceModel:
    def __init__(self, max_len=5):
        self.max_len = max_len
        self.counts = defaultdict(int)
        self.total_counts_by_len = defaultdict(int)
        self.unique_patterns_by_len = defaultdict(set)

    def train(self, sequence):
        n = len(sequence)
        for length in range(1, self.max_len + 1):
            for i in range(n - length + 1):
                sub = tuple(sequence[i : i + length])
                self.counts[sub] += 1
                self.total_counts_by_len[length] += 1
                self.unique_patterns_by_len[length].add(sub)

    def get_probability(self, sequence_tuple):
        length = len(sequence_tuple)
        if length == 0 or length > self.max_len: return 0.0
        count = self.counts.get(sequence_tuple, 0)
        num_unique = len(self.unique_patterns_by_len[length])
        if num_unique == 0: return 0.0
        total_occurrences = self.total_counts_by_len[length]
        avg_occurrence = total_occurrences / num_unique
        if avg_occurrence == 0: return 0.0
        return count / avg_occurrence

# ==========================================
# 3. Trader Class
# ==========================================
class SequenceTrader:
    def __init__(self, max_seq_len=5, edit_depth=1, n_categories=20):
        self.max_seq_len = max_seq_len
        self.edit_depth = edit_depth
        self.n_categories = n_categories
        self.cat_model = SequenceModel(max_seq_len)
        self.dir_model = SequenceModel(max_seq_len)
        self.bin_edges = None
        self.unique_dirs = set()
        self.equity = [1000.0]
        
    def fit(self, train_prices):
        print(f"Training on {len(train_prices)} bars...")
        _, self.bin_edges = pd.qcut(train_prices, self.n_categories, retbins=True, duplicates='drop')
        train_cats = pd.cut(train_prices, bins=self.bin_edges, labels=False, include_lowest=True)
        train_cats = np.nan_to_num(train_cats, nan=0).astype(int)
        train_dirs = [0] * len(train_cats)
        for i in range(1, len(train_cats)):
            train_dirs[i] = train_cats[i] - train_cats[i-1]
        self.unique_dirs = set(train_dirs)
        self.cat_model.train(list(train_cats))
        self.dir_model.train(train_dirs)
        print("Training Complete.")

    def discretize_single_window(self, prices):
        cats = pd.cut(prices, bins=self.bin_edges, labels=False, include_lowest=True)
        max_cat = len(self.bin_edges) - 2 
        cats = np.nan_to_num(cats, nan=0)
        cats = np.clip(cats, 0, max_cat).astype(int)
        dirs = [0] * len(cats)
        for i in range(1, len(cats)):
            dirs[i] = cats[i] - cats[i-1]
        return list(cats), dirs

    def generate_edits(self, seq, alphabet):
        candidates = [{'seq': seq, 'type': 'original', 'meta': None}]
        if self.edit_depth < 1: return candidates
        seq_len = len(seq)
        for i in range(seq_len + 1):
            for token in alphabet:
                new_seq = seq[:i] + (token,) + seq[i:]
                candidates.append({'seq': new_seq, 'type': 'insert', 'meta': (i, token)})
        for i in range(seq_len):
            new_seq = seq[:i] + seq[i+1:]
            candidates.append({'seq': new_seq, 'type': 'remove', 'meta': i})
        for i in range(seq_len):
            for token in alphabet:
                if token != seq[i]:
                    new_seq = seq[:i] + (token,) + seq[i+1:]
                    candidates.append({'seq': new_seq, 'type': 'swap', 'meta': (i, token)})
        return candidates

    def get_best_variation(self, input_seq, model, alphabet):
        variations = self.generate_edits(input_seq, alphabet)
        best_var = None
        best_prob = -1.0
        for var in variations:
            prob = model.get_probability(var['seq'])
            if prob > best_prob:
                best_prob = prob
                best_var = var
        return best_var, best_prob

    def predict(self, window_prices):
        cat_seq, dir_seq = self.discretize_single_window(window_prices)
        input_len = self.max_seq_len - 1
        if len(cat_seq) < input_len: return 0
        input_c = tuple(cat_seq[-input_len:])
        input_d = tuple(dir_seq[-input_len:])
        alph_c = list(range(len(self.bin_edges)-1))
        alph_d = list(self.unique_dirs)
        
        best_c, prob_c = self.get_best_variation(input_c, self.cat_model, alph_c)
        best_d, prob_d = self.get_best_variation(input_d, self.dir_model, alph_d)
        
        signal_score = 0
        c_is_ext = (best_c['type'] == 'insert' and best_c['meta'][0] == len(input_c))
        d_is_ext = (best_d['type'] == 'insert' and best_d['meta'][0] == len(input_d))
        combined_prob = prob_c + prob_d
        
        if c_is_ext:
            pred_cat = best_c['meta'][1]
            curr_cat = input_c[-1]
            if pred_cat > curr_cat: signal_score += combined_prob
            elif pred_cat < curr_cat: signal_score -= combined_prob
            
        if d_is_ext:
            pred_dir = best_d['meta'][1]
            if pred_dir > 0: signal_score += combined_prob
            elif pred_dir < 0: signal_score -= combined_prob
            
        if signal_score > 2.5: return 1
        if signal_score < -2.5: return -1
        return 0

    def run_backtest(self, prices):
        n = len(prices)
        split_idx = int(n * 0.5)
        train_data = prices[:split_idx]
        test_data = prices[split_idx:] # Not strictly used, just for logging
        
        self.fit(train_data)
        print(f"Testing on {len(test_data)} bars (Bars {split_idx} to {n})...")
        
        position = 0
        entry_price = 0
        test_indices = range(split_idx, n-1)
        
        for t in test_indices:
            window_start = t - self.max_seq_len
            window = prices[window_start : t + 1]
            sig = self.predict(window)
            curr_p = prices[t]
            if position != 0:
                pnl = (curr_p - entry_price) * position
                self.equity.append(self.equity[-1] + pnl)
                position = 0
            else:
                self.equity.append(self.equity[-1])
            if sig != 0:
                position = sig
                entry_price = curr_p
        return self.equity

# ==========================================
# 4. GitHub Upload Logic
# ==========================================
def upload_plot_to_github(filename, repo, token, branch="main"):
    print(f"Preparing to upload {filename} to {repo}...")
    
    with open(filename, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    url = f"https://api.github.com/repos/{repo}/contents/{filename}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Check if file exists to get SHA (needed for update/overwrite)
    check_response = requests.get(url, headers=headers)
    sha = None
    if check_response.status_code == 200:
        sha = check_response.json().get("sha")
        print("File exists, overwriting...")

    data = {
        "message": f"Update equity plot: {filename}",
        "content": encoded_string,
        "branch": branch
    }
    if sha:
        data["sha"] = sha

    response = requests.put(url, headers=headers, data=json.dumps(data))
    
    if response.status_code in [200, 201]:
        print("Successfully uploaded to GitHub.")
    else:
        print(f"Failed to upload. Status: {response.status_code}")
        print(response.text)

# ==========================================
# 5. Main Execution
# ==========================================
def main():
    # 0. Load Environment Variables
    load_env()
    
    # 1. Generate Data
    df = generate_price_data(n=2000)
    prices = df['price'].values
    
    # 2. Run Backtest
    trader = SequenceTrader(max_seq_len=4, edit_depth=1, n_categories=20)
    equity = trader.run_backtest(prices)
    
    # 3. Stats
    eq_series = pd.Series(equity)
    sharpe = 0.0
    if len(eq_series) > 1 and eq_series.pct_change().std() > 0:
        sharpe = (eq_series.pct_change().mean() / eq_series.pct_change().std()) * np.sqrt(252)
        
    print("-" * 30)
    print(f"Final Equity: ${equity[-1]:.2f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print("-" * 30)

    # 4. Create Low-Res Plot
    plt.figure(figsize=(10, 6))
    plt.plot(equity, label='Strategy Equity')
    plt.title(f'Sequence Trader Results (Sharpe: {sharpe:.2f})')
    plt.xlabel('Trades')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    
    plot_filename = "equity_curve.png"
    # Low resolution: 50 DPI
    plt.savefig(plot_filename, dpi=50) 
    print(f"Plot saved locally as {plot_filename} (Low Res)")
    
    # 5. Upload to GitHub
    pat = os.environ.get("PAT")
    repo = "constantinbender51-cmyk/Models"
    
    if pat:
        try:
            upload_plot_to_github(plot_filename, repo, pat)
        except Exception as e:
            print(f"Error during upload: {e}")
    else:
        print("Warning: 'PAT' not found in environment variables or .env file. Skipping upload.")

if __name__ == "__main__":
    main()