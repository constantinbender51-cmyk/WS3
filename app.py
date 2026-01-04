import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import requests
import base64
import json
import yfinance as yf
import time
import math
import itertools

# ==========================================
# 0. Configuration & Parameters
# ==========================================
CONFIG = {
    # Data Fetching
    "TICKER": "SPY",
    "START_DATE": "2020-01-01",
    "END_DATE": "2025-01-01",
    
    # Strategy / Model Parameters (Defaults, will be overwritten by Grid Search)
    "MAX_SEQ_LEN": 4,        
    "EDIT_DEPTH": 1,         
    "N_CATEGORIES": 20,      
    "SIGNAL_THRESHOLD": 2.5, 
    
    # Debugging & Output
    "DEBUG_PRINTS": 1000000, 
    "PRINT_DELAY": 0.05,     
    
    # Grid Search
    "ENABLE_GRID_SEARCH": True,
    
    # Plotting
    "PLOT_FILENAME_EQUITY": "equity_curve.png",
    "PLOT_FILENAME_PRICE": "price_categories.png",
    "PLOT_DPI": 100,         
    
    # GitHub Upload
    "GITHUB_REPO": "constantinbender51-cmyk/Models",
    "GITHUB_BRANCH": "main"
}

# ==========================================
# 0.1 Helper: Load .env manually
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
# 0.2 Helper: Slow Print
# ==========================================
def slow_print(text, delay=CONFIG["PRINT_DELAY"]):
    print(text)
    time.sleep(delay)

# ==========================================
# 1. Data Fetching (Real Data)
# ==========================================
def fetch_price_data(ticker, start, end):
    print(f"Downloading data for {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}.")

    price_series = None
    if isinstance(df.columns, pd.MultiIndex):
        try:
            price_series = df.xs('Adj Close', axis=1, level=0)
        except KeyError:
            try:
                price_series = df.xs('Close', axis=1, level=0)
            except KeyError:
                pass
        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.iloc[:, 0]
    else:
        if 'Adj Close' in df.columns:
            price_series = df['Adj Close']
        elif 'Close' in df.columns:
            price_series = df['Close']

    if price_series is None:
         raise ValueError("Could not locate price data.")

    clean_df = pd.DataFrame({'price': price_series.values})
    clean_df.dropna(inplace=True)
    return clean_df

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
    def __init__(self, max_seq_len, edit_depth, n_categories, signal_threshold, debug_limit):
        self.max_seq_len = max_seq_len
        self.edit_depth = edit_depth
        self.n_categories = n_categories
        self.signal_threshold = signal_threshold
        self.debug_limit = debug_limit
        
        self.cat_model = SequenceModel(max_seq_len)
        self.dir_model = SequenceModel(max_seq_len)
        
        self.train_min = None
        self.train_max = None
        self.bin_width = None
        self.bin_edges = None 
        
        self.unique_dirs = set()
        self.equity = [1000.0]
        self.debug_count = 0
        
    def fit(self, train_prices):
        # Determine Min/Max and Bin Width
        self.train_min = np.min(train_prices)
        self.train_max = np.max(train_prices)
        price_range = self.train_max - self.train_min
        if price_range == 0: price_range = 1.0
        
        self.bin_width = price_range / self.n_categories
        self.bin_edges = [self.train_min + i * self.bin_width for i in range(self.n_categories + 1)]
        
        # Discretize Training Data
        train_cats_raw = np.floor((train_prices - self.train_min) / self.bin_width).astype(int)
        train_cats = np.clip(train_cats_raw, 0, self.n_categories - 1)
        
        train_dirs = [0] * len(train_cats)
        for i in range(1, len(train_cats)):
            train_dirs[i] = train_cats[i] - train_cats[i-1]
            
        self.unique_dirs = set(train_dirs)
        self.cat_model.train(list(train_cats))
        self.dir_model.train(train_dirs)

    def discretize_single_window(self, prices):
        raw_cats = np.floor((prices - self.train_min) / self.bin_width).astype(int)
        cats = list(raw_cats)
        dirs = [0] * len(cats)
        for i in range(1, len(cats)):
            dirs[i] = cats[i] - cats[i-1]
        return cats, dirs

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
        
        alph_c = list(range(self.n_categories))
        alph_d = list(self.unique_dirs)
        
        best_c, prob_c = self.get_best_variation(input_c, self.cat_model, alph_c)
        best_d, prob_d = self.get_best_variation(input_d, self.dir_model, alph_d)
        
        # Detect Extensions
        c_is_ext = (best_c['type'] == 'insert' and best_c['meta'][0] == len(input_c))
        d_is_ext = (best_d['type'] == 'insert' and best_d['meta'][0] == len(input_d))
        
        # --- Filtered Debug Print ---
        if (c_is_ext or d_is_ext) and self.debug_count < self.debug_limit:
            msg = f">>> PREDICTION EVENT {self.debug_count + 1} <<<\n"
            if c_is_ext:
                msg += f"  [Category] Predict: Append {best_c['meta'][1]} (Prob: {prob_c:.4f})\n"
            else:
                msg += f"  [Category] No Extension (Type: {best_c['type']})\n"
                
            if d_is_ext:
                msg += f"  [Directn]  Predict: Append {best_d['meta'][1]} (Prob: {prob_d:.4f})\n"
            else:
                msg += f"  [Directn]  No Extension (Type: {best_d['type']})\n"
            
            slow_print(msg)
            self.debug_count += 1
        # -------------------------------

        signal_score = 0
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
        
        if signal_score > self.signal_threshold: return 1
        if signal_score < -self.signal_threshold: return -1
        return 0

    def calculate_sharpe(self):
        eq_series = pd.Series(self.equity)
        if len(eq_series) < 2: return 0.0
        returns = eq_series.pct_change().dropna()
        if len(returns) < 2 or returns.std() == 0: return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    def run_backtest(self, prices):
        n = len(prices)
        split_idx = int(n * 0.5)
        train_data = prices[:split_idx]
        test_data = prices[split_idx:]
        
        self.fit(train_data)
        
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
# 4. Grid Search
# ==========================================
def perform_grid_search(train_data):
    """
    Optimizes params on the Training Data (which is further split 50/50 inside run_backtest).
    """
    print("="*40)
    print("STARTING GRID SEARCH")
    print("="*40)
    
    # Ranges
    r_len = range(2, 11)        # 2 to 10
    r_depth = range(1, 5)       # 1 to 4
    r_cats = range(10, 101, 10) # 10 to 100 in steps of 10
    
    best_sharpe = -999.0
    best_params = {}
    
    total_iterations = len(r_len) * len(r_depth) * len(r_cats)
    count = 0
    
    for length, depth, cats in itertools.product(r_len, r_depth, r_cats):
        count += 1
        
        # Initialize trader with debug=0 to be silent
        trader = SequenceTrader(
            max_seq_len=length,
            edit_depth=depth,
            n_categories=cats,
            signal_threshold=CONFIG["SIGNAL_THRESHOLD"],
            debug_limit=0 
        )
        
        # We pass the 'train_data' (which is the first 50% of the full dataset).
        # run_backtest will split THIS data into 50% train (quarter 1) and 50% val (quarter 2).
        # This provides a valid OOS metric for parameter selection.
        trader.run_backtest(train_data)
        sharpe = trader.calculate_sharpe()
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = {
                "MAX_SEQ_LEN": length,
                "EDIT_DEPTH": depth,
                "N_CATEGORIES": cats
            }
            print(f"[{count}/{total_iterations}] New Best: {best_params} -> Sharpe: {sharpe:.4f}")
        else:
            # Optional: Print progress every N steps
            if count % 20 == 0:
                print(f"[{count}/{total_iterations}] Current: L={length}, D={depth}, C={cats} -> Sharpe: {sharpe:.4f}")
                
    print("\n" + "="*40)
    print(f"GRID SEARCH COMPLETE. Best Params: {best_params} (Sharpe: {best_sharpe:.4f})")
    print("="*40 + "\n")
    
    return best_params

# ==========================================
# 5. GitHub Upload Logic
# ==========================================
def upload_plot_to_github(filename, repo, token, branch="main"):
    print(f"Preparing to upload {filename} to {repo}...")
    try:
        with open(filename, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"File {filename} not found, skipping upload.")
        return

    url = f"https://api.github.com/repos/{repo}/contents/{filename}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    
    check_response = requests.get(url, headers=headers)
    sha = None
    if check_response.status_code == 200:
        sha = check_response.json().get("sha")
        print(f"{filename} exists, overwriting...")

    data = {"message": f"Update plot: {filename}", "content": encoded_string, "branch": branch}
    if sha: data["sha"] = sha

    response = requests.put(url, headers=headers, data=json.dumps(data))
    if response.status_code in [200, 201]: print(f"Successfully uploaded {filename} to GitHub.")
    else: print(f"Failed to upload {filename}. Status: {response.status_code}")

# ==========================================
# 6. Main Execution
# ==========================================
def main():
    load_env()
    
    # 1. Fetch Data
    try:
        df = fetch_price_data(
            ticker=CONFIG["TICKER"], 
            start=CONFIG["START_DATE"], 
            end=CONFIG["END_DATE"]
        )
        prices = df['price'].values
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    print(f"Loaded {len(prices)} data points.")
    
    # 2. Grid Search
    if CONFIG["ENABLE_GRID_SEARCH"]:
        # Isolate the training portion (first 50%)
        train_len = int(len(prices) * 0.5)
        training_data_subset = prices[:train_len]
        
        best_params = perform_grid_search(training_data_subset)
        
        # Apply Best Params
        CONFIG.update(best_params)
    
    # 3. Final Backtest (on full data, testing on 2nd half)
    print("Running Final Backtest with Optimized Parameters...")
    trader = SequenceTrader(
        max_seq_len=CONFIG["MAX_SEQ_LEN"],
        edit_depth=CONFIG["EDIT_DEPTH"],
        n_categories=CONFIG["N_CATEGORIES"],
        signal_threshold=CONFIG["SIGNAL_THRESHOLD"],
        debug_limit=CONFIG["DEBUG_PRINTS"]
    )
    equity = trader.run_backtest(prices)
    sharpe = trader.calculate_sharpe()
        
    print("-" * 30)
    print(f"Final Equity: ${equity[-1]:.2f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print("-" * 30)

    # 4. Plots
    plt.figure(figsize=(10, 6))
    plt.plot(equity, label='Strategy Equity')
    plt.title(f'Sequence Trader Results (Sharpe: {sharpe:.2f})')
    plt.xlabel('Trades')
    plt.ylabel('Equity ($)')
    plt.legend(); plt.grid(True)
    plt.savefig(CONFIG["PLOT_FILENAME_EQUITY"], dpi=CONFIG["PLOT_DPI"]) 
    print(f"Equity plot saved as {CONFIG['PLOT_FILENAME_EQUITY']}")
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(prices, label='Price', color='black', linewidth=1)
    if trader.bin_edges is not None:
        for i, edge in enumerate(trader.bin_edges):
            plt.axhline(y=edge, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
    split_idx = int(len(prices) * 0.5)
    plt.axvline(x=split_idx, color='blue', linestyle='-', linewidth=2, label='Train/Test Split')
    plt.title(f'Price History vs {CONFIG["N_CATEGORIES"]} Equal-Width Bins')
    plt.xlabel('Bars (Days)'); plt.ylabel('Price'); plt.legend()
    plt.savefig(CONFIG["PLOT_FILENAME_PRICE"], dpi=CONFIG["PLOT_DPI"])
    print(f"Price plot saved as {CONFIG['PLOT_FILENAME_PRICE']}")
    plt.close()

    # 5. Upload
    pat = os.environ.get("PAT")
    repo = CONFIG["GITHUB_REPO"]
    if pat:
        upload_plot_to_github(CONFIG["PLOT_FILENAME_EQUITY"], repo, pat, CONFIG["GITHUB_BRANCH"])
        upload_plot_to_github(CONFIG["PLOT_FILENAME_PRICE"], repo, pat, CONFIG["GITHUB_BRANCH"])
    else:
        print("Warning: 'PAT' not found. Skipping upload.")

if __name__ == "__main__":
    main()