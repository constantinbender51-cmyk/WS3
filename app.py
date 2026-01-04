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
import random

# ==========================================
# 0. Configuration & Parameters
# ==========================================
CONFIG = {
    # Data Fetching
    "TICKER": "SPY",
    "START_DATE": "2020-01-01",
    "END_DATE": "2025-01-01",
    
    # Strategy / Model Parameters (Defaults, overwritten by Grid Search)
    "MAX_SEQ_LEN": 4,        
    "EDIT_DEPTH": 1,         
    "N_CATEGORIES": 20,      
    "SIGNAL_THRESHOLD": 2.5,
    "PREDICTION_BOOST": 1.5, # Multiplier for "Insert at End" probabilities
    
    # Debugging & Output
    "DEBUG_PRINTS": 0,       
    "PRINT_DELAY": 0.1,     
    
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
        self.train_min = np.min(train_prices)
        self.train_max = np.max(train_prices)
        price_range = self.train_max - self.train_min
        if price_range == 0: price_range = 1.0
        
        self.bin_width = price_range / self.n_categories
        self.bin_edges = [self.train_min + i * self.bin_width for i in range(self.n_categories + 1)]
        
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

    def derive_direction_sequence(self, cat_sequence):
        if not cat_sequence:
            return tuple()
        dirs = [0] * len(cat_sequence)
        for i in range(1, len(cat_sequence)):
            dirs[i] = cat_sequence[i] - cat_sequence[i-1]
        return tuple(dirs)

    def get_best_joint_variation(self, input_c, cat_model, dir_model, cat_alphabet):
        """
        1. Generate edits on CATEGORIES.
        2. Derive DIRECTIONS.
        3. Score = P(cat) + P(dir).
        Apply Boost only to extensions that predict a move.
        """
        cat_variations = self.generate_edits(input_c, cat_alphabet)
        best_var = None
        best_joint_prob = -1.0
        input_len = len(input_c)
        current_cat = input_c[-1] if input_len > 0 else None
        
        for var in cat_variations:
            seq_c = var['seq']
            prob_c = cat_model.get_probability(seq_c)
            seq_d = self.derive_direction_sequence(seq_c)
            prob_d = dir_model.get_probability(seq_d)
            
            # --- Selective Prediction Boost ---
            multiplier = 1.0
            is_extension = (var['type'] == 'insert' and var['meta'][0] == input_len)
            
            if is_extension:
                predicted_cat = var['meta'][1]
                # Only boost if the predicted cat represents a MOVE away from current cat
                if predicted_cat != current_cat:
                    multiplier = CONFIG["PREDICTION_BOOST"]
                
            joint_prob = (prob_c + prob_d) * multiplier
            
            if joint_prob > best_joint_prob:
                best_joint_prob = joint_prob
                var['derived_dir_seq'] = seq_d
                var['prob_c'] = prob_c
                var['prob_d'] = prob_d
                best_var = var
                
        return best_var, best_joint_prob

    def _analyze_window(self, window_prices):
        cat_seq, dir_seq = self.discretize_single_window(window_prices)
        input_len = self.max_seq_len - 1
        if len(cat_seq) < input_len: return None
        input_c = tuple(cat_seq[-input_len:])
        alph_c = list(range(self.n_categories))
        best_var, joint_prob = self.get_best_joint_variation(input_c, self.cat_model, self.dir_model, alph_c)
        return {"input_c": input_c, "best_var": best_var, "joint_prob": joint_prob}

    def predict(self, window_prices):
        data = self._analyze_window(window_prices)
        if not data: return 0
        best_var = data["best_var"]
        joint_prob = data["joint_prob"]
        input_c = data["input_c"]
        is_ext = (best_var['type'] == 'insert' and best_var['meta'][0] == len(input_c))
        
        if is_ext and self.debug_count < self.debug_limit:
            predicted_cat = best_var['meta'][1]
            msg = f">>> PREDICTION EVENT {self.debug_count + 1} <<<\n"
            msg += f"  Input: {input_c}\n"
            msg += f"  Predict Append: {predicted_cat}\n"
            msg += f"  Joint Score: {joint_prob:.4f} (P_c: {best_var['prob_c']:.2f} + P_d: {best_var['prob_d']:.2f})\n"
            slow_print(msg)
            self.debug_count += 1

        if is_ext:
            pred_cat = best_var['meta'][1]
            curr_cat = input_c[-1]
            if pred_cat > curr_cat:
                if joint_prob > self.signal_threshold: return 1
            elif pred_cat < curr_cat:
                if joint_prob > self.signal_threshold: return -1
        return 0

    def print_random_test_samples(self, prices, count=5):
        n = len(prices)
        split_idx = int(n * 0.5)
        test_range = range(split_idx, n - 1)
        if len(test_range) < count: return
        random_indices = random.sample(test_range, count)
        random_indices.sort()
        print("\n" + "="*60 + "\nRANDOM SAMPLES FROM TESTING SET (JOINT OPTIMIZATION)\n" + "="*60)
        for idx in random_indices:
            window_start = idx - self.max_seq_len
            if window_start < 0: continue
            window = prices[window_start : idx + 1]
            data = self._analyze_window(window)
            if not data: continue
            var = data['best_var']
            out_str = f"Append {var['meta'][1]}" if (var['type'] == 'insert' and var['meta'][0] == len(data['input_c'])) else f"{var['type']}"
            slow_print(f"Sample Index: {idx}\n  Input Cat:  {data['input_c']}\n  Result Cat: {var['seq']}\n  Result Dir: {var['derived_dir_seq']}\n  Operation:  {out_str}\n  Joint Score: {data['joint_prob']:.4f}\n" + "-"*60)

    def calculate_sharpe(self):
        eq_series = pd.Series(self.equity)
        if len(eq_series) < 2: return 0.0
        returns = eq_series.pct_change().dropna()
        if len(returns) < 2 or returns.std() == 0: return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    def run_backtest(self, prices):
        n = len(prices); split_idx = int(n * 0.5); train_data = prices[:split_idx]
        self.fit(train_data)
        position = 0; entry_price = 0; test_indices = range(split_idx, n-1)
        for t in test_indices:
            window_start = t - self.max_seq_len; window = prices[window_start : t + 1]
            sig = self.predict(window); curr_p = prices[t]
            if position != 0:
                self.equity.append(self.equity[-1] + (curr_p - entry_price) * position)
                position = 0
            else: self.equity.append(self.equity[-1])
            if sig != 0: position = sig; entry_price = curr_p
        return self.equity

# ==========================================
# 4. Grid Search
# ==========================================
def perform_grid_search(train_data):
    print("="*40 + "\nSTARTING GRID SEARCH\n" + "="*40)
    r_len = range(2, 11); r_depth = range(1, 5); r_cats = range(10, 101, 10); r_boost = np.linspace(1, 5, 10)
    best_sharpe = -999.0; best_params = {}
    total_iterations = len(r_len) * len(r_depth) * len(r_cats) * len(r_boost); count = 0
    for length, depth, cats, boost in itertools.product(r_len, r_depth, r_cats, r_boost):
        count += 1; CONFIG["PREDICTION_BOOST"] = boost
        trader = SequenceTrader(length, depth, cats, CONFIG["SIGNAL_THRESHOLD"], 0)
        trader.run_backtest(train_data); sharpe = trader.calculate_sharpe()
        if sharpe > best_sharpe:
            best_sharpe = sharpe; best_params = {"MAX_SEQ_LEN": length, "EDIT_DEPTH": depth, "N_CATEGORIES": cats, "PREDICTION_BOOST": boost}
            print(f"[{count}/{total_iterations}] New Best: {best_params} -> Sharpe: {sharpe:.4f}")
        elif count % 200 == 0: print(f"[{count}/{total_iterations}] Current: L={length}, D={depth}, C={cats}, B={boost:.1f} -> Sharpe: {sharpe:.4f}")
    return best_params

# ==========================================
# 5. GitHub Upload Logic
# ==========================================
def upload_plot_to_github(filename, repo, token, branch="main"):
    try:
        with open(filename, "rb") as f: encoded = base64.b64encode(f.read()).decode("utf-8")
    except: return
    url = f"https://api.github.com/repos/{repo}/contents/{filename}"; headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    resp = requests.get(url, headers=headers); sha = resp.json().get("sha") if resp.status_code == 200 else None
    data = {"message": f"Update plot: {filename}", "content": encoded, "branch": branch}
    if sha: data["sha"] = sha
    requests.put(url, headers=headers, data=json.dumps(data))

# ==========================================
# 6. Main Execution
# ==========================================
def main():
    load_env()
    try:
        df = fetch_price_data(CONFIG["TICKER"], CONFIG["START_DATE"], CONFIG["END_DATE"])
        prices = df['price'].values
    except Exception as e: print(e); return
    if CONFIG["ENABLE_GRID_SEARCH"]:
        best_params = perform_grid_search(prices[:int(len(prices)*0.5)])
        CONFIG.update(best_params)
    trader = SequenceTrader(CONFIG["MAX_SEQ_LEN"], CONFIG["EDIT_DEPTH"], CONFIG["N_CATEGORIES"], CONFIG["SIGNAL_THRESHOLD"], CONFIG["DEBUG_PRINTS"])
    equity = trader.run_backtest(prices)
    trader.print_random_test_samples(prices, count=5)
    sharpe = trader.calculate_sharpe()
    print(f"Final Equity: ${equity[-1]:.2f}\nSharpe Ratio: {sharpe:.4f}")
    plt.figure(figsize=(10, 6)); plt.plot(equity); plt.title(f'Equity (Sharpe: {sharpe:.2f})'); plt.savefig(CONFIG["PLOT_FILENAME_EQUITY"], dpi=CONFIG["PLOT_DPI"]); plt.close()
    plt.figure(figsize=(12, 8)); plt.plot(prices); plt.axvline(x=int(len(prices)*0.5), color='blue')
    if trader.bin_edges:
        for edge in trader.bin_edges: plt.axhline(y=edge, color='red', alpha=0.3, linewidth=0.5)
    plt.savefig(CONFIG["PLOT_FILENAME_PRICE"], dpi=CONFIG["PLOT_DPI"]); plt.close()
    pat = os.environ.get("PAT"); repo = CONFIG["GITHUB_REPO"]
    if pat:
        upload_plot_to_github(CONFIG["PLOT_FILENAME_EQUITY"], repo, pat, CONFIG["GITHUB_BRANCH"])
        upload_plot_to_github(CONFIG["PLOT_FILENAME_PRICE"], repo, pat, CONFIG["GITHUB_BRANCH"])

if __name__ == "__main__": main()