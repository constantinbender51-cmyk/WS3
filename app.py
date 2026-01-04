import yfinance as yf
import time
import math
import numpy as np
import pandas as pd
from collections import Counter
import itertools

# ==============================================================================
# 1. DATA HANDLER
# ==============================================================================

class MarketDataHandler:
    def __init__(self):
        self.last_min = 0
        self.last_step = 0

    def fetch_candles(self, symbol="SPY", period="5y"):
        print(f"--- Fetching {period} data for {symbol} ---")
        try:
            df = yf.download(symbol, period=period, interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    prices = df.xs('Close', level=0, axis=1)[symbol].tolist()
                except KeyError:
                    prices = df['Close'].iloc[:, 0].tolist()
            elif 'Close' in df.columns:
                prices = df['Close'].tolist()
            else:
                prices = df.iloc[:, 0].tolist()
            prices = [p for p in prices if not math.isnan(p)]
            if not prices: raise ValueError("Empty data")
            print(f"Loaded {len(prices)} candles.")
            return prices
        except Exception as e:
            print(f"Error fetching data: {e}")
            return []

    def discretize_sequence(self, prices, num_bins=40):
        if not prices or len(prices) < 2: return []
        min_p = min(prices) * 0.95
        max_p = max(prices) * 1.05
        step = (max_p - min_p) / num_bins
        
        self.last_min = min_p
        self.last_step = step

        sequence = []
        prev_cat = -1
        for i, p in enumerate(prices):
            cat = int((p - min_p) / step)
            cat = max(1, min(num_bins, cat))
            if i == 0: mom = 'FLAT'
            else:
                if cat > prev_cat: mom = 'UP'
                elif cat < prev_cat: mom = 'DOWN'
                else: mom = 'FLAT' 
            prev_cat = cat
            sequence.append((mom, cat))
        return sequence

# ==============================================================================
# 2. PROBABILISTIC ENGINE
# ==============================================================================

class ZScoreEngine:
    def __init__(self, num_categories=40):
        self.num_categories = num_categories
        self.mom_map = {k: v for v, k in enumerate(['UP', 'DOWN', 'FLAT'])}
        self.mom_trans = np.zeros((3, 3))
        self.mom_start = np.zeros(3)
        self.cat_trans = np.zeros((num_categories, num_categories))
        self.cat_start = np.zeros(num_categories)
        self.stats = {} 

    def train(self, sequence):
        print("Training Markov Probabilities...")
        mom_counts = np.ones((3, 3)) 
        cat_counts = np.ones((self.num_categories, self.num_categories)) 
        for i in range(len(sequence) - 1):
            curr_mom, curr_cat = sequence[i]
            next_mom, next_cat = sequence[i+1]
            m1, m2 = self.mom_map[curr_mom], self.mom_map[next_mom]
            c1, c2 = curr_cat - 1, next_cat - 1
            mom_counts[m1][m2] += 1
            cat_counts[c1][c2] += 1
            self.mom_start[m1] += 1
            self.cat_start[c1] += 1

        self.mom_trans = mom_counts / mom_counts.sum(axis=1, keepdims=True)
        self.cat_trans = cat_counts / cat_counts.sum(axis=1, keepdims=True)
        self.mom_start = self.mom_start / self.mom_start.sum()
        self.cat_start = self.cat_start / self.cat_start.sum()
        
    def _get_raw_log_prob(self, seq):
        if not seq: return -999.0
        m_idxs = [self.mom_map[m] for m, c in seq]
        c_idxs = [c-1 for m, c in seq]
        
        log_prob = 0.0
        log_prob += math.log(self.mom_start[m_idxs[0]] + 1e-9)
        log_prob += math.log(self.cat_start[c_idxs[0]] + 1e-9)
        for i in range(len(seq) - 1):
            log_prob += math.log(self.mom_trans[m_idxs[i]][m_idxs[i+1]] + 1e-9)
            log_prob += math.log(self.cat_trans[c_idxs[i]][c_idxs[i+1]] + 1e-9)
        return log_prob

    def calibrate(self, max_len=20, samples=2000):
        print(f"Calibrating baseline statistics (Max Len: {max_len})...")
        for length in range(1, max_len + 1):
            logs = []
            for _ in range(samples):
                m = np.random.choice(3, p=self.mom_start)
                c = np.random.choice(self.num_categories, p=self.cat_start)
                m_hist, c_hist = [m], [c]
                for _ in range(length - 1):
                    m = np.random.choice(3, p=self.mom_trans[m])
                    c = np.random.choice(self.num_categories, p=self.cat_trans[c])
                    m_hist.append(m)
                    c_hist.append(c)
                
                lp = math.log(self.mom_start[m_hist[0]] + 1e-9) + math.log(self.cat_start[c_hist[0]] + 1e-9)
                for i in range(length - 1):
                    lp += math.log(self.mom_trans[m_hist[i]][m_hist[i+1]] + 1e-9)
                    lp += math.log(self.cat_trans[c_hist[i]][c_hist[i+1]] + 1e-9)
                logs.append(lp)
            self.stats[length] = {'mu': np.mean(logs), 'std': np.std(logs)}

    def get_z_score(self, sequence):
        if not sequence: return 99.0
        L = len(sequence)
        if L not in self.stats: return 50.0 
        raw_lp = self._get_raw_log_prob(sequence)
        mu = self.stats[L]['mu']
        std = self.stats[L]['std']
        return 0 if std == 0 else -((raw_lp - mu) / std)

# ==============================================================================
# 3. EVOLUTIONARY SOLVER
# ==============================================================================

class EvolutionarySolver:
    def __init__(self, engine, num_bins):
        self.engine = engine
        self.num_bins = num_bins

    def _repair_physics(self, seq):
        repaired = []
        for i in range(len(seq)):
            cat = seq[i][1]
            if i == 0: mom = seq[i][0]
            else:
                prev_cat = repaired[-1][1]
                if cat > prev_cat: mom = 'UP'
                elif cat < prev_cat: mom = 'DOWN'
                else: mom = 'FLAT'
            repaired.append((mom, cat))
        return repaired

    def generate_mutations(self, seq):
        candidates = []
        L = len(seq)
        
        # 1. Modify Value
        for i in range(L):
            curr_mom, curr_cat = seq[i]
            if curr_cat < self.num_bins:
                candidates.append(seq[:i] + [(curr_mom, curr_cat + 1)] + seq[i+1:])
            if curr_cat > 1:
                candidates.append(seq[:i] + [(curr_mom, curr_cat - 1)] + seq[i+1:])

        # 2. Swap Adjacent
        for i in range(L - 1):
            new_seq = seq[:]
            new_seq[i], new_seq[i+1] = new_seq[i+1], new_seq[i]
            candidates.append(new_seq)
            
        # 3. Delete Index
        if L > 2:
            for i in range(L):
                candidates.append(seq[:i] + seq[i+1:])
                
        # 4. Insert (Average neighbor)
        if L < 15: # Prevent infinite growth in mutation phase
            for i in range(L):
                candidates.append(seq[:i] + [seq[i]] + seq[i:])

        return candidates

    def solve(self, input_seq, horizon_steps=2):
        # 1. BASE POOL: Start with the Input Sequence
        base_pool = [{'seq': input_seq, 'type': 'orig'}]
        
        # 2. SHORTENING: Generate all sub-sequences
        L = len(input_seq)
        for length in range(3, L): # Min length 3, up to L-1
            for i in range(L - length + 1):
                sub = input_seq[i : i+length]
                base_pool.append({'seq': sub, 'type': 'sub'})

        # 3. MUTATION: Add mutations of the original sequence
        # (Optional: could also mutate the sub-sequences, but that explodes complexity)
        muts = self.generate_mutations(input_seq)
        for m in muts:
            base_pool.append({'seq': m, 'type': 'mut'})

        # 4. EXTENSION (FORECASTING): 
        # Crucial update: We now try to extend EVERYTHING in the base pool.
        # This means we can predict future from full seq, OR predict future from a sub-seq.
        final_pool = []
        
        # Add the base items themselves (in case holding is the best option)
        final_pool.extend(base_pool)
        
        for item in base_pool:
            current_tips = [item['seq']]
            
            # Extend this specific candidate by N steps
            for _ in range(horizon_steps):
                new_tips = []
                for tip in current_tips:
                    last_cat = tip[-1][1]
                    # Try UP, DOWN, FLAT extensions
                    options = [last_cat]
                    if last_cat < self.num_bins: options.append(last_cat + 1)
                    if last_cat > 1: options.append(last_cat - 1)
                    
                    for opt in options:
                        new_tip = tip + [('FLAT', opt)] 
                        new_tips.append(new_tip)
                        
                        # Add to final pool
                        final_pool.append({'seq': new_tip, 'type': item['type'] + '+ext'})
                
                current_tips = new_tips # Continue extending from the new tips

        # 5. EVALUATE
        scored = []
        unique_hashes = set()

        for cand in final_pool:
            # Repair logic (ensure UP/DOWN text matches numbers)
            fixed_seq = self._repair_physics(cand['seq'])
            
            s_hash = str(fixed_seq)
            if s_hash in unique_hashes: continue
            unique_hashes.add(s_hash)
            
            z = self.engine.get_z_score(fixed_seq)
            scored.append({
                'seq': fixed_seq,
                'z': z,
                'type': cand['type']
            })

        # Lowest Z-score wins (Most "Normal"/Probable sequence)
        scored.sort(key=lambda x: x['z'])
        return scored[0]

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    SYMBOL = "SPY"
    PERIOD = "2y"
    NUM_BINS = 50
    WINDOW_SIZE = 8
    HORIZON = 2 
    
    market = MarketDataHandler()
    raw_data = market.fetch_candles(SYMBOL, PERIOD)
    if not raw_data: exit()

    full_seq = market.discretize_sequence(raw_data, num_bins=NUM_BINS)
    split = int(len(full_seq) * 0.8)
    train_seq = full_seq[:split]
    test_seq = full_seq[split:]
    
    engine = ZScoreEngine(num_categories=NUM_BINS)
    engine.train(train_seq)
    
    # Calibrate for a wide range of lengths (Shortest sub=3, Longest ext=Window+Horizon+Mutations)
    engine.calibrate(max_len=WINDOW_SIZE + HORIZON + 5)
    
    solver = EvolutionarySolver(engine, NUM_BINS)
    
    print(f"\n{'IDX':<5} | {'WINNER SEQUENCE (Cats)':<35} | {'ORIG LEN':<8} | {'NEW LEN':<8} | {'SOURCE':<10}")
    print("-" * 100)

    # Run last 50 for demo
    start_idx = max(0, len(test_seq) - WINDOW_SIZE - 50)
    
    for i in range(start_idx, len(test_seq) - WINDOW_SIZE - 1, 1):
        input_window = test_seq[i : i + WINDOW_SIZE]
        winner = solver.solve(input_window, horizon_steps=HORIZON)
        
        best_seq = winner['seq']
        orig_len = len(input_window)
        new_len = len(best_seq)
        
        # Display
        out_str = str([x[1] for x in best_seq])
        if len(out_str) > 33: out_str = "..." + out_str[-30:]
        
        print(f"{i:<5} | {out_str:<35} | {orig_len:<8} | {new_len:<8} | {winner['type']:<10}")