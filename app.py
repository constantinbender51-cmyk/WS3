import yfinance as yf
import time
import math
import numpy as np
import random
import copy
import pandas as pd

# Override print for railway/console flushing
_builtin_print = print
def print(*args, **kwargs):
    _builtin_print(*args, **kwargs)

# ==============================================================================
# 1. DATA HANDLER
# ==============================================================================

class MarketDataHandler:
    def __init__(self):
        self.last_min = 0
        self.last_step = 0

    def fetch_candles(self, symbol="SPY", period="10y"):
        """Fetches dataset."""
        print(f"Fetching {period} of data for {symbol} via Yahoo Finance...")
        
        try:
            df = yf.download(symbol, period=period, interval="1d", progress=False)
            if df.empty:
                print("Error: No data found.")
                return []
                
            # Robust data extraction for different yfinance versions
            prices = []
            if 'Close' in df.columns:
                if isinstance(df.columns, pd.MultiIndex):
                    # MultiIndex case: ('Close', 'SPY')
                    try:
                        prices = df['Close'][symbol].dropna().tolist()
                    except KeyError:
                        prices = df['Close'].iloc[:, 0].dropna().tolist()
                else:
                    # Standard case
                    prices = df['Close'].dropna().tolist()
            else:
                # Fallback
                prices = df.iloc[:, 0].dropna().tolist()
                
        except Exception as e:
            print(f"Data fetch error: {e}")
            return []
            
        print(f"Loaded {len(prices)} data points.")
        return prices

    def discretize_sequence(self, prices, num_bins=40):
        if not prices or len(prices) < 2: return []

        # Dynamic normalization
        min_p = min(prices) * 0.95
        max_p = max(prices) * 1.05
        price_range = max_p - min_p
        step = price_range / num_bins
        
        self.last_min = min_p
        self.last_step = step

        sequence = []
        prev_cat = -1
        
        for i, p in enumerate(prices):
            cat = int((p - min_p) / step)
            cat = max(1, min(num_bins, cat))
            
            # Determine momentum (UP/DOWN/FLAT)
            if i == 0:
                mom = 'FLAT'
            else:
                if cat > prev_cat: mom = 'UP'
                elif cat < prev_cat: mom = 'DOWN'
                else: mom = 'FLAT' 
            
            prev_cat = cat
            sequence.append((mom, cat))
            
        return sequence

# ==============================================================================
# 2. PROBABILISTIC ENGINE (The "Grammar" Checker)
# ==============================================================================

class ZScoreEngine:
    def __init__(self, num_categories=40):
        self.num_categories = num_categories
        self.mom_map = {k: v for v, k in enumerate(['UP', 'DOWN', 'FLAT'])}
        
        # Transition Matrices (Markov Chain)
        self.mom_trans = np.ones((3, 3)) 
        self.mom_start = np.ones(3) 
        self.cat_trans = np.ones((num_categories, num_categories)) 
        self.cat_start = np.ones(num_categories) 
        
        # Statistics for Z-Score Normalization
        self.stats = {'mom': {}, 'cat': {}}

    def train(self, sequence):
        print("Training Markov chains on historical physics...")
        
        # Reset counts with Laplace smoothing
        mom_counts = np.ones((3, 3)) * 0.1
        cat_counts = np.ones((self.num_categories, self.num_categories)) * 0.1
        mom_start_counts = np.ones(3) * 0.1
        cat_start_counts = np.ones(self.num_categories) * 0.1

        for i in range(len(sequence) - 1):
            curr_mom, curr_cat = sequence[i]
            next_mom, next_cat = sequence[i+1]
            
            curr_m_idx = self.mom_map[curr_mom]
            next_m_idx = self.mom_map[next_mom]
            
            curr_c_idx = curr_cat - 1
            next_c_idx = next_cat - 1
            
            if 0 <= curr_c_idx < self.num_categories and 0 <= next_c_idx < self.num_categories:
                cat_counts[curr_c_idx][next_c_idx] += 1
                cat_start_counts[curr_c_idx] += 1
            
            mom_counts[curr_m_idx][next_m_idx] += 1
            mom_start_counts[curr_m_idx] += 1

        # Normalize to probabilities
        self.mom_trans = mom_counts / mom_counts.sum(axis=1, keepdims=True)
        self.cat_trans = cat_counts / cat_counts.sum(axis=1, keepdims=True)
        self.mom_start = mom_start_counts / mom_start_counts.sum()
        self.cat_start = cat_start_counts / cat_start_counts.sum()
        
        print("Training Complete.")

    def _get_log_prob(self, seq, trans, start):
        if not seq: return 0.0
        
        first_idx = seq[0]
        if first_idx >= len(start) or first_idx < 0: return -50.0 
        
        log_prob = math.log(start[first_idx] + 1e-9)
        
        for i in range(len(seq)-1):
            curr = seq[i]
            next_val = seq[i+1]
            
            if curr >= len(trans) or next_val >= len(trans) or curr < 0 or next_val < 0:
                log_prob += -50.0 
            else:
                prob = trans[curr][next_val]
                log_prob += math.log(prob + 1e-9)
                
        return log_prob

    def calibrate(self):
        """Calibrates 'Normal' probability for lengths 2 to 30."""
        print("Calibrating Z-Score baselines...")
        for length in range(2, 31): 
            m_logs, c_logs = [], []
            for _ in range(100): 
                # Simulate Momentum Path
                s_m = [np.random.choice(3, p=self.mom_start)]
                for _ in range(length-1): 
                    s_m.append(np.random.choice(3, p=self.mom_trans[s_m[-1]]))
                m_logs.append(self._get_log_prob(s_m, self.mom_trans, self.mom_start))
                
                # Simulate Category Path
                s_c = [np.random.choice(self.num_categories, p=self.cat_start)]
                for _ in range(length-1): 
                    s_c.append(np.random.choice(self.num_categories, p=self.cat_trans[s_c[-1]]))
                c_logs.append(self._get_log_prob(s_c, self.cat_trans, self.cat_start))
                
            self.stats['mom'][length] = {'mu': np.mean(m_logs), 'var': np.var(m_logs)}
            self.stats['cat'][length] = {'mu': np.mean(c_logs), 'var': np.var(c_logs)}

    def get_z_score(self, sequence):
        length = len(sequence)
        # Handle lengths outside calibration by clamping
        calib_len = max(2, min(length, 30))
        
        m_idxs = [self.mom_map[m] for m, c in sequence]
        c_idxs = [c-1 for m, c in sequence]
        
        lp_m = self._get_log_prob(m_idxs, self.mom_trans, self.mom_start)
        lp_c = self._get_log_prob(c_idxs, self.cat_trans, self.cat_start)
        
        mu = self.stats['mom'][calib_len]['mu'] + self.stats['cat'][calib_len]['mu']
        std = math.sqrt(self.stats['mom'][calib_len]['var'] + self.stats['cat'][calib_len]['var'])
        
        if std == 0: return 0
        return (lp_m + lp_c - mu) / std

# ==============================================================================
# 3. SEQUENCE MUTATOR (The Editor)
# ==============================================================================

class SequenceEditor:
    def __init__(self, engine, edit_penalty=0.4):
        self.engine = engine
        self.edit_penalty = edit_penalty 

    def repair_momentum(self, sequence):
        """Ensures the (UP/DOWN/FLAT) labels match the actual category numbers."""
        if not sequence: return []
        repaired = [sequence[0]] 
        for i in range(1, len(sequence)):
            prev_mom, prev_cat = repaired[-1]
            curr_mom, curr_cat = sequence[i]
            
            if curr_cat > prev_cat: mom = 'UP'
            elif curr_cat < prev_cat: mom = 'DOWN'
            else: mom = 'FLAT'
            
            repaired.append((mom, curr_cat))
        return repaired

    def mutate(self, sequence):
        """Randomly alters the sequence (Edit, Insert, Delete)."""
        seq = copy.deepcopy(sequence)
        if len(seq) < 2: return seq
        
        op = random.random()
        idx = random.randint(0, len(seq) - 1)
        
        # 1. NUDGE VALUE (Modify) - 60% chance
        if op < 0.6:
            _, cat = seq[idx]
            change = random.choice([-1, 1, -2, 2])
            new_cat = max(1, min(self.engine.num_categories, cat + change))
            seq[idx] = (seq[idx][0], new_cat)
            
        # 2. INSERT/EXTEND (Add) - 20% chance
        elif op < 0.8:
            _, cat = seq[idx]
            # Insert a duplicate of the current value next to it
            seq.insert(idx + 1, ('FLAT', cat)) 
            
        # 3. DELETE (Remove) - 20% chance
        else:
            if len(seq) > 2: 
                del seq[idx]

        return self.repair_momentum(seq)

    def solve(self, input_sequence, iterations=300):
        """
        Stochastic Hill Climbing to 'Spell Check' the sequence.
        """
        current_seq = self.repair_momentum(input_sequence)
        
        # Initial Score
        current_z = self.engine.get_z_score(current_seq)
        
        best_seq = current_seq
        best_score = current_z
        
        for i in range(iterations):
            # Try a mutation
            candidate_seq = self.mutate(current_seq)
            cand_z = self.engine.get_z_score(candidate_seq)
            
            # Improvement calculation
            z_diff = cand_z - current_z
            
            # Acceptance Threshold (Must improve score by margin to justify edit)
            if z_diff > self.edit_penalty:
                current_seq = candidate_seq
                current_z = cand_z
                
                if current_z > best_score:
                    best_seq = candidate_seq
                    best_score = current_z
            
            # Random Restart prevents getting stuck in local maxima
            if i % 50 == 0:
                current_seq = best_seq
                current_z = best_score

        return best_seq

# ==============================================================================
# 4. RUNNER
# ==============================================================================

if __name__ == "__main__":
    
    SYMBOL = "SPY" 
    PERIOD = "5y"
    NUM_BINS = 50
    
    # 1. Load Data
    market = MarketDataHandler()
    raw_prices = market.fetch_candles(SYMBOL, period=PERIOD)
    if not raw_prices: exit()

    full_seq = market.discretize_sequence(raw_prices, num_bins=NUM_BINS)
    
    # 2. Train Engine 
    split = int(len(full_seq) * 0.8)
    train_data = full_seq[:split]
    test_data = full_seq[split:]
    
    engine = ZScoreEngine(num_categories=NUM_BINS)
    engine.train(train_data)
    engine.calibrate()
    
    # 3. The Corrector
    # edit_penalty: Cost of making a change. 
    # High = Strict (changes must be VERY good). Low = Creative (changes can be minor).
    corrector = SequenceEditor(engine, edit_penalty=0.15) 
    
    print("\n" + "="*60)
    print(f"MARKET SPELL CHECKER DEMO ({SYMBOL})")
    print("="*60)
    
    # Test on a few random slices
    for i in range(0, len(test_data) - 10, 50):
        # Grab a real slice
        input_slice = test_data[i : i+10]
        
        # --- Inject Artificial "Typos" (Noise) ---
        noisy_slice = copy.deepcopy(input_slice)
        
        # Typo 1: Add a massive unrealistic spike in the middle
        mid_idx = 5
        noisy_slice[mid_idx] = ('UP', min(NUM_BINS, noisy_slice[mid_idx][1] + 10))
        noisy_slice = corrector.repair_momentum(noisy_slice)

        # Run Correction
        corrected_slice = corrector.solve(noisy_slice, iterations=500)
        
        # --- Output ---
        input_vals = [x[1] for x in noisy_slice]
        output_vals = [x[1] for x in corrected_slice]
        
        print(f"INPUT (With Typo): {input_vals}")
        print(f"OUTPUT (Edited):   {output_vals}")
        
        if len(output_vals) != len(input_vals):
             print(f"--> Length Changed: {len(input_vals)} -> {len(output_vals)}")

        # Check if the spike was removed/smoothed
        original_val = input_slice[mid_idx][1]
        typo_val = input_vals[mid_idx]
        try:
            # Note: Index might shift if deletions occurred, simplified check
            corrected_val = output_vals[mid_idx] 
            print(f"--> Value at Index 5: Real={original_val}, Typo={typo_val}, Fixed={corrected_val}")
        except IndexError:
            print("--> Index 5 was deleted by editor.")
            
        print("-" * 60)