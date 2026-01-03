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

    def fetch_candles(self, symbol="SPY", period="5y"): # Reduced period for speed
        """Fetches dataset."""
        print(f"Fetching {period} of data for {symbol} via Yahoo Finance...")
        
        try:
            df = yf.download(symbol, period=period, interval="1d", progress=False)
            if df.empty:
                print("Error: No data found.")
                return []
                
            # Handle different yfinance return formats (MultiIndex vs Standard)
            if 'Close' in df.columns:
                if isinstance(df.columns, pd.MultiIndex):
                    # For newer yfinance versions where columns are (Price, Ticker)
                    prices = df['Close'][symbol].tolist()
                else:
                    prices = df['Close'].values.flatten().tolist()
            else:
                prices = df.iloc[:, 0].tolist()
        except Exception as e:
            print(f"Data fetch error: {e}")
            return []
            
        prices = [p for p in prices if not math.isnan(p)]
        print(f"Loaded {len(prices)} data points.")
        return prices

    def discretize_sequence(self, prices, num_bins=40):
        if not prices or len(prices) < 2: return []

        # Dynamic normalization based on the window provided
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

    def sequence_to_prices(self, sequence):
        """Converts the categorical sequence back to price levels."""
        return [self.last_min + (cat * self.last_step) for _, cat in sequence]

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
        
        # Reset counts with Laplace smoothing (start at 0.1)
        mom_counts = np.ones((3, 3)) * 0.1
        cat_counts = np.ones((self.num_categories, self.num_categories)) * 0.1
        mom_start_counts = np.ones(3) * 0.1
        cat_start_counts = np.ones(self.num_categories) * 0.1

        for i in range(len(sequence) - 1):
            curr_mom, curr_cat = sequence[i]
            next_mom, next_cat = sequence[i+1]
            
            curr_m_idx = self.mom_map[curr_mom]
            next_m_idx = self.mom_map[next_mom]
            
            # categories are 1-based, so subtract 1 for index
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
        if first_idx >= len(start) or first_idx < 0: return -50.0 # Penalty for out of bounds
        
        log_prob = math.log(start[first_idx] + 1e-9)
        
        for i in range(len(seq)-1):
            curr = seq[i]
            next_val = seq[i+1]
            
            if curr >= len(trans) or next_val >= len(trans) or curr < 0 or next_val < 0:
                log_prob += -50.0 # Heavy penalty for invalid transitions
            else:
                prob = trans[curr][next_val]
                log_prob += math.log(prob + 1e-9)
                
        return log_prob

    def calibrate(self):
        """
        Runs Monte Carlo simulations to determine what a 'normal' probability
        looks like for sequences of different lengths.
        """
        print("Calibrating Z-Score baselines...")
        # We calibrate for lengths 2 to 30
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
        # Fallback for uncalibrated lengths (extrapolate)
        if length not in self.stats['mom']: 
            return -2.0 # Assume slightly improbable if too long/short
        
        m_idxs = [self.mom_map[m] for m, c in sequence]
        c_idxs = [c-1 for m, c in sequence]
        
        lp_m = self._get_log_prob(m_idxs, self.mom_trans, self.mom_start)
        lp_c = self._get_log_prob(c_idxs, self.cat_trans, self.cat_start)
        
        mu = self.stats['mom'][length]['mu'] + self.stats['cat'][length]['mu']
        std = math.sqrt(self.stats['mom'][length]['var'] + self.stats['cat'][length]['var'])
        
        if std == 0: return 0
        return (lp_m + lp_c - mu) / std

# ==============================================================================
# 3. SEQUENCE MUTATOR (The Editor)
# ==============================================================================

class SequenceEditor:
    def __init__(self, engine, edit_penalty=0.4):
        self.engine = engine
        self.edit_penalty = edit_penalty # Cost per edit (Standard Deviations)

    def repair_momentum(self, sequence):
        """Ensures the (UP/DOWN/FLAT) labels match the actual category numbers."""
        if not sequence: return []
        repaired = [sequence[0]] # Keep start momentum as is usually
        for i in range(1, len(sequence)):
            prev_mom, prev_cat = repaired[-1]
            curr_mom, curr_cat = sequence[i]
            
            if curr_cat > prev_cat: mom = 'UP'
            elif curr_cat < prev_cat: mom = 'DOWN'
            else: mom = 'FLAT' # or keep prev_mom if desired
            
            repaired.append((mom, curr_cat))
        return repaired

    def mutate(self, sequence):
        """Randomly alters the sequence."""
        seq = copy.deepcopy(sequence)
        if len(seq) < 2: return seq
        
        op = random.random()
        idx = random.randint(0, len(seq) - 1)
        
        # 1. NUDGE VALUE (Modify) - 60% chance
        if op < 0.6:
            _, cat = seq[idx]
            # +/- 1 or 2 steps
            change = random.choice([-1, 1, -1, 1, -2, 2])
            new_cat = max(1, min(self.engine.num_categories, cat + change))
            seq[idx] = (seq[idx][0], new_cat)
            
        # 2. INSERT/EXTEND (Add) - 20% chance
        elif op < 0.8:
            # If idx is end, it's a prediction/extension
            # If idx is middle, it's an interpolation
            _, cat = seq[idx]
            seq.insert(idx + 1, ('FLAT', cat)) # Dupe value, physics will fix momentum
            
        # 3. DELETE (Remove) - 20% chance
        else:
            if len(seq) > 2: # Don't delete if too short
                del seq[idx]

        return self.repair_momentum(seq)

    def solve(self, input_sequence, iterations=300):
        """
        Hill Climbing Search.
        Input: Noisy/Gibberish sequence.
        Output: High probability sequence close to input.
        """
        current_seq = self.repair_momentum(input_sequence)
        current_edits = 0
        
        # Calculate initial score
        # Score = Z_Score - (Edit_Count * Penalty)
        # We assume initial input has 0 edits.
        current_z = self.engine.get_z_score(current_seq)
        current_score = current_z
        
        best_seq = current_seq
        best_score = current_score
        
        # We track edits relative to the original input length to avoid 
        # infinite growth or shrinking loops just for score hacking.
        original_len = len(input_sequence)

        for i in range(iterations):
            # Generate a mutation of the CURRENT best (local search)
            candidate_seq = self.mutate(current_seq)
            
            # Heuristic for Edit Distance (approximate)
            # We penalize length deviation and values that differ from original indices
            # Ideally this is Levenshtein, but for this simpler version:
            # We just apply a flat penalty for every mutation accepted.
            
            cand_z = self.engine.get_z_score(candidate_seq)
            
            # Simple cooling schedule or fixed penalty
            # We want to maximize Z-Score, but pay for edits.
            # Since we are mutating 'current_seq', we are effectively walking away from input.
            # We need to decide if the new step is worth it.
            
            # Compare candidate to CURRENT
            # If Z-score improves significantly, take the step.
            
            z_diff = cand_z - current_z
            
            # Acceptance threshold: 
            # We need the Z-score to improve by at least 'edit_penalty' to justify a change.
            if z_diff > self.edit_penalty:
                current_seq = candidate_seq
                current_z = cand_z
                current_score = cand_z # We reset score tracking to the new baseline
                
                # Keep track of absolute best found
                if current_z > best_score:
                    best_seq = candidate_seq
                    best_score = current_z
            
            # Random Restart (Simulated Annealing-lite)
            # Occasionally revert to best found to prevent getting stuck in a bad local max
            if i % 50 == 0:
                current_seq = best_seq
                current_z = best_score

        return best_seq, best_score

# ==============================================================================
# 4. RUNNER
# ==============================================================================

if __name__ == "__main__":
    
    # SETUP
    SYMBOL = "BTC-USD" # Crypto is noisier, good for testing correction
    PERIOD = "2y"
    NUM_BINS = 50
    
    # 1. Load Data
    market = MarketDataHandler()
    raw_prices = market.fetch_candles(SYMBOL, period=PERIOD)
    if not raw_prices: exit()

    full_seq = market.discretize_sequence(raw_prices, num_bins=NUM_BINS)
    
    # 2. Train Engine (Learn "Spelling" Rules of the Market)
    split = int(len(full_seq) * 0.8)
    train_data = full_seq[:split]
    test_data = full_seq[split:]
    
    engine = ZScoreEngine(num_categories=NUM_BINS)
    engine.train(train_data)
    engine.calibrate()
    
    # 3. The Corrector
    # Lower penalty = more creativity/editing. Higher penalty = strict adherence to input.
    corrector = SequenceEditor(engine, edit_penalty=0.15) 
    
    print("\n" + "="*60)
    print(f"RUNNING CORRECTION DEMO (Input Length: 10)")
    print("="*60)
    
    # Take random slices of the test data and try to "correct" them
    for i in range(0, len(test_data) - 10, 20):
        # We take a real slice, maybe add some artificial noise?
        input_slice = test_data[i : i+10]
        
        # Let's add artificial noise to see if it fixes it
        noisy_slice = copy.deepcopy(input_slice)
        # Introduce a massive spike in the middle (Gibberish)
        noisy_slice[5] = ('UP', min(NUM_BINS, noisy_slice[5][1] + 15)) 
        noisy_slice = corrector.repair_momentum(noisy_slice)

        # Run Correction
        corrected_slice, score = corrector.solve(noisy_slice, iterations=400)
        
        # Format output
        input_vals = [x[1] for x in noisy_slice]
        output_vals = [x[1] for x in corrected_slice]
        
        print(f"INPUT (Noisy):     {input_vals}")
        print(f"OUTPUT (Corrected):{output_vals}")
        
        # Check diff
        if len(output_vals) > len(input_vals):
            print(f"--> Extended by {len(output_vals)-len(input_vals)} steps")
        elif len(output_vals) < len(input_vals):
            print(f"--> Shortened by {len(input_vals)-len(output_vals)} steps")
            
        print("-" * 60)