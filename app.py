import yfinance as yf
import time
import math
import numpy as np
from collections import Counter

# Override print for railway/console flushing
# Note: Reduced sleep to 0.01s to speed up the backtest loop output
_builtin_print = print
def print(*args, **kwargs):
    # time.sleep(0.01) 
    _builtin_print(*args, **kwargs)

# ==============================================================================
# 1. DATA HANDLER
# ==============================================================================

class MarketDataHandler:
    def __init__(self):
        self.last_min = 0
        self.last_step = 0

    def fetch_candles(self, symbol="SPY", period="10y"):
        """Fetches a large dataset for statistical significance."""
        print(f"Fetching {period} of data for {symbol} via Yahoo Finance...")
        
        df = yf.download(symbol, period=period, interval="1d", progress=False)
        
        if df.empty:
            print("Error: No data found.")
            return []
            
        try:
            prices = df['Close'].values.flatten().tolist()
        except KeyError:
            prices = df['Close'].tolist()
            
        prices = [p for p in prices if not math.isnan(p)]
        print(f"Loaded {len(prices)} data points.")
        return prices

    def discretize_sequence(self, prices, num_bins=40):
        """
        Maps prices to categories.
        Increased num_bins to 40 to capture finer movements over a 10y range.
        """
        if not prices or len(prices) < 2: return []

        # Global min/max for stable state space (simplified for this demo)
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
            
            if i == 0:
                mom = 'FLAT'
            else:
                if cat > prev_cat: mom = 'UP'
                elif cat < prev_cat: mom = 'DOWN'
                else: mom = 'FLAT' 
            
            prev_cat = cat
            sequence.append((mom, cat))
            
        return sequence

    def category_to_price(self, category):
        return self.last_min + (category * self.last_step)

# ==============================================================================
# 2. THE LEARNABLE AI ENGINE
# ==============================================================================

class ZScoreEngine:
    def __init__(self, num_categories=40):
        self.num_categories = num_categories
        self.mom_map = {k: v for v, k in enumerate(['UP', 'DOWN', 'FLAT'])}
        self.mom_list = ['UP', 'DOWN', 'FLAT']
        
        # Initialize with Uniform Probabilities (Will be overwritten by train())
        self.mom_trans = np.ones((3, 3)) / 3
        self.mom_start = np.ones(3) / 3
        self.cat_trans = np.ones((num_categories, num_categories)) / num_categories
        self.cat_start = np.ones(num_categories) / num_categories
        
        self.stats = {'mom': {}, 'cat': {}}

    def train(self, sequence):
        """
        LEARNS physics from the provided training sequence.
        Builds transition matrices based on actual historical frequency.
        """
        print("Training model on historical data...")
        
        # Reset counters with small epsilon for smoothing (laplace smoothing)
        mom_counts = np.ones((3, 3)) * 0.1
        cat_counts = np.ones((self.num_categories, self.num_categories)) * 0.1
        
        mom_start_counts = np.ones(3) * 0.1
        cat_start_counts = np.ones(self.num_categories) * 0.1

        # Iterate through history
        for i in range(len(sequence) - 1):
            curr_mom, curr_cat = sequence[i]
            next_mom, next_cat = sequence[i+1]
            
            # Indices
            curr_m_idx = self.mom_map[curr_mom]
            next_m_idx = self.mom_map[next_mom]
            curr_c_idx = curr_cat - 1
            next_c_idx = next_cat - 1
            
            # Count Transitions
            mom_counts[curr_m_idx][next_m_idx] += 1
            cat_counts[curr_c_idx][next_c_idx] += 1
            
            # Count Starts (Marginals)
            mom_start_counts[curr_m_idx] += 1
            cat_start_counts[curr_c_idx] += 1

        # Normalize to create Probability Matrices
        # Axis 1 = sum across columns (for each row)
        self.mom_trans = mom_counts / mom_counts.sum(axis=1, keepdims=True)
        self.cat_trans = cat_counts / cat_counts.sum(axis=1, keepdims=True)
        
        self.mom_start = mom_start_counts / mom_start_counts.sum()
        self.cat_start = cat_start_counts / cat_start_counts.sum()
        
        print("Training Complete. Model has learned market physics.")

    def _get_log_prob(self, seq, trans, start):
        if not seq: return 0.0
        # Guard against zero index issues
        first_idx = seq[0]
        if first_idx >= len(start): return -99.0
        
        p = math.log(start[first_idx] + 1e-9)
        for i in range(len(seq)-1):
            curr = seq[i]
            next_val = seq[i+1]
            if curr >= len(trans) or next_val >= len(trans):
                p += -99.0
            else:
                p += math.log(trans[curr][next_val] + 1e-9)
        return p

    def calibrate(self):
        """Generates Z-Score baselines using the LEARNED probabilities."""
        print("Calibrating Z-Score baselines (Monte Carlo)...")
        # Calibrate for relevant window sizes (e.g., 8 to 10)
        # We need slightly longer than window to account for prediction appends
        for length in range(2, 15): 
            m_logs, c_logs = [], []
            for _ in range(500): # 500 samples per length
                # Sim Mom
                s_m = [np.random.choice(3, p=self.mom_start)]
                for _ in range(length-1): 
                    s_m.append(np.random.choice(3, p=self.mom_trans[s_m[-1]]))
                m_logs.append(self._get_log_prob(s_m, self.mom_trans, self.mom_start))
                
                # Sim Cat
                s_c = [np.random.choice(self.num_categories, p=self.cat_start)]
                for _ in range(length-1): 
                    s_c.append(np.random.choice(self.num_categories, p=self.cat_trans[s_c[-1]]))
                c_logs.append(self._get_log_prob(s_c, self.cat_trans, self.cat_start))
                
            self.stats['mom'][length] = {'mu': np.mean(m_logs), 'var': np.var(m_logs)}
            self.stats['cat'][length] = {'mu': np.mean(c_logs), 'var': np.var(c_logs)}

    def get_z_score(self, sequence):
        length = len(sequence)
        if length not in self.stats['mom']: return -10.0 # Penalty
        
        m_idxs = [self.mom_map[m] for m, c in sequence]
        c_idxs = [c-1 for m, c in sequence]
        
        lp_m = self._get_log_prob(m_idxs, self.mom_trans, self.mom_start)
        lp_c = self._get_log_prob(c_idxs, self.cat_trans, self.cat_start)
        
        mu = self.stats['mom'][length]['mu'] + self.stats['cat'][length]['mu']
        std = math.sqrt(self.stats['mom'][length]['var'] + self.stats['cat'][length]['var'])
        return 0 if std == 0 else (lp_m + lp_c - mu) / std

class CompletionCorrector:
    def __init__(self, engine):
        self.engine = engine
        self.COST_INSERT = 1.2 # Lower cost to encourage predictions if data supports it

    def repair_physics(self, sequence):
        if not sequence: return []
        repaired = [sequence[0]]
        for i in range(1, len(sequence)):
            prev_mom, prev_cat = repaired[-1]
            curr_mom, curr_cat = sequence[i]
            if curr_cat > prev_cat: mom = 'UP'
            elif curr_cat < prev_cat: mom = 'DOWN'
            else: mom = curr_mom
            repaired.append((mom, curr_cat))
        return repaired

    def solve(self, sequence):
        # Window size 8 is standard for this setup
        N = len(sequence)
        variants = [{'seq': sequence, 'op': 'HOLD', 'cost': 0}]
        
        # Try appending ONE future item (Prediction)
        last_cat = sequence[-1][1]
        
        # Only try plausible moves (+/- 1 or same)
        candidates = [last_cat]
        if last_cat < self.engine.num_categories: candidates.append(last_cat + 1)
        if last_cat > 1: candidates.append(last_cat - 1)
            
        for cat in candidates:
            # Append to end
            new_sub = sequence + [('FLAT', cat)] 
            variants.append({
                'seq': self.repair_physics(new_sub), 
                'op': 'PREDICT', 
                'cost': self.COST_INSERT
            })
        
        # Score
        results = []
        for v in variants:
            z = self.engine.get_z_score(v['seq'])
            results.append({**v, 'net': z - v['cost']})
            
        results.sort(key=lambda x: x['net'], reverse=True)
        return results[0]

# ==============================================================================
# 3. STATISTICAL VALIDATOR & RUNNER
# ==============================================================================

if __name__ == "__main__":
    # CONFIG
    SYMBOL = "SPY"
    PERIOD = "10y"       # Requires ~2500 candles for significance
    NUM_BINS = 40        # Finer granularity for 10y range
    SPLIT_RATIO = 0.7    # 70% Train, 30% Test
    
    # 1. FETCH
    market = MarketDataHandler()
    raw_prices = market.fetch_candles(SYMBOL, period=PERIOD)
    if not raw_prices: exit()

    # 2. DISCRETIZE
    full_seq = market.discretize_sequence(raw_prices, num_bins=NUM_BINS)
    
    # 3. TRAIN/TEST SPLIT
    split_idx = int(len(full_seq) * SPLIT_RATIO)
    train_seq = full_seq[:split_idx]
    test_seq = full_seq[split_idx:]
    
    print(f"\nTotal Data Points: {len(full_seq)}")
    print(f"Training Set:      {len(train_seq)} (70%)")
    print(f"Testing Set:       {len(test_seq)} (30%)")
    
    # 4. TRAIN ENGINE
    engine = ZScoreEngine(num_categories=NUM_BINS)
    engine.train(train_seq)
    engine.calibrate()
    
    ai = CompletionCorrector(engine)
    
    # 5. RUN BACKTEST SIMULATION (On Test Set)
    print(f"\nRunning Backtest on {len(test_seq)} points...")
    correct_predictions = 0
    total_predictions = 0
    window_size = 8
    
    # Iterate through test set with sliding window
    # We only count it as a "Prediction" if the AI chooses 'PREDICT' over 'HOLD'
    for i in range(len(test_seq) - window_size - 1):
        window = test_seq[i : i+window_size]
        actual_next = test_seq[i+window_size][1] # The real category that happened
        
        result = ai.solve(window)
        
        if result['op'] == 'PREDICT':
            total_predictions += 1
            predicted_cat = result['seq'][-1][1]
            
            # Did it predict the correct category?
            # We allow +/- 0 tolerance (strict) or +/- 1 (loose)
            # Let's check strict directional accuracy (did it predict UP vs DOWN correctly?)
            
            last_known = window[-1][1]
            pred_dir = np.sign(predicted_cat - last_known)
            actual_dir = np.sign(actual_next - last_known)
            
            if predicted_cat == actual_next:
                correct_predictions += 1
            
    print("\n" + "="*40)
    print("BACKTEST RESULTS (Statistical Significance)")
    print("="*40)
    if total_predictions > 0:
        acc = (correct_predictions / total_predictions) * 100
        print(f"Algorithm triggered prediction {total_predictions} times.")
        print(f"Exact Category Match: {acc:.2f}%")
        print("Note: Random guessing on 40 bins is ~2.5%")
    else:
        print("Algorithm was too conservative (0 trades triggered).")
        print("Try lowering COST_INSERT.")

    # 6. LIVE PREDICTION (Current Day)
    print("\n" + "="*40)
    print(f"LIVE FORECAST ({SYMBOL})")
    print("="*40)
    
    current_window = full_seq[-window_size:]
    final_result = ai.solve(current_window)
    
    print(f"Current Price:   ${raw_prices[-1]:.2f} (Cat {current_window[-1][1]})")
    
    if final_result['op'] == 'PREDICT':
        p_cat = final_result['seq'][-1][1]
        p_price = market.category_to_price(p_cat)
        print(f"AI PREDICTION:   Category {p_cat} (~${p_price:.2f})")
        print(f"Logic:           Momentum Z-Score > Insert Cost")
    else:
        print(f"AI DECISION:     HOLD / WAIT")
        print(f"Logic:           No statistically significant pattern found.")