import yfinance as yf
import math
import time
import numpy as np
import pandas as pd
from collections import Counter

# Override print to slow down output for readability (0.1s)
_builtin_print = print
def print(*args, **kwargs):
    time.sleep(0.1)
    _builtin_print(*args, **kwargs)

# ==============================================================================
# 1. DATA HANDLER (Preserved)
# ==============================================================================

class MarketDataHandler:
    def __init__(self):
        self.last_min = 0
        self.last_step = 0

    def fetch_candles(self, symbol="SPY", period="2y"):
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
# 2. FREQUENCY ENGINE (Exact Counting)
# ==============================================================================

class FrequencyEngine:
    def __init__(self, num_categories=40):
        self.num_categories = num_categories
        self.history = []

    def train(self, sequence):
        """Stores the historical sequence for pattern matching."""
        print("Training Frequency Engine (Storing History)...")
        self.history = sequence

    def get_score(self, sub_seq):
        """
        Calculates Probability = Count / Total_Possible_Of_Length.
        """
        if not self.history or not sub_seq: return 0, 0.0
        
        len_sub = len(sub_seq)
        len_hist = len(self.history)
        
        if len_sub > len_hist: return 0, 0.0
        
        # Exact Sequence Matching
        count = 0
        # We scan the history
        for i in range(len_hist - len_sub + 1):
            if self.history[i : i + len_sub] == sub_seq:
                count += 1
        
        # Total possible sequences of this specific length
        total_possible = len_hist - len_sub + 1
        
        prob = count / total_possible if total_possible > 0 else 0.0
        return count, prob

# ==============================================================================
# 3. MUTATION SOLVER (The Evolutionary Core)
# ==============================================================================

class MutationSolver:
    def __init__(self, engine, num_bins):
        self.engine = engine
        self.num_bins = num_bins

    def _repair_physics(self, seq):
        """
        Re-calculates Momentum tags to match Price Categories.
        Essential after swapping/inserting to ensure sequence is valid for lookup.
        """
        if not seq: return []
        repaired = []
        for i in range(len(seq)):
            cat = seq[i][1]
            if i == 0:
                # Keep original start momentum if possible, else FLAT
                mom = seq[i][0]
            else:
                prev_cat = repaired[-1][1]
                if cat > prev_cat: mom = 'UP'
                elif cat < prev_cat: mom = 'DOWN'
                else: mom = 'FLAT'
            repaired.append((mom, cat))
        return repaired

    def generate_variants(self, input_seq):
        """Generates all mutations: Swap, Del, Mod, Insert, Sub, Ext."""
        pool = []
        L = len(input_seq)
        
        # 1. Original
        pool.append({'seq': input_seq, 'type': 'orig'})

        # 2. Extensions (Forecasts) - Try UP/DOWN/FLAT
        last_cat = input_seq[-1][1]
        options = [last_cat]
        if last_cat < self.num_bins: options.append(last_cat + 1)
        if last_cat > 1: options.append(last_cat - 1)
        
        for opt in options:
            # We append 'FLAT' as placeholder, _repair_physics fixes it later
            ext_seq = input_seq + [('FLAT', opt)]
            pool.append({'seq': ext_seq, 'type': 'ext'})

        # 3. Sub-sequences (Shortening)
        # We take slices from the END (most recent data must be preserved usually)
        # But per user instruction "cde wins over Ababcde", we try all substrings
        for start in range(L):
            for end in range(start + 2, L + 1): # Min length 2
                if start == 0 and end == L: continue # Skip original (already added)
                sub = input_seq[start:end]
                pool.append({'seq': sub, 'type': 'sub'})

        # 4. Modifications (Noise adjust)
        for i in range(L):
            curr_mom, curr_cat = input_seq[i]
            # +1
            if curr_cat < self.num_bins:
                m_seq = input_seq[:]
                m_seq[i] = (curr_mom, curr_cat + 1)
                pool.append({'seq': m_seq, 'type': 'mod'})
            # -1
            if curr_cat > 1:
                m_seq = input_seq[:]
                m_seq[i] = (curr_mom, curr_cat - 1)
                pool.append({'seq': m_seq, 'type': 'mod'})

        # 5. Swaps (Temporal dislocation)
        for i in range(L - 1):
            s_seq = input_seq[:]
            s_seq[i], s_seq[i+1] = s_seq[i+1], s_seq[i]
            pool.append({'seq': s_seq, 'type': 'swap'})

        # 6. Deletions (Remove outlier)
        if L > 2:
            for i in range(L):
                d_seq = input_seq[:i] + input_seq[i+1:]
                pool.append({'seq': d_seq, 'type': 'del'})
        
        # 7. Insertions (Interpolation)
        # Insert average of neighbors
        if L < 10: # Limit complexity
            for i in range(L - 1):
                c1 = input_seq[i][1]
                c2 = input_seq[i+1][1]
                avg = int((c1 + c2) / 2)
                i_seq = input_seq[:i+1] + [('FLAT', avg)] + input_seq[i+1:]
                pool.append({'seq': i_seq, 'type': 'ins'})

        return pool

    def solve(self, input_seq):
        variants = self.generate_variants(input_seq)
        scored = []
        seen_hashes = set()

        for v in variants:
            # 1. Repair Physics
            fixed_seq = self._repair_physics(v['seq'])
            
            # 2. Deduplicate
            s_hash = str(fixed_seq)
            if s_hash in seen_hashes: continue
            seen_hashes.add(s_hash)
            
            # 3. Score (Frequency / Total_Possible)
            count, prob = self.engine.get_score(fixed_seq)
            
            # 4. Filter impossible sequences
            if count > 0:
                scored.append({
                    'seq': fixed_seq,
                    'count': count,
                    'prob': prob,
                    'type': v['type']
                })
        
        # Sort by Probability (Desc)
        scored.sort(key=lambda x: x['prob'], reverse=True)
        
        if not scored:
            return {'seq': input_seq, 'prob': 0.0, 'type': 'none', 'count': 0}
            
        return scored[0]

# ==============================================================================
# 4. BACKTESTER UTILS (Preserved)
# ==============================================================================

class Backtester:
    def __init__(self, name):
        self.name = name
        self.equity = 10000.0
        self.returns = []
        self.wins = 0
        self.total = 0

    def step(self, action, pct_change):
        step_ret = 0.0
        if action == 'BUY': step_ret = pct_change
        elif action == 'SELL': step_ret = -pct_change
        
        self.equity *= (1 + step_ret)
        self.returns.append(step_ret)
        
        if action != 'FLAT':
            self.total += 1
            if step_ret > 0: self.wins += 1

    def get_sharpe(self):
        if not self.returns: return 0.0
        rets = np.array(self.returns)
        mean = np.mean(rets)
        std = np.std(rets)
        return (mean / std) * math.sqrt(252) if std > 0 else 0.0

# ==============================================================================
# 5. MAIN COMPARISON RUNNER
# ==============================================================================

if __name__ == "__main__":
    SYMBOL = "SPY"
    PERIOD = "2y"
    NUM_BINS = 50
    LENGTHS_TO_TEST = [4, 5, 6, 7]
    
    market = MarketDataHandler()
    raw_prices = market.fetch_candles(SYMBOL, PERIOD)
    if not raw_prices: exit()

    full_seq = market.discretize_sequence(raw_prices, num_bins=NUM_BINS)
    
    # 70% Train, 30% Test
    split_idx = int(len(full_seq) * 0.7)
    train_seq = full_seq[:split_idx]
    
    engine = FrequencyEngine(num_categories=NUM_BINS)
    engine.train(train_seq)
    
    solver = MutationSolver(engine, NUM_BINS)
    
    # Initialize Backtesters
    backtesters = {L: Backtester(f"Len-{L}") for L in LENGTHS_TO_TEST}
    
    print(f"\nComparing Evolutionary Frequency (Mutations): {LENGTHS_TO_TEST} on {SYMBOL}")
    print(f"Training Data: {split_idx} candles | Test Data: {len(full_seq) - split_idx} candles")
    print("\n" + "="*100)

    for i in range(split_idx, len(full_seq) - 1):
        curr_price = raw_prices[i]
        next_price = raw_prices[i+1]
        pct_change = (next_price - curr_price) / curr_price
        
        actual_move = "FLAT"
        if pct_change > 0.001: actual_move = "UP"
        elif pct_change < -0.001: actual_move = "DOWN"

        print(f"IDX: {i:<5} | Actual: {actual_move:<5} ({pct_change*100:+.2f}%)")
        
        # Run each length
        for L in LENGTHS_TO_TEST:
            if i < L: continue
                
            input_window = full_seq[i - L + 1 : i + 1]
            input_cats = [x[1] for x in input_window]
            
            # SOLVE (Generate Mutations -> Compare Probs -> Pick Winner)
            winner = solver.solve(input_window)
            best_seq = winner['seq']
            
            # DECISION LOGIC
            # We only trade if the winner EXTENDS beyond the current timeframe
            action = "FLAT"
            
            # Length check: Did it forecast?
            if len(best_seq) > len(input_window):
                # Check the first new value (index = len(input))
                pred_cat = best_seq[len(input_window)][1]
                curr_cat = input_window[-1][1]
                
                if pred_cat > curr_cat: action = "BUY"
                elif pred_cat < curr_cat: action = "SELL"
            
            # Backtest
            backtesters[L].step(action, pct_change)
            
            # Formatting
            output_cats = [x[1] for x in best_seq]
            
            # Truncate strings if too long
            in_s = str(input_cats)
            out_s = str(output_cats)
            if len(out_s) > 30: out_s = out_s[:27] + "..."
            
            stats = f"[Type={winner['type']}, P={winner['prob']:.3f}]"
            print(f"  L={L}: {in_s:<20} -> {out_s:<30} | {action:<4} {stats}")
            
        print("-" * 100)

    # ==========================================================================
    # FINAL RESULTS
    # ==========================================================================
    print("="*100)
    print(f"{'METRIC':<15} | {'LEN=4':<15} | {'LEN=5':<15} | {'LEN=6':<15} | {'LEN=7':<15}")
    print("-" * 100)
    
    equity_row = "Final Equity    "
    sharpe_row = "Sharpe Ratio    "
    winrate_row = "Win Rate        "
    return_row = "Total Return    "
    
    for L in LENGTHS_TO_TEST:
        bt = backtesters[L]
        tot_ret = (bt.equity - 10000) / 10000 * 100
        wr = (bt.wins / bt.total * 100) if bt.total > 0 else 0
        
        equity_row += f"| ${bt.equity:,.0f}".ljust(18)
        sharpe_row += f"| {bt.get_sharpe():.2f}".ljust(18)
        winrate_row += f"| {wr:.1f}% ({bt.wins}/{bt.total})".ljust(18)
        return_row += f"| {tot_ret:+.2f}%".ljust(18)

    print(equity_row)
    print(return_row)
    print(sharpe_row)
    print(winrate_row)
    print("="*100)