import yfinance as yf
import math
import time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

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
# 2. RELATIVE FREQUENCY ENGINE (Step 3 & 7)
# ==============================================================================

class RelativeFrequencyEngine:
    def __init__(self, name):
        self.name = name
        self.history = []
        # cache stats: {length: {'total_count': X, 'unique_count': Y, 'avg': Z}}
        self.stats = {} 

    def train(self, sequence):
        """
        Stores history and pre-computes statistics for all lengths up to ~15.
        """
        print(f"Training {self.name} Engine...")
        self.history = sequence
        
        # Pre-compute Average Occurrences for lengths 1 to 15
        max_len_stat = 15
        for L in range(1, max_len_stat + 1):
            counts = Counter()
            # Sliding window count
            for i in range(len(self.history) - L + 1):
                sub = tuple(self.history[i : i + L])
                counts[sub] += 1
            
            unique_seqs = len(counts)
            total_occurrences = sum(counts.values())
            
            if unique_seqs > 0:
                avg = total_occurrences / unique_seqs
            else:
                avg = 1.0
                
            self.stats[L] = {
                'avg': avg,
                'counts': counts
            }

    def get_relative_score(self, sub_seq):
        """
        Score = Occurrence / Average_Occurrence_For_This_Length
        """
        if not sub_seq: return 0.0
        L = len(sub_seq)
        
        # Check stats
        if L not in self.stats:
            return 0.0 # Length not seen/trained
        
        t_seq = tuple(sub_seq)
        count = self.stats[L]['counts'][t_seq]
        avg = self.stats[L]['avg']
        
        if avg == 0: return 0.0
        return count / avg

# ==============================================================================
# 3. RECURSIVE EVOLUTIONARY SOLVER (Steps 4, 5, 6, 7)
# ==============================================================================

class RecursiveSolver:
    def __init__(self, cat_engine, dir_engine, num_bins):
        self.cat_engine = cat_engine
        self.dir_engine = dir_engine
        self.num_bins = num_bins
        self.max_depth = 2 # "Edit n times" - Limit depth for performance

    def _repair_physics(self, seq):
        """Ensures directions match category changes after mutations."""
        repaired = []
        for i in range(len(seq)):
            mom, cat = seq[i]
            if i == 0:
                # Keep original start mom
                pass
            else:
                prev_cat = repaired[-1][1]
                if cat > prev_cat: mom = 'UP'
                elif cat < prev_cat: mom = 'DOWN'
                else: mom = 'FLAT'
            repaired.append((mom, cat))
        return repaired

    def _generate_edits(self, seq):
        """
        Generates single-step edits: Insert, Delete, Swap.
        Used recursively.
        """
        edits = []
        L = len(seq)
        
        # 1. SWAP
        for i in range(L - 1):
            new_s = seq[:]
            new_s[i], new_s[i+1] = new_s[i+1], new_s[i]
            edits.append(new_s)

        # 2. DELETE
        if L > 2:
            for i in range(L):
                new_s = seq[:i] + seq[i+1:]
                edits.append(new_s)

        # 3. INSERT (Forecast at end)
        # We explicitly add possible next steps at the tail
        last_cat = seq[-1][1]
        options = [last_cat]
        if last_cat < self.num_bins: options.append(last_cat + 1)
        if last_cat > 1: options.append(last_cat - 1)
        
        for opt in options:
            # We add a placeholder direction, _repair_physics fixes it
            edits.append(seq + [('FLAT', opt)])
            
        # 3b. INSERT (Interpolate/Duplicate)
        # "abc -> aabc" logic
        if L < 12: # Prevent explosion
            for i in range(L):
                # Duplicate current item at i
                edits.append(seq[:i] + [seq[i]] + seq[i:])
                
        return edits

    def _get_variants_recursive(self, current_seq, current_depth, max_depth, collected):
        """
        Recursively generates variants up to max_depth.
        Stores them in 'collected' set to avoid duplicates.
        """
        # Add current sequence to collection
        # Repair first to ensure we only store valid physics
        repaired = tuple(self._repair_physics(current_seq))
        collected.add(repaired)
        
        if current_depth >= max_depth:
            return

        # Generate 1-step edits
        edits = self._generate_edits(list(repaired))
        
        for edit in edits:
            # Recurse
            self._get_variants_recursive(edit, current_depth + 1, max_depth, collected)

    def solve(self, input_seq):
        # 1. Generate ALL variants (Recursively)
        variant_pool = set()
        self._get_variants_recursive(input_seq, 0, self.max_depth, variant_pool)
        
        scored_candidates = []
        
        for seq_tuple in variant_pool:
            seq = list(seq_tuple)
            
            # Extract Categories and Directions (Step 7)
            cats = [x[1] for x in seq]
            dirs = [x[0] for x in seq]
            
            # Score Categories (Step 3)
            # Score = Occurrence / Avg
            cat_score = self.cat_engine.get_relative_score(cats)
            
            # Score Directions (Step 7)
            dir_score = self.dir_engine.get_relative_score(dirs)
            
            # Total Score (Sum)
            total_score = cat_score + dir_score
            
            if total_score > 0:
                scored_candidates.append({
                    'seq': seq,
                    'score': total_score,
                    'cat_score': cat_score,
                    'dir_score': dir_score,
                    'is_longer': len(seq) > len(input_seq)
                })
        
        # Sort by Total Score Descending
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        if not scored_candidates:
            return None
            
        return scored_candidates[0] # Return Winner

# ==============================================================================
# 4. BACKTESTER (Preserved)
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
        if std == 0: return 0.0
        return (mean / std) * math.sqrt(252)

# ==============================================================================
# 5. MAIN EXECUTION
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
    
    # Split
    split_idx = int(len(full_seq) * 0.7)
    train_seq = full_seq[:split_idx]
    
    # 1. Train Dual Engines (Categories + Directions)
    cat_engine = RelativeFrequencyEngine("Category")
    dir_engine = RelativeFrequencyEngine("Direction")
    
    cat_train = [x[1] for x in train_seq]
    dir_train = [x[0] for x in train_seq]
    
    cat_engine.train(cat_train)
    dir_engine.train(dir_train)
    
    # 2. Init Solver
    solver = RecursiveSolver(cat_engine, dir_engine, NUM_BINS)
    
    # 3. Init Backtesters
    backtesters = {L: Backtester(f"Len-{L}") for L in LENGTHS_TO_TEST}
    
    print(f"\nComparing Recursive Frequency Solver (Dual Engine): {LENGTHS_TO_TEST}")
    print("\n" + "="*120)

    # Main Loop
    for i in range(split_idx, len(full_seq) - 1):
        curr_price = raw_prices[i]
        next_price = raw_prices[i+1]
        pct_change = (next_price - curr_price) / curr_price
        
        actual_move = "FLAT"
        if pct_change > 0.001: actual_move = "UP"
        elif pct_change < -0.001: actual_move = "DOWN"

        print(f"IDX: {i:<5} | Actual: {actual_move:<5} ({pct_change*100:+.2f}%)")
        
        for L in LENGTHS_TO_TEST:
            if i < L: continue
                
            input_window = full_seq[i - L + 1 : i + 1]
            input_cats = [x[1] for x in input_window]
            
            # SOLVE
            winner = solver.solve(input_window)
            
            action = "FLAT"
            stats_str = "No Signal"
            
            if winner:
                best_seq = winner['seq']
                
                # Step 6: Signal Logic
                # "If the output results in an insert at the end... It is a signal"
                # We check if the sequence grew AND the tail logic implies direction
                if len(best_seq) > len(input_window):
                    # Check what the value is at the 'future' index
                    future_idx = len(input_window)
                    if future_idx < len(best_seq):
                        pred_cat = best_seq[future_idx][1]
                        curr_cat = input_window[-1][1]
                        
                        if pred_cat > curr_cat: action = "BUY"
                        elif pred_cat < curr_cat: action = "SELL"
                
                stats_str = f"S={winner['score']:.2f} (C:{winner['cat_score']:.1f}+D:{winner['dir_score']:.1f})"
                out_cats_str = str([x[1] for x in best_seq])
                if len(out_cats_str) > 25: out_cats_str = out_cats_str[:22] + "..."
            else:
                out_cats_str = "None"

            # Backtest
            backtesters[L].step(action, pct_change)
            
            # Format
            print(f"  L={L}: {str(input_cats):<20} -> {out_cats_str:<25} | {action:<4} | {stats_str}")
            
        print("-" * 120)

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