import yfinance as yf
import math
import numpy as np
import pandas as pd
from collections import Counter

# ==============================================================================
# 1. DATA HANDLER
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
# 2. PROBABILISTIC ENGINE (Likelihood Based)
# ==============================================================================

class MarkovEngine:
    def __init__(self, num_categories=40):
        self.num_categories = num_categories
        self.mom_map = {k: v for v, k in enumerate(['UP', 'DOWN', 'FLAT'])}
        
        # Probabilities
        self.mom_trans = np.zeros((3, 3))
        self.cat_trans = np.zeros((num_categories, num_categories))
        
        # We need start probs to handle the very first item's existence probability
        self.mom_start = np.zeros(3)
        self.cat_start = np.zeros(num_categories)

    def train(self, sequence):
        print("Training Markov Probabilities...")
        mom_counts = np.ones((3, 3)) # Laplace smoothing (start at 1)
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

        # Normalize to probabilities (Sum of row = 1.0)
        self.mom_trans = mom_counts / mom_counts.sum(axis=1, keepdims=True)
        self.cat_trans = cat_counts / cat_counts.sum(axis=1, keepdims=True)
        self.mom_start = self.mom_start / self.mom_start.sum()
        self.cat_start = self.cat_start / self.cat_start.sum()
        
    def get_average_log_likelihood(self, seq):
        """
        Calculates the Average Log Likelihood per Transition.
        This metric is independent of length.
        """
        if not seq: return -999.0
        L = len(seq)
        if L < 2: return -999.0 # Need at least 2 points for a transition
        
        m_idxs = [self.mom_map[m] for m, c in seq]
        c_idxs = [c-1 for m, c in seq]
        
        total_log_prob = 0.0
        
        # 1. Starting Probability (How likely is the start state?)
        # We include this so "rare" starting points are penalized slightly
        total_log_prob += math.log(self.mom_start[m_idxs[0]] + 1e-9)
        total_log_prob += math.log(self.cat_start[c_idxs[0]] + 1e-9)
        
        # 2. Transition Probabilities
        count = 1 # We counted the start
        
        for i in range(L - 1):
            p_m = self.mom_trans[m_idxs[i]][m_idxs[i+1]]
            p_c = self.cat_trans[c_idxs[i]][c_idxs[i+1]]
            
            total_log_prob += math.log(p_m + 1e-9)
            total_log_prob += math.log(p_c + 1e-9)
            count += 1
            
        # Return Average Log Likelihood
        # Higher (closer to 0) is better.
        return total_log_prob / count

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
        # Modify
        for i in range(L):
            curr_mom, curr_cat = seq[i]
            if curr_cat < self.num_bins: candidates.append(seq[:i] + [(curr_mom, curr_cat + 1)] + seq[i+1:])
            if curr_cat > 1: candidates.append(seq[:i] + [(curr_mom, curr_cat - 1)] + seq[i+1:])
        # Swap
        for i in range(L - 1):
            new_seq = seq[:]
            new_seq[i], new_seq[i+1] = new_seq[i+1], new_seq[i]
            candidates.append(new_seq)
        # Delete
        if L > 3:
            for i in range(L): candidates.append(seq[:i] + seq[i+1:])
        return candidates

    def solve(self, input_seq, horizon_steps=2):
        base_pool = [{'seq': input_seq, 'type': 'orig'}]
        L = len(input_seq)
        
        # Sub-sequences (Shortening)
        for length in range(3, L): 
            for i in range(L - length + 1):
                base_pool.append({'seq': input_seq[i : i+length], 'type': 'sub'})

        # Mutations
        muts = self.generate_mutations(input_seq)
        for m in muts:
            base_pool.append({'seq': m, 'type': 'mut'})

        # Extensions
        final_pool = []
        final_pool.extend(base_pool)
        
        for item in base_pool:
            current_tips = [item['seq']]
            for _ in range(horizon_steps):
                new_tips = []
                for tip in current_tips:
                    last_cat = tip[-1][1]
                    # Allowed moves: +1, -1, 0
                    options = [last_cat]
                    if last_cat < self.num_bins: options.append(last_cat + 1)
                    if last_cat > 1: options.append(last_cat - 1)
                    
                    for opt in options:
                        new_tip = tip + [('FLAT', opt)] 
                        new_tips.append(new_tip)
                        final_pool.append({'seq': new_tip, 'type': item['type'] + '+ext'})
                current_tips = new_tips

        # Evaluate based on Average Likelihood
        scored = []
        unique_hashes = set()
        
        for cand in final_pool:
            fixed_seq = self._repair_physics(cand['seq'])
            s_hash = str(fixed_seq)
            if s_hash in unique_hashes: continue
            unique_hashes.add(s_hash)
            
            # Score is Average Log Likelihood (Max is best, e.g. -0.5 is better than -2.0)
            score = self.engine.get_average_log_likelihood(fixed_seq)
            
            scored.append({
                'seq': fixed_seq, 
                'score': score, 
                'type': cand['type']
            })

        # Sort Descending (Highest Average Probability first)
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[0]

# ==============================================================================
# 4. BACKTESTER
# ==============================================================================

class Backtester:
    def __init__(self, initial_capital=10000.0):
        self.initial_capital = initial_capital
        self.equity = [initial_capital]
        self.returns = []
    
    def on_market_step(self, predicted_signal, actual_pct_change):
        step_return = predicted_signal * actual_pct_change
        curr_equity = self.equity[-1]
        new_equity = curr_equity * (1 + step_return)
        self.equity.append(new_equity)
        self.returns.append(step_return)
        
    def get_stats(self):
        if not self.returns: return {}
        equity_curve = np.array(self.equity)
        returns = np.array(self.returns)
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = (mean_ret / std_ret) * math.sqrt(252) if std_ret > 0 else 0
        peak = equity_curve[0]
        max_dd = 0
        for val in equity_curve:
            if val > peak: peak = val
            dd = (peak - val) / peak
            if dd > max_dd: max_dd = dd
        return {
            "Total Return %": total_return * 100,
            "Sharpe Ratio": sharpe,
            "Max Drawdown %": max_dd * 100,
            "Win Rate %": (np.sum(returns > 0) / np.sum(returns != 0)) * 100 if np.sum(returns != 0) > 0 else 0,
            "Final Equity": equity_curve[-1]
        }

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    SYMBOL = "SPY"
    PERIOD = "2y"
    NUM_BINS = 50
    WINDOW_SIZE = 8
    HORIZON = 2 
    
    market = MarketDataHandler()
    raw_prices = market.fetch_candles(SYMBOL, PERIOD)
    if not raw_prices: exit()

    full_seq = market.discretize_sequence(raw_prices, num_bins=NUM_BINS)
    split_idx = int(len(full_seq) * 0.7)
    train_seq = full_seq[:split_idx]
    
    # Init Engine
    engine = MarkovEngine(num_categories=NUM_BINS)
    engine.train(train_seq)
    
    # Init Solver
    solver = EvolutionarySolver(engine, NUM_BINS)
    backtester = Backtester()
    
    print(f"\nRunning Evolutionary Likelihood Solver...")
    print(f"{'IDX':<5} | {'WINNER LEN':<10} | {'ACTION':<10} | {'AVG LOG-LIK':<12} | {'EQUITY':<12}")
    print("-" * 65)
    
    actions_count = Counter()
    
    for i in range(split_idx, len(full_seq) - WINDOW_SIZE - 1):
        input_window = full_seq[i : i + WINDOW_SIZE]
        
        curr_price = raw_prices[i + WINDOW_SIZE]
        next_price = raw_prices[i + WINDOW_SIZE + 1]
        actual_pct_change = (next_price - curr_price) / curr_price
        
        # Solve
        winner = solver.solve(input_window, horizon_steps=HORIZON)
        best_seq = winner['seq']
        
        # Decide
        signal = 0
        action = "FLAT"
        
        if len(best_seq) > len(input_window):
            pred_cat = best_seq[len(input_window)][1]
            curr_cat = input_window[-1][1]
            if pred_cat > curr_cat: 
                signal = 1
                action = "BUY"
            elif pred_cat < curr_cat: 
                signal = -1
                action = "SELL"
        elif len(best_seq) < len(input_window):
            action = "NOISE"
            
        actions_count[action] += 1
        backtester.on_market_step(signal, actual_pct_change)
        
        if i % 25 == 0:
            len_delta = len(best_seq) - len(input_window)
            len_fmt = f"{len(best_seq)} ({len_delta:+})"
            print(f"{i:<5} | {len_fmt:<10} | {action:<10} | {winner['score']:.4f}       | {backtester.equity[-1]:,.0f}")

    stats = backtester.get_stats()
    
    print("\n" + "="*40)
    print(f"BACKTEST RESULTS ({SYMBOL})")
    print("="*40)
    print(f"Action Distribution: {dict(actions_count)}")
    print(f"Final Equity:   ${stats['Final Equity']:,.2f}")
    print(f"Total Return:   {stats['Total Return %']:.2f}%")
    print(f"Sharpe Ratio:   {stats['Sharpe Ratio']:.4f}")
    print(f"Max Drawdown:   {stats['Max Drawdown %']:.2f}%")
    
    bh_ret = (raw_prices[-1] - raw_prices[split_idx+WINDOW_SIZE]) / raw_prices[split_idx+WINDOW_SIZE] * 100
    print(f"Buy & Hold Ret: {bh_ret:.2f}%")