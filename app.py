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
# 2. PURE MARKOV ENGINE (No Evolutionary Logic)
# ==============================================================================

class MarkovEngine:
    def __init__(self, num_categories=40):
        self.num_categories = num_categories
        self.mom_map = {k: v for v, k in enumerate(['UP', 'DOWN', 'FLAT'])}
        self.mom_trans = np.zeros((3, 3))
        self.cat_trans = np.zeros((num_categories, num_categories))
        
        # We need smoothing to prevent log(0)
        self.epsilon = 1e-9

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

        # Normalize
        self.mom_trans = mom_counts / mom_counts.sum(axis=1, keepdims=True)
        self.cat_trans = cat_counts / cat_counts.sum(axis=1, keepdims=True)

    def calculate_log_prob(self, seq):
        """Calculates log probability of a specific sequence occurring."""
        if not seq or len(seq) < 2: return -999.0
        
        m_idxs = [self.mom_map[m] for m, c in seq]
        c_idxs = [c-1 for m, c in seq]
        
        log_prob = 0.0
        for i in range(len(seq) - 1):
            p_m = self.mom_trans[m_idxs[i]][m_idxs[i+1]]
            p_c = self.cat_trans[c_idxs[i]][c_idxs[i+1]]
            log_prob += math.log(p_m + self.epsilon)
            log_prob += math.log(p_c + self.epsilon)
            
        return log_prob

    def predict_next(self, input_seq):
        """
        Given an input sequence, try all 3 possible next moves (UP, DOWN, FLAT).
        Return the one with the highest total log probability.
        """
        last_val = input_seq[-1][1]
        candidates = []
        
        # 1. Try UP
        if last_val < self.num_categories:
            next_seq = input_seq + [('UP', last_val + 1)]
            score = self.calculate_log_prob(next_seq)
            candidates.append({
                'action': 'BUY', 
                'score': score, 
                'next_cat': last_val + 1
            })
            
        # 2. Try DOWN
        if last_val > 1:
            next_seq = input_seq + [('DOWN', last_val - 1)]
            score = self.calculate_log_prob(next_seq)
            candidates.append({
                'action': 'SELL', 
                'score': score, 
                'next_cat': last_val - 1
            })
            
        # 3. Try FLAT
        next_seq = input_seq + [('FLAT', last_val)]
        score = self.calculate_log_prob(next_seq)
        candidates.append({
            'action': 'FLAT', 
            'score': score, 
            'next_cat': last_val
        })
        
        # Sort by score (Highest Log Prob is best)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[0]

# ==============================================================================
# 3. BACKTESTER UTILS
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
# 4. MAIN COMPARISON RUNNER
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
    
    engine = MarkovEngine(num_categories=NUM_BINS)
    engine.train(train_seq)
    
    # Initialize Backtesters
    backtesters = {L: Backtester(f"Len-{L}") for L in LENGTHS_TO_TEST}
    
    print(f"\nComparing Fixed Markov Windows: {LENGTHS_TO_TEST} on {SYMBOL}")
    print(f"Training Data: {split_idx} candles | Test Data: {len(full_seq) - split_idx} candles")
    print("\n" + "="*100)

    # We iterate through the test set.
    # We must ensure we have enough history for the max length
    max_L = max(LENGTHS_TO_TEST)
    
    for i in range(split_idx, len(full_seq) - 1):
        # We need next price for PnL
        curr_price = raw_prices[i]
        next_price = raw_prices[i+1]
        pct_change = (next_price - curr_price) / curr_price
        
        actual_move = "FLAT"
        if pct_change > 0.001: actual_move = "UP"
        elif pct_change < -0.001: actual_move = "DOWN"

        print(f"IDX: {i:<5} | Actual: {actual_move:<5} ({pct_change*100:+.2f}%)")
        
        # Run each length
        for L in LENGTHS_TO_TEST:
            if i < L: 
                continue
                
            input_window = full_seq[i - L + 1 : i + 1]
            input_cats = [x[1] for x in input_window]
            
            # Predict
            prediction = engine.predict_next(input_window)
            action = prediction['action']
            next_cat = prediction['next_cat']
            
            output_cats = input_cats + [next_cat]
            
            # Backtest
            backtesters[L].step(action, pct_change)
            
            # Formatting (Convert lists to strings for nice display)
            print(f"  L={L}: {str(input_cats):<35} -> {str(output_cats):<40} | {action}")
            
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