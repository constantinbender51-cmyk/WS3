import numpy as np
import pandas as pd
from collections import defaultdict

# ==========================================
# 1. Data Generation
# ==========================================
def generate_price_data(n=2000, seed=42):
    """
    Generates a synthetic price series.
    Increased N to support 20-bin categorization better.
    """
    np.random.seed(seed)
    t = np.linspace(0, 100, n)
    
    # Components: Trend + Cycle + Random Walk
    trend = t * 0.2
    cycle = 5 * np.sin(t) + 3 * np.sin(t * 3) # Complex cycle
    noise = np.random.normal(0, 1, n)
    random_walk = np.cumsum(np.random.normal(0, 0.5, n))
    
    price = 100 + trend + cycle + noise + random_walk
    price = np.maximum(price, 10.0) # Ensure positive
    
    return pd.DataFrame({'price': price})

# ==========================================
# 2. Sequence Analysis Engine
# ==========================================
class SequenceModel:
    def __init__(self, max_len=5):
        self.max_len = max_len
        # Stores counts of sequences: {(1,2): 10, (1,3): 50...}
        self.counts = defaultdict(int)
        # Stores total counts for a given length
        self.total_counts_by_len = defaultdict(int)
        # Stores unique patterns seen
        self.unique_patterns_by_len = defaultdict(set)

    def train(self, sequence):
        """
        Ingests a full list of tokens (ints or chars) and builds probability maps.
        """
        n = len(sequence)
        # We assume the sequence passed is the full history available up to t
        for length in range(1, self.max_len + 1):
            for i in range(n - length + 1):
                sub = tuple(sequence[i : i + length])
                self.counts[sub] += 1
                self.total_counts_by_len[length] += 1
                self.unique_patterns_by_len[length].add(sub)

    def get_probability(self, sequence_tuple):
        """
        Prob = Occurrence / Avg Occurrence of sequences of that length
        Avg Occurrence = Total Occurrences of Len L / Count of Unique Patterns of Len L
        """
        length = len(sequence_tuple)
        if length == 0 or length > self.max_len:
            return 0.0
            
        count = self.counts.get(sequence_tuple, 0)
        
        # Denominator logic
        num_unique = len(self.unique_patterns_by_len[length])
        if num_unique == 0:
            return 0.0
            
        total_occurrences = self.total_counts_by_len[length]
        avg_occurrence = total_occurrences / num_unique
        
        if avg_occurrence == 0:
            return 0.0
            
        return count / avg_occurrence

class SequenceTrader:
    def __init__(self, df, lookback=200, max_seq_len=5, edit_depth=1, n_categories=20):
        self.df = df.copy()
        self.lookback = lookback
        self.max_seq_len = max_seq_len
        self.edit_depth = edit_depth
        self.n_categories = n_categories
        
        # Results
        self.equity = [1000.0]
        self.signals = []
        
    def discretize_window(self, prices):
        """
        1. Breaks prices into N categories (0 to 19) using Quantiles (qcut).
        2. Calculates Directions as diff of Categories.
        """
        # 1. Categorize
        # qcut tries to divide into equal-sized buckets
        try:
            cats = pd.qcut(prices, self.n_categories, labels=False, duplicates='drop')
        except:
            # Fallback if too many duplicates
            cats = pd.cut(prices, self.n_categories, labels=False)
            
        cats = list(cats)
        
        # 2. Directions
        # Dir[0] = 0 (Flat)
        # Dir[t] = Cat[t] - Cat[t-1]
        dirs = [0] * len(cats)
        for i in range(1, len(cats)):
            dirs[i] = cats[i] - cats[i-1]
            
        return cats, dirs

    def generate_edits(self, seq, alphabet):
        """
        Generates variations of 'seq' (tuple) using 'alphabet' (list of valid tokens).
        Returns list of objects: {'seq': tuple, 'type': str, 'meta': val}
        """
        candidates = []
        
        # 0. Original (No Edit)
        candidates.append({'seq': seq, 'type': 'original', 'meta': None})
        
        if self.edit_depth < 1:
            return candidates

        seq_len = len(seq)
        
        # 1. Insert (Input: abc -> Output: aabc, abac... abcd)
        # We can insert any token from the alphabet at any position
        for i in range(seq_len + 1):
            for token in alphabet:
                # Construct new tuple
                new_seq = seq[:i] + (token,) + seq[i:]
                candidates.append({'seq': new_seq, 'type': 'insert', 'meta': (i, token)})
        
        # 2. Remove (Input: abc -> bc, ac, ab)
        for i in range(seq_len):
            new_seq = seq[:i] + seq[i+1:]
            candidates.append({'seq': new_seq, 'type': 'remove', 'meta': i})
            
        # 3. Swap (Input: abc -> xbc, axc, abx)
        for i in range(seq_len):
            for token in alphabet:
                if token != seq[i]:
                    new_seq = seq[:i] + (token,) + seq[i+1:]
                    candidates.append({'seq': new_seq, 'type': 'swap', 'meta': (i, token)})
        
        return candidates

    def get_best_variation(self, input_seq, model, alphabet):
        """
        Generates edits, scores them, returns the best one.
        """
        variations = self.generate_edits(input_seq, alphabet)
        
        best_var = None
        best_prob = -1.0
        
        for var in variations:
            prob = model.get_probability(var['seq'])
            if prob > best_prob:
                best_prob = prob
                best_var = var
                
        return best_var, best_prob

    def analyze_signal(self, current_prices):
        # Need enough data
        if len(current_prices) < self.lookback:
            return 0
            
        # 1. Get sequences for the current window
        cat_seq_full, dir_seq_full = self.discretize_window(current_prices)
        
        # 2. Prepare Training Data
        # We train on the history in this window to learn local probabilities
        cat_model = SequenceModel(self.max_seq_len)
        dir_model = SequenceModel(self.max_seq_len)
        cat_model.train(cat_seq_full)
        dir_model.train(dir_seq_full)
        
        # 3. Define Alphabets for the Editor
        # Categories: 0 to N-1
        alph_c = list(range(self.n_categories))
        
        # Directions: All unique directions seen in this window (e.g., -2, 0, 1, 5)
        # We limit alphabet to what is actually observed to avoid testing impossible jumps
        alph_d = list(set(dir_seq_full))
        
        # 4. Input Sequence (The "Tail" to be edited)
        # If max_len is 5, we look at the last 4 items to see if we can append a 5th
        # or edit the last 5. 
        # Standard approach: Input length M-1 to see if we Insert at end to make M
        input_len = self.max_seq_len - 1
        input_c = tuple(cat_seq_full[-input_len:])
        input_d = tuple(dir_seq_full[-input_len:])
        
        # 5. Find Best Variations
        best_c, prob_c = self.get_best_variation(input_c, cat_model, alph_c)
        best_d, prob_d = self.get_best_variation(input_d, dir_model, alph_d)
        
        # 6. Combined Analysis
        # Prompt: "Select the sequence as output that has the highest category probability + direction probability"
        # However, C and D are separate optimizations. We effectively have a "Winning Pair".
        
        signal = 0
        
        # Check if Category Result is an "Extension" (Insert at End)
        c_is_ext = (best_c['type'] == 'insert' and best_c['meta'][0] == len(input_c))
        
        # Check if Direction Result is an "Extension"
        d_is_ext = (best_d['type'] == 'insert' and best_d['meta'][0] == len(input_d))
        
        # If either (or both) implies an extension, we evaluate the signal
        if c_is_ext:
            predicted_next_cat = best_c['meta'][1]
            current_cat = input_c[-1]
            
            # Logic: Is the predicted category higher or lower?
            # We weight this by the combined probability
            strength = prob_c + prob_d
            
            if predicted_next_cat > current_cat:
                signal += strength # Buy bias
            elif predicted_next_cat < current_cat:
                signal -= strength # Sell bias
                
        if d_is_ext:
            predicted_next_dir = best_d['meta'][1]
            
            strength = prob_c + prob_d
            
            # Logic: If predicted direction is positive -> Buy
            if predicted_next_dir > 0:
                signal += strength
            elif predicted_next_dir < 0:
                signal -= strength

        # 7. Final Decision Threshold
        # Arbitrary threshold to ensure signal quality (Sequence must be > avg likelihood)
        if signal > 2.0: return 1  # Buy
        if signal < -2.0: return -1 # Sell
        return 0

    def run_backtest(self):
        prices = self.df['price'].values
        n = len(prices)
        position = 0 
        entry_price = 0
        
        print(f"Starting Backtest on {n} bars with {self.n_categories} categories...")
        
        # Start after lookback
        for t in range(self.lookback, n - 1):
            window = prices[t - self.lookback : t + 1]
            
            sig = self.analyze_signal(window)
            
            curr_p = prices[t]
            next_p = prices[t+1]
            
            # 1-Bar Hold Strategy
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

    def calculate_sharpe(self):
        eq = pd.Series(self.equity)
        rets = eq.pct_change().dropna()
        if len(rets) < 2 or rets.std() == 0: return 0.0
        return (rets.mean() / rets.std()) * np.sqrt(252)

# ==========================================
# 3. Main
# ==========================================
def main():
    # 1. Data
    df = generate_price_data(n=1000)
    
    # 2. Config
    # Categories: 20
    # Length M: 4 (computes probability of length 1 to 4)
    trader = SequenceTrader(df, lookback=100, max_seq_len=4, edit_depth=1, n_categories=20)
    
    # 3. Run
    equity = trader.run_backtest()
    
    # 4. Stats
    sharpe = trader.calculate_sharpe()
    print("-" * 30)
    print(f"Final Equity: {equity[-1]:.2f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()