import random
import collections
import string
import time
import os

# --- CONFIGURATION ---
MIN_PRICE = 0
MAX_PRICE = 200000
NUM_BUCKETS = 20
BUCKET_SIZE = (MAX_PRICE - MIN_PRICE) / NUM_BUCKETS
BUCKET_CHARS = string.ascii_uppercase[:NUM_BUCKETS] # A-T
WINDOW_SIZE = 4
SIMULATION_YEARS = 50
PORT = int(os.environ.get('PORT', 8080))

class StockPatternCompleter:
    def __init__(self, price_history, window_size=WINDOW_SIZE):
        self.prices = price_history
        self.window = window_size
        
        # 1. Ingest Data
        # Every step i has a Loc[i] and a Mom[i] (change from i-1 to i)
        self.loc_stream, self.mom_stream = self.discretize_data(self.prices)
        
        # 2. Train Memory Banks
        # Both look for patterns of exactly WINDOW_SIZE
        self.loc_probs = collections.defaultdict(collections.Counter)
        self.mom_probs = collections.defaultdict(collections.Counter)
        self.train()

    def get_bucket_index(self, price):
        if price >= MAX_PRICE: return NUM_BUCKETS - 1
        if price <= MIN_PRICE: return 0
        return int((price - MIN_PRICE) / BUCKET_SIZE)

    def discretize_data(self, prices):
        """
        Translates raw prices into two synced streams.
        Step i contains Location at i and the Momentum that led to i.
        """
        locs = []
        moms = []
        # We start from index 1 so every entry has a preceding price to determine momentum
        for i in range(1, len(prices)):
            prev_idx = self.get_bucket_index(prices[i-1])
            curr_idx = self.get_bucket_index(prices[i])
            
            locs.append(BUCKET_CHARS[curr_idx])
            
            if curr_idx > prev_idx: moms.append('U')
            elif curr_idx < prev_idx: moms.append('D')
            else: moms.append('F')
        return locs, moms

    def train(self):
        time.sleep(0.1)
        print(f"Training on {len(self.loc_stream)} market steps...")
        
        # Both models train on the same window size
        for i in range(len(self.loc_stream) - self.window):
            # Location pattern and the next location
            loc_pattern = tuple(self.loc_stream[i : i + self.window])
            next_loc = self.loc_stream[i + self.window]
            self.loc_probs[loc_pattern][next_loc] += 1

            # Momentum pattern and the next momentum
            mom_pattern = tuple(self.mom_stream[i : i + self.window])
            next_mom = self.mom_stream[i + self.window]
            self.mom_probs[mom_pattern][next_mom] += 1

    def get_probability(self, counter, target):
        total = sum(counter.values())
        if total == 0: return 0.0
        return counter[target] / total

    def complete_pattern(self, input_locs, input_moms):
        """
        Takes a synced pattern of locations and the momentums that reached them.
        Returns the likely next location.
        """
        if len(input_locs) != self.window or len(input_moms) != self.window:
            return "Error: Input must match window size"

        hist_loc = tuple(input_locs)
        hist_mom = tuple(input_moms)
        
        time.sleep(0.1)
        print(f"\n[Input Word] Loc: {''.join(input_locs)} | Mom: {''.join(input_moms)}")

        last_char = input_locs[-1]
        last_idx = BUCKET_CHARS.index(last_char)
        candidates = []
        
        # Possible next steps
        transitions = [('U', 1), ('D', -1), ('F', 0)]
        
        for move, idx_change in transitions:
            next_idx = last_idx + idx_change
            if 0 <= next_idx < NUM_BUCKETS:
                next_loc = BUCKET_CHARS[next_idx]
                
                # Probability from Location history (ABC part)
                p_abc = self.get_probability(self.loc_probs[hist_loc], next_loc)
                
                # Probability from Momentum history (UDF part)
                p_udf = self.get_probability(self.mom_probs[hist_mom], move)
                
                # Combined dependent pattern completion probability
                total_score = p_abc + p_udf
                
                candidates.append({
                    'loc': next_loc, 'mom': move, 'score': total_score,
                    'p_abc': p_abc, 'p_udf': p_udf
                })

        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        print("  Candidate Scores:")
        for c in candidates:
            time.sleep(0.1)
            print(f"    -> {c['loc']} (via {c['mom']}): {c['score']:.2f} [ABC: {c['p_abc']:.2f}, UDF: {c['p_udf']:.2f}]")

        winner = candidates[0]
        time.sleep(0.1)
        print(f"[Prediction] Next Location: {winner['loc']}")
        return winner['loc']

if __name__ == "__main__":
    # 1. Setup Data
    print(f"--- SIMULATING {SIMULATION_YEARS} YEARS OF MARKET DATA ---")
    prices = [100000] # Start at 100k
    for _ in range(365 * SIMULATION_YEARS):
        change = random.gauss(50, 2500)
        prices.append(max(MIN_PRICE, min(MAX_PRICE, prices[-1] + change)))

    # 2. Train Completer
    completer = StockPatternCompleter(prices)

    # 3. Demonstration
    # Note: For the input word, we provide the Location AND the Momentum that reached it.
    
    print(f"\n--- TEST 1: Sideways in Zone K ---")
    # Location is K, reached by Flat moves
    completer.complete_pattern(['K', 'K', 'K', 'K'], ['F', 'F', 'F', 'F'])

    print(f"\n--- TEST 2: Rally in Rare Zone (A to D) ---")
    # Location moves A->B->C->D, reached by Up moves
    # Even if ABC history is 0, UDF history will drive the result.
    completer.complete_pattern(['A', 'B', 'C', 'D'], ['U', 'U', 'U', 'U'])

    print(f"\n--- TEST 3: Volatile Reversal ---")
    # Down Down Down reaching Zone J, but then Flat
    completer.complete_pattern(['M', 'L', 'K', 'J'], ['D', 'D', 'D', 'F'])