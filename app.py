import random
import collections
import string

# --- CONFIGURATION ---
MIN_PRICE = 0
MAX_PRICE = 200000
NUM_BUCKETS = 20
BUCKET_SIZE = (MAX_PRICE - MIN_PRICE) / NUM_BUCKETS
BUCKET_CHARS = string.ascii_uppercase[:NUM_BUCKETS] # A-T
WINDOW_SIZE = 4
SIMULATION_YEARS = 50

class StockPatternCompleter:
    def __init__(self, price_history, window_size=WINDOW_SIZE):
        self.prices = price_history
        self.window = window_size
        
        # 1. Ingest Data
        # We need both streams to build our probability distributions
        self.loc_stream, self.mom_stream = self.discretize_data(self.prices)
        
        # 2. Train Memory Banks
        self.loc_probs = collections.defaultdict(collections.Counter)
        self.mom_probs = collections.defaultdict(collections.Counter)
        self.train()

    def get_bucket_index(self, price):
        if price >= MAX_PRICE: return NUM_BUCKETS - 1
        if price <= MIN_PRICE: return 0
        return int((price - MIN_PRICE) / BUCKET_SIZE)

    def discretize_data(self, prices):
        """
        Translates raw prices into the two fundamental languages.
        """
        locs = []
        moms = []
        for i in range(1, len(prices)):
            prev = self.get_bucket_index(prices[i-1])
            curr = self.get_bucket_index(prices[i])
            
            # Location (The "Word")
            locs.append(BUCKET_CHARS[curr])
            
            # Momentum (The "Grammar")
            if curr > prev: moms.append('U')
            elif curr < prev: moms.append('D')
            else: moms.append('F')
        return locs, moms

    def train(self):
        print(f"Training on {len(self.loc_stream)} cycles...")
        # Train Location Model: P(NextLocation | PreviousLocations)
        for i in range(len(self.loc_stream) - self.window):
            pattern = tuple(self.loc_stream[i : i + self.window])
            next_val = self.loc_stream[i + self.window]
            self.loc_probs[pattern][next_val] += 1

        # Train Momentum Model: P(NextMove | PreviousMoves)
        for i in range(len(self.mom_stream) - self.window):
            pattern = tuple(self.mom_stream[i : i + self.window])
            next_val = self.mom_stream[i + self.window]
            self.mom_probs[pattern][next_val] += 1

    def get_probability(self, counter, target):
        total = sum(counter.values())
        if total == 0: return 0.0
        return counter[target] / total

    def complete_pattern(self, incomplete_loc_pattern):
        """
        Takes an incomplete Location Pattern (e.g. ['K', 'K', 'L'])
        And returns the most likely Completed Location Pattern.
        """
        # 1. Derive the Momentum Pattern from the Location Pattern
        # (Translation Step: Location -> Momentum)
        current_moms = []
        for i in range(1, len(incomplete_loc_pattern)):
            prev_char = incomplete_loc_pattern[i-1]
            curr_char = incomplete_loc_pattern[i]
            prev_idx = BUCKET_CHARS.index(prev_char)
            curr_idx = BUCKET_CHARS.index(curr_char)
            
            if curr_idx > prev_idx: current_moms.append('U')
            elif curr_idx < prev_idx: current_moms.append('D')
            else: current_moms.append('F')

        # 2. Setup History Tuples
        hist_loc = tuple(incomplete_loc_pattern)
        hist_mom = tuple(current_moms) # Note: Momentum history is 1 shorter than Loc history

        # We need the momentum window to match the training size
        # If input is short, we use what we have.
        
        print(f"\n[Input] Location Pattern: {incomplete_loc_pattern}")
        print(f"[Derived] Momentum Pattern: {current_moms}")

        # 3. Determine Candidates for the NEXT Location
        last_char = incomplete_loc_pattern[-1]
        last_idx = BUCKET_CHARS.index(last_char)
        
        candidates = []
        
        # Physics: We can only move Up, Down, or Flat from current location
        transitions = [('U', 1), ('D', -1), ('F', 0)]
        
        for move, idx_change in transitions:
            next_idx = last_idx + idx_change
            
            # Boundary Check
            if 0 <= next_idx < NUM_BUCKETS:
                next_loc = BUCKET_CHARS[next_idx]
                
                # --- PROBABILITY CALCULATION ---
                # We calculate the likelihood of this specific Next Location
                
                # A. Location Probability (Specific History)
                # "How often does K,K,L lead to M?"
                # If pattern is new, this is 0.0
                p_loc = self.get_probability(self.loc_probs[hist_loc], next_loc)
                
                # B. Momentum Probability (General History)
                # "How often does F,U lead to U?"
                # This provides the "Grammar" when the specific word is unknown.
                p_mom = self.get_probability(self.mom_probs[hist_mom], move)
                
                total_score = p_loc + p_mom
                
                candidates.append({
                    'next_loc': next_loc,
                    'move': move,
                    'score': total_score,
                    'debug': f"P_loc({p_loc:.2f}) + P_mom({p_mom:.2f})"
                })

        # 4. Pick Winner
        candidates.sort(key=lambda x: x['score'], reverse=True)
        winner = candidates[0]
        
        print(f"  Candidates:")
        for c in candidates:
             print(f"    -> Extend to '{c['next_loc']}' ({c['move']}): Score {c['score']:.2f} [{c['debug']}]")

        # 5. Return the Completed Location Pattern
        completed_pattern = list(incomplete_loc_pattern)
        completed_pattern.append(winner['next_loc'])
        
        print(f"[Result] Completed Location Pattern: {completed_pattern}")
        return completed_pattern

# --- SIMULATION ---

if __name__ == "__main__":
    # Generate Data
    print(f"--- GENERATING MARKET DATA ({SIMULATION_YEARS} Years) ---")
    prices = [100000]
    for _ in range(365 * SIMULATION_YEARS):
        change = random.gauss(50, 2000)
        prices.append(max(MIN_PRICE, min(MAX_PRICE, prices[-1] + change)))

    completer = StockPatternCompleter(prices)

    # TEST 1: Common Pattern
    # Input: K -> K -> K -> K (Consolidation in Middle)
    print(f"\n--- TEST 1: Middle Consolidation ---")
    completer.complete_pattern(['K', 'K', 'K', 'K'])

    # TEST 2: The "New Area" Test
    # Input: A -> B -> C -> D (Strong Rally at Bottom)
    # The Model likely has NEVER seen 'A,B,C,D'. P_loc will be 0.
    # But it has seen 'U,U,U' many times. P_mom will be high for 'U'.
    # Result should be extension to 'E'.
    print(f"\n--- TEST 2: Rally in Rare Zone (Zone A->D) ---")
    completer.complete_pattern(['A', 'B', 'C', 'D'])