import random
import collections
import string

# --- CONFIGURATION PARAMETERS ---

# Linear Bucketing Settings
MIN_PRICE = 0
MAX_PRICE = 200000
NUM_BUCKETS = 20
BUCKET_SIZE = (MAX_PRICE - MIN_PRICE) / NUM_BUCKETS

# Generate Bucket Alphabet (A-T)
BUCKET_CHARS = string.ascii_uppercase[:NUM_BUCKETS]

# Movement Alphabet
MOVE_CHARS = ['U', 'D', 'F']

# Pattern Recognition Settings
WINDOW_SIZE = 4          # A "word" is 4 days of (Bucket+Move) tokens
                         # Example: ['KF', 'KF', 'LU', 'LF']

# Simulation Settings
SIMULATION_YEARS = 50    # Increased years to help fill the larger state space
START_PRICE = 100000     
DAILY_VOLATILITY = 2000  
DAILY_DRIFT = 50         

# --- LOGIC ---

class StockSpellChecker:
    def __init__(self, price_history, window_size=WINDOW_SIZE):
        self.prices = price_history
        self.window_size = window_size
        
        # 1. Translate Prices to "Language" (List of Combined Tokens)
        self.sequence = self.encode_prices(self.prices)
        
        # 2. Train the "Dictionary"
        self.pattern_counts = collections.Counter()
        self.valid_tokens = set(self.sequence) # Remember which tokens actually exist
        self.train_patterns()

    def get_bucket_index(self, price):
        """Helper to find which absolute zone a price is in."""
        if price >= MAX_PRICE: return NUM_BUCKETS - 1
        if price <= MIN_PRICE: return 0
        return int((price - MIN_PRICE) / BUCKET_SIZE)
        
    def encode_prices(self, prices):
        """
        Converts price history to a list of tokens.
        Token Format: "{BucketChar}{MoveChar}" (e.g., "KF")
        """
        encoded_tokens = []
        for i in range(1, len(prices)):
            prev_idx = self.get_bucket_index(prices[i-1])
            curr_idx = self.get_bucket_index(prices[i])
            
            # Determine Movement
            if curr_idx > prev_idx:
                move = 'U' # Crossed Up
            elif curr_idx < prev_idx:
                move = 'D' # Crossed Down
            else:
                move = 'F' # Flat (Same Bucket)
            
            # Determine Bucket Char
            bucket_char = BUCKET_CHARS[curr_idx]
            
            # Combine: "K" + "F" -> "KF"
            token = f"{bucket_char}{move}"
            encoded_tokens.append(token)
            
        return encoded_tokens

    def train_patterns(self):
        """
        Learns all historical patterns of length `window_size`.
        Keys are now Tuples of strings, e.g. ('KF', 'KF', 'LU', 'LF')
        """
        print(f"Training on {len(self.sequence)} days of data...")
        print(f"Vocabulary: {len(self.valid_tokens)} unique states seen (out of {NUM_BUCKETS*3} possible).")
        
        self.transitions = collections.defaultdict(collections.Counter)
        
        for i in range(len(self.sequence) - self.window_size):
            # Create a tuple for the pattern (hashable)
            pattern = tuple(self.sequence[i : i + self.window_size])
            next_move = self.sequence[i + self.window_size]
            
            self.transitions[pattern][next_move] += 1
            self.pattern_counts[pattern] += 1

    def predict_next(self, recent_pattern_list):
        """
        Finds the most likely next token.
        input: list of tokens e.g. ['KF', 'KF', 'LU', 'LF']
        """
        # Convert list to tuple for dictionary lookup
        pattern_tuple = tuple(recent_pattern_list)

        if len(pattern_tuple) != self.window_size:
            print(f"Error: Pattern length must be {self.window_size}")
            return None

        if pattern_tuple in self.transitions:
            candidates = self.transitions[pattern_tuple]
            total = sum(candidates.values())
            
            print(f"\nAnalysis for pattern {pattern_tuple}:")
            print(f"  Historical occurrences: {self.pattern_counts[pattern_tuple]}")
            
            predictions = []
            for token, count in candidates.most_common():
                prob = count / total
                predictions.append((token, prob))
                
                # Contextual description
                b_char = token[0]
                m_char = token[1]
                desc = "Unknown"
                if m_char == 'U': desc = "Up into"
                elif m_char == 'D': desc = "Down into"
                elif m_char == 'F': desc = "Staying in"
                
                print(f"  -> Next: {token} ({desc} Zone {b_char}): {prob*100:.1f}%")
            
            return predictions[0][0]
        else:
            return self.find_fuzzy_match(pattern_tuple)

    def find_fuzzy_match(self, pattern_tuple):
        """
        Edits tokens to find a similar historical sequence.
        """
        print(f"\nPattern {pattern_tuple} never seen. Fuzzy matching...")
        
        candidates = []
        
        # 1-Edit Distance (Substitute 1 token in the sequence)
        # We only substitute with 'valid_tokens' that we've actually seen in history
        # to avoid searching 60 possibilities every time.
        for i in range(len(pattern_tuple)):
            for token in self.valid_tokens:
                if token == pattern_tuple[i]: continue
                
                # Create modified tuple
                edited_list = list(pattern_tuple)
                edited_list[i] = token
                edited_tuple = tuple(edited_list)
                
                if edited_tuple in self.transitions:
                    weight = self.pattern_counts[edited_tuple]
                    candidates.append((edited_tuple, weight))
        
        if not candidates:
            return "Unknown"
            
        best_match = max(candidates, key=lambda x: x[1])[0]
        print(f"  Closest historical match: {best_match}")
        return self.predict_next(best_match)

# --- SIMULATION ---

if __name__ == "__main__":
    print(f"--- PARAMETERS ---")
    print(f"Buckets: {NUM_BUCKETS} (Size: ${BUCKET_SIZE:,.0f})")
    print(f"Token Structure: [Bucket][Move] (e.g., 'KF' = Zone K, Flat)")
    
    # 1. Generate Fake Stock Data
    print(f"\nGenerating {SIMULATION_YEARS} years of data...")
    prices = [START_PRICE]
    for _ in range(365 * SIMULATION_YEARS):
        change = random.gauss(DAILY_DRIFT, DAILY_VOLATILITY)
        new_price = prices[-1] + change
        # Keep inside bounds
        if new_price < MIN_PRICE: new_price = MIN_PRICE + 100
        if new_price > MAX_PRICE: new_price = MAX_PRICE - 100
        prices.append(new_price)

    # 2. Initialize
    model = StockSpellChecker(prices)

    # 3. Test Scenarios
    
    # Scenario 1: Middle of the pack, doing nothing.
    # K is ~100k. F is Flat.
    test_1 = ['KF', 'KF', 'KF', 'KF']
    print(f"\n--- TEST 1: Consolidation in Zone K ({test_1}) ---")
    model.predict_next(test_1)

    # Scenario 2: Moving Up the ladder.
    # J(Up) -> K(Up) -> L(Up) -> M(Up)
    # This implies a very strong trend crossing buckets rapidly.
    test_2 = ['JU', 'KU', 'LU', 'MU']
    print(f"\n--- TEST 2: Breakout Rally ({test_2}) ---")
    model.predict_next(test_2)
    
    # Scenario 3: A Crash followed by a bounce (rare/complex)
    # L(Down) -> K(Down) -> J(Flat) -> J(Up)
    test_3 = ['LD', 'KD', 'JF', 'JU']
    print(f"\n--- TEST 3: Dip and Bounce ({test_3}) ---")
    model.predict_next(test_3)
    
    print("\n--- Bucket Legend (Partial) ---")
    for i in range(0, NUM_BUCKETS, 5): 
        char = BUCKET_CHARS[i]
        start = MIN_PRICE + (i * BUCKET_SIZE)
        print(f"Zone {char}: ${start:,.0f}+")