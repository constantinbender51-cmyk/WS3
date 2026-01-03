import random
import collections

# --- CONFIGURATION PARAMETERS ---

# Linear Bucketing Settings
MIN_PRICE = 0
MAX_PRICE = 200000
NUM_BUCKETS = 20
BUCKET_SIZE = (MAX_PRICE - MIN_PRICE) / NUM_BUCKETS  # 10,000 per bucket

# Pattern Recognition Settings
WINDOW_SIZE = 4          # A "word" is 4 days of bucket movements (e.g., "FUUD")
ALPHABET_CHARS = ['U', 'D', 'F'] 

# Simulation Settings
SIMULATION_YEARS = 20    
START_PRICE = 100000     
DAILY_VOLATILITY = 2000  # $2000 std dev to ensure we cross bucket lines occasionally
DAILY_DRIFT = 50         # Slight upward bias ($)

# --- LOGIC ---

class StockSpellChecker:
    def __init__(self, price_history, window_size=WINDOW_SIZE):
        self.prices = price_history
        self.window_size = window_size
        
        # 1. Translate Prices to "Language" (Bucket Transitions)
        self.sequence_string = self.encode_prices(self.prices)
        
        # 2. Train the "Dictionary"
        self.pattern_counts = collections.Counter()
        self.train_patterns()

    def get_bucket_index(self, price):
        """Helper to find which absolute zone a price is in."""
        if price >= MAX_PRICE: return NUM_BUCKETS - 1
        if price <= MIN_PRICE: return 0
        return int((price - MIN_PRICE) / BUCKET_SIZE)
        
    def encode_prices(self, prices):
        """
        Converts price history to U/D/F based on crossing absolute bucket lines.
        """
        encoded = []
        for i in range(1, len(prices)):
            prev_idx = self.get_bucket_index(prices[i-1])
            curr_idx = self.get_bucket_index(prices[i])
            
            if curr_idx > prev_idx:
                char = 'U' # Crossed into a higher bucket
            elif curr_idx < prev_idx:
                char = 'D' # Crossed into a lower bucket
            else:
                char = 'F' # Stayed in the same absolute bucket
                
            encoded.append(char)
            
        return "".join(encoded)

    def train_patterns(self):
        """
        Learns all historical patterns of length `window_size`.
        """
        print(f"Training on {len(self.sequence_string)} days of data...")
        print(f"Definition: 'U' = Cross Bucket Line Up | 'D' = Cross Line Down | 'F' = Stay in Bucket")
        
        self.transitions = collections.defaultdict(collections.Counter)
        
        for i in range(len(self.sequence_string) - self.window_size):
            pattern = self.sequence_string[i : i + self.window_size]
            next_move = self.sequence_string[i + self.window_size]
            
            self.transitions[pattern][next_move] += 1
            self.pattern_counts[pattern] += 1

    def predict_next(self, recent_pattern):
        """
        Finds the most likely next bucket movement.
        """
        if len(recent_pattern) != self.window_size:
            print(f"Error: Pattern length must be {self.window_size}")
            return None

        if recent_pattern in self.transitions:
            candidates = self.transitions[recent_pattern]
            total = sum(candidates.values())
            
            print(f"\nAnalysis for pattern [{recent_pattern}]:")
            print(f"  Historical occurrences: {self.pattern_counts[recent_pattern]}")
            
            predictions = []
            for move, count in candidates.most_common():
                prob = count / total
                predictions.append((move, prob))
                
                # Contextual description
                desc = ""
                if move == 'U': desc = "(Jump to higher zone)"
                elif move == 'D': desc = "(Drop to lower zone)"
                else: desc = "(Stay in current zone)"
                
                print(f"  -> Next move '{move}' {desc}: {prob*100:.1f}%")
            
            return predictions[0][0]
        else:
            return self.find_fuzzy_match(recent_pattern)

    def find_fuzzy_match(self, pattern):
        """
        Corrects 'typos' in market patterns (rare events) to find closest history.
        """
        print(f"\nPattern [{pattern}] never seen before. Fuzzy matching...")
        
        candidates = []
        
        # 1-Edit Distance
        for i in range(len(pattern)):
            for char in ALPHABET_CHARS:
                if char == pattern[i]: continue
                
                edited_pattern = pattern[:i] + char + pattern[i+1:]
                
                if edited_pattern in self.transitions:
                    weight = self.pattern_counts[edited_pattern]
                    candidates.append((edited_pattern, weight))
        
        if not candidates:
            return "Unknown"
            
        best_match = max(candidates, key=lambda x: x[1])[0]
        print(f"  Closest historical match: [{best_match}]")
        return self.predict_next(best_match)

# --- SIMULATION ---

if __name__ == "__main__":
    print(f"--- PARAMETERS ---")
    print(f"Buckets: {NUM_BUCKETS} (Size: ${BUCKET_SIZE:,.0f})")
    
    # 1. Generate Fake Stock Data
    # We need high volatility or specific movement to trigger bucket crossings
    print(f"\nGenerating {SIMULATION_YEARS} years of synthetic stock data...")
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
    
    # Scenario 1: Consolidation (Staying in same bucket for 4 days)
    # Even if price wiggles, if it doesn't cross $10k lines, it is "FFFF"
    test_1 = "FFFF"
    print(f"\n--- TEST 1: Consolidation ({test_1}) ---")
    model.predict_next(test_1)

    # Scenario 2: Rally (Crossing lines upwards repeatedly)
    # e.g., $18k -> $21k (U) -> $31k (U) -> $42k (U) -> $55k (U)
    test_2 = "UUUU" 
    print(f"\n--- TEST 2: Breakout Rally ({test_2}) ---")
    model.predict_next(test_2)
    
    # Scenario 3: Volatile Chop (Crossing lines up and down)
    test_3 = "UDUD"
    print(f"\n--- TEST 3: Volatility ({test_3}) ---")
    model.predict_next(test_3)