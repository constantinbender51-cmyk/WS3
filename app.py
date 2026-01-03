import random
import collections
import math

# --- CONFIGURATION PARAMETERS ---

# Simulation Settings
SIMULATION_YEARS = 10
START_PRICE = 10000
DAILY_VOLATILITY = 1.5   # Standard deviation in %
DAILY_DRIFT = 0.05       # Mean daily return in %

# Pattern Recognition Settings
WINDOW_SIZE = 4          # How many days make up a "word" (e.g., 4 days = "ABCD")
ALPHABET_CHARS = 'ABCDEFG'

# Encoding Thresholds (% change to Letter mapping)
# Format: (Upper Bound Limit, Character)
# Note: The logic processes these in order. Anything above the last limit gets the final character.
THRESHOLDS = [
    (-2.0, 'A'), # Crash (<= -2.0%)
    (-1.0, 'B'), # Drop  (<= -1.0%)
    (-0.2, 'C'), # Dip   (<= -0.2%)
    ( 0.2, 'D'), # Flat  (<=  0.2%)
    ( 1.0, 'E'), # Rise  (<=  1.0%)
    ( 2.0, 'F'), # Rally (<=  2.0%)
]
DEFAULT_CHAR = 'G' # Moon  (>   2.0%)

# --- LOGIC ---

class StockSpellChecker:
    def __init__(self, price_history, window_size=WINDOW_SIZE):
        self.prices = price_history
        self.window_size = window_size
        
        # 1. Translate Prices to "Language"
        self.sequence_string = self.encode_prices(self.prices)
        
        # 2. Train the "Dictionary"
        self.pattern_counts = collections.Counter()
        self.train_patterns()
        
    def encode_prices(self, prices):
        """
        Converts price changes to a string of characters based on global THRESHOLDS.
        """
        encoded = []
        for i in range(1, len(prices)):
            pct_change = (prices[i] - prices[i-1]) / prices[i-1] * 100
            
            char = DEFAULT_CHAR
            for limit, c in THRESHOLDS:
                if pct_change <= limit:
                    char = c
                    break
            encoded.append(char)
            
        return "".join(encoded)

    def train_patterns(self):
        """
        Learns all historical patterns of length `window_size`.
        """
        print(f"Training on {len(self.sequence_string)} days of data using window size {self.window_size}...")
        
        # Store patterns and what came *immediately after* them
        self.transitions = collections.defaultdict(collections.Counter)
        
        for i in range(len(self.sequence_string) - self.window_size):
            pattern = self.sequence_string[i : i + self.window_size]
            next_move = self.sequence_string[i + self.window_size]
            
            self.transitions[pattern][next_move] += 1
            self.pattern_counts[pattern] += 1

    def predict_next(self, recent_pattern):
        """
        The 'Autocomplete' logic.
        """
        if len(recent_pattern) != self.window_size:
            print(f"Error: Pattern length must be {self.window_size}")
            return None

        if recent_pattern in self.transitions:
            candidates = self.transitions[recent_pattern]
            total = sum(candidates.values())
            
            print(f"\nAnalysis for pattern [{recent_pattern}]:")
            print(f"  Historical occurrences: {self.pattern_counts[recent_pattern]}")
            
            # Calculate Probability P(Next | Pattern)
            predictions = []
            for move, count in candidates.most_common():
                prob = count / total
                predictions.append((move, prob))
                print(f"  -> Next day '{move}': {prob*100:.1f}%")
            
            return predictions[0][0] # Return most likely
        else:
            return self.find_fuzzy_match(recent_pattern)

    def find_fuzzy_match(self, pattern):
        """
        The 'Spell Correction' logic.
        If the exact pattern never happened, look for 1-edit distance historical matches.
        """
        print(f"\nPattern [{pattern}] never seen before. Looking for similar historical patterns (Edits)...")
        
        candidates = []
        
        # 1-Edit Distance (Hamming distance logic for time series)
        for i in range(len(pattern)):
            for char in ALPHABET_CHARS:
                if char == pattern[i]: continue
                
                # Form a new pattern (e.g., change day 2 from 'A' to 'B')
                edited_pattern = pattern[:i] + char + pattern[i+1:]
                
                if edited_pattern in self.transitions:
                    # Weight the probability by how common the "corrected" pattern is
                    weight = self.pattern_counts[edited_pattern]
                    candidates.append((edited_pattern, weight))
        
        if not candidates:
            return "Unknown"
            
        # Get the most common similar pattern
        best_match = max(candidates, key=lambda x: x[1])[0]
        print(f"  Closest historical match: [{best_match}]")
        return self.predict_next(best_match)

# --- SIMULATION ---

if __name__ == "__main__":
    # 1. Generate Fake Stock Data (Random Walk)
    print(f"Generating {SIMULATION_YEARS} years of synthetic stock data...")
    prices = [START_PRICE]
    for _ in range(365 * SIMULATION_YEARS):
        change = random.gauss(DAILY_DRIFT, DAILY_VOLATILITY)
        new_price = prices[-1] * (1 + change/100)
        prices.append(new_price)

    # 2. Initialize the "Corrector"
    model = StockSpellChecker(prices)

    # 3. Test scenarios
    
    # Common Pattern Construction (e.g., Flat Flat Rise Rise)
    # We use characters from ALPHABET_CHARS to build test cases
    test_pattern_common = "DDEE" 
    print(f"\n--- TEST 1: Common Pattern ({test_pattern_common}) ---")
    prediction = model.predict_next(test_pattern_common)
    print(f"Prediction: Market will likely move '{prediction}' next.")

    # Rare Pattern Construction (e.g., Crash Moon Crash Moon)
    test_pattern_rare = "AGAG"
    print(f"\n--- TEST 2: Rare/Unknown Pattern ({test_pattern_rare}) ---")
    prediction = model.predict_next(test_pattern_rare)
    print(f"Prediction: Market will likely move '{prediction}' next.")

    print("\n--- Legend ---")
    for limit, char in THRESHOLDS:
        print(f"{char}: <= {limit}%")
    print(f"{DEFAULT_CHAR}: > {THRESHOLDS[-1][0]}%")