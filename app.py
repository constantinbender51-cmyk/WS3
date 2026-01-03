import random
import collections
import math

class StockSpellChecker:
    def __init__(self, price_history):
        self.prices = price_history
        # 1. Translate Prices to "Language"
        self.sequence_string = self.encode_prices(self.prices)
        # 2. Train the "Dictionary" (Corpus of 5-day patterns)
        self.pattern_counts = collections.Counter()
        self.train_patterns(window_size=4) # Learn 4-day sequences
        
    def encode_prices(self, prices):
        """
        Converts price changes to a string of characters (The Alphabet).
        We don't use raw price (0-10k), we use % change.
        """
        encoded = []
        for i in range(1, len(prices)):
            pct_change = (prices[i] - prices[i-1]) / prices[i-1] * 100
            
            # Discretize into categories (The Alphabet)
            if pct_change <= -2.0: char = 'A' # Crash
            elif pct_change <= -1.0: char = 'B' # Drop
            elif pct_change <= -0.2: char = 'C' # Dip
            elif pct_change <=  0.2: char = 'D' # Flat
            elif pct_change <=  1.0: char = 'E' # Rise
            elif pct_change <=  2.0: char = 'F' # Rally
            else:                    char = 'G' # Moon
            encoded.append(char)
        return "".join(encoded)

    def train_patterns(self, window_size):
        """
        Learns all historical patterns of length `window_size`.
        This creates our 'Dictionary'.
        """
        print(f"Training on {len(self.sequence_string)} days of data...")
        # Store patterns and what came *immediately after* them
        self.transitions = collections.defaultdict(collections.Counter)
        
        for i in range(len(self.sequence_string) - window_size):
            pattern = self.sequence_string[i : i + window_size]
            next_move = self.sequence_string[i + window_size]
            self.transitions[pattern][next_move] += 1
            self.pattern_counts[pattern] += 1

    def predict_next(self, recent_pattern):
        """
        The 'Autocomplete' logic.
        Given a pattern like 'CCDE', what is the most probable next letter?
        """
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
        If the exact pattern 'ABBA' never happened, look for 'ABBB' or 'ABCA'.
        This is equivalent to Norvig's edits1 function.
        """
        print(f"\nPattern [{pattern}] never seen before. Looking for similar historical patterns (Edits)...")
        
        candidates = []
        alphabet = 'ABCDEFG'
        
        # 1-Edit Distance (Hamming distance logic for time series)
        # We try changing one day in the past to see if that pattern exists in history
        for i in range(len(pattern)):
            for char in alphabet:
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

# --- Simulation ---

# 1. Generate Fake Stock Data (Random Walk)
print("Generating 10 years of synthetic stock data...")
prices = [10000] # Start at 10k
for _ in range(365 * 10):
    change = random.gauss(0.05, 1.5) # Slight upward drift, 1.5% volatility
    new_price = prices[-1] * (1 + change/100)
    prices.append(new_price)

# 2. Initialize the "Corrector"
model = StockSpellChecker(prices)

# 3. Test a few scenarios
print("\n--- TEST 1: Common Pattern ---")
# Let's say market was Flat, Flat, Rise, Rise (DDEE)
prediction = model.predict_next("DDEE")
print(f"Prediction: Market will likely move '{prediction}' next.")

print("\n--- TEST 2: Rare/Unknown Pattern (Requires 'Spelling Correction') ---")
# Let's say we had a Crash, then Moon, then Crash, then Moon (AGAG) - highly unlikely
prediction = model.predict_next("AGAG")
print(f"Prediction: Market will likely move '{prediction}' next.")

print("\n--- Legend ---")
print("A: Crash (<-2%) | B: Drop | C: Dip | D: Flat | E: Rise | F: Rally | G: Moon (>+2%)")