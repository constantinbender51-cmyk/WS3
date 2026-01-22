import requests
import math
import pandas as pd
from collections import Counter

# ==========================================
# PARAMETERS
# ==========================================
TIMEFRAME = '1h'       # e.g., '1m', '5m', '1h', '1d'
SYMBOL = 'BTCUSDT'     # Binance symbol
START = '2023-01-01'   # Start Date
END = '2023-06-01'     # End Date

A_ROUND = 0.5          # a%: Rounding step (floor)
B_SPLIT = 70           # b%: Percentage of data for Split 1 (Training)
C_TOP = 10             # c%: Percentage of top frequent sequences to keep
D_LEN = 3              # d: Total length of candle sequence
E_SIM = 0.1            # e%: Similarity threshold (0.1 = 10% difference)

# ==========================================
# FUNCTIONS
# ==========================================

def fetch(timeframe, symbol, start, end):
    """
    Fetches OHLC data from Binance.
    Returns a list of [Open, High, Low, Close] floats.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Convert dates to milliseconds timestamps
    start_ts = int(pd.Timestamp(start).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end).timestamp() * 1000)
    
    data = []
    current_start = start_ts
    
    print(f"Fetching {symbol} {timeframe} from {start} to {end}...")
    
    while current_start < end_ts:
        params = {
            'symbol': symbol,
            'interval': timeframe,
            'startTime': current_start,
            'endTime': end_ts,
            'limit': 1000
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            klines = response.json()
            
            if not klines:
                break
                
            # Parse Open, High, Low, Close (indices 1, 2, 3, 4)
            # kline format: [open_time, open, high, low, close, vol, ...]
            for k in klines:
                # We strictly need O, H, L, C as floats
                ohlc = [float(k[1]), float(k[2]), float(k[3]), float(k[4])]
                data.append(ohlc)
            
            # Update start time for next batch (last close time + 1ms)
            current_start = klines[-1][6] + 1
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    print(f"Fetched {len(data)} candles.")
    return data

def deriveround(ohlc_data, a):
    """
    Applies (Candle[i] - Candle[i-1]) / Candle[i-1] for O, H, L, C.
    Rounds result to floor based on a.
    Returns list of [dO, dH, dL, dC].
    """
    derived = []
    
    # Start from 1 since we need i-1
    for i in range(1, len(ohlc_data)):
        curr = ohlc_data[i]
        prev = ohlc_data[i-1]
        
        # Calculate % change * 100 to make 'a' comparable (e.g. 1% = 1.0)
        # Using element-wise calculation
        d_row = []
        for j in range(4): # 0:Open, 1:High, 2:Low, 3:Close
            if prev[j] == 0:
                change = 0.0
            else:
                change = ((curr[j] - prev[j]) / prev[j]) * 100.0
            
            # Round to floor based on a
            # logic: floor(1.6 / 1) * 1 = 1.0
            rounded = math.floor(change / a) * a
            d_row.append(rounded)
            
        derived.append(tuple(d_row)) # Use tuple to make it hashable later
        
    return derived

def split(derived_data, b):
    """
    Splits data into b% training and (100-b)% testing.
    """
    split_idx = int(len(derived_data) * (b / 100.0))
    train = derived_data[:split_idx]
    test = derived_data[split_idx:]
    return train, test

def gettop(train_data, c, d):
    """
    Finds sequences of length d in train_data.
    Returns the top c% most frequent sequences.
    """
    sequences = []
    
    # Generate all sequences of length d
    for i in range(len(train_data) - d + 1):
        # Create a tuple of tuples to represent the sequence (hashable)
        seq = tuple(train_data[i : i+d])
        sequences.append(seq)
        
    if not sequences:
        return []

    # Count frequencies
    counts = Counter(sequences)
    unique_seqs = list(counts.items()) 
    
    # Sort by frequency (descending)
    unique_seqs.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate how many to keep (c%)
    # If c=10 and there are 100 unique sequences, keep top 10.
    limit = int(len(unique_seqs) * (c / 100.0))
    if limit < 1:
        limit = 1
        
    top_sequences = [item[0] for item in unique_seqs[:limit]]
    
    print(f"Identified {len(top_sequences)} top sequences from {len(unique_seqs)} unique patterns.")
    return top_sequences

def is_similar(seq1, seq2, e):
    """
    Helper to check if two sequences (beginning parts) are similar by e%.
    seq1, seq2 are lists of [dO, dH, dL, dC] tuples.
    Similarity: abs(v2 - v1) / abs(v1) < e (for every component)
    """
    # They must be same length
    if len(seq1) != len(seq2):
        return False
        
    # Iterate through every candle in the sequence
    for k in range(len(seq1)):
        candle1 = seq1[k]
        candle2 = seq2[k]
        
        # Iterate through O, H, L, C
        for val1, val2 in zip(candle1, candle2):
            # Avoid division by zero
            if val1 == 0:
                # If base is 0, strict match required or epsilon handling?
                # If val1 is 0 and val2 is 0, difference is 0. 
                # If val1 is 0 and val2 is not, error is infinite -> Fail
                if val2 != 0:
                    return False
                continue
            
            diff = abs(val2 - val1)
            rel_diff = diff / abs(val1)
            
            if rel_diff >= e:
                return False
                
    return True

def completesimilarbeginnings(test_data, top_sequences, d, e):
    """
    Scans test_data for beginnings similar to top_sequences.
    Predicts the d-th candle.
    Returns list of (Prediction_Close_Change, Actual_Close_Change)
    """
    predictions = []
    
    # The beginning length is d-1
    begin_len = d - 1
    
    if begin_len < 1:
        print("Sequence length d must be at least 2.")
        return []

    # Iterate through test data
    # We need enough data for the window (begin_len) AND the outcome (1 candle)
    # Total needed: d
    for i in range(len(test_data) - d + 1):
        current_window = test_data[i : i + begin_len]
        actual_outcome = test_data[i + begin_len] # This is the d-th candle
        
        prediction = None
        
        # Check against top sequences (prioritizing most frequent as they are sorted)
        for seq in top_sequences:
            top_beginning = seq[:begin_len]
            
            if is_similar(top_beginning, current_window, e):
                # Match found. 
                # Predicted outcome is the d-th candle of the top sequence
                # We specifically care about the Close change (index 3) for direction
                prediction_candle = seq[begin_len]
                prediction = prediction_candle[3] # Close change
                break # Stop after finding the most frequent match
        
        if prediction is not None:
            actual = actual_outcome[3] # Close change
            predictions.append((prediction, actual))
            
    return predictions

def printaccuracy(results):
    """
    Calculates accuracy of directional prediction.
    Excludes flat predictions (0) and flat outcomes (0).
    """
    valid_count = 0
    correct_count = 0
    
    for pred, actual in results:
        # Exclude flats
        if pred == 0 or actual == 0:
            continue
            
        valid_count += 1
        
        # Check direction
        if (pred > 0 and actual > 0) or (pred < 0 and actual < 0):
            correct_count += 1
            
    if valid_count == 0:
        print("No valid non-flat predictions found.")
    else:
        accuracy = (correct_count / valid_count) * 100
        print(f"Total Predictions (matched patterns): {len(results)}")
        print(f"Valid Directional Pairs (non-flat): {valid_count}")
        print(f"Accuracy: {accuracy:.2f}%")

def main():
    # 1. Fetch
    raw_data = fetch(TIMEFRAME, SYMBOL, START, END)
    
    if len(raw_data) < D_LEN:
        print("Not enough data fetched.")
        return

    # 2. Derive & Round
    derived_data = deriveround(raw_data, A_ROUND)
    
    # 3. Split
    train_data, test_data = split(derived_data, B_SPLIT)
    print(f"Split data: {len(train_data)} training, {len(test_data)} testing.")
    
    # 4. Get Top Sequences
    # Warning: d must be >= 2 for "beginning" logic to work
    if D_LEN < 2:
        print("D_LEN must be >= 2")
        return
        
    top_seqs = gettop(train_data, C_TOP, D_LEN)
    
    if not top_seqs:
        print("No sequences found.")
        return

    # 5. Complete Similar Beginnings & Predict
    results = completesimilarbeginnings(test_data, top_seqs, D_LEN, E_SIM)
    
    # 6. Print Accuracy
    printaccuracy(results)

if __name__ == "__main__":
    main()
