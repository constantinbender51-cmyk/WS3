import requests
import math
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import http.server
import socketserver
import os

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

PORT = 8080

# ==========================================
# FUNCTIONS
# ==========================================

def fetch(timeframe, symbol, start, end):
    """
    Fetches OHLC data from Binance.
    Returns a list of [Open, High, Low, Close] floats.
    """
    base_url = "https://api.binance.com/api/v3/klines"
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
                
            for k in klines:
                # O, H, L, C are at indices 1, 2, 3, 4
                ohlc = [float(k[1]), float(k[2]), float(k[3]), float(k[4])]
                data.append(ohlc)
            
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
    
    for i in range(1, len(ohlc_data)):
        curr = ohlc_data[i]
        prev = ohlc_data[i-1]
        
        d_row = []
        for j in range(4): 
            if prev[j] == 0:
                change = 0.0
            else:
                change = ((curr[j] - prev[j]) / prev[j]) * 100.0
            
            # Floor round
            rounded = math.floor(change / a) * a
            d_row.append(rounded)
            
        derived.append(tuple(d_row))
        
    return derived

def split(derived_data, b):
    split_idx = int(len(derived_data) * (b / 100.0))
    train = derived_data[:split_idx]
    test = derived_data[split_idx:]
    return train, test

def gettop(train_data, c, d):
    """
    Finds top c% most frequent sequences of length d in train_data.
    """
    sequences = []
    for i in range(len(train_data) - d + 1):
        seq = tuple(train_data[i : i+d])
        sequences.append(seq)
        
    if not sequences:
        return []

    counts = Counter(sequences)
    unique_seqs = list(counts.items()) 
    unique_seqs.sort(key=lambda x: x[1], reverse=True)
    
    limit = int(len(unique_seqs) * (c / 100.0))
    if limit < 1: limit = 1
        
    top_sequences = [item[0] for item in unique_seqs[:limit]]
    print(f"Identified {len(top_sequences)} top sequences.")
    return top_sequences

def is_similar(seq1, seq2, e):
    """
    Checks if seq1 and seq2 are similar by e%.
    """
    if len(seq1) != len(seq2): return False
        
    for k in range(len(seq1)):
        candle1 = seq1[k]
        candle2 = seq2[k]
        for val1, val2 in zip(candle1, candle2):
            if val1 == 0:
                if val2 != 0: return False
                continue
            
            diff = abs(val2 - val1)
            rel_diff = diff / abs(val1)
            
            if rel_diff >= e:
                return False
    return True

def completesimilarbeginnings(test_data, top_sequences, d, e):
    """
    Predicts using similarity logic. 
    Returns list of (Prediction, Actual) tuples.
    """
    predictions = []
    begin_len = d - 1
    
    for i in range(len(test_data) - d + 1):
        current_window = test_data[i : i + begin_len]
        actual_outcome = test_data[i + begin_len]
        
        prediction = None
        
        for seq in top_sequences:
            top_beginning = seq[:begin_len]
            if is_similar(top_beginning, current_window, e):
                prediction_candle = seq[begin_len]
                prediction = prediction_candle[3] # Predict Close change
                break 
        
        if prediction is not None:
            actual = actual_outcome[3] # Actual Close change
            predictions.append((prediction, actual))
            
    return predictions

def generate_plots_and_serve(results):
    """
    Calculates metrics, plots them, saves to file, and serves on port 8080.
    """
    valid_predictions = []
    
    # Process results for plotting
    cumulative_correct = []
    rolling_accuracy = []
    
    correct_count = 0
    total_valid = 0
    
    for pred, actual in results:
        # Exclude flat predictions or outcomes
        if pred == 0 or actual == 0:
            continue
            
        total_valid += 1
        
        # Check direction match
        is_correct = (pred > 0 and actual > 0) or (pred < 0 and actual < 0)
        
        if is_correct:
            correct_count += 1
            
        cumulative_correct.append(correct_count)
        rolling_accuracy.append((correct_count / total_valid) * 100)

    if total_valid == 0:
        print("No valid predictions to plot.")
        return

    # Create Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Subplot 1: Accuracy Over Time
    ax1.plot(rolling_accuracy, color='blue', label='Cumulative Accuracy %')
    ax1.set_title(f'Directional Accuracy Over Time (Final: {rolling_accuracy[-1]:.2f}%)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xlabel('Number of Trades')
    ax1.grid(True)
    ax1.legend()
    
    # Subplot 2: Correct Predictions Count
    ax2.plot(cumulative_correct, color='green', label='Correct Predictions')
    ax2.set_title('Cumulative Correct Predictions')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Number of Trades')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results.png')
    print("Plot saved to results.png")

    # Serve
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                html = f"""
                <html>
                <head><title>Trading Strategy Results</title></head>
                <body>
                    <h1>Strategy Performance</h1>
                    <p><strong>Total Valid Trades:</strong> {total_valid}</p>
                    <p><strong>Final Accuracy:</strong> {rolling_accuracy[-1]:.2f}%</p>
                    <img src="results.png" alt="Results Graph" style="max-width:100%;">
                </body>
                </html>
                """
                self.wfile.write(html.encode('utf-8'))
            else:
                return http.server.SimpleHTTPRequestHandler.do_GET(self)

    print(f"Serving results on http://localhost:{PORT} ...")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

def main():
    raw_data = fetch(TIMEFRAME, SYMBOL, START, END)
    if len(raw_data) < D_LEN: return

    derived_data = deriveround(raw_data, A_ROUND)
    train_data, test_data = split(derived_data, B_SPLIT)
    
    if D_LEN < 2:
        print("D_LEN must be >= 2")
        return
        
    top_seqs = gettop(train_data, C_TOP, D_LEN)
    if not top_seqs: return

    results = completesimilarbeginnings(test_data, top_seqs, D_LEN, E_SIM)
    generate_plots_and_serve(results)

if __name__ == "__main__":
    main()
