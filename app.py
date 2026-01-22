import requests
import math
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import http.server
import socketserver
import os
from datetime import datetime, timedelta

# ==========================================
# PARAMETERS
# ==========================================
TIMEFRAME = '1h'       # e.g., '1m', '5m', '1h', '1d'
SYMBOL = 'BTCUSDT'     # Binance symbol
START = '2023-01-01'   # Training/Backtest Start
END = '2023-06-01'     # Training/Backtest End

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
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Handle both string and datetime objects
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
                # O, H, L, C
                ohlc = [float(k[1]), float(k[2]), float(k[3]), float(k[4])]
                data.append(ohlc)
            
            # Update start time: last close time + 1ms
            current_start = klines[-1][6] + 1
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    print(f"Fetched {len(data)} candles.")
    return data

def deriveround(ohlc_data, a):
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
    sequences = []
    for i in range(len(train_data) - d + 1):
        seq = tuple(train_data[i : i+d])
        sequences.append(seq)
        
    if not sequences: return []

    counts = Counter(sequences)
    unique_seqs = list(counts.items()) 
    unique_seqs.sort(key=lambda x: x[1], reverse=True)
    
    limit = int(len(unique_seqs) * (c / 100.0))
    if limit < 1: limit = 1
        
    top_sequences = [item[0] for item in unique_seqs[:limit]]
    print(f"Identified {len(top_sequences)} top sequences (The Model).")
    return top_sequences

def is_similar(seq1, seq2, e):
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
            if rel_diff >= e: return False
    return True

def completesimilarbeginnings(test_data, top_sequences, d, e):
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
                prediction = prediction_candle[3] 
                break 
        
        if prediction is not None:
            actual = actual_outcome[3] 
            predictions.append((prediction, actual))
            
    return predictions

def process_results_for_display(results, filename_prefix):
    """
    Calculates metrics and generates a plot for a given set of results.
    Returns: (stats_dict, html_table_rows)
    """
    trade_log = []
    cumulative_correct = []
    rolling_accuracy = []
    
    correct_count = 0
    total_valid = 0
    total_pnl = 0.0
    
    table_rows = ""
    
    for i, (pred, actual) in enumerate(results):
        if pred == 0 or actual == 0:
            continue
            
        total_valid += 1
        
        direction = 1 if pred > 0 else -1
        pnl = direction * actual
        total_pnl += pnl
        
        is_correct = (pred > 0 and actual > 0) or (pred < 0 and actual < 0)
        if is_correct:
            correct_count += 1
            
        cumulative_correct.append(correct_count)
        rolling_accuracy.append((correct_count / total_valid) * 100)
        
        color = "green" if pnl > 0 else "red"
        table_rows += f"""
        <tr>
            <td>{total_valid}</td>
            <td>{pred:.2f}%</td>
            <td>{actual:.2f}%</td>
            <td style="color:{color}; font-weight:bold;">{pnl:.2f}%</td>
            <td>{"Yes" if is_correct else "No"}</td>
        </tr>
        """

    if total_valid == 0:
        return {'valid': 0}, ""

    # Generate Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(rolling_accuracy, color='blue', label='Cumulative Accuracy %')
    ax1.set_title(f'Directional Accuracy (Final: {rolling_accuracy[-1]:.2f}%)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True)
    
    ax2.plot(cumulative_correct, color='green', label='Correct Predictions')
    ax2.set_title(f'Cumulative Correct Predictions (Total: {correct_count})')
    ax2.set_ylabel('Count')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}.png')
    plt.close(fig) # Close to free memory
    
    stats = {
        'valid': total_valid,
        'accuracy': rolling_accuracy[-1],
        'pnl': total_pnl
    }
    
    return stats, table_rows

def serve_results(hist_stats, hist_table, recent_stats, recent_table):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                # HTML Structure
                html = f"""
                <html>
                <head>
                    <title>Model Performance Report</title>
                    <style>
                        body {{ font-family: sans-serif; padding: 20px; max-width: 1200px; margin: auto; }}
                        .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                        .section {{ flex: 1; min-width: 500px; border: 1px solid #ddd; padding: 20px; border-radius: 8px; }}
                        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; font-size: 0.9em; }}
                        th, td {{ border: 1px solid #eee; padding: 6px; text-align: center; }}
                        th {{ background-color: #f8f9fa; }}
                        h1 {{ text-align: center; color: #333; }}
                        h2 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
                        .stats {{ background: #f0f4f8; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
                        img {{ max-width: 100%; height: auto; border: 1px solid #eee; margin-top: 10px; }}
                    </style>
                </head>
                <body>
                    <h1>Trading Strategy Performance Report</h1>
                    
                    <div class="container">
                        <div class="section">
                            <h2>1. Historical Validation ({START} to {END})</h2>
                            <div class="stats">
                                <p><strong>Total Valid Trades:</strong> {hist_stats.get('valid', 0)}</p>
                                <p><strong>Final Accuracy:</strong> {hist_stats.get('accuracy', 0):.2f}%</p>
                                <p><strong>Total Realized PnL:</strong> {hist_stats.get('pnl', 0):.2f}%</p>
                            </div>
                            <img src="historical.png" alt="Historical Results">
                            
                            <h3>Trade Log</h3>
                            <div style="max-height: 400px; overflow-y: scroll;">
                                <table>
                                    <thead>
                                        <tr><th>#</th><th>Pred</th><th>Actual</th><th>PnL</th><th>Correct</th></tr>
                                    </thead>
                                    <tbody>{hist_table}</tbody>
                                </table>
                            </div>
                        </div>

                        <div class="section">
                            <h2>2. Recent Performance (Last 14 Days)</h2>
                            <div class="stats">
                                <p><strong>Total Valid Trades:</strong> {recent_stats.get('valid', 0)}</p>
                                <p><strong>Final Accuracy:</strong> {recent_stats.get('accuracy', 0):.2f}%</p>
                                <p><strong>Total Realized PnL:</strong> {recent_stats.get('pnl', 0):.2f}%</p>
                            </div>
                            <img src="recent.png" alt="Recent Results">
                            
                            <h3>Trade Log</h3>
                            <div style="max-height: 400px; overflow-y: scroll;">
                                <table>
                                    <thead>
                                        <tr><th>#</th><th>Pred</th><th>Actual</th><th>PnL</th><th>Correct</th></tr>
                                    </thead>
                                    <tbody>{recent_table}</tbody>
                                </table>
                            </div>
                        </div>
                    </div>
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
    # -----------------------------------------
    # 1. Historical Training & Validation
    # -----------------------------------------
    raw_data = fetch(TIMEFRAME, SYMBOL, START, END)
    if len(raw_data) < D_LEN: return

    derived_data = deriveround(raw_data, A_ROUND)
    train_data, test_data = split(derived_data, B_SPLIT)
    
    if D_LEN < 2:
        print("D_LEN must be >= 2")
        return
        
    # TRAIN THE MODEL (Identify Sequences)
    top_seqs = gettop(train_data, C_TOP, D_LEN)
    if not top_seqs: return

    # TEST ON HISTORY
    hist_results = completesimilarbeginnings(test_data, top_seqs, D_LEN, E_SIM)
    hist_stats, hist_table = process_results_for_display(hist_results, "historical")

    # -----------------------------------------
    # 2. Recent Data Prediction (Last 14 Days)
    # -----------------------------------------
    now = datetime.now()
    recent_start = now - timedelta(days=14)
    
    # Fetch data
    recent_raw = fetch(TIMEFRAME, SYMBOL, recent_start, now)
    
    recent_stats = {}
    recent_table = ""
    
    if len(recent_raw) > D_LEN:
        recent_derived = deriveround(recent_raw, A_ROUND)
        # Use the SAME top_seqs (Model) on NEW data
        recent_results = completesimilarbeginnings(recent_derived, top_seqs, D_LEN, E_SIM)
        recent_stats, recent_table = process_results_for_display(recent_results, "recent")
    else:
        print("Not enough recent data fetched.")

    # -----------------------------------------
    # 3. Serve Report
    # -----------------------------------------
    serve_results(hist_stats, hist_table, recent_stats, recent_table)

if __name__ == "__main__":
    main()
