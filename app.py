import itertools
import numpy as np
import time
import urllib.request
import json
import matplotlib.pyplot as plt
import os
import http.server
import socketserver
from threading import Thread

# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================
PAIR = 'BTCUSDT' 
WINDOW_SIZE = 12
SWITCHING_PENALTY_WEIGHT = 35
ACTIONS = ['Long', 'Hold', 'Short']
PORT = int(os.environ.get("PORT", 8080)) # Railway provides PORT env var

# ==========================================

def get_binance_monthly_close(symbol):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1M&limit=1000"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return [float(entry[4]) for entry in data]
    except Exception as e:
        print(f"Error: {e}")
        return None

def optimize_segment(segment_prices, last_action=None):
    n_intervals = len(segment_prices) - 1
    price_diffs = np.diff(segment_prices)
    best_score, best_seq = -float('inf'), None
    
    for sequence in itertools.product(ACTIONS, repeat=n_intervals):
        multipliers = np.array([1 if a == 'Long' else (-1 if a == 'Short' else 0) for a in sequence])
        strategy_returns = price_diffs * multipliers
        total_return = np.sum(strategy_returns)
        std_dev = np.std(strategy_returns)
        risk_adj_return = total_return / (std_dev + 1e-9)
        
        switches = 0
        if last_action and sequence[0] != last_action: switches += 1
        for i in range(1, n_intervals):
            if sequence[i] != sequence[i-1]: switches += 1
        
        penalty_score = (switches * SWITCHING_PENALTY_WEIGHT) / n_intervals
        current_score = risk_adj_return - penalty_score
        
        if current_score > best_score:
            best_score, best_seq = current_score, sequence
    return best_seq

def plot_results(prices, full_sequence):
    """Generates a plot of the price and the signals."""
    plt.figure(figsize=(12, 6))
    prices_plot = np.array(prices[:len(full_sequence)])
    indices = np.arange(len(prices_plot))
    
    plt.plot(indices, prices_plot, label='BTC Price', color='black', alpha=0.3)
    
    # Scatter plot for actions
    for i, action in enumerate(full_sequence):
        color = 'green' if action == 'Long' else ('red' if action == 'Short' else 'gray')
        marker = '^' if action == 'Long' else ('v' if action == 'Short' else 'o')
        if i < len(prices_plot):
            plt.scatter(i, prices_plot[i], color=color, marker=marker, s=50)

    plt.title(f"Trading Strategy Results: {PAIR}")
    plt.xlabel("Months")
    plt.ylabel("Price (USDT)")
    plt.legend(['Price', 'Long (Green)', 'Short (Red)', 'Hold (Gray)'])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save as PNG
    plt.savefig('report.png')
    print("Graph saved as report.png")

def start_server():
    """Starts a simple HTTP server to serve the report.png."""
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Server started at http://0.0.0.0:{PORT}")
        print("View your report at /report.png")
        httpd.serve_forever()

def run_backtest():
    prices = get_binance_monthly_close(PAIR)
    if not prices or len(prices) < 2: return

    full_sequence = []
    last_action = None
    step_size = WINDOW_SIZE - 1
    
    for i in range(0, len(prices) - 1, step_size):
        end_idx = min(i + WINDOW_SIZE, len(prices))
        segment = prices[i:end_idx]
        if len(segment) < 2: break
        window_best_seq = optimize_segment(segment, last_action)
        full_sequence.extend(window_best_seq)
        last_action = window_best_seq[-1]

    # Plotting
    plot_results(prices, full_sequence)
    
    # Start Web Server in a background thread or main loop
    print("Starting web server...")
    start_server()

if __name__ == "__main__":
    run_backtest()
