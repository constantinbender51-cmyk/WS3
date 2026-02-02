import ccxt
import pandas as pd
import numpy as np
import pygad
import time
from datetime import datetime, timedelta

# ==========================================
# 1. Fetch Data
# ==========================================
def fetch_data():
    exchange = ccxt.binance()
    symbol = 'ETH/USDT'
    timeframe = '1h'
    
    # Calculate timestamp for 30 days ago
    since = exchange.parse8601((datetime.now() - timedelta(days=30)).isoformat())
    
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        since = ohlcv[-1][0] + 1
        all_ohlcv += ohlcv
        if len(ohlcv) < 1000:
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Return numpy arrays for speed in backtesting
    return df['open'].values, df['high'].values, df['low'].values, df['close'].values

# ==========================================
# 2. Backtest Engine (Vectorized)
# ==========================================
def run_strategy(solution, open_arr, high_arr, low_arr, close_arr):
    """
    Simulates placing a Long and Short every hour.
    Exit Logic:
    - Long: TP at Open * (1 + tp_pct), SL at Open * (1 - sl_pct)
    - Short: TP at Open * (1 - tp_pct), SL at Open * (1 + sl_pct)
    - Time Exit: If neither hit, close at candle Close.
    - Constraint: If Low <= SL and High >= TP in same candle, assume SL hit first (conservative).
    """
    sl_pct, tp_pct = solution[0], solution[1]
    
    # --- LONG POSITIONS ---
    long_tp_price = open_arr * (1 + tp_pct)
    long_sl_price = open_arr * (1 - sl_pct)
    
    # Check hits
    long_hit_sl = low_arr <= long_sl_price
    long_hit_tp = high_arr >= long_tp_price
    
    # Result calculation
    # Default: Close at market close
    long_pnl = (close_arr - open_arr) / open_arr
    
    # Overwrite with TP/SL logic
    # Priority: SL hit (conservative assumption) -> then TP -> else Market Close
    # If SL hit: Return is -SL%
    long_pnl = np.where(long_hit_sl, -sl_pct, long_pnl)
    # If TP hit AND NOT SL hit: Return is +TP%
    long_pnl = np.where(long_hit_tp & (~long_hit_sl), tp_pct, long_pnl)

    # --- SHORT POSITIONS ---
    short_tp_price = open_arr * (1 - tp_pct)
    short_sl_price = open_arr * (1 + sl_pct)
    
    short_hit_sl = high_arr >= short_sl_price
    short_hit_tp = low_arr <= short_tp_price
    
    # Default: Close at market close (inverse logic for short)
    short_pnl = (open_arr - close_arr) / open_arr
    
    # Overwrite
    short_pnl = np.where(short_hit_sl, -sl_pct, short_pnl)
    short_pnl = np.where(short_hit_tp & (~short_hit_sl), tp_pct, short_pnl)
    
    # Combine Returns (Hourly rebalance implies sum of returns)
    total_returns = long_pnl + short_pnl
    
    return total_returns

def calculate_sharpe(returns):
    if np.std(returns) == 0:
        return -9999
    # Annualized Sharpe (assuming hourly data: 24 * 365)
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(24 * 365)
    return sharpe

# ==========================================
# 3. Genetic Algorithm Optimization
# ==========================================
# Load data once globally to avoid passing large arrays repeatedly
try:
    OPEN, HIGH, LOW, CLOSE = fetch_data()
    print(f"Fetched {len(OPEN)} candles.")
except Exception as e:
    print(f"Data fetch failed: {e}")
    exit()

def fitness_func(ga_instance, solution, solution_idx):
    # Genes: [SL_pct, TP_pct]
    # Penalize negative SL/TP or illogical values
    if solution[0] <= 0 or solution[1] <= 0:
        return -9999
    
    returns = run_strategy(solution, OPEN, HIGH, LOW, CLOSE)
    sharpe = calculate_sharpe(returns)
    return sharpe

# Configuration
fitness_function = fitness_func
num_generations = 30
num_parents_mating = 4
sol_per_pop = 20
num_genes = 2

# Gene space: SL and TP between 0.1% and 10%
gene_space = [{'low': 0.001, 'high': 0.1}, {'low': 0.001, 'high': 0.1}]

ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    gene_space=gene_space,
    parent_selection_type="rws",
    keep_parents=1,
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=10
)

print("Starting optimization...")
ga_instance.run()

# ==========================================
# Output
# ==========================================
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("\n--- Optimization Results ---")
print(f"Best Solution Parameters:")
print(f"  SL: {solution[0]*100:.2f}%")
print(f"  TP: {solution[1]*100:.2f}%")
print(f"Max Annualized Sharpe Ratio: {solution_fitness:.4f}")
