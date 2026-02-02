import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from deap import base, creator, tools, algorithms
import random
import multiprocessing

# -----------------------------------------------------------------------------
# 1. Data Fetching (Binance)
# -----------------------------------------------------------------------------
def fetch_binance_data(symbol='BTC/USDT', timeframe='1h', start_date='2020-01-01T00:00:00Z'):
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601(start_date)
    all_candles = []
    
    print(f"Fetching {symbol} {timeframe} data from {start_date}...")
    
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not candles:
                break
            
            since = candles[-1][0] + 1  # Increment to prevent overlap
            all_candles += candles
            
            # Stop if reached present (approximate check)
            if candles[-1][0] > (datetime.now().timestamp() * 1000) - 3600000:
                break
                
            # Rate limit handling is internal to ccxt, but explicit sleep is safe
            time.sleep(0.1) 
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Drop potential duplicates from pagination
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"Data fetched: {len(df)} rows.")
    return df

# -----------------------------------------------------------------------------
# 2. Indicators & Regime Classification
# -----------------------------------------------------------------------------
def prepare_data(df):
    # SMA 365 calculation
    df['sma365'] = df['close'].rolling(window=365).mean()
    
    # Drop NaN values resulting from rolling window
    df.dropna(inplace=True)
    
    # Classify Regimes
    # Bull: Close > SMA 365
    # Bear: Close < SMA 365
    bull_df = df[df['close'] > df['sma365']].copy()
    bear_df = df[df['close'] < df['sma365']].copy()
    
    return bull_df, bear_df

# -----------------------------------------------------------------------------
# 3. Vectorized Backtest Engine
# -----------------------------------------------------------------------------
def run_strategy(df, sl_pct, tp_pct):
    """
    Simulates placing a LONG and a SHORT at the open of every candle.
    Positions are closed at TP, SL, or End of Candle (Close).
    
    Pessimistic Assumption: If both SL and TP are hit in the same candle, 
    we assume SL is hit first.
    """
    if df.empty:
        return 0.0

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    # --- LONG POSITIONS ---
    long_sl_price = opens * (1 - sl_pct)
    long_tp_price = opens * (1 + tp_pct)
    
    # Check hits
    l_sl_hit = lows <= long_sl_price
    l_tp_hit = highs >= long_tp_price
    
    # Logic: 
    # 1. If SL hit (regardless of TP) -> Loss (-sl_pct) (Pessimistic)
    # 2. Else If TP hit -> Profit (+tp_pct)
    # 3. Else -> Market Close ((close - open) / open)
    
    long_pnl = np.where(l_sl_hit, -sl_pct,
                 np.where(l_tp_hit, tp_pct,
                          (closes - opens) / opens))

    # --- SHORT POSITIONS ---
    short_sl_price = opens * (1 + sl_pct)
    short_tp_price = opens * (1 - tp_pct)
    
    # Check hits
    s_sl_hit = highs >= short_sl_price
    s_tp_hit = lows <= short_tp_price
    
    # Logic:
    # 1. If SL hit -> Loss (-sl_pct)
    # 2. Else If TP hit -> Profit (+tp_pct)
    # 3. Else -> Market Close ((open - close) / open)
    
    short_pnl = np.where(s_sl_hit, -sl_pct,
                 np.where(s_tp_hit, tp_pct,
                          (opens - closes) / opens))
    
    # Total PnL per hour is sum of both (hedged)
    total_pnl_series = long_pnl + short_pnl
    
    # Fitness metric: Total Cumulative Return (Sum of PnL %)
    return np.sum(total_pnl_series)

# -----------------------------------------------------------------------------
# 4. Genetic Algorithm (DEAP)
# -----------------------------------------------------------------------------
def optimize_params(df, market_type):
    print(f"\n--- Starting GA Optimization for {market_type} Market ({len(df)} candles) ---")
    
    # Constants
    POPULATION_SIZE = 50
    GENERATIONS = 10
    
    # Creator Setup
    # We want to maximize Profit. 
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Gene: Float between 0.001 (0.1%) and 0.05 (5%)
    toolbox.register("attr_float", random.uniform, 0.001, 0.05)
    
    # Individual: [SL_Percent, TP_Percent]
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation Function
    def evaluate(individual):
        sl, tp = individual
        # Constraint: SL and TP must be positive
        if sl <= 0 or tp <= 0:
            return -99999,
        
        profit = run_strategy(df, sl, tp)
        return profit,

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run GA
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # Run Algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, 
                                   ngen=GENERATIONS, stats=stats, 
                                   halloffame=hof, verbose=True)

    best_ind = hof[0]
    print(f"Best {market_type} Params -> SL: {best_ind[0]*100:.2f}%, TP: {best_ind[1]*100:.2f}%")
    print(f"Total Return for period: {best_ind.fitness.values[0]:.4f} units")
    
    return best_ind

# -----------------------------------------------------------------------------
# 5. Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Fetch
    try:
        data = fetch_binance_data()
    except Exception as e:
        print("Fatal error fetching data.")
        exit()

    # 2. Prepare
    bull_data, bear_data = prepare_data(data)
    
    print(f"Bull Market Data Points: {len(bull_data)}")
    print(f"Bear Market Data Points: {len(bear_data)}")

    # 3. Optimize Bull
    if len(bull_data) > 0:
        bull_best = optimize_params(bull_data, "BULL")
    else:
        print("Not enough Bull data to optimize.")

    # 4. Optimize Bear
    if len(bear_data) > 0:
        bear_best = optimize_params(bear_data, "BEAR")
    else:
        print("Not enough Bear data to optimize.")

    print("\n--- Final Results ---")
    if 'bull_best' in locals():
        print(f"BULL Market Optimal: StopLoss={bull_best[0]*100:.3f}%, TakeProfit={bull_best[1]*100:.3f}%")
    if 'bear_best' in locals():
        print(f"BEAR Market Optimal: StopLoss={bear_best[0]*100:.3f}%, TakeProfit={bear_best[1]*100:.3f}%")
