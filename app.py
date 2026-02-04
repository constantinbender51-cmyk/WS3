import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from flask import Flask, jsonify, send_file
import threading
import time
import io
import random

# Configuration
SYMBOL = 'ETH/USDT'
TIMEFRAME = '1h'
START_YEAR = 2018
SMA_WINDOW_HOURS = 1460 * 24  # 35,040 hours
TRAIN_SPLIT = 0.7
PORT = 8080

# Trading Parameters
FEE = 0.0002
SL_PCT = 0.01
TP_PCT = 0.02
BAND_PCT = 0.005

# ---------------------------------------------------------
# 1. Fetch
# ---------------------------------------------------------
def fetch_data():
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601(f'{START_YEAR}-01-01T00:00:00Z')
    all_candles = []
    
    print(f"Fetching {SYMBOL} from {START_YEAR}...")
    while True:
        try:
            candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not candles:
                break
            since = candles[-1][0] + 1
            all_candles += candles
            # Break if reached current time (handled by fetch_ohlcv returning empty or near current)
            if len(candles) < 1000:
                break
        except Exception as e:
            print(f"Fetch error: {e}")
            time.sleep(5)
            continue
            
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df.astype(float)

# ---------------------------------------------------------
# 2. Detrend
# ---------------------------------------------------------
def apply_trend_logic(df):
    # Split Data
    n = len(df)
    train_idx = int(n * TRAIN_SPLIT)
    train_df = df.iloc[:train_idx].copy()
    
    # Calculate SMA on Train
    # Note: We need sufficient data. 35040 hours is ~4 years. 
    # If train set < 4 years, this will result in NaNs.
    train_df['sma'] = train_df['close'].rolling(window=SMA_WINDOW_HOURS).mean()
    
    # Shift SMA: Value at t becomes value at t - SMA_WINDOW
    # This aligns the "future" average to the current point for training
    train_df['sma_shifted'] = train_df['sma'].shift(-SMA_WINDOW_HOURS)
    
    # Linear Regression on valid Shifted SMA data
    valid_train = train_df.dropna(subset=['sma_shifted'])
    
    if valid_train.empty:
        raise ValueError("Insufficient training data to form one complete SMA window.")

    X = np.arange(len(valid_train)).reshape(-1, 1)
    y = valid_train['sma_shifted'].values
    
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    
    # Project Trend Line over ENTIRE dataset (Train + Test)
    # X_full needs to align with the indices 0 to len(df)
    X_full = np.arange(len(df)).reshape(-1, 1)
    trend_values = reg.predict(X_full)
    
    df['trend'] = trend_values
    df['detrended'] = df['close'] - df['trend']
    
    return df, slope, intercept

# ---------------------------------------------------------
# 3. Optimize (Genetic Algorithm)
# ---------------------------------------------------------
def backtest(levels, df):
    # levels: np array of 100 detrended price levels
    # Logic: 
    # 1. Identify active line (nearest).
    # 2. Band check (0.5% of RAW price).
    # 3. Position logic.
    
    closes = df['close'].values
    trends = df['trend'].values
    
    balance = 1000.0
    position = 0 # 0: Flat, 1: Long, -1: Short
    entry_price = 0.0
    active_line_val = None 
    
    # Vectorized approaches are hard with state-dependent "active line". Using Numba or loop.
    # Using loop for correctness of logic specified.
    
    equity_curve = []
    
    for i in range(len(closes)):
        price = closes[i]
        trend = trends[i]
        
        # Determine specific line levels in Raw Price terms
        # raw_levels = levels + trend
        # But we only need the nearest one if we are "entering a zone"
        
        # If we don't have an active line, find the nearest one
        raw_levels = levels + trend
        
        # Find distance to all lines
        dists = np.abs(raw_levels - price)
        nearest_idx = np.argmin(dists)
        nearest_line = raw_levels[nearest_idx]
        
        # Band calculation: 0.5% of Raw Price
        band = price * BAND_PCT
        
        in_band = dists[nearest_idx] < band
        
        # Update active line if we enter a band
        if in_band:
            active_line_val = nearest_line
        
        # Trading Logic
        if in_band:
            # Close position if open
            if position != 0:
                # Close logic
                pnl = (price - entry_price) / entry_price * position - FEE
                balance *= (1 + pnl)
                position = 0
        else:
            # Outside band
            if active_line_val is not None:
                # We have a reference line
                # Recalculate band relative to the ACTIVE line? 
                # Prompt: "If a zone is entered we use that line to determine the position relative to that"
                # "0.5% from the raw price... plot those"
                # Assuming the band is around the line. 
                
                upper_bound = active_line_val + (active_line_val * BAND_PCT)
                lower_bound = active_line_val - (active_line_val * BAND_PCT)
                
                # Check Signals
                if position == 0:
                    if price > upper_bound:
                        position = 1
                        entry_price = price
                        balance *= (1 - FEE) # Entry fee
                    elif price < lower_bound:
                        position = -1
                        entry_price = price
                        balance *= (1 - FEE) # Entry fee
                
                # Check SL/TP if in position
                elif position == 1:
                    # Long
                    pnl_pct = (price - entry_price) / entry_price
                    if pnl_pct <= -SL_PCT or pnl_pct >= TP_PCT:
                        balance *= (1 + pnl_pct - FEE)
                        position = 0
                elif position == -1:
                    # Short
                    pnl_pct = (entry_price - price) / entry_price
                    if pnl_pct <= -SL_PCT or pnl_pct >= TP_PCT:
                        balance *= (1 + pnl_pct - FEE)
                        position = 0

    return balance

class GeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate, df):
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.df = df
        self.population = []
        
        # Initialize population: 100 levels between min and max detrended price
        min_dt = df['detrended'].min()
        max_dt = df['detrended'].max()
        for _ in range(population_size):
            # Random uniform initialization
            individual = np.random.uniform(min_dt, max_dt, 100)
            self.population.append(individual)

    def optimize(self):
        best_fitness = -np.inf
        best_sol = None
        
        for gen in range(self.generations):
            scores = []
            for ind in self.population:
                fitness = backtest(ind, self.df)
                scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_sol = ind
            
            # Selection (Tournament)
            next_pop = []
            for _ in range(self.pop_size):
                p1, p2 = random.sample(list(zip(self.population, scores)), 2)
                parent1 = p1[0] if p1[1] > p2[1] else p2[0]
                p3, p4 = random.sample(list(zip(self.population, scores)), 2)
                parent2 = p3[0] if p3[1] > p4[1] else p4[0]
                
                # Crossover
                cut = random.randint(1, 99)
                child = np.concatenate((parent1[:cut], parent2[cut:]))
                
                # Mutation
                if random.random() < self.mutation_rate:
                    idx = random.randint(0, 99)
                    # Mutate one level
                    child[idx] = random.uniform(self.df['detrended'].min(), self.df['detrended'].max())
                
                next_pop.append(child)
            
            self.population = next_pop
            print(f"Gen {gen+1}/{self.generations} | Best Balance: {best_fitness:.2f}")
            
        return best_sol

# ---------------------------------------------------------
# 4. Serve
# ---------------------------------------------------------
app = Flask(__name__)
final_levels = None
global_df = None

@app.route('/api/status', methods=['GET'])
def get_status():
    if final_levels is None:
        return jsonify({"status": "Optimizing or Loading..."})
    
    # Convert numpy array to list
    return jsonify({
        "status": "Ready",
        "optimized_levels_detrended": final_levels.tolist(),
        "parameters": {
            "fee": FEE,
            "sl": SL_PCT,
            "tp": TP_PCT,
            "band": BAND_PCT,
            "sma_shift": -SMA_WINDOW_HOURS
        }
    })

@app.route('/plot.png')
def plot_chart():
    if global_df is None or final_levels is None:
        return "Data not ready", 503
    
    # Plotting latest portion of data for visibility
    plot_df = global_df.iloc[-500:].copy() # Last 500 hours
    
    plt.figure(figsize=(15, 8))
    plt.plot(plot_df.index, plot_df['close'], label='Price', color='black', alpha=0.5)
    
    # Plot Trend
    plt.plot(plot_df.index, plot_df['trend'], label='Trend Line', color='blue', linestyle='--')
    
    # Plot Optimized Lines (Raw)
    # Only plotting a subset or it becomes unreadable, but prompt implies visual
    # calculating raw levels for the last timestamp to show distribution
    trend_last = plot_df['trend'].iloc[-1]
    
    # We need to plot the lines as they move with the trend.
    # It's expensive to plot 100 lines for every timestamp. 
    # We will just plot 10 representative lines or the active ones if we tracked them.
    # To strictly follow "API with lines that have been optimized", we plot the grid for the visible range.
    
    t_values = plot_df['trend'].values
    time_values = plot_df.index
    
    # Plotting 100 lines is computationally heavy for matplotlib backend, 
    # we will plot them as a collection or just a few for validation.
    # Given instructions "Accuracy", we plot all.
    
    for lvl in final_levels:
        line_data = t_values + lvl
        plt.plot(time_values, line_data, color='green', alpha=0.1, linewidth=0.5)

    plt.title(f"ETH/USDT Optimized (Last 500h) - Best Levels")
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

def main_pipeline():
    global global_df, final_levels
    
    # 1. Fetch
    df = fetch_data()
    
    # 2. Detrend (Train/Test split logic handled inside or passed explicitly)
    # The requirement: "On 70% of the data: ... line fitted". 
    # We fit the line on 70%, but we calculate detrended values for the WHOLE dataset based on that fit.
    df_processed, slope, intercept = apply_trend_logic(df)
    
    # Split for optimization
    train_size = int(len(df_processed) * TRAIN_SPLIT)
    train_df = df_processed.iloc[:train_size]
    
    print(f"Data Processed. Slope: {slope:.5f}, Intercept: {intercept:.2f}")
    
    # 3. Optimize
    # Run GA on the TRAINING set
    ga = GeneticAlgorithm(population_size=20, generations=10, mutation_rate=0.1, df=train_df)
    print("Starting Optimization...")
    best_levels = ga.optimize()
    final_levels = best_levels
    global_df = df_processed
    
    print("Optimization Complete.")
    
if __name__ == "__main__":
    # Run pipeline in separate thread to allow server to start
    t = threading.Thread(target=main_pipeline)
    t.start()
    
    # 4. Serve
    app.run(host='0.0.0.0', port=PORT)
