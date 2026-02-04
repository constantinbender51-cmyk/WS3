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
SMA_WINDOW_HOURS = 1460 * 24 
TRAIN_SPLIT = 0.7
PORT = 8080

# Trading Parameters
FEE = 0.0002
SL_PCT = 0.01
TP_PCT = 0.02
BAND_PCT = 0.005

full_history = {
    "dates": [],
    "price": [],
    "equity": [],
    "position": [],
    "trend": [],
    "levels": None
}

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

def apply_trend_logic(df):
    n = len(df)
    train_idx = int(n * TRAIN_SPLIT)
    train_df = df.iloc[:train_idx].copy()
    
    train_df['sma'] = train_df['close'].rolling(window=SMA_WINDOW_HOURS).mean()
    train_df['sma_shifted'] = train_df['sma'].shift(-SMA_WINDOW_HOURS)
    
    valid_train = train_df.dropna(subset=['sma_shifted'])
    
    # Strict Mode: Crash if insufficient data
    if valid_train.empty:
        raise ValueError(f"Insufficient data for 1460-day SMA. Got {len(train_df)} hours.")

    X = np.arange(len(valid_train)).reshape(-1, 1)
    y = valid_train['sma_shifted'].values
    
    reg = LinearRegression().fit(X, y)
    
    X_full = np.arange(len(df)).reshape(-1, 1)
    trend_values = reg.predict(X_full)
    
    df['trend'] = trend_values
    df['detrended'] = df['close'] - df['trend']
    
    return df

def backtest(levels, df, record_history=False):
    closes = df['close'].values
    trends = df['trend'].values
    
    balance = 1000.0
    position = 0 
    entry_price = 0.0
    active_line_val = None 
    
    history_equity = []
    history_position = []
    
    for i in range(len(closes)):
        price = closes[i]
        trend = trends[i]
        
        raw_levels = levels + trend
        dists = np.abs(raw_levels - price)
        nearest_idx = np.argmin(dists)
        nearest_line = raw_levels[nearest_idx]
        
        band = price * BAND_PCT
        in_band = dists[nearest_idx] < band
        
        if in_band:
            active_line_val = nearest_line
        
        if in_band:
            if position != 0:
                pnl = (price - entry_price) / entry_price * position - FEE
                balance *= (1 + pnl)
                position = 0
        else:
            if active_line_val is not None:
                upper_bound = active_line_val + (active_line_val * BAND_PCT)
                lower_bound = active_line_val - (active_line_val * BAND_PCT)
                
                if position == 0:
                    if price > upper_bound:
                        position = 1
                        entry_price = price
                        balance *= (1 - FEE)
                    elif price < lower_bound:
                        position = -1
                        entry_price = price
                        balance *= (1 - FEE)
                elif position == 1:
                    pnl_pct = (price - entry_price) / entry_price
                    if pnl_pct <= -SL_PCT or pnl_pct >= TP_PCT:
                        balance *= (1 + pnl_pct - FEE)
                        position = 0
                elif position == -1:
                    pnl_pct = (entry_price - price) / entry_price
                    if pnl_pct <= -SL_PCT or pnl_pct >= TP_PCT:
                        balance *= (1 + pnl_pct - FEE)
                        position = 0

        current_equity = balance
        if position != 0:
            unrealized_pnl = (price - entry_price) / entry_price * position
            current_equity = balance * (1 + unrealized_pnl)
        
        if record_history:
            history_equity.append(current_equity)
            history_position.append(position)
            
    if record_history:
        return balance, history_equity, history_position
    return balance

class GeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate, df):
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.df = df
        self.population = []
        
        min_dt = df['detrended'].min()
        max_dt = df['detrended'].max()
        for _ in range(population_size):
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
            
            next_pop = []
            for _ in range(self.pop_size):
                p1, p2 = random.sample(list(zip(self.population, scores)), 2)
                parent1 = p1[0] if p1[1] > p2[1] else p2[0]
                p3, p4 = random.sample(list(zip(self.population, scores)), 2)
                parent2 = p3[0] if p3[1] > p4[1] else p4[0]
                
                cut = random.randint(1, 99)
                child = np.concatenate((parent1[:cut], parent2[cut:]))
                
                if random.random() < self.mutation_rate:
                    idx = random.randint(0, 99)
                    child[idx] = random.uniform(self.df['detrended'].min(), self.df['detrended'].max())
                
                next_pop.append(child)
            self.population = next_pop
            print(f"Gen {gen+1}/{self.generations} | Best Balance: {best_fitness:.2f}")
            
        return best_sol

app = Flask(__name__)

@app.route('/plot.png')
def plot_chart():
    # Fixed Boolean Check
    if len(full_history["dates"]) == 0:
        return "Data processing not complete", 503
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    dates = full_history["dates"]
    price = np.array(full_history["price"])
    trend = np.array(full_history["trend"])
    equity = full_history["equity"]
    pos = np.array(full_history["position"])
    levels = full_history["levels"]
    
    ax1.plot(dates, price, label='Price', color='black', linewidth=1)
    ax1.plot(dates, trend, label='Trend', color='blue', linestyle='--', linewidth=1.5)
    
    if levels is not None:
        for i, lvl in enumerate(levels):
            if i % 5 == 0: 
                ax1.plot(dates, trend + lvl, color='green', alpha=0.05, linewidth=0.5)
    
    y_min, y_max = ax1.get_ylim()
    ax1.fill_between(dates, y_min, y_max, where=(pos==1), color='green', alpha=0.1, label='Long')
    ax1.fill_between(dates, y_min, y_max, where=(pos==-1), color='red', alpha=0.1, label='Short')
    
    ax1.set_title(f"{SYMBOL} Strategy Execution (2018-Present)")
    ax1.set_ylabel("Price")
    ax1.legend(loc='upper left')
    
    ax2.plot(dates, equity, color='purple', linewidth=1.5, label='Account Balance')
    ax2.set_ylabel("Equity ($)")
    ax2.set_xlabel("Date")
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

def main_pipeline():
    global full_history
    
    df = fetch_data()
    df = apply_trend_logic(df)
    
    train_size = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:train_size]
    
    # Increased to 50 Generations
    ga = GeneticAlgorithm(population_size=50, generations=50, mutation_rate=0.1, df=train_df)
    print("Optimizing (50 Gens)...")
    best_levels = ga.optimize()
    
    print("Generating full history...")
    _, hist_equity, hist_pos = backtest(best_levels, df, record_history=True)
    
    full_history["dates"] = df.index
    full_history["price"] = df['close'].values
    full_history["trend"] = df['trend'].values
    full_history["equity"] = hist_equity
    full_history["position"] = hist_pos
    full_history["levels"] = best_levels
    
    print("Ready to serve.")

if __name__ == "__main__":
    t = threading.Thread(target=main_pipeline)
    t.start()
    app.run(host='0.0.0.0', port=PORT)
