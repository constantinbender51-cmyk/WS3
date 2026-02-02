import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import http.server
import socketserver
import os
import random
from datetime import datetime

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
class Config:
    # Exchange Settings
    SYMBOL = "BTCUSDT"
    INTERVAL = "4h"
    START_TIME = "2025-01-01"
    
    # Backtest Settings
    LEVELS_COUNT = 3        # Number of Long and Short levels (each)
    TRAIN_SPLIT = 0.6       # Data split (0.6 = 60% training)
    
    # Genetic Algorithm Settings
    POPULATION_SIZE = 50
    GENERATIONS = 15
    MUTATION_RATE = 0.2
    
    # GA Mutation/Bound Constraints
    SL_MIN = 0.01           # 1% Stop Loss Min
    SL_MAX = 0.10           # 10% Stop Loss Max
    SL_MUTATION_SIGMA = 0.01
    PRICE_MUTATION_PCT = 0.05  # Percentage of price range for mutation std dev
    
    # Server Settings
    PORT = 8080
    OUTPUT_DIR = "server_root"

# ==========================================
# 1. DATA FETCHING (Binance)
# ==========================================
def fetch_ohlc(symbol, interval, start_str):
    base_url = "https://api.binance.com/api/v3/klines"
    dt_obj = datetime.strptime(start_str, "%Y-%m-%d")
    start_ts = int(dt_obj.timestamp() * 1000)
    
    data = []
    current_ts = start_ts
    
    print(f"[SYSTEM] Fetching data for {symbol} from {start_str}...")
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_ts,
            "limit": 1000
        }
        try:
            r = requests.get(base_url, params=params)
            r.raise_for_status()
            chunk = r.json()
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            break

        if not chunk:
            break
            
        data.extend(chunk)
        last_ts = chunk[-1][0]
        current_ts = last_ts + 1
        
        if len(chunk) < 1000:
            break
            
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    numeric_cols = ["open", "high", "low", "close"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    
    print(f"[SYSTEM] Fetched {len(df)} candles.")
    return df[["open_time", "open", "high", "low", "close"]]

# ==========================================
# 2. BACKTEST ENGINE
# ==========================================
def backtest(df, long_levels, short_levels, stop_loss_pct):
    highs = df["high"].values
    lows = df["low"].values
    times = df["open_time"].values
    closes = df["close"].values
    
    trades = [] 
    active_trades = [] 
    equity = 0.0
    equity_curve = [0.0]
    
    for i in range(len(df)):
        h = highs[i]
        l = lows[i]
        t = times[i]
        
        # Entries
        for level in long_levels:
            if l <= level <= h:
                if not any(tr['entry_price'] == level and tr['type'] == 'long' for tr in active_trades):
                    active_trades.append({
                        'type': 'long',
                        'entry_price': level,
                        'entry_time': t,
                        'sl_price': level * (1 - stop_loss_pct),
                        'status': 'open'
                    })

        for level in short_levels:
            if l <= level <= h:
                if not any(tr['entry_price'] == level and tr['type'] == 'short' for tr in active_trades):
                    active_trades.append({
                        'type': 'short',
                        'entry_price': level,
                        'entry_time': t,
                        'sl_price': level * (1 + stop_loss_pct),
                        'status': 'open'
                    })
        
        # Exits
        for trade in active_trades[:]:
            pnl = 0.0
            closed = False
            
            if trade['type'] == 'long':
                if l <= trade['sl_price']:
                    exit_p = trade['sl_price']
                    pnl = (exit_p - trade['entry_price']) / trade['entry_price']
                    closed = True
            
            elif trade['type'] == 'short':
                if h >= trade['sl_price']:
                    exit_p = trade['sl_price']
                    pnl = (trade['entry_price'] - exit_p) / trade['entry_price']
                    closed = True
            
            if closed:
                trade['exit_price'] = exit_p
                trade['exit_time'] = t
                trade['pnl'] = pnl
                trade['status'] = 'closed'
                trades.append(trade)
                active_trades.remove(trade)
                equity += pnl

        equity_curve.append(equity)

    # Force Close at End
    last_price = closes[-1]
    last_time = times[-1]
    for trade in active_trades:
        pnl = 0.0
        if trade['type'] == 'long':
            pnl = (last_price - trade['entry_price']) / trade['entry_price']
        else:
            pnl = (trade['entry_price'] - last_price) / trade['entry_price']
        
        trade['exit_price'] = last_price
        trade['exit_time'] = last_time
        trade['pnl'] = pnl
        trade['status'] = 'force_closed'
        trades.append(trade)
        equity += pnl
        
    return np.array(equity_curve), trades, equity

# ==========================================
# 3. GENETIC ALGORITHM
# ==========================================
class GeneticOptimizer:
    def __init__(self, data):
        self.data = data
        self.price_min = data['low'].min()
        self.price_max = data['high'].max()
        self.levels_cnt = Config.LEVELS_COUNT
        
    def init_population(self):
        pop = []
        for _ in range(Config.POPULATION_SIZE):
            longs = np.random.uniform(self.price_min, self.price_max, self.levels_cnt)
            shorts = np.random.uniform(self.price_min, self.price_max, self.levels_cnt)
            sl = np.random.uniform(Config.SL_MIN, Config.SL_MAX)
            genome = np.concatenate([longs, shorts, [sl]])
            pop.append(genome)
        return pop

    def fitness(self, genome):
        longs = genome[:self.levels_cnt]
        shorts = genome[self.levels_cnt:2*self.levels_cnt]
        sl = genome[-1]
        
        if sl <= 0: return -9999.0
        
        _, _, total_pnl = backtest(self.data, longs, shorts, sl)
        return total_pnl

    def mutate(self, genome):
        idx = random.randint(0, len(genome)-1)
        if idx < 2 * self.levels_cnt:
            # Price level mutation
            shift = np.random.normal(0, (self.price_max - self.price_min) * Config.PRICE_MUTATION_PCT)
            genome[idx] += shift
        else:
            # SL mutation
            genome[idx] += np.random.normal(0, Config.SL_MUTATION_SIGMA)
            genome[idx] = np.clip(genome[idx], 0.001, 0.2)
        return genome

    def crossover(self, p1, p2):
        pt = random.randint(1, len(p1)-1)
        c1 = np.concatenate([p1[:pt], p2[pt:]])
        c2 = np.concatenate([p2[:pt], p1[pt:]])
        return c1, c2

    def run(self):
        pop = self.init_population()
        best_genome = None
        best_score = -np.inf
        
        for g in range(Config.GENERATIONS):
            scores = [self.fitness(ind) for ind in pop]
            
            max_s = np.max(scores)
            if max_s > best_score:
                best_score = max_s
                best_genome = pop[np.argmax(scores)]
            
            print(f"[GA] Gen {g+1}/{Config.GENERATIONS} | Max PnL: {max_s:.4f}")
            
            new_pop = []
            while len(new_pop) < Config.POPULATION_SIZE:
                # Tournament Selection
                p1 = pop[np.argmax([scores[random.randint(0, len(pop)-1)] for _ in range(3)])]
                p2 = pop[np.argmax([scores[random.randint(0, len(pop)-1)] for _ in range(3)])]
                
                c1, c2 = self.crossover(p1, p2)
                
                if random.random() < Config.MUTATION_RATE: c1 = self.mutate(c1)
                if random.random() < Config.MUTATION_RATE: c2 = self.mutate(c2)
                
                new_pop.extend([c1, c2])
            
            pop = new_pop[:Config.POPULATION_SIZE]
            
        return best_genome

# ==========================================
# 4. PLOTTING & REPORTING
# ==========================================
def generate_plots(df, trades, equity_curve, long_levels, short_levels):
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)

    # Price Plot
    plt.figure(figsize=(14, 8))
    plt.plot(df['open_time'], df['close'], label='Price', color='black', alpha=0.6, linewidth=1)
    
    for l in long_levels:
        plt.axhline(l, color='green', linestyle='--', alpha=0.5)
    for s in short_levels:
        plt.axhline(s, color='red', linestyle='--', alpha=0.5)
        
    long_entries = [t for t in trades if t['type'] == 'long']
    short_entries = [t for t in trades if t['type'] == 'short']
    
    if long_entries:
        plt.scatter([t['entry_time'] for t in long_entries], [t['entry_price'] for t in long_entries], 
                    marker='^', color='green', s=50)
        plt.scatter([t['exit_time'] for t in long_entries], [t['exit_price'] for t in long_entries], 
                    marker='x', color='black', s=30)

    if short_entries:
        plt.scatter([t['entry_time'] for t in short_entries], [t['entry_price'] for t in short_entries], 
                    marker='v', color='red', s=50)
        plt.scatter([t['exit_time'] for t in short_entries], [t['exit_price'] for t in short_entries], 
                    marker='x', color='black', s=30)
                    
    plt.title(f"Strategy Execution: {Config.SYMBOL}")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{Config.OUTPUT_DIR}/price_plot.png")
    plt.close()

    # PnL Plot
    plt.figure(figsize=(14, 6))
    plt.plot(equity_curve, color='blue', label='Cumulative PnL')
    plt.title("Equity Curve")
    plt.grid(True)
    plt.savefig(f"{Config.OUTPUT_DIR}/pnl_plot.png")
    plt.close()

def generate_html(trades, pnl):
    html = f"""
    <html>
    <head>
        <title>Backtest Report</title>
        <style>
            body {{ font-family: monospace; background: #f4f4f4; padding: 20px; }}
            .container {{ max-width: 1000px; margin: auto; background: white; padding: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 12px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #333; color: white; }}
            .profit {{ color: green; font-weight: bold; }}
            .loss {{ color: red; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Strategy Report: {Config.SYMBOL}</h1>
            <h2>Total PnL: <span class="{ 'profit' if pnl > 0 else 'loss' }">{pnl:.4f} R</span></h2>
            <h3>Price Action</h3><img src="price_plot.png" style="max-width:100%">
            <h3>Equity Curve</h3><img src="pnl_plot.png" style="max-width:100%">
            <h3>Trade Log</h3>
            <table>
                <tr><th>Type</th><th>Entry</th><th>Exit</th><th>PnL</th></tr>
    """
    
    for t in trades:
        color = "profit" if t['pnl'] > 0 else "loss"
        html += f"<tr><td>{t['type']}</td><td>{t['entry_price']:.2f}</td><td>{t['exit_price']:.2f}</td><td class='{color}'>{t['pnl']:.4f}</td></tr>"
        
    html += "</table></div></body></html>"
    
    with open(f"{Config.OUTPUT_DIR}/index.html", "w") as f:
        f.write(html)

# ==========================================
# 5. SERVER
# ==========================================
def serve_results():
    os.chdir(Config.OUTPUT_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", Config.PORT), handler) as httpd:
        print(f"[SERVER] Serving at http://localhost:{Config.PORT}")
        httpd.serve_forever()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    df = fetch_ohlc(Config.SYMBOL, Config.INTERVAL, Config.START_TIME)
    
    split_idx = int(len(df) * Config.TRAIN_SPLIT)
    train_data = df.iloc[:split_idx].reset_index(drop=True)
    test_data = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"[SYSTEM] Data Split: Train={len(train_data)}, Test={len(test_data)}")
    
    print("[SYSTEM] Starting Genetic Optimization...")
    ga = GeneticOptimizer(train_data)
    best_genome = ga.run()
    
    opt_longs = best_genome[:Config.LEVELS_COUNT]
    opt_shorts = best_genome[Config.LEVELS_COUNT:2*Config.LEVELS_COUNT]
    opt_sl = best_genome[-1]
    
    print(f"[RESULT] Optimized Params:\n Longs: {opt_longs}\n Shorts: {opt_shorts}\n SL: {opt_sl:.4f}")
    
    print("[SYSTEM] Running Test on Out-of-Sample Data...")
    equity_curve, trades, total_pnl = backtest(test_data, opt_longs, opt_shorts, opt_sl)
    
    generate_plots(test_data, trades, equity_curve, opt_longs, opt_shorts)
    generate_html(trades, total_pnl)
    
    serve_results()
