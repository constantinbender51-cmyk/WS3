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
    START_TIME = "2024-01-01"
    
    # Backtest Settings
    LEVELS_COUNT = 5        # Increased count slightly for grid density
    TRAIN_SPLIT = 0.6       
    ANNUALIZATION_FACTOR = 365 * 6 
    
    # Genetic Algorithm Settings
    POPULATION_SIZE = 50
    GENERATIONS = 15
    MUTATION_RATE = 0.2
    ELITISM_COUNT = 2
    
    # GA Mutation/Bound Constraints
    SL_MIN = 0.01           
    SL_MAX = 0.10           
    TP_MIN = 0.02           
    TP_MAX = 0.30           
    
    SL_MUTATION_SIGMA = 0.01
    TP_MUTATION_SIGMA = 0.02
    PRICE_MUTATION_PCT = 0.05
    
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
# 2. BACKTEST ENGINE (HEDGING ENABLED)
# ==========================================
def backtest(df, levels, stop_loss_pct, take_profit_pct):
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
        
        # Entries: Check every level
        for level in levels:
            if l <= level <= h:
                # Hedge Mode: Try to open BOTH Long and Short at this level
                
                # Open Long if not already Long at this level
                if not any(tr['entry_price'] == level and tr['type'] == 'long' for tr in active_trades):
                    active_trades.append({
                        'type': 'long',
                        'entry_price': level,
                        'entry_time': t,
                        'sl_price': level * (1 - stop_loss_pct),
                        'tp_price': level * (1 + take_profit_pct),
                        'status': 'open'
                    })

                # Open Short if not already Short at this level
                if not any(tr['entry_price'] == level and tr['type'] == 'short' for tr in active_trades):
                    active_trades.append({
                        'type': 'short',
                        'entry_price': level,
                        'entry_time': t,
                        'sl_price': level * (1 + stop_loss_pct),
                        'tp_price': level * (1 - take_profit_pct),
                        'status': 'open'
                    })
        
        # Exits
        for trade in active_trades[:]:
            pnl = 0.0
            closed = False
            exit_p = 0.0
            
            if trade['type'] == 'long':
                sl_hit = l <= trade['sl_price']
                tp_hit = h >= trade['tp_price']
                
                if sl_hit and tp_hit:
                    # Conservative: SL hit first
                    exit_p = trade['sl_price']
                    pnl = (exit_p - trade['entry_price']) / trade['entry_price']
                    closed = True
                elif sl_hit:
                    exit_p = trade['sl_price']
                    pnl = (exit_p - trade['entry_price']) / trade['entry_price']
                    closed = True
                elif tp_hit:
                    exit_p = trade['tp_price']
                    pnl = (exit_p - trade['entry_price']) / trade['entry_price']
                    closed = True
            
            elif trade['type'] == 'short':
                sl_hit = h >= trade['sl_price']
                tp_hit = l <= trade['tp_price']
                
                if sl_hit and tp_hit:
                    exit_p = trade['sl_price']
                    pnl = (trade['entry_price'] - exit_p) / trade['entry_price']
                    closed = True
                elif sl_hit:
                    exit_p = trade['sl_price']
                    pnl = (trade['entry_price'] - exit_p) / trade['entry_price']
                    closed = True
                elif tp_hit:
                    exit_p = trade['tp_price']
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

    # Force Close
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
            # Genome: [Level1, Level2, ..., LevelN, SL, TP]
            levels = np.random.uniform(self.price_min, self.price_max, self.levels_cnt)
            sl = np.random.uniform(Config.SL_MIN, Config.SL_MAX)
            tp = np.random.uniform(Config.TP_MIN, Config.TP_MAX)
            
            genome = np.concatenate([levels, [sl, tp]])
            pop.append(genome)
        return pop

    def calculate_sharpe(self, equity_curve):
        returns = np.diff(equity_curve)
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0 
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = (mean_ret / std_ret) * np.sqrt(Config.ANNUALIZATION_FACTOR)
        return sharpe

    def fitness(self, genome):
        levels = genome[:self.levels_cnt]
        sl = genome[-2]
        tp = genome[-1]
        
        if sl <= 0 or tp <= 0: return -99.0
        
        equity_curve, _, _ = backtest(self.data, levels, sl, tp)
        return self.calculate_sharpe(equity_curve)

    def mutate(self, genome):
        idx = random.randint(0, len(genome)-1)
        
        # Indices: 0 to CNT-1 are prices. 
        # CNT is SL. 
        # CNT+1 is TP.
        
        price_cutoff = self.levels_cnt
        
        if idx < price_cutoff:
            shift = np.random.normal(0, (self.price_max - self.price_min) * Config.PRICE_MUTATION_PCT)
            genome[idx] += shift
        elif idx == price_cutoff:
            genome[idx] += np.random.normal(0, Config.SL_MUTATION_SIGMA)
            genome[idx] = np.clip(genome[idx], 0.001, 0.2)
        else:
            genome[idx] += np.random.normal(0, Config.TP_MUTATION_SIGMA)
            genome[idx] = np.clip(genome[idx], 0.001, 0.5)
            
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
            best_idx = np.argmax(scores)
            if max_s > best_score:
                best_score = max_s
                best_genome = pop[best_idx]
            
            print(f"[GA] Gen {g+1}/{Config.GENERATIONS} | Max Sharpe: {max_s:.4f}")
            
            sorted_indices = np.argsort(scores)[::-1]
            new_pop = [pop[i] for i in sorted_indices[:Config.ELITISM_COUNT]]
            
            while len(new_pop) < Config.POPULATION_SIZE:
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
def generate_plots(df, trades, equity_curve, levels):
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)

    # Price Plot
    plt.figure(figsize=(14, 8))
    plt.plot(df['open_time'], df['close'], label='Price', color='gray', alpha=0.5, linewidth=1)
    
    # Unified Levels
    for l in levels:
        plt.axhline(l, color='gold', linestyle='-', alpha=0.8, linewidth=1.5, label='Trigger Level' if l==levels[0] else "")
        
    long_entries = [t for t in trades if t['type'] == 'long']
    short_entries = [t for t in trades if t['type'] == 'short']
    
    if long_entries:
        plt.scatter([t['entry_time'] for t in long_entries], [t['entry_price'] for t in long_entries], 
                    marker='^', color='green', s=30, alpha=0.7)
    if short_entries:
        plt.scatter([t['entry_time'] for t in short_entries], [t['entry_price'] for t in short_entries], 
                    marker='v', color='red', s=30, alpha=0.7)
                    
    plt.title(f"Strategy Execution: {Config.SYMBOL} (Hedge/Grid Mode)")
    plt.legend()
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

def generate_html(trades, pnl, sharpe):
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
            <h2>Sharpe Ratio: {sharpe:.4f}</h2>
            <h3>Price Action</h3><img src="price_plot.png" style="max-width:100%">
            <h3>Equity Curve</h3><img src="pnl_plot.png" style="max-width:100%">
            <h3>Trade Log</h3>
            <table>
                <tr><th>Type</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Status</th></tr>
    """
    
    for t in trades:
        color = "profit" if t['pnl'] > 0 else "loss"
        html += f"""
        <tr>
            <td>{t['type']}</td>
            <td>{t['entry_price']:.2f}</td>
            <td>{t['exit_price']:.2f}</td>
            <td class='{color}'>{t['pnl']:.4f}</td>
            <td>{t['status']}</td>
        </tr>"""
        
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
    
    print("[SYSTEM] Starting Genetic Optimization (Hedge Mode)...")
    ga = GeneticOptimizer(train_data)
    best_genome = ga.run()
    
    opt_levels = best_genome[:Config.LEVELS_COUNT]
    opt_sl = best_genome[-2]
    opt_tp = best_genome[-1]
    
    print(f"[RESULT] Optimized Params:\n Levels: {opt_levels}")
    print(f" SL: {opt_sl:.4f} | TP: {opt_tp:.4f}")
    
    print("[SYSTEM] Running Test on Out-of-Sample Data...")
    equity_curve, trades, total_pnl = backtest(test_data, opt_levels, opt_sl, opt_tp)
    
    test_sharpe = ga.calculate_sharpe(equity_curve)
    print(f"[RESULT] Test Sharpe: {test_sharpe:.4f}")

    generate_plots(test_data, trades, equity_curve, opt_levels)
    generate_html(trades, total_pnl, test_sharpe)
    
    serve_results()
