import ccxt
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import io
import http.server
import socketserver
from deap import base, creator, tools, algorithms
from datetime import datetime
import time
import warnings

# Configuration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
SINCE_STR = '2022-01-01 00:00:00'
POPULATION_SIZE = 50
GENERATIONS = 10  # Kept low for execution speed in demo; scale up for real use
N_LEVELS = 5      # Number of price levels to optimize
TRAIN_SPLIT = 0.85
PORT = 8080

warnings.filterwarnings("ignore")

# 1. Fetch OHLCV Data
def fetch_data(symbol, timeframe, since_str):
    print(f"Fetching {timeframe} data for {symbol} since {since_str}...")
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601(since_str)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            # Check if reached present (approximate check using last timestamp)
            if len(ohlcv) < 1000:
                break
            print(f"Fetched {len(all_ohlcv)} candles...", end='\r')
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(5)
            continue
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    print(f"\nTotal candles: {len(df)}")
    return df

# 2. Backtest Engine
def backtest(df, levels, sl_pct, tp_pct):
    # Logic:
    # Levels are absolute prices. 
    # If Previous_Close < Level and Current_Close > Level -> BUY
    # If Previous_Close > Level and Current_Close < Level -> SELL
    # FIFO for overlapping trades or simple single-trade logic?
    # Complexity: Multi-trade allowed. Each entry tracks its own SL/TP.
    
    trades = [] # {'entry_price': float, 'direction': 1/-1, 'sl': float, 'tp': float, 'entry_time': timestamp, 'status': 'open'}
    closed_trades = []
    
    # Vectorized approach is hard with individual SL/TP logic per trade without look-ahead bias
    # Iterative approach (dense/slow but accurate)
    
    # Pre-calculate crosses
    # We iterate row by row to simulate real execution
    
    active_trades = []
    equity = [0]
    
    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    times = df.index
    
    # Optimization: Convert levels to array for broadcasting
    levels = np.array(sorted(levels))
    
    for i in range(1, len(df)):
        current_price = prices[i]
        prev_price = prices[i-1]
        current_high = highs[i]
        current_low = lows[i]
        ts = times[i]
        
        # Check Exits first
        still_active = []
        for trade in active_trades:
            pnl = 0
            closed = False
            
            if trade['direction'] == 1: # Long
                if current_low <= trade['sl']:
                    pnl = (trade['sl'] - trade['entry_price']) / trade['entry_price']
                    closed = True
                elif current_high >= trade['tp']:
                    pnl = (trade['tp'] - trade['entry_price']) / trade['entry_price']
                    closed = True
            else: # Short
                if current_high >= trade['sl']:
                    pnl = (trade['entry_price'] - trade['sl']) / trade['entry_price']
                    closed = True
                elif current_low <= trade['tp']:
                    pnl = (trade['entry_price'] - trade['tp']) / trade['entry_price']
                    closed = True
            
            if closed:
                trade['exit_time'] = ts
                trade['pnl'] = pnl
                closed_trades.append(trade)
                equity.append(equity[-1] + pnl)
            else:
                still_active.append(trade)
        active_trades = still_active

        # Check Entries
        # Cross Up
        cross_up = (prev_price < levels) & (current_price > levels)
        for lvl in levels[cross_up]:
            active_trades.append({
                'entry_price': current_price,
                'direction': 1,
                'sl': current_price * (1 - sl_pct),
                'tp': current_price * (1 + tp_pct),
                'entry_time': ts
            })
            
        # Cross Down
        cross_down = (prev_price > levels) & (current_price < levels)
        for lvl in levels[cross_down]:
            active_trades.append({
                'entry_price': current_price,
                'direction': -1,
                'sl': current_price * (1 + sl_pct),
                'tp': current_price * (1 - tp_pct),
                'entry_time': ts
            })
            
    total_pnl = equity[-1]
    return total_pnl, closed_trades, equity

# 3. GA Setup
data = fetch_data(SYMBOL, TIMEFRAME, SINCE_STR)
split_idx = int(len(data) * TRAIN_SPLIT)
train_data = data.iloc[:split_idx]
test_data = data.iloc[split_idx:]

# Price range for gene initialization
min_price = train_data['low'].min()
max_price = train_data['high'].max()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Genes: [Level1, Level2, ..., LevelN, SL_pct, TP_pct]
toolbox.register("attr_price", random.uniform, min_price, max_price)
toolbox.register("attr_pct", random.uniform, 0.01, 0.20) # 1% to 20%
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_price,) * N_LEVELS + (toolbox.attr_pct, toolbox.attr_pct), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    levels = individual[:N_LEVELS]
    sl = individual[N_LEVELS]
    tp = individual[N_LEVELS+1]
    # Penalize negative SL/TP or illogical values
    if sl <= 0 or tp <= 0: return -9999,
    
    pnl, _, _ = backtest(train_data, levels, sl, tp)
    return pnl,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=[500]*N_LEVELS + [0.01, 0.01], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 4. Run Optimization
print("Starting GA Optimization...")
pop = toolbox.population(n=POPULATION_SIZE)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=GENERATIONS, 
                               stats=stats, halloffame=hof, verbose=True)

best_ind = hof[0]
best_levels = sorted(best_ind[:N_LEVELS])
best_sl = best_ind[N_LEVELS]
best_tp = best_ind[N_LEVELS+1]

print(f"Best Levels: {best_levels}")
print(f"Best SL: {best_sl:.2%}, Best TP: {best_tp:.2%}")

# 5. Test on Test Set
print("Running on Test Set...")
test_pnl, test_trades, test_equity = backtest(test_data, best_levels, best_sl, best_tp)
print(f"Test Set PnL: {test_pnl:.4f}")

# 6. Visualization & Serving
def generate_plot():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Price and Levels
    ax1.plot(test_data.index, test_data['close'], label='Price', color='black', alpha=0.6)
    for lvl in best_levels:
        ax1.axhline(lvl, color='blue', linestyle='--', alpha=0.5, label=f'Level {int(lvl)}')
    
    # Entries
    longs = [t for t in test_trades if t['direction'] == 1]
    shorts = [t for t in test_trades if t['direction'] == -1]
    
    if longs:
        ax1.scatter([t['entry_time'] for t in longs], [t['entry_price'] for t in longs], 
                   marker='^', color='green', label='Buy', zorder=5)
    if shorts:
        ax1.scatter([t['entry_time'] for t in shorts], [t['entry_price'] for t in shorts], 
                   marker='v', color='red', label='Sell', zorder=5)
        
    ax1.set_title('Price & Entries')
    ax1.legend(loc='upper left')
    
    # Equity Curve
    # Align equity curve to trade exit times for plotting
    eq_times = [test_data.index[0]] + [t['exit_time'] for t in test_trades]
    # This is a simplification; equity is list of cumulative sums. 
    # Match lengths for plot:
    ax2.plot(range(len(test_equity)), test_equity, label='Equity', color='purple')
    ax2.set_title(f'Cumulative PnL (Test Set): {test_pnl:.2f}')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Generate Trade List HTML
            trade_rows = ""
            for t in test_trades[-50:]: # Show last 50
                trade_rows += f"<tr><td>{t['entry_time']}</td><td>{t['direction']}</td><td>{t['entry_price']:.2f}</td><td>{t['pnl']:.4f}</td></tr>"
            
            html = f"""
            <html>
                <body>
                    <h1>Optimization Results</h1>
                    <p>Best Levels: {best_levels}</p>
                    <p>SL: {best_sl:.2%} | TP: {best_tp:.2%}</p>
                    <img src="/plot.png" width="1000" />
                    <h2>Last 50 Trades (Test Set)</h2>
                    <table border="1">
                        <tr><th>Time</th><th>Dir</th><th>Price</th><th>PnL</th></tr>
                        {trade_rows}
                    </table>
                </body>
            </html>
            """
            self.wfile.write(html.encode())
            
        elif self.path == '/plot.png':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            plot_buf = generate_plot()
            self.wfile.write(plot_buf.getvalue())
            plot_buf.close()

print(f"Serving on port {PORT}...")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
