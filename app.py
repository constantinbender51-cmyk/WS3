import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from deap import base, creator, tools, algorithms
import random
import http.server
import socketserver
import io
import time
import warnings
import threading

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_YEAR = 2018
SMA_PERIOD = 1460
SMA_OFFSET = -1460     
GA_POP_SIZE = 100
GA_NGEN = 15
GA_CXPB = 0.5
GA_MUTPB = 0.2
N_REVERSAL_LEVELS = 15
INITIAL_BALANCE = 10000
SERVER_PORT = 8080

# GENE RANGES
GENE_LEVEL_MIN = -40000
GENE_LEVEL_MAX = 40000
GENE_SL_MIN = 0.01
GENE_SL_MAX = 0.15
GENE_TP_MIN = 0.01
GENE_TP_MAX = 0.30

# DATES
TRAIN_END_DATE = '2025-12-31'
TEST_START_DATE = '2026-01-01'

# --- FETCH ---
def fetch_ohlc():
    exchange = ccxt.binance()
    since = exchange.parse8601(f'{START_YEAR}-01-01T00:00:00Z')
    all_candles = []
    limit = 1000 
    
    print(f"Fetching {SYMBOL} data from {START_YEAR}...")
    while True:
        try:
            candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since, limit=limit)
            if not candles:
                break
            all_candles += candles
            since = candles[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
            
            if len(candles) < limit:
                break
        except Exception as e:
            print(f"Fetch error: {e}")
            break
            
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    print(f"Total Rows Fetched: {len(df)}")
    return df

# --- PROCESS ---
def apply_indicators(df):
    df['sma'] = df['close'].rolling(window=SMA_PERIOD).mean()
    df['sma_shifted'] = df['sma'].shift(SMA_OFFSET)
    
    fit_df = df.dropna(subset=['sma_shifted'])
    
    if fit_df.empty:
        raise ValueError(f"Insufficient data. Total Rows: {len(df)}.")

    fit_df['ordinal'] = fit_df.index.map(pd.Timestamp.toordinal)
    
    def linear_func(x, m, c):
        return m * x + c
    
    popt, _ = curve_fit(linear_func, fit_df['ordinal'], fit_df['sma_shifted'])
    m, c = popt
    
    df['ordinal'] = df.index.map(pd.Timestamp.toordinal)
    df['trend_line'] = linear_func(df['ordinal'], m, c)
    df['detrended'] = df['close'] - df['trend_line']
    
    return df

# --- STRATEGY ENGINE ---
def run_strategy_logic(individual, data, record_trades=False):
    balance = INITIAL_BALANCE
    equity = []
    position = 0 
    entry_price = 0.0
    active_sl = 0.0
    active_tp = 0.0
    
    trades = {'entry_dt': [], 'entry_price': [], 'exit_dt': [], 'exit_price': [], 'type': []}
    
    levels = []
    for i in range(0, len(individual), 3):
        levels.append({'thresh': individual[i], 'sl': individual[i+1], 'tp': individual[i+2]})
    
    detrended = data['detrended'].values
    closes = data['close'].values
    timestamps = data.index
    
    for i in range(1, len(data)):
        curr_dt = detrended[i]
        prev_dt = detrended[i-1]
        price = closes[i]
        ts = timestamps[i]
        
        # Check Exit
        if position != 0:
            pnl_pct = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
            if pnl_pct <= -active_sl or pnl_pct >= active_tp:
                balance *= (1 + pnl_pct)
                if record_trades:
                    trades['exit_dt'].append(ts)
                    trades['exit_price'].append(curr_dt)
                position = 0
        
        # Check Entry
        if position == 0:
            for lvl in levels:
                threshold = lvl['thresh']
                # Cross UP -> Short
                if prev_dt < threshold and curr_dt >= threshold:
                    position = -1
                    entry_price = price
                    active_sl = lvl['sl']
                    active_tp = lvl['tp']
                    if record_trades:
                        trades['entry_dt'].append(ts)
                        trades['entry_price'].append(curr_dt)
                        trades['type'].append('short')
                    break
                # Cross DOWN -> Long
                elif prev_dt > threshold and curr_dt <= threshold:
                    position = 1
                    entry_price = price
                    active_sl = lvl['sl']
                    active_tp = lvl['tp']
                    if record_trades:
                        trades['entry_dt'].append(ts)
                        trades['entry_price'].append(curr_dt)
                        trades['type'].append('long')
                    break
                    
        equity.append(balance)
    
    return equity, trades

def backtest_strategy(individual, data):
    train_data = data.loc[:TRAIN_END_DATE]
    if train_data.empty: return -999,
    
    equity, _ = run_strategy_logic(individual, train_data, record_trades=False)
    
    if not equity: return -999,
    returns = pd.Series(equity).pct_change().dropna()
    if len(returns) < 2 or returns.std() == 0: return -999,
    
    sharpe = (returns.mean() / returns.std()) * np.sqrt(365)
    return sharpe,

def optimize(df):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_lvl", random.uniform, GENE_LEVEL_MIN, GENE_LEVEL_MAX)
    toolbox.register("attr_sl", random.uniform, GENE_SL_MIN, GENE_SL_MAX)
    toolbox.register("attr_tp", random.uniform, GENE_TP_MIN, GENE_TP_MAX)
    
    def create_ind():
        ind = []
        for _ in range(N_REVERSAL_LEVELS):
            ind.extend([toolbox.attr_lvl(), toolbox.attr_sl(), toolbox.attr_tp()])
        return ind
    
    toolbox.register("individual", tools.initIterate, creator.Individual, create_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", backtest_strategy, data=df)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=500, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=GA_POP_SIZE)
    algorithms.eaSimple(pop, toolbox, cxpb=GA_CXPB, mutpb=GA_MUTPB, ngen=GA_NGEN, verbose=False)
    
    return tools.selBest(pop, 1)[0]

# --- HELPER FOR PLOTTING LOGIC ---
def plot_strategy_performance(ax, data, best_genes, title):
    # Run Simulation
    equity_curve, trades = run_strategy_logic(best_genes, data, record_trades=True)
    
    # Plot Price (Detrended)
    ax.plot(data.index, data['detrended'], label='Detrended Price', color='grey', lw=0.8, alpha=0.8)
    
    # Plot Levels
    colors = ['r', 'g', 'b', 'orange', 'purple']
    for i in range(N_REVERSAL_LEVELS):
        lvl = best_genes[i*3]
        ax.axhline(lvl, color=colors[i%len(colors)], ls='--', alpha=0.5)
    
    # Plot Trades
    long_entries = [(t, p) for t, p, type_ in zip(trades['entry_dt'], trades['entry_price'], trades['type']) if type_ == 'long']
    short_entries = [(t, p) for t, p, type_ in zip(trades['entry_dt'], trades['entry_price'], trades['type']) if type_ == 'short']
    
    if long_entries:
        lx, ly = zip(*long_entries)
        ax.scatter(lx, ly, marker='^', color='green', s=60, label='Long', zorder=5)
    if short_entries:
        sx, sy = zip(*short_entries)
        ax.scatter(sx, sy, marker='v', color='red', s=60, label='Short', zorder=5)
    if trades['exit_dt']:
        ax.scatter(trades['exit_dt'], trades['exit_price'], marker='x', color='black', s=40, label='Exit', zorder=5)

    ax.set_title(title)
    ax.set_ylabel('Detrended Price ($)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot PnL (Twin Axis)
    ax_pnl = ax.twinx()
    # Align equity curve
    if len(equity_curve) > 0:
        align_len = min(len(equity_curve), len(data))
        # Usually equity curve is len(data)-1 because loop starts at 1
        plot_idx = data.index[-align_len:]
        plot_eq = equity_curve[-align_len:]
        
        ax_pnl.plot(plot_idx, plot_eq, color='blue', lw=1.5, alpha=0.6, label='Equity')
        ax_pnl.set_ylabel('Equity ($)', color='blue')
        ax_pnl.tick_params(axis='y', labelcolor='blue')
        
        # Add PnL legend manually or via helper
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_pnl.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')

# --- PLOT GENERATION ---
def generate_plot(df, best_genes):
    plt.figure(figsize=(16, 14))
    
    # 1. Training Phase
    ax1 = plt.subplot(3, 1, 1)
    train_data = df.loc[:TRAIN_END_DATE]
    if not train_data.empty:
        plot_strategy_performance(ax1, train_data, best_genes, f'Training Phase (2018-2025): Trades & PnL')
    
    # 2. Testing Phase
    ax2 = plt.subplot(3, 1, 2)
    test_data = df.loc[TEST_START_DATE:]
    if not test_data.empty:
        plot_strategy_performance(ax2, test_data, best_genes, f'Test Phase (2026): Unseen Data Performance')
    else:
        ax2.text(0.5, 0.5, 'No Test Data', ha='center')

    # 3. Parameter Table
    ax3 = plt.subplot(3, 1, 3)
    ax3.axis('tight')
    ax3.axis('off')
    
    table_data = []
    col_labels = ['Level', 'Threshold', 'Stop Loss', 'Take Profit']
    
    for i in range(0, len(best_genes), 3):
        idx = (i//3) + 1
        thresh = best_genes[i]
        sl = best_genes[i+1] * 100
        tp = best_genes[i+2] * 100
        table_data.append([f"Lvl {idx}", f"{thresh:.2f}", f"{sl:.2f}%", f"{tp:.2f}%"])
        
    table = ax3.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.scale(1, 1.5)
    table.set_fontsize(12)
    ax3.set_title("Strategy Parameters", pad=10)
        
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

class PlotHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.end_headers()
        self.wfile.write(plot_buffer.getvalue())

if __name__ == "__main__":
    df = fetch_ohlc()
    if df.empty: exit()
    
    try:
        df = apply_indicators(df)
    except ValueError as e:
        print(f"Error: {e}")
        exit()
        
    print("Running GA...")
    best_ind = optimize(df)
    print("Serving 8080...")
    
    plot_buffer = generate_plot(df, best_ind)
    
    with socketserver.TCPServer(("", SERVER_PORT), PlotHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
