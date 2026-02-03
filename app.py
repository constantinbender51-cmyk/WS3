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
import datetime
import sys
from collections import deque

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_YEAR = 2018
SMA_PERIOD = 1460
SMA_OFFSET = -1460     
GA_POP_SIZE = 50
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

# GLOBAL STATE FOR LIVE VIEW
live_state = {
    'price': 0.0,
    'detrended': 0.0,
    'balance': INITIAL_BALANCE,
    'position': 0,
    'last_update': 'Waiting...',
    'logs': deque(maxlen=20)  # Keep last 20 logs
}
state_lock = threading.Lock()

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
    
    return df, m, c

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
    
    # Custom Mutation to handle different gene scales
    def custom_mutate(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                if i % 3 == 0: # Level Gene
                    sigma = 500
                else: # SL or TP Gene
                    sigma = 0.02 
                individual[i] += random.gauss(0, sigma)
        return individual,

    # Bounds Checker Decorator
    def checkBounds(min_lvl, max_lvl, min_sl, max_sl, min_tp, max_tp):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(len(child)):
                        if i % 3 == 0: # Level
                            if child[i] > max_lvl: child[i] = max_lvl
                            elif child[i] < min_lvl: child[i] = min_lvl
                        elif i % 3 == 1: # SL
                            if child[i] > max_sl: child[i] = max_sl
                            elif child[i] < min_sl: child[i] = min_sl
                        elif i % 3 == 2: # TP
                            if child[i] > max_tp: child[i] = max_tp
                            elif child[i] < min_tp: child[i] = min_tp
                return offspring
            return wrapper
        return decorator
    
    toolbox.register("individual", tools.initIterate, creator.Individual, create_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", backtest_strategy, data=df)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutate, indpb=0.1) # Register custom mutation
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Apply Bounds Checking
    toolbox.decorate("mate", checkBounds(GENE_LEVEL_MIN, GENE_LEVEL_MAX, GENE_SL_MIN, GENE_SL_MAX, GENE_TP_MIN, GENE_TP_MAX))
    toolbox.decorate("mutate", checkBounds(GENE_LEVEL_MIN, GENE_LEVEL_MAX, GENE_SL_MIN, GENE_SL_MAX, GENE_TP_MIN, GENE_TP_MAX))
    
    pop = toolbox.population(n=GA_POP_SIZE)
    algorithms.eaSimple(pop, toolbox, cxpb=GA_CXPB, mutpb=GA_MUTPB, ngen=GA_NGEN, verbose=False)
    
    return tools.selBest(pop, 1)[0]

# --- HELPER FOR PLOTTING LOGIC ---
def plot_strategy_performance(ax, data, best_genes, title):
    equity_curve, trades = run_strategy_logic(best_genes, data, record_trades=True)
    
    ax.plot(data.index, data['detrended'], label='Detrended Price', color='grey', lw=0.8, alpha=0.8)
    
    colors = ['r', 'g', 'b', 'orange', 'purple']
    for i in range(N_REVERSAL_LEVELS):
        lvl = best_genes[i*3]
        ax.axhline(lvl, color=colors[i%len(colors)], ls='--', alpha=0.5)
    
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
    
    ax_pnl = ax.twinx()
    if len(equity_curve) > 0:
        align_len = min(len(equity_curve), len(data))
        plot_idx = data.index[-align_len:]
        plot_eq = equity_curve[-align_len:]
        
        ax_pnl.plot(plot_idx, plot_eq, color='blue', lw=1.5, alpha=0.6, label='Equity')
        ax_pnl.set_ylabel('Equity ($)', color='blue')
        ax_pnl.tick_params(axis='y', labelcolor='blue')
        
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_pnl.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')

# --- PLOT GENERATION ---
def generate_plot(df, best_genes):
    plt.figure(figsize=(16, 14))
    
    ax1 = plt.subplot(3, 1, 1)
    train_data = df.loc[:TRAIN_END_DATE]
    if not train_data.empty:
        plot_strategy_performance(ax1, train_data, best_genes, f'Training Phase (2018-2025): Trades & PnL')
    
    ax2 = plt.subplot(3, 1, 2)
    test_data = df.loc[TEST_START_DATE:]
    if not test_data.empty:
        plot_strategy_performance(ax2, test_data, best_genes, f'Test Phase (2026): Unseen Data Performance')
    else:
        ax2.text(0.5, 0.5, 'No Test Data', ha='center')

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

# --- LIVE TRADING ---
def log_message(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    full_msg = f"[{ts}] {msg}"
    print(full_msg)
    with state_lock:
        live_state['logs'].append(full_msg)

def live_trading_loop(best_genes, m, c):
    log_message("--- INITIALIZING LIVE TRADING ENGINE ---")
    exchange = ccxt.binance()
    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0.0
    active_sl = 0.0
    active_tp = 0.0
    
    # Parse levels
    levels = []
    for i in range(0, len(best_genes), 3):
        levels.append({'thresh': best_genes[i], 'sl': best_genes[i+1], 'tp': best_genes[i+2]})

    # Initialize previous state
    log_message("Pre-fetching initial state...")
    try:
        init_candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=2)
        if len(init_candles) >= 2:
            prev_candle = init_candles[-2]
            prev_ts = pd.to_datetime(prev_candle[0], unit='ms')
            prev_ord = prev_ts.toordinal()
            prev_detrended = prev_candle[4] - (m * prev_ord + c)
            log_message(f"State initialized. Prev Detrended: {prev_detrended:.2f}")
        else:
            prev_detrended = 0
            log_message("Warning: Insufficient history for initialization.")
    except Exception as e:
        log_message(f"Init Error: {e}")
        prev_detrended = 0

    while True:
        try:
            # Sync to 2 seconds after minute
            now = datetime.datetime.now()
            next_minute = (now + datetime.timedelta(minutes=1)).replace(second=2, microsecond=0)
            sleep_sec = (next_minute - now).total_seconds()
            if sleep_sec < 0: sleep_sec += 60
            
            time.sleep(sleep_sec)
            
            # Fetch Latest Candle
            candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1)
            if not candles:
                continue
                
            candle = candles[0]
            ts = pd.to_datetime(candle[0], unit='ms')
            price = candle[4]
            ordinal = ts.toordinal()
            
            # Calculate Detrended
            trend = m * ordinal + c
            curr_detrended = price - trend
            
            # Update Global State
            with state_lock:
                live_state['price'] = price
                live_state['detrended'] = curr_detrended
                live_state['balance'] = balance
                live_state['position'] = position
                live_state['last_update'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            log_message(f"Price: {price:.2f} | Detrended: {curr_detrended:.2f} | Bal: {balance:.2f} | Pos: {position}")
            
            # Logic
            # Check Exit
            if position != 0:
                pnl_pct = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
                if pnl_pct <= -active_sl or pnl_pct >= active_tp:
                    balance *= (1 + pnl_pct)
                    log_message(f">>> EXIT {'LONG' if position==1 else 'SHORT'}: PnL {pnl_pct*100:.2f}% | New Balance: {balance:.2f}")
                    position = 0
            
            # Check Entry
            if position == 0:
                for lvl in levels:
                    threshold = lvl['thresh']
                    # Cross UP -> Short
                    if prev_detrended < threshold and curr_detrended >= threshold:
                        position = -1
                        entry_price = price
                        active_sl = lvl['sl']
                        active_tp = lvl['tp']
                        log_message(f">>> ENTRY SHORT @ {price:.2f} (Thresh: {threshold:.2f})")
                        break
                    # Cross DOWN -> Long
                    elif prev_detrended > threshold and curr_detrended <= threshold:
                        position = 1
                        entry_price = price
                        active_sl = lvl['sl']
                        active_tp = lvl['tp']
                        log_message(f">>> ENTRY LONG @ {price:.2f} (Thresh: {threshold:.2f})")
                        break
            
            prev_detrended = curr_detrended
            
        except Exception as e:
            log_message(f"Live Loop Error: {e}")
            time.sleep(5)

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/plot.png':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            self.wfile.write(plot_buffer.getvalue())
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            with state_lock:
                current_price = live_state['price']
                current_detrend = live_state['detrended']
                current_bal = live_state['balance']
                current_pos = "NEUTRAL"
                if live_state['position'] == 1: current_pos = "LONG"
                elif live_state['position'] == -1: current_pos = "SHORT"
                last_up = live_state['last_update']
                logs_html = "<br>".join(reversed(list(live_state['logs'])))
            
            html = f"""
            <html>
            <head>
                <title>Trading Strategy Dashboard</title>
                <style>
                    body {{ font-family: monospace; background-color: #1e1e1e; color: #d4d4d4; padding: 20px; }}
                    .stats {{ display: flex; gap: 20px; background: #252526; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .stat-box {{ border: 1px solid #3e3e42; padding: 10px; min-width: 150px; }}
                    .label {{ color: #858585; font-size: 0.9em; }}
                    .value {{ font-size: 1.2em; font-weight: bold; color: #4ec9b0; }}
                    .log-box {{ background: #000; padding: 10px; height: 200px; overflow-y: scroll; border: 1px solid #333; font-size: 0.9em; color: #ce9178; }}
                    img {{ max-width: 100%; border: 1px solid #333; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <h2>Strategy Live Execution: {SYMBOL}</h2>
                <div class="stats">
                    <div class="stat-box"><div class="label">Last Update</div><div class="value">{last_up}</div></div>
                    <div class="stat-box"><div class="label">Price</div><div class="value">${current_price:.2f}</div></div>
                    <div class="stat-box"><div class="label">Detrended</div><div class="value">{current_detrend:.2f}</div></div>
                    <div class="stat-box"><div class="label">Balance</div><div class="value">${current_bal:.2f}</div></div>
                    <div class="stat-box"><div class="label">Position</div><div class="value">{current_pos}</div></div>
                </div>
                
                <h3>Live Logs</h3>
                <div class="log-box">
                    {logs_html}
                </div>
                
                <img src="/plot.png" alt="Strategy Performance">
            </body>
            </html>
            """
            self.wfile.write(html.encode())

if __name__ == "__main__":
    df = fetch_ohlc()
    if df.empty: exit()
    
    try:
        df, trend_m, trend_c = apply_indicators(df)
    except ValueError as e:
        print(f"Error: {e}")
        exit()
        
    print("Running GA...")
    best_ind = optimize(df)
    
    print("Generating Plot...")
    plot_buffer = generate_plot(df, best_ind)
    
    # Start Live Loop Thread
    live_thread = threading.Thread(target=live_trading_loop, args=(best_ind, trend_m, trend_c), daemon=True)
    live_thread.start()
    
    print(f"Serving dashboard at http://localhost:{SERVER_PORT} ... (Ctrl+C to stop)")
    with socketserver.TCPServer(("", SERVER_PORT), DashboardHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
