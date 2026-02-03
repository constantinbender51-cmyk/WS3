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

warnings.filterwarnings("ignore")

# 1. Fetch OHLC (2018-Present)
def fetch_data(symbol='BTC/USDT', timeframe='1d', since_year=2018):
    exchange = ccxt.binance()
    since = exchange.parse8601(f'{since_year}-01-01T00:00:00Z')
    all_candles = []
    
    print(f"Fetching {symbol}...")
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since)
            if not candles:
                break
            all_candles += candles
            since = candles[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
            if len(candles) < 1000: break
        except:
            break
            
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

# 2, 3, 4. Process: SMA, Shift -1460, Fit Line, Detrend
def process_data(df):
    # SMA 1460
    df['sma_1460'] = df['close'].rolling(window=1460).mean()
    
    # Shift -1460 (Moves future values to current index)
    # The last 1460 rows will be NaN
    df['curve_future'] = df['sma_1460'].shift(-1460)
    
    # Ordinal dates for regression
    df['ordinal'] = df.index.map(pd.Timestamp.toordinal)
    
    # Fit line ONLY to the valid shifted curve (historical data that has 'future' confirmation)
    fit_mask = df['curve_future'].notna()
    fit_data = df[fit_mask]
    
    if fit_data.empty:
        raise ValueError("Insufficient data length for -1460 shift + SMA window")

    def linear_func(x, m, c):
        return m * x + c
    
    popt, _ = curve_fit(linear_func, fit_data['ordinal'], fit_data['curve_future'])
    m, c = popt
    
    # Extrapolate line to entire dataset (including the recent 1460 days and 2026)
    df['trend_line'] = linear_func(df['ordinal'], m, c)
    
    # Subtract line from OHLC
    df['close_dt'] = df['close'] - df['trend_line']
    
    return df

# 5. GA Optimization
N_LEVELS = 3 # n reversal price levels

def backtest(individual, data):
    # individual: [thresh1, sl1, tp1, thresh2, sl2, tp2, ...]
    balance = 10000
    equity = []
    pos = 0 # 1=Long, -1=Short
    entry_price = 0.0
    active_sl = 0.0
    active_tp = 0.0
    
    # Parse genes into levels
    levels = []
    for i in range(0, len(individual), 3):
        levels.append({'val': individual[i], 'sl': individual[i+1], 'tp': individual[i+2]})
        
    # Run on Training Data (Up to end of 2025)
    train_slice = data.loc[:'2025-12-31']
    closes = train_slice['close'].values
    dt_closes = train_slice['close_dt'].values
    
    for i in range(1, len(train_slice)):
        price = closes[i]
        dt_curr = dt_closes[i]
        dt_prev = dt_closes[i-1]
        
        # Manage Open Position
        if pos != 0:
            pnl = (price - entry_price) / entry_price if pos == 1 else (entry_price - price) / entry_price
            if pnl <= -active_sl or pnl >= active_tp:
                balance *= (1 + pnl)
                pos = 0
        
        # Entry Logic
        if pos == 0:
            for lvl in levels:
                thresh = lvl['val']
                # Short: From below cross (upwards) -> Reversal Short
                # Assuming thresh is positive (upper bound)
                if dt_prev < thresh and dt_curr >= thresh:
                    pos = -1
                    entry_price = price
                    active_sl = lvl['sl']
                    active_tp = lvl['tp']
                    break
                
                # Long: From above cross (downwards) -> Reversal Long
                # Using negative threshold for lower bounds
                # If gene generates -5000:
                elif dt_prev > thresh and dt_curr <= thresh:
                    pos = 1
                    entry_price = price
                    active_sl = lvl['sl']
                    active_tp = lvl['tp']
                    break
        
        equity.append(balance)
        
    s = pd.Series(equity)
    if len(s) < 2: return -999,
    ret = s.pct_change().dropna()
    if ret.std() == 0: return -999,
    return (ret.mean() / ret.std()) * np.sqrt(365),

def run_ga(df):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    # Level: Deviation from linear trend
    toolbox.register("attr_lvl", random.uniform, -20000, 20000)
    toolbox.register("attr_sl", random.uniform, 0.01, 0.15)
    toolbox.register("attr_tp", random.uniform, 0.01, 0.30)
    
    def create_ind():
        l = []
        for _ in range(N_LEVELS):
            l.extend([toolbox.attr_lvl(), toolbox.attr_sl(), toolbox.attr_tp()])
        return l

    toolbox.register("ind", tools.initIterate, creator.Individual, create_ind)
    toolbox.register("pop", tools.initRepeat, list, toolbox.ind)
    toolbox.register("eval", backtest, data=df)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mut", tools.mutGaussian, mu=0, sigma=1000, indpb=0.1)
    toolbox.register("sel", tools.selTournament, tournsize=3)
    
    pop = toolbox.pop(n=50)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=15, verbose=False)
    
    return tools.selBest(pop, 1)[0]

# 6 & 7. Plot and Test Unseen
def serve_plot(df, best):
    plt.figure(figsize=(15, 10))
    
    # Main Plot: Detrended Prices + Levels
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df['close_dt'], label='Detrended Close', lw=0.8, color='black')
    
    # Plot Levels
    cols = ['r', 'g', 'b']
    for i in range(N_LEVELS):
        lvl = best[i*3]
        ax1.axhline(lvl, color=cols[i%3], ls='--', label=f'Lvl {i}: {lvl:.0f}')
        
    ax1.set_title('Detrended Price vs Linear Fit (Training Data)')
    ax1.legend()
    
    # Test Data Zoom (2026+)
    ax2 = plt.subplot(2, 1, 2)
    test = df.loc['2026-01-01':]
    
    if not test.empty:
        ax2.plot(test.index, test['close_dt'], label='2026 Unseen', lw=1.5, color='blue')
        for i in range(N_LEVELS):
            lvl = best[i*3]
            ax2.axhline(lvl, color=cols[i%3], ls='--', alpha=0.6)
        ax2.set_title(f'Performance on Unseen Data (Jan 2026 - Present) | Days: {len(test)}')
    else:
        ax2.text(0.5, 0.5, 'No 2026 Data', ha='center')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Simple Server
    class H(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type','image/png')
            self.end_headers()
            self.wfile.write(buf.getvalue())
            
    print("Serving on 8080...")
    try:
        http.server.HTTPServer(("", 8080), H).serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    df = fetch_data()
    if not df.empty:
        df = process_data(df)
        print("Optimizing...")
        best = run_ga(df)
        print(f"Best: {best}")
        serve_plot(df, best)
