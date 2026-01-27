import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import random
import http.server
import socketserver
import warnings
from deap import base, creator, tools, algorithms
from copy import deepcopy

# --- Configuration ---
DATA_URL = "https://ohlcendpoint.up.railway.app/data/btc1m.csv"
PORT = 8080
N_LINES = 1000
POPULATION_SIZE = 20  # Adjusted for execution speed; increase for deeper search
GENERATIONS = 5
RISK_FREE_RATE = 0.0

# Ranges
STOP_PCT_RANGE = (0.001, 0.02)   # 0.1% to 2%
PROFIT_PCT_RANGE = (0.0004, 0.05) # 0.04% to 5%

warnings.filterwarnings("ignore")

# --- 1. Data Ingestion & Processing ---
def get_data():
    print(f"Downloading data from {DATA_URL}...")
    try:
        df = pd.read_csv(DATA_URL)
        # Ensure column mapping matches source
        df.columns = [c.lower() for c in df.columns]
        if 'timestamp' not in df.columns:
            # Fallback if no header, assuming standard structure or first col is time
            df = pd.read_csv(DATA_URL, header=None)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample to 1H
        df_1h = df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        # Split Train (70%) / Test (30%)
        split_idx = int(len(df_1h) * 0.7)
        train = df_1h.iloc[:split_idx]
        test = df_1h.iloc[split_idx:]
        
        return train, test
    except Exception as e:
        print(f"Error getting data: {e}")
        exit()

# --- 2. Strategy Logic ---
def run_backtest(df, stop_pct, profit_pct, lines):
    """
    Executes the Grid Reversal Strategy.
    lines: np.array of 1000 price levels.
    """
    # Pre-computation for speed
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    times = df.index
    
    equity = 10000.0
    equity_curve = [equity]
    position = 0 # 1 (Long), -1 (Short), 0 (Flat)
    entry_price = 0.0
    entry_line_val = -1.0 # The specific line value that triggered the current trade
    
    trades = []
    
    # Sort lines for efficient searching (though we iterate, sorted helps visualization/logic)
    # Note: The GA can optimize lines to be anywhere, we sort them for the logic check if needed,
    # but strictly the genome order doesn't matter, just values.
    # To optimize lookup:
    lines = np.sort(lines)
    
    for i in range(1, len(df)):
        current_o = opens[i]
        current_h = highs[i]
        current_l = lows[i]
        current_c = closes[i]
        prev_c = closes[i-1]
        
        # 1. Check Exit conditions (SL/TP) if in position
        if position != 0:
            pn_l = 0
            exit_price = 0
            triggered_exit = False
            
            # Long Exit Logic
            if position == 1:
                sl_price = entry_price * (1 - stop_pct)
                tp_price = entry_price * (1 + profit_pct)
                
                # Check Low for SL
                if current_l <= sl_price:
                    exit_price = sl_price
                    pn_l = (exit_price - entry_price) / entry_price
                    triggered_exit = True
                    reason = "SL"
                # Check High for TP
                elif current_h >= tp_price:
                    exit_price = tp_price
                    pn_l = (exit_price - entry_price) / entry_price
                    triggered_exit = True
                    reason = "TP"
            
            # Short Exit Logic
            elif position == -1:
                sl_price = entry_price * (1 + stop_pct)
                tp_price = entry_price * (1 - profit_pct)
                
                # Check High for SL
                if current_h >= sl_price:
                    exit_price = sl_price
                    pn_l = (entry_price - exit_price) / entry_price
                    triggered_exit = True
                    reason = "SL"
                # Check Low for TP
                elif current_l <= tp_price:
                    exit_price = tp_price
                    pn_l = (entry_price - exit_price) / entry_price
                    triggered_exit = True
                    reason = "TP"
            
            if triggered_exit:
                equity *= (1 + pn_l)
                position = 0
                trades.append({
                    'time': times[i], 'type': 'Exit', 'price': exit_price, 
                    'pnl': pn_l, 'equity': equity, 'reason': reason
                })
                # Position is now closed, continue to check for new entries within same candle?
                # To avoid complexity of multiple trades per candle, we assume we are flat for rest of candle
                equity_curve.append(equity)
                continue 

        # 2. Check Grid Crossings (Entry or Reversal)
        # We look for lines intersected by the candle range [current_l, current_h]
        # Logic: 
        # Short Trigger: Line is ABOVE prev_close (Resistance). Price hits it (goes up).
        # Long Trigger: Line is BELOW prev_close (Support). Price hits it (goes down).
        
        # Optimization: Find lines within range [Low, High]
        # usage of searchsorted is faster than where for sorted arrays
        idx_start = np.searchsorted(lines, current_l)
        idx_end = np.searchsorted(lines, current_h, side='right')
        touched_lines = lines[idx_start:idx_end]
        
        if len(touched_lines) > 0:
            for line in touched_lines:
                # Determine direction relative to previous close
                # If Line > Prev Close -> We hit it from below -> Short
                # If Line < Prev Close -> We hit it from above -> Long
                
                new_signal = 0
                if line > prev_c:
                    new_signal = -1
                elif line < prev_c:
                    new_signal = 1
                
                if new_signal == 0: continue # Line exactly equal to prev close, unlikely but ignore
                
                # REVERSAL / ENTRY LOGIC
                if position == 0:
                    # Enter
                    position = new_signal
                    entry_price = line
                    entry_line_val = line
                    trades.append({
                        'time': times[i], 'type': 'Short' if position == -1 else 'Long', 
                        'price': entry_price, 'pnl': 0, 'equity': equity, 'reason': 'Entry'
                    })
                    break # Take first valid signal in candle logic
                
                elif position != 0:
                    # Check Reversal Constraint 13: A line can't flip itself
                    if line == entry_line_val:
                        continue
                    
                    # Reverse Logic: If we hit a new line
                    # Close current
                    exit_price = line
                    if position == 1:
                        pn_l = (exit_price - entry_price) / entry_price
                    else:
                        pn_l = (entry_price - exit_price) / entry_price
                        
                    equity *= (1 + pn_l)
                    trades.append({
                        'time': times[i], 'type': 'ReversalClose', 'price': exit_price, 
                        'pnl': pn_l, 'equity': equity, 'reason': 'Reverse'
                    })
                    
                    # Open Opposite
                    position = new_signal
                    entry_price = line
                    entry_line_val = line
                    trades.append({
                        'time': times[i], 'type': 'Short' if position == -1 else 'Long', 
                        'price': entry_price, 'pnl': 0, 'equity': equity, 'reason': 'ReverseEntry'
                    })
                    break # Process one reversal per candle max

        equity_curve.append(equity)

    return equity_curve, trades

def calculate_sharpe(equity_curve):
    if len(equity_curve) < 2: return -999.0
    returns = pd.Series(equity_curve).pct_change().dropna()
    if returns.std() == 0: return -999.0
    # Annualized Sharpe (assuming hourly data -> 24*365 = 8760)
    sharpe = np.sqrt(8760) * (returns.mean() / returns.std())
    return sharpe

# --- 3. Genetic Algorithm ---

def setup_ga(min_price, max_price):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generators
    toolbox.register("attr_stop", random.uniform, STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    toolbox.register("attr_profit", random.uniform, PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    toolbox.register("attr_line", random.uniform, min_price, max_price)

    # Structure: [Stop, Profit, Line1, Line2, ... Line1000]
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_stop, toolbox.attr_profit) + (toolbox.attr_line,)*N_LINES, n=1)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox

def evaluate_genome(individual, df_train):
    stop_pct = individual[0]
    profit_pct = individual[1]
    lines = np.array(individual[2:])
    
    # Enforce constraints softly or hard reset?
    # GA might push values out of bounds, clip them
    stop_pct = np.clip(stop_pct, STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    profit_pct = np.clip(profit_pct, PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    
    eq_curve, _ = run_backtest(df_train, stop_pct, profit_pct, lines)
    sharpe = calculate_sharpe(eq_curve)
    return (sharpe,)

def mutate_custom(individual, indpb, min_p, max_p):
    # Mutation for Stop/Profit
    if random.random() < indpb:
        individual[0] += random.gauss(0, 0.005)
        individual[0] = np.clip(individual[0], STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    if random.random() < indpb:
        individual[1] += random.gauss(0, 0.005)
        individual[1] = np.clip(individual[1], PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    
    # Mutation for Lines
    for i in range(2, len(individual)):
        if random.random() < (indpb / 10): # Lower prob for lines to keep stability
            # Move line slightly
            shift = random.gauss(0, (max_p - min_p) * 0.01) 
            individual[i] += shift
            individual[i] = np.clip(individual[i], min_p, max_p)
            
    return individual,

# --- 4. Server & Visualization ---

def generate_report(best_ind, train_data, test_data, train_curve, test_curve, test_trades):
    # Plotting
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.title("Equity Curve: Training (Blue) vs Test (Orange)")
    plt.plot(train_curve, label='Training Equity')
    plt.plot(range(len(train_curve), len(train_curve)+len(test_curve)), test_curve, label='Test Equity')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.title("Price Action & Optimized Grid (Last 500 hours of Test)")
    subset = test_data.iloc[-500:]
    plt.plot(subset.index, subset['close'], color='black', alpha=0.5)
    
    # Plot a subset of lines to avoid clutter, or all if feasible
    lines = best_ind[2:]
    # Only plot lines near the price action of the subset
    min_sub = subset['low'].min()
    max_sub = subset['high'].max()
    active_lines = [l for l in lines if min_sub * 0.95 < l < max_sub * 1.05]
    
    for l in active_lines:
        plt.axhline(y=l, color='blue', alpha=0.1, linewidth=0.5)
        
    plt.tight_layout()
    
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plot_url = base64.b64encode(img_io.getvalue()).decode()
    plt.close()
    
    # Tables
    trades_df = pd.DataFrame(test_trades)
    if not trades_df.empty:
        trades_html = trades_df.to_html(classes='table table-striped', index=False)
    else:
        trades_html = "<p>No trades generated in Test set.</p>"
        
    full_data_html = test_data.head(100).to_html(classes='table table-sm') # Show sample of OHLC
    
    params_html = f"""
    <ul>
        <li><strong>Stop Loss:</strong> {best_ind[0]*100:.4f}%</li>
        <li><strong>Take Profit:</strong> {best_ind[1]*100:.4f}%</li>
        <li><strong>Total Lines:</strong> {N_LINES}</li>
    </ul>
    """

    html_content = f"""
    <html>
    <head>
        <title>Grid Reversal Strategy Results</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    </head>
    <body class="p-4">
        <h1>Grid Reversal GA Results</h1>
        <hr>
        <h3>Parameters</h3>
        {params_html}
        <h3>Performance Charts</h3>
        <img src="data:image/png;base64,{plot_url}" style="width:100%; max-width:1200px;">
        <hr>
        <h3>Trade Log (Test Set)</h3>
        <div style="max-height: 500px; overflow-y: scroll;">
            {trades_html}
        </div>
        <hr>
        <h3>Data Sample (First 100 Test Candles)</h3>
        {full_data_html}
    </body>
    </html>
    """
    return html_content

class ResultsHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(HTML_REPORT.encode('utf-8'))

# --- Main Execution Flow ---
if __name__ == "__main__":
    # 1. Load Data
    train_df, test_df = get_data()
    print(f"Data Loaded. Train: {len(train_df)} rows, Test: {len(test_df)} rows.")
    
    min_price = train_df['low'].min()
    max_price = train_df['high'].max()
    print(f"Price Bounds for Grid: {min_price} - {max_price}")

    # 2. Setup GA
    toolbox = setup_ga(min_price, max_price)
    toolbox.register("evaluate", evaluate_genome, df_train=train_df)
    toolbox.register("mate", tools.cxTwoPoint) # Simple crossover for array
    toolbox.register("mutate", mutate_custom, indpb=0.1, min_p=min_price, max_p=max_price)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 3. Run GA
    print("Starting Genetic Algorithm Optimization...")
    pop = toolbox.population(n=POPULATION_SIZE)
    
    # Store Hall of Fame
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=GENERATIONS, 
                                   stats=stats, halloffame=hof, verbose=True)

    best_ind = hof[0]
    print(f"Best Sharpe on Train: {best_ind.fitness.values[0]:.4f}")
    
    # 4. Final Tests
    print("Running Final Test on Test Set...")
    train_curve, _ = run_backtest(train_df, best_ind[0], best_ind[1], np.array(best_ind[2:]))
    test_curve, test_trades = run_backtest(test_df, best_ind[0], best_ind[1], np.array(best_ind[2:]))
    
    test_sharpe = calculate_sharpe(test_curve)
    print(f"Sharpe on Test Set: {test_sharpe:.4f}")

    # 5. Serve Results
    HTML_REPORT = generate_report(best_ind, train_df, test_df, train_curve, test_curve, test_trades)
    
    print(f"Serving results on port {PORT}...")
    with socketserver.TCPServer(("", PORT), ResultsHandler) as httpd:
        print("Server running. Open http://localhost:8080")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.server_close()
            print("Server stopped.")
