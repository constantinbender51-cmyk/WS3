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

# --- Configuration ---
DATA_URL = "https://ohlcendpoint.up.railway.app/data/btc1m.csv"
PORT = 8080
N_LINES = 1000
POPULATION_SIZE = 20
GENERATIONS = 5
RISK_FREE_RATE = 0.0

# Ranges
STOP_PCT_RANGE = (0.001, 0.02)   # 0.1% to 2%
PROFIT_PCT_RANGE = (0.0004, 0.05) # 0.04% to 5%

warnings.filterwarnings("ignore")

# --- 1. Precise Data Ingestion ---
def get_data():
    print(f"Downloading data from {DATA_URL}...")
    try:
        # Read CSV. We know headers exist: timestamp, open, high, low, close, volume, datetime
        df = pd.read_csv(DATA_URL)
        
        # Standardize column names
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Parse Dates: Prefer 'datetime' column, fallback to 'timestamp' (ms)
        if 'datetime' in df.columns:
            df['dt'] = pd.to_datetime(df['datetime'], errors='coerce')
        elif 'timestamp' in df.columns:
            df['dt'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        else:
            raise ValueError("No valid date column found (checked 'datetime' and 'timestamp').")

        # Set Index and Drop Invalid
        df.dropna(subset=['dt', 'open', 'high', 'low', 'close'], inplace=True)
        df.set_index('dt', inplace=True)
        df.sort_index(inplace=True)

        print(f"Raw 1m Data: {len(df)} rows")

        # Resample to 1H
        # Aggregation rules: Open=First, High=Max, Low=Min, Close=Last
        df_1h = df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        print(f"Resampled 1H Data: {len(df_1h)} rows")
        
        if len(df_1h) < 100:
            raise ValueError("Data insufficient after resampling.")

        # Split Train (70%) / Test (30%)
        split_idx = int(len(df_1h) * 0.7)
        train = df_1h.iloc[:split_idx]
        test = df_1h.iloc[split_idx:]
        
        return train, test

    except Exception as e:
        print(f"CRITICAL DATA ERROR: {e}")
        exit(1)

# --- 2. Strategy Logic (Grid Reversal with Self-Rejection) ---
def run_backtest(df, stop_pct, profit_pct, lines):
    """
    Executes the Grid Reversal Strategy.
    """
    # Pre-fetch numpy arrays for speed
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    times = df.index
    
    equity = 10000.0
    equity_curve = [equity]
    position = 0 # 1 (Long), -1 (Short), 0 (Flat)
    entry_price = 0.0
    entry_line_val = -1.0 # Tracks which line triggered the current position
    
    trades = []
    
    # Sort lines for binary search (searchsorted)
    lines = np.sort(lines)
    
    for i in range(1, len(df)):
        current_h = highs[i]
        current_l = lows[i]
        prev_c = closes[i-1]
        
        # 1. Check Exit conditions (SL/TP)
        if position != 0:
            pn_l = 0
            exit_price = 0
            triggered_exit = False
            
            if position == 1: # Long
                sl_price = entry_price * (1 - stop_pct)
                tp_price = entry_price * (1 + profit_pct)
                
                if current_l <= sl_price:
                    exit_price = sl_price
                    pn_l = (exit_price - entry_price) / entry_price
                    triggered_exit = True
                    reason = "SL"
                elif current_h >= tp_price:
                    exit_price = tp_price
                    pn_l = (exit_price - entry_price) / entry_price
                    triggered_exit = True
                    reason = "TP"
            
            elif position == -1: # Short
                sl_price = entry_price * (1 + stop_pct)
                tp_price = entry_price * (1 - profit_pct)
                
                if current_h >= sl_price:
                    exit_price = sl_price
                    pn_l = (entry_price - exit_price) / entry_price
                    triggered_exit = True
                    reason = "SL"
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
                # Proceed to next candle; assume no re-entry in same candle for simplicity
                equity_curve.append(equity)
                continue 

        # 2. Check Grid Crossings
        # Find all lines intersected by the candle [Low, High]
        idx_start = np.searchsorted(lines, current_l)
        idx_end = np.searchsorted(lines, current_h, side='right')
        touched_lines = lines[idx_start:idx_end]
        
        if len(touched_lines) > 0:
            for line in touched_lines:
                new_signal = 0
                # Determine crossing direction relative to Previous Close
                if line > prev_c:
                    new_signal = -1 # Price moved up to hit line -> Resistance -> Short
                elif line < prev_c:
                    new_signal = 1 # Price moved down to hit line -> Support -> Long
                
                if new_signal == 0: continue
                
                if position == 0:
                    # New Entry
                    position = new_signal
                    entry_price = line
                    entry_line_val = line
                    trades.append({
                        'time': times[i], 'type': 'Short' if position == -1 else 'Long', 
                        'price': entry_price, 'pnl': 0, 'equity': equity, 'reason': 'Entry'
                    })
                    break 
                
                elif position != 0:
                    # REVERSAL LOGIC
                    # Constraint 13: A line cannot flip itself
                    if line == entry_line_val:
                        continue
                    
                    # Execute Reversal: Close Current -> Open Opposite
                    exit_price = line
                    
                    # Close PnL
                    if position == 1:
                        pn_l = (exit_price - entry_price) / entry_price
                    else:
                        pn_l = (entry_price - exit_price) / entry_price
                        
                    equity *= (1 + pn_l)
                    trades.append({
                        'time': times[i], 'type': 'ReversalClose', 'price': exit_price, 
                        'pnl': pn_l, 'equity': equity, 'reason': 'Reverse'
                    })
                    
                    # Open New
                    position = new_signal
                    entry_price = line
                    entry_line_val = line
                    trades.append({
                        'time': times[i], 'type': 'Short' if position == -1 else 'Long', 
                        'price': entry_price, 'pnl': 0, 'equity': equity, 'reason': 'ReverseEntry'
                    })
                    break # Process max one reversal per candle to preserve path integrity

        equity_curve.append(equity)

    return equity_curve, trades

def calculate_sharpe(equity_curve):
    if len(equity_curve) < 2: return -999.0
    returns = pd.Series(equity_curve).pct_change().dropna()
    if returns.std() == 0: return -999.0
    # Annualized Sharpe (hourly data -> 8760 periods/year)
    sharpe = np.sqrt(8760) * (returns.mean() / returns.std())
    return sharpe

# --- 3. Genetic Algorithm ---

def setup_ga(min_price, max_price):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Genome: [Stop%, Profit%, Line1, Line2, ... Line1000]
    toolbox.register("attr_stop", random.uniform, STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    toolbox.register("attr_profit", random.uniform, PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    toolbox.register("attr_line", random.uniform, min_price, max_price)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_stop, toolbox.attr_profit) + (toolbox.attr_line,)*N_LINES, n=1)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox

def evaluate_genome(individual, df_train):
    stop_pct = np.clip(individual[0], STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    profit_pct = np.clip(individual[1], PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    lines = np.array(individual[2:])
    
    eq_curve, _ = run_backtest(df_train, stop_pct, profit_pct, lines)
    sharpe = calculate_sharpe(eq_curve)
    return (sharpe,)

def mutate_custom(individual, indpb, min_p, max_p):
    # Mutate Risk Params
    if random.random() < indpb:
        individual[0] += random.gauss(0, 0.005)
        individual[0] = np.clip(individual[0], STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    if random.random() < indpb:
        individual[1] += random.gauss(0, 0.005)
        individual[1] = np.clip(individual[1], PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    
    # Mutate Lines (Sparse mutation)
    for i in range(2, len(individual)):
        if random.random() < (indpb / 10.0): 
            shift = random.gauss(0, (max_p - min_p) * 0.01) 
            individual[i] += shift
            individual[i] = np.clip(individual[i], min_p, max_p)
            
    return individual,

# --- 4. Server & Visualization ---

def generate_report(best_ind, train_data, test_data, train_curve, test_curve, test_trades):
    # Plotting
    plt.figure(figsize=(14, 10))
    
    # Equity Curve
    plt.subplot(2, 1, 1)
    plt.title("Equity Curve: Training (Blue) vs Test (Orange)")
    plt.plot(train_curve, label='Training Equity')
    plt.plot(range(len(train_curve), len(train_curve)+len(test_curve)), test_curve, label='Test Equity')
    plt.legend()
    plt.grid(True)
    
    # Price Action & Grid
    plt.subplot(2, 1, 2)
    plt.title("Price Action & Optimized Grid (Last 200 Hours)")
    subset = test_data.iloc[-200:]
    plt.plot(subset.index, subset['close'], color='black', alpha=0.6, label='Price')
    
    lines = best_ind[2:]
    min_sub = subset['low'].min()
    max_sub = subset['high'].max()
    # Filter lines visible in the window
    active_lines = [l for l in lines if min_sub * 0.95 < l < max_sub * 1.05]
    
    for l in active_lines:
        plt.axhline(y=l, color='blue', alpha=0.15, linewidth=0.8)
    
    plt.tight_layout()
    
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plot_url = base64.b64encode(img_io.getvalue()).decode()
    plt.close()
    
    # Data Tables
    trades_df = pd.DataFrame(test_trades)
    if not trades_df.empty:
        trades_html = trades_df.to_html(classes='table table-striped table-sm', index=False, max_rows=500)
    else:
        trades_html = "<p class='alert alert-warning'>No trades generated in Test set.</p>"
        
    full_data_html = test_data.head(50).to_html(classes='table table-bordered table-sm')
    
    params_html = f"""
    <ul class="list-group">
        <li class="list-group-item"><strong>Stop Loss:</strong> {best_ind[0]*100:.4f}%</li>
        <li class="list-group-item"><strong>Take Profit:</strong> {best_ind[1]*100:.4f}%</li>
        <li class="list-group-item"><strong>Active Grid Lines:</strong> {N_LINES}</li>
    </ul>
    """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Grid Reversal Strategy Results</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>body {{ padding: 20px; }} h3 {{ margin-top: 30px; }}</style>
    </head>
    <body>
        <div class="container-fluid">
            <h1 class="mb-4">Grid Reversal GA Results</h1>
            <div class="row">
                <div class="col-md-4">{params_html}</div>
                <div class="col-md-8 text-right">
                    <h5>Test Sharpe: {calculate_sharpe(test_curve):.4f}</h5>
                </div>
            </div>
            
            <hr>
            <h3>Performance Charts</h3>
            <img src="data:image/png;base64,{plot_url}" class="img-fluid border rounded">
            
            <hr>
            <h3>Trade Log (Test Set - First 500)</h3>
            <div style="max-height: 400px; overflow-y: scroll; border: 1px solid #ddd;">
                {trades_html}
            </div>
            
            <hr>
            <h3>Data Sample</h3>
            {full_data_html}
        </div>
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
    print(f"Data Loaded Successfully. Train: {len(train_df)} rows, Test: {len(test_df)} rows.")
    
    min_price = train_df['low'].min()
    max_price = train_df['high'].max()
    print(f"Price Bounds for Grid: {min_price} - {max_price}")

    # 2. Setup GA
    toolbox = setup_ga(min_price, max_price)
    toolbox.register("evaluate", evaluate_genome, df_train=train_df)
    toolbox.register("mate", tools.cxTwoPoint) 
    toolbox.register("mutate", mutate_custom, indpb=0.1, min_p=min_price, max_p=max_price)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 3. Run GA
    print(f"Starting Genetic Algorithm (Pop: {POPULATION_SIZE}, Gens: {GENERATIONS})...")
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=GENERATIONS, 
                                   stats=stats, halloffame=hof, verbose=True)

    best_ind = hof[0]
    print(f"Best Sharpe on Train: {best_ind.fitness.values[0]:.4f}")
    
    # 4. Final Verification
    print("Running Final Test on Test Set...")
    train_curve, _ = run_backtest(train_df, best_ind[0], best_ind[1], np.array(best_ind[2:]))
    test_curve, test_trades = run_backtest(test_df, best_ind[0], best_ind[1], np.array(best_ind[2:]))
    
    test_sharpe = calculate_sharpe(test_curve)
    print(f"Sharpe on Test Set: {test_sharpe:.4f}")

    # 5. Serve
    HTML_REPORT = generate_report(best_ind, train_df, test_df, train_curve, test_curve, test_trades)
    
    print(f"Serving results on port {PORT}...")
    with socketserver.TCPServer(("", PORT), ResultsHandler) as httpd:
        print("Server running. Open http://localhost:8080")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.server_close()
            print("Server stopped.")
