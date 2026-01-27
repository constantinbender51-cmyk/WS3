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
N_LINES = 100
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
        df = pd.read_csv(DATA_URL)
        df.columns = [c.lower().strip() for c in df.columns]
        
        if 'datetime' in df.columns:
            df['dt'] = pd.to_datetime(df['datetime'], errors='coerce')
        elif 'timestamp' in df.columns:
            df['dt'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        else:
            raise ValueError("No valid date column found.")

        df.dropna(subset=['dt', 'open', 'high', 'low', 'close'], inplace=True)
        df.set_index('dt', inplace=True)
        df.sort_index(inplace=True)

        print(f"Raw 1m Data: {len(df)} rows")

        df_1h = df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        print(f"Resampled 1H Data: {len(df_1h)} rows")
        
        if len(df_1h) < 100:
            raise ValueError("Data insufficient after resampling.")

        split_idx = int(len(df_1h) * 0.7)
        train = df_1h.iloc[:split_idx]
        test = df_1h.iloc[split_idx:]
        
        return train, test

    except Exception as e:
        print(f"CRITICAL DATA ERROR: {e}")
        exit(1)

# --- 2. Strategy Logic ---
def run_backtest(df, stop_pct, profit_pct, lines, detailed_log_trades=0):
    """
    Executes the Grid Strategy with Intra-Candle Logic.
    - Opens on line cross.
    - Closes on NEXT line cross OR SL/TP (whichever comes first).
    """
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    times = df.index
    
    equity = 10000.0
    equity_curve = [equity]
    position = 0 
    entry_price = 0.0
    entry_line_val = -1.0 
    
    trades = []
    hourly_log = []
    
    # Sort lines for binary search
    lines = np.sort(lines)
    
    trades_completed = 0
    
    for i in range(1, len(df)):
        current_l = lows[i]
        current_h = highs[i]
        current_c = closes[i]
        prev_c = closes[i-1]
        ts = times[i]
        
        # --- Detailed Logging (Hourly Snapshot) ---
        if detailed_log_trades > 0 and trades_completed < detailed_log_trades:
            idx = np.searchsorted(lines, current_c)
            val_below = lines[idx-1] if idx > 0 else -999.0
            val_above = lines[idx] if idx < len(lines) else 999999.0
            
            act_sl = np.nan
            act_tp = np.nan
            pos_str = "FLAT"
            
            if position == 1:
                pos_str = "LONG"
                act_sl = entry_price * (1 - stop_pct)
                act_tp = entry_price * (1 + profit_pct)
            elif position == -1:
                pos_str = "SHORT"
                act_sl = entry_price * (1 + stop_pct)
                act_tp = entry_price * (1 - profit_pct)
            
            log_entry = {
                "Timestamp": str(ts),
                "Price": f"{current_c:.2f}",
                "Nearest Below": f"{val_below:.2f}" if val_below != -999 else "None",
                "Nearest Above": f"{val_above:.2f}" if val_above != 999999 else "None",
                "Position": pos_str,
                "Active SL": f"{act_sl:.2f}" if not np.isnan(act_sl) else "-",
                "Active TP": f"{act_tp:.2f}" if not np.isnan(act_tp) else "-",
                "Equity": f"{equity:.2f}"
            }
            hourly_log.append(log_entry)

        # --- Intra-Candle Simulation ---
        # 1. Identify all lines touched in this candle
        idx_start = np.searchsorted(lines, current_l)
        idx_end = np.searchsorted(lines, current_h, side='right')
        touched_lines = lines[idx_start:idx_end]
        
        # 2. Sort lines by proximity to PREVIOUS CLOSE to simulate path
        #    We assume price moves from PrevClose -> Line A -> Line B...
        if len(touched_lines) > 0:
            touched_lines = sorted(touched_lines, key=lambda x: abs(x - prev_c))
        
        # Track the "last processed price" to calculate SL/TP hits between lines
        last_price_ref = prev_c if position == 0 else entry_price
        
        for line in touched_lines:
            
            # --- A. CHECK EXIT (If Position is Open) ---
            if position != 0:
                # We are moving from last_price_ref -> line.
                # Did we hit SL or TP in this segment?
                
                sl_hit = False
                tp_hit = False
                exit_p = 0.0
                reason = ""
                
                if position == 1: # Long
                    sl_price = entry_price * (1 - stop_pct)
                    tp_price = entry_price * (1 + profit_pct)
                    
                    # Logic: If SL is between last ref and current line (or beyond line in adverse dir)
                    # Note: We sort lines by distance. 
                    # If we are Long, SL is below entry.
                    # If 'line' is below 'sl_price', we hit SL.
                    if line <= sl_price: 
                        sl_hit = True
                        exit_p = sl_price
                    elif line >= tp_price:
                        tp_hit = True
                        exit_p = tp_price
                        
                elif position == -1: # Short
                    sl_price = entry_price * (1 + stop_pct)
                    tp_price = entry_price * (1 - profit_pct)
                    
                    if line >= sl_price: 
                        sl_hit = True
                        exit_p = sl_price
                    elif line <= tp_price:
                        tp_hit = True
                        exit_p = tp_price

                # Execute SL/TP Exit if triggered
                if sl_hit or tp_hit:
                    pn_l = (exit_p - entry_price) / entry_price if position == 1 else (entry_price - exit_p) / entry_price
                    equity *= (1 + pn_l)
                    trades.append({'time': ts, 'type': 'Exit', 'price': exit_p, 'pnl': pn_l, 'equity': equity, 'reason': 'SL' if sl_hit else 'TP'})
                    position = 0
                    trades_completed += 1
                    # After SL/TP, we are Flat. The loop continues to process this 'line' as a potential NEW entry point?
                    # If we hit SL *before* reaching the line, we effectively stopped there.
                    # Ideally we stop processing this path to be conservative, or assume price continued to 'line'.
                    # We will continue, allowing re-entry if the line itself is a valid signal.
                
                else:
                    # We reached the grid line without hitting SL/TP.
                    # Rule: "If a line_prices value is crossed during a trade. Close the position."
                    # We exclude the entry line itself to avoid instant closing on signal noise.
                    if line != entry_line_val:
                        exit_p = line
                        pn_l = (exit_p - entry_price) / entry_price if position == 1 else (entry_price - exit_p) / entry_price
                        equity *= (1 + pn_l)
                        trades.append({'time': ts, 'type': 'GridExit', 'price': exit_p, 'pnl': pn_l, 'equity': equity, 'reason': 'LineCross'})
                        position = 0
                        trades_completed += 1

            # --- B. CHECK ENTRY (If Position is Flat) ---
            if position == 0:
                # Rule: Open on first, etc. 
                # We check signal relative to last reference (simulating crossing)
                
                # Signal Logic:
                # If Line > Prev (Upside Cross) -> Short (Fade)
                # If Line < Prev (Downside Cross) -> Long (Fade)
                
                new_signal = 0
                if line > last_price_ref: new_signal = -1
                elif line < last_price_ref: new_signal = 1
                
                if new_signal != 0:
                    position = new_signal
                    entry_price = line
                    entry_line_val = line
                    trades.append({'time': ts, 'type': 'Short' if position == -1 else 'Long', 'price': entry_price, 'pnl': 0, 'equity': equity, 'reason': 'Entry'})
            
            # Update reference for next iteration in this candle
            last_price_ref = line

        # End of Candle Equity Record
        equity_curve.append(equity)

    return equity_curve, trades, hourly_log

def calculate_sharpe(equity_curve):
    if len(equity_curve) < 2: return -999.0
    returns = pd.Series(equity_curve).pct_change().dropna()
    if returns.std() == 0: return -999.0
    return np.sqrt(8760) * (returns.mean() / returns.std())

# --- 3. Genetic Algorithm ---
def setup_ga(min_price, max_price):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
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
    eq_curve, _, _ = run_backtest(df_train, stop_pct, profit_pct, lines, detailed_log_trades=0)
    return (calculate_sharpe(eq_curve),)

def mutate_custom(individual, indpb, min_p, max_p):
    if random.random() < indpb:
        individual[0] = np.clip(individual[0] + random.gauss(0, 0.005), STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    if random.random() < indpb:
        individual[1] = np.clip(individual[1] + random.gauss(0, 0.005), PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    for i in range(2, len(individual)):
        if random.random() < (indpb / 10.0): 
            individual[i] = np.clip(individual[i] + random.gauss(0, (max_p - min_p) * 0.01), min_p, max_p)
    return individual,

# --- 4. Server & Visualization ---
def generate_report(best_ind, train_data, test_data, train_curve, test_curve, test_trades, hourly_log):
    plt.figure(figsize=(14, 12))
    
    # 1. Equity Curve
    plt.subplot(2, 1, 1)
    plt.title("Equity Curve: Training (Blue) vs Test (Orange)")
    plt.plot(train_curve, label='Training Equity')
    plt.plot(range(len(train_curve), len(train_curve)+len(test_curve)), test_curve, label='Test Equity')
    plt.legend()
    plt.grid(True)
    
    # 2. Full Price Action with Lines
    plt.subplot(2, 1, 2)
    plt.title("Full Test Set Price Action & Grid Lines")
    plt.plot(test_data.index, test_data['close'], color='black', alpha=0.6, label='Price', linewidth=0.8)
    
    lines = best_ind[2:]
    min_test = test_data['low'].min()
    max_test = test_data['high'].max()
    margin = (max_test - min_test) * 0.1
    visible_lines = [l for l in lines if (min_test - margin) < l < (max_test + margin)]
    
    for l in visible_lines:
        plt.axhline(y=l, color='blue', alpha=0.1, linewidth=0.5)
    
    plt.tight_layout()
    
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png', dpi=100)
    img_io.seek(0)
    plot_url = base64.b64encode(img_io.getvalue()).decode()
    plt.close()
    
    trades_df = pd.DataFrame(test_trades)
    trades_html = trades_df.to_html(classes='table table-striped table-sm', index=False, max_rows=500) if not trades_df.empty else "No trades."
    
    # Hourly Log Table
    hourly_df = pd.DataFrame(hourly_log)
    hourly_html = hourly_df.to_html(classes='table table-bordered table-sm table-hover', index=False) if not hourly_df.empty else "No hourly data recorded."

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
        <title>Grid Strategy Results (Dynamic Exit)</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>body {{ padding: 20px; }} h3 {{ margin-top: 30px; }} th {{ position: sticky; top: 0; background: white; }}</style>
    </head>
    <body>
        <div class="container-fluid">
            <h1 class="mb-4">Grid Strategy (Dynamic Exit) GA Results</h1>
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
            <h3>Trade Log (Test Set)</h3>
            <div style="max-height: 400px; overflow-y: scroll; border: 1px solid #ddd;">{trades_html}</div>
            
            <hr>
            <h3>Hourly Details (First 5 Trades Timeline)</h3>
            <p class="text-muted">Hourly snapshot of price, nearest grid lines, and active orders for the duration of the first 5 trades.</p>
            <div style="max-height: 600px; overflow-y: scroll; border: 1px solid #ddd;">
                {hourly_html}
            </div>
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

if __name__ == "__main__":
    train_df, test_df = get_data()
    print(f"Data Loaded. Train: {len(train_df)}, Test: {len(test_df)}")
    
    min_p, max_p = train_df['low'].min(), train_df['high'].max()
    toolbox = setup_ga(min_p, max_p)
    toolbox.register("evaluate", evaluate_genome, df_train=train_df)
    toolbox.register("mate", tools.cxTwoPoint) 
    toolbox.register("mutate", mutate_custom, indpb=0.1, min_p=min_p, max_p=max_p)
    toolbox.register("select", tools.selTournament, tournsize=3)

    print("Starting GA...")
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    best_ind = hof[0]
    print(f"Best Sharpe Train: {best_ind.fitness.values[0]:.4f}")
    
    # Run backtest again, this time logging detailed hourly info for first 5 trades
    print("Running Final Test...")
    train_curve, _, _ = run_backtest(train_df, best_ind[0], best_ind[1], np.array(best_ind[2:]), detailed_log_trades=0)
    test_curve, test_trades, hourly_log = run_backtest(test_df, best_ind[0], best_ind[1], np.array(best_ind[2:]), detailed_log_trades=5)
    print(f"Test Sharpe: {calculate_sharpe(test_curve):.4f}")

    HTML_REPORT = generate_report(best_ind, train_df, test_df, train_curve, test_curve, test_trades, hourly_log)
    
    print(f"Serving on {PORT}...")
    with socketserver.TCPServer(("", PORT), ResultsHandler) as httpd:
        try: httpd.serve_forever()
        except KeyboardInterrupt: httpd.server_close()
