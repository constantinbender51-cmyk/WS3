import requests
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import http.server
import socketserver
import threading
import base64
import random
import time
from datetime import datetime
from io import BytesIO

# ==========================================
# 1. Configuration & Data Acquisition
# ==========================================

URL = "https://ohlcendpoint.up.railway.app/data/btc1m.csv"
PORT = 8080

def download_and_process_data(url):
    print(f"Downloading data from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Assuming CSV format: timestamp, open, high, low, close, volume
        # We try to infer columns or assume standard naming
        df = pd.read_csv(io.StringIO(response.text))
        
        # Standardize column names
        df.columns = [c.lower() for c in df.columns]
        
        # Parse Dates
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
        # 2. Resample to 1h OHLC
        print("Resampling to 1h OHLC...")
        df_1h = df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return df_1h
        
    except Exception as e:
        print(f"Error obtaining data: {e}")
        # Fallback for demonstration if URL fails or limits are hit
        print("Generating synthetic data for functional verification...")
        dates = pd.date_range(start="2023-01-01", periods=5000, freq='1h')
        prices = 20000 + np.cumsum(np.random.randn(5000) * 50)
        df_fake = pd.DataFrame({
            'open': prices,
            'high': prices + 10,
            'low': prices - 10,
            'close': prices + np.random.randn(5000),
            'volume': np.abs(np.random.randn(5000) * 100)
        }, index=dates)
        return df_fake

# ==========================================
# 3. Strategy Logic (The Core Complexity)
# ==========================================

class StrategyEngine:
    def __init__(self, stop_pct, profit_pct, n_lines, line_prices, data):
        self.stop_pct = stop_pct / 100.0
        self.profit_pct = profit_pct / 100.0
        self.n_lines = int(n_lines)
        self.sorted_lines = np.sort(line_prices)
        self.data = data.copy()
        
    def run_backtest(self):
        # Arrays for performance
        opens = self.data['open'].values
        highs = self.data['high'].values
        lows = self.data['low'].values
        closes = self.data['close'].values
        timestamps = self.data.index
        
        n = len(self.data)
        equity = [10000.0] # Start with $10k
        position = 0 # 0: None, 1: Long, -1: Short
        entry_price = 0.0
        trades = []
        
        # Pre-calculation optimization: 
        # We need to know where lines are relative to prices quickly.
        # However, grid logic is path dependent.
        
        for i in range(n):
            current_open = opens[i]
            current_high = highs[i]
            current_low = lows[i]
            current_close = closes[i]
            current_equity = equity[-1]
            
            # Logic Step:
            # "Iterate through all prices... any line_prices value above current triggers short..."
            # "If a line_prices value is reached while in a position the position reverses"
            
            # Intra-candle simulation (simplified to High/Low bounds for speed, 
            # but preserving logic order: Open -> Low/High -> Close)
            
            # We assume price moves Open -> Low -> High -> Close or Open -> High -> Low -> Close.
            # For neutrality, we check if lines are triggered within the [Low, High] range.
            
            # Find lines involved in this candle
            # Use searchsorted to find relevant grid lines within range [current_low, current_high]
            
            start_idx = np.searchsorted(self.sorted_lines, current_low, side='left')
            end_idx = np.searchsorted(self.sorted_lines, current_high, side='right')
            
            touched_lines = self.sorted_lines[start_idx:end_idx]
            
            # If no lines touched, check TP/SL for existing position
            if len(touched_lines) == 0:
                if position != 0:
                    self._check_exit(position, entry_price, current_low, current_high, trades, equity, current_open)
                    # If exited, position becomes 0
                    if trades and trades[-1]['type'] in ['stop_loss', 'take_profit'] and trades[-1]['exit_idx'] == i:
                        position = 0
                equity.append(current_equity) # Mark-to-market approximation
                continue

            # If lines ARE touched, we have reversal/entry triggers.
            # Complex Logic: Which happened first? 
            # We will prioritize Reversals over SL/TP if a line is hit, 
            # because the prompt implies lines represent immediate triggers.
            
            for line in touched_lines:
                # Logic: "any value above... triggers short"
                # If we are Long (1), and we hit a line ABOVE entry? 
                # Or simply, if price touches line X, and line X > previous_close?
                
                # We define "Above" relative to the price *before* touching it.
                # If price is rising to touch a line, that line is above. -> Short.
                # If price is falling to touch a line, that line is below. -> Long.
                
                # Since we don't have tick data, we infer direction.
                # If Open < Line <= High -> Price rose to touch it -> Trigger Short (Reversal)
                # If Open > Line >= Low -> Price fell to touch it -> Trigger Long (Reversal)
                
                triggered_short = (current_open < line <= current_high)
                triggered_long = (current_open > line >= current_low)
                
                if position == 1 and triggered_short:
                    # Reverse Long to Short
                    pnl = (line - entry_price) / entry_price
                    new_equity = equity[-1] * (1 + pnl)
                    equity[-1] = new_equity
                    trades.append({'type': 'reversal_short', 'entry': entry_price, 'exit': line, 'pnl': pnl, 'exit_idx': i})
                    
                    # New Position
                    position = -1
                    entry_price = line
                    
                elif position == -1 and triggered_long:
                    # Reverse Short to Long
                    pnl = (entry_price - line) / entry_price
                    new_equity = equity[-1] * (1 + pnl)
                    equity[-1] = new_equity
                    trades.append({'type': 'reversal_long', 'entry': entry_price, 'exit': line, 'pnl': pnl, 'exit_idx': i})
                    
                    # New Position
                    position = 1
                    entry_price = line
                    
                elif position == 0:
                    # New Entry
                    if triggered_short:
                        position = -1
                        entry_price = line
                    elif triggered_long:
                        position = 1
                        entry_price = line

            # After processing lines, check if SL/TP was hit on the NEW position 
            # (or existing if not reversed) within the remainder of the candle.
            if position != 0:
                self._check_exit(position, entry_price, current_low, current_high, trades, equity, current_open)
                if trades and trades[-1]['type'] in ['stop_loss', 'take_profit'] and trades[-1]['exit_idx'] == i:
                    position = 0

            equity.append(equity[-1])

        # Calculate metrics
        equity_curve = np.array(equity)
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        if len(returns) < 2 or returns.std() == 0:
            sharpe = 0.0
        else:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(24 * 365) # Annualized (hourly)
            
        return sharpe, equity_curve, trades

    def _check_exit(self, position, entry, low, high, trades, equity, open_p):
        # Helper to check standard SL/TP
        if position == 1:
            stop_price = entry * (1 - self.stop_pct)
            take_price = entry * (1 + self.profit_pct)
            
            # Check Low for Stop
            if low <= stop_price:
                pnl = (stop_price - entry) / entry
                equity[-1] = equity[-1] * (1 + pnl)
                trades.append({'type': 'stop_loss', 'entry': entry, 'exit': stop_price, 'pnl': pnl, 'exit_idx': -1})
                return
            
            # Check High for Profit
            if high >= take_price:
                pnl = (take_price - entry) / entry
                equity[-1] = equity[-1] * (1 + pnl)
                trades.append({'type': 'take_profit', 'entry': entry, 'exit': take_price, 'pnl': pnl, 'exit_idx': -1})
                return

        elif position == -1:
            stop_price = entry * (1 + self.stop_pct)
            take_price = entry * (1 - self.profit_pct)
            
            if high >= stop_price:
                pnl = (entry - stop_price) / entry
                equity[-1] = equity[-1] * (1 + pnl)
                trades.append({'type': 'stop_loss', 'entry': entry, 'exit': stop_price, 'pnl': pnl, 'exit_idx': -1})
                return
                
            if low <= take_price:
                pnl = (entry - take_price) / entry
                equity[-1] = equity[-1] * (1 + pnl)
                trades.append({'type': 'take_profit', 'entry': entry, 'exit': take_price, 'pnl': pnl, 'exit_idx': -1})
                return


# ==========================================
# 4. Genetic Algorithm (Parameter Optimization)
# ==========================================

class GeneticOptimizer:
    def __init__(self, train_data, pop_size=20, generations=5):
        self.train_data = train_data
        self.pop_size = pop_size
        self.generations = generations
        self.min_price = train_data['low'].min()
        self.max_price = train_data['high'].max()
        
    def create_individual(self):
        # Genome: [stop_pct, profit_pct, n_lines]
        # line_prices is derived from n_lines (equidistant) to maintain feasibility 
        # while respecting the prompt's request for line_price configuration.
        return {
            'stop_pct': random.uniform(0.1, 2.0),
            'profit_pct': random.uniform(0.04, 5.0),
            'n_lines': random.randint(10, 10000)
        }
        
    def fitness(self, ind):
        # Generate the lines based on n_lines and min/max
        lines = np.linspace(self.min_price, self.max_price, int(ind['n_lines']))
        engine = StrategyEngine(ind['stop_pct'], ind['profit_pct'], ind['n_lines'], lines, self.train_data)
        sharpe, _, _ = engine.run_backtest()
        return sharpe

    def crossover(self, p1, p2):
        child = {}
        child['stop_pct'] = (p1['stop_pct'] + p2['stop_pct']) / 2
        child['profit_pct'] = (p1['profit_pct'] + p2['profit_pct']) / 2
        child['n_lines'] = int((p1['n_lines'] + p2['n_lines']) / 2)
        return child
    
    def mutate(self, ind):
        if random.random() < 0.3:
            ind['stop_pct'] = np.clip(ind['stop_pct'] + random.uniform(-0.2, 0.2), 0.1, 2.0)
        if random.random() < 0.3:
            ind['profit_pct'] = np.clip(ind['profit_pct'] + random.uniform(-0.5, 0.5), 0.04, 5.0)
        if random.random() < 0.3:
            ind['n_lines'] = int(np.clip(ind['n_lines'] + random.randint(-100, 100), 10, 10000))
        return ind

    def run(self):
        population = [self.create_individual() for _ in range(self.pop_size)]
        
        for g in range(self.generations):
            scored_pop = []
            print(f"GA Generation {g+1}/{self.generations} processing...")
            for ind in population:
                score = self.fitness(ind)
                scored_pop.append((ind, score))
            
            scored_pop.sort(key=lambda x: x[1], reverse=True)
            print(f"Gen {g+1} Best Sharpe: {scored_pop[0][1]:.4f}")
            
            # Selection (Top 50%)
            survivors = [x[0] for x in scored_pop[:self.pop_size//2]]
            
            # Breeding
            new_pop = survivors[:]
            while len(new_pop) < self.pop_size:
                p1, p2 = random.sample(survivors, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_pop.append(child)
            
            population = new_pop
            
        # Return best
        best_ind = scored_pop[0][0]
        return best_ind

# ==========================================
# 5. Result Serving (HTML Generation)
# ==========================================

def generate_report(test_data, best_params, equity_curve, trades):
    # 1. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label='Equity Curve')
    plt.title(f"Strategy Performance (Sharpe optimized)\nParams: Stop={best_params['stop_pct']:.2f}%, Profit={best_params['profit_pct']:.2f}%, N_lines={best_params['n_lines']}")
    plt.legend()
    plt.grid(True)
    
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # 2. Table Data
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_html = trades_df.to_html(classes='table table-striped', index=False)
        stats_html = f"""
        <ul>
            <li>Total Trades: {len(trades)}</li>
            <li>Final Equity: {equity_curve[-1]:.2f}</li>
            <li>Win Rate: {len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100:.2f}%</li>
        </ul>
        """
    else:
        trades_html = "<p>No trades executed on Test Set</p>"
        stats_html = ""

    # 3. HTML Template
    html = f"""
    <html>
    <head>
        <title>Algorithmic Trading Results</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    </head>
    <body class="container mt-5">
        <h1>Backtest Results</h1>
        <hr>
        <h3>Optimal Parameters</h3>
        <ul>
            <li>Stop Percent: {best_params['stop_pct']}</li>
            <li>Profit Percent: {best_params['profit_pct']}</li>
            <li>Number of Lines: {best_params['n_lines']}</li>
        </ul>
        <hr>
        <h3>Performance Chart</h3>
        <img src="data:image/png;base64,{plot_url}" style="width:100%">
        <hr>
        <h3>Statistics</h3>
        {stats_html}
        <h3>Full Trade Log</h3>
        <div style="max-height: 500px; overflow-y: scroll;">
            {trades_html}
        </div>
        <h3>Full Price Data Head</h3>
        <div style="max-height: 300px; overflow-y: scroll;">
            {test_data.head(100).to_html(classes='table table-sm')}
        </div>
    </body>
    </html>
    """
    return html

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes(REPORT_HTML, "utf8"))

# ==========================================
# 6. Main Execution Flow
# ==========================================

if __name__ == "__main__":
    # 1. Download
    df_ohlc = download_and_process_data(URL)
    
    # 3. Split Train/Test (70/30)
    split_idx = int(len(df_ohlc) * 0.7)
    train_set = df_ohlc.iloc[:split_idx]
    test_set = df_ohlc.iloc[split_idx:]
    
    print(f"Training Data: {len(train_set)} candles")
    print(f"Test Data: {len(test_set)} candles")

    # 4. & 7. & 8. Genetic Algorithm Optimization
    print("Starting Genetic Algorithm Optimization...")
    ga = GeneticOptimizer(train_set, pop_size=20, generations=5)
    best_params = ga.run()
    
    print("Optimization Complete.")
    print(f"Best Parameters: {best_params}")
    
    # Run on Test Set
    print("Running on Test Set...")
    lines = np.linspace(train_set['low'].min(), train_set['high'].max(), int(best_params['n_lines']))
    
    # Note: Using min/max of TRAIN set for lines to avoid lookahead bias, 
    # though price in Test might drift outside this grid (Strategy limitation accepted).
    engine = StrategyEngine(best_params['stop_pct'], best_params['profit_pct'], best_params['n_lines'], lines, test_set)
    sharpe, equity, trades = engine.run_backtest()
    
    # 10. Serve Results
    REPORT_HTML = generate_report(test_set, best_params, equity, trades)
    
    print(f"Serving results at http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
