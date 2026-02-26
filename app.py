import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for web server plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import io
import base64
import http.server
import socketserver
import urllib.parse
import socket
import time

# --- Configuration ---
SYMBOL = 'PEPE/USDT'     
TIMEFRAME = '1h'
DAYS_BACK = 30           
STARTING_BALANCE = 10000
PORT = 8000

# Global variable to hold market data so we only fetch it once
GLOBAL_DF = None

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def fetch_binance_data(symbol, timeframe, days):
    print(f"Fetching {days} days of {timeframe} data for {symbol}...")
    exchange = ccxt.binance()
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Pre-calculate Wick and Body metrics
    df['body'] = abs(df['close'] - df['open'])
    df['top_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['bot_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['max_wick'] = df[['top_wick', 'bot_wick']].max(axis=1)
    
    df['wick_pct'] = df['max_wick'] / df['open']
    df['wick_body_ratio'] = np.where(df['body'] == 0, np.inf, df['max_wick'] / df['body'])
    
    return df

# ==========================================
# 1. CORE ENGINE (Detailed UI Tracking)
# ==========================================
def run_backtest(df, sl_pct, tsl_pct, f_pct, g_pct, u_pct):
    balance = STARTING_BALANCE
    sl_history, tsl_history = [False], [False]
    position_history, exit_price_history = [0], [0.0]
    state_history, equity_history = [2], [balance]
    
    state = 2
    low_profit_count = 0
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # State Transitions
        if prev['wick_pct'] > f_pct or prev['wick_body_ratio'] > g_pct:
            state = 1
            low_profit_count = 0 
        elif low_profit_count >= 3:
            state = 2
            
        if state == 2:
            sl_history.append(False); tsl_history.append(False); position_history.append(0)
            exit_price_history.append(current['close']); state_history.append(2); equity_history.append(balance)
            continue
            
        # State 1 Trading
        position = 1 if prev['close'] >= prev['open'] else -1
        entry_price = current['open']
        sl_price = entry_price * (1 - sl_pct) if position == 1 else entry_price * (1 + sl_pct)
        
        sl_hit, tsl_hit = False, False
        exit_price, pnl_pct = current['close'], 0.0
        
        if position == 1:
            if current['low'] <= sl_price:
                sl_hit, exit_price, pnl_pct = True, sl_price, -sl_pct
            else:
                act_price = entry_price * (1 + tsl_pct)
                if current['high'] >= act_price:
                    trailing_sl = current['high'] * (1 - tsl_pct)
                    if current['close'] <= trailing_sl:
                        tsl_hit, exit_price, pnl_pct = True, trailing_sl, (trailing_sl - entry_price) / entry_price
                    else:
                        pnl_pct = (current['close'] - entry_price) / entry_price
                else:
                    pnl_pct = (current['close'] - entry_price) / entry_price
        else:
            if current['high'] >= sl_price:
                sl_hit, exit_price, pnl_pct = True, sl_price, -sl_pct
            else:
                act_price = entry_price * (1 - tsl_pct)
                if current['low'] <= act_price:
                    trailing_sl = current['low'] * (1 + tsl_pct)
                    if current['close'] >= trailing_sl:
                        tsl_hit, exit_price, pnl_pct = True, trailing_sl, (entry_price - trailing_sl) / entry_price
                    else:
                        pnl_pct = (entry_price - current['close']) / entry_price
                else:
                    pnl_pct = (entry_price - current['close']) / entry_price
                    
        # Triggers
        if pnl_pct < u_pct: low_profit_count += 1
        else: low_profit_count = 0
                
        balance *= (1 + pnl_pct) 
            
        sl_history.append(sl_hit); tsl_history.append(tsl_hit); position_history.append(position)
        exit_price_history.append(exit_price); state_history.append(1); equity_history.append(balance)
                
    return balance, sl_history, tsl_history, position_history, exit_price_history, state_history, equity_history

# ==========================================
# 2. FAST ENGINE (For Genetic Algorithm)
# ==========================================
def fast_ga_backtest(opens, highs, lows, closes, wick_pcts, wick_body_ratios, sl_pct, tsl_pct, f_pct, g_pct, u_pct):
    """Numpy-optimized engine to run 10,000x faster during GA optimization"""
    balance = 10000.0
    equity = np.empty(len(opens))
    equity[0] = balance
    state, low_profit_count = 2, 0
    
    for i in range(1, len(opens)):
        if wick_pcts[i-1] > f_pct or wick_body_ratios[i-1] > g_pct:
            state = 1
            low_profit_count = 0
        elif low_profit_count >= 3:
            state = 2
            
        if state == 2:
            equity[i] = balance
            continue
            
        position = 1 if closes[i-1] >= opens[i-1] else -1
        c_open, c_high, c_low, c_close = opens[i], highs[i], lows[i], closes[i]
        
        sl_price = c_open * (1 - sl_pct) if position == 1 else c_open * (1 + sl_pct)
        pnl = 0.0
        
        if position == 1:
            if c_low <= sl_price: pnl = -sl_pct
            else:
                if c_high >= c_open * (1 + tsl_pct):
                    tsl = c_high * (1 - tsl_pct)
                    pnl = (tsl - c_open)/c_open if c_close <= tsl else (c_close - c_open)/c_open
                else: pnl = (c_close - c_open)/c_open
        else:
            if c_high >= sl_price: pnl = -sl_pct
            else:
                if c_low <= c_open * (1 - tsl_pct):
                    tsl = c_low * (1 + tsl_pct)
                    pnl = (c_open - tsl)/c_open if c_close >= tsl else (c_open - c_close)/c_open
                else: pnl = (c_open - c_close)/c_open
                    
        if pnl < u_pct: low_profit_count += 1
        else: low_profit_count = 0
            
        balance *= (1 + pnl)
        equity[i] = balance
        
    return equity

def run_ga_optimization(df):
    print("\n🧬 Starting Genetic Algorithm (Optimizing for Sharpe Ratio)...")
    start_time = time.time()
    
    # Extract Numpy Arrays
    opens, highs, lows, closes = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    wick_pcts, wick_body_ratios = df['wick_pct'].values, df['wick_body_ratio'].values

    # GA Settings
    POP_SIZE = 50
    GENERATIONS = 15
    
    # Bounds: [f, g, u, sl, tsl]
    bounds = np.array([[0.1, 4.0], [50.0, 400.0], [-1.0, 2.0], [0.5, 6.0], [0.5, 6.0]])
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (POP_SIZE, 5))

    def evaluate(ind):
        equity = fast_ga_backtest(opens, highs, lows, closes, wick_pcts, wick_body_ratios, 
                                  ind[3]/100.0, ind[4]/100.0, ind[0]/100.0, ind[1]/100.0, ind[2]/100.0)
        returns = np.diff(equity) / equity[:-1]
        std = np.std(returns)
        if std == 0: return -999.0
        return (np.mean(returns) / std) * np.sqrt(365*24) # Annualized Hourly Sharpe

    for gen in range(GENERATIONS):
        fitness = np.array([evaluate(ind) for ind in pop])
        
        # Elitism
        best_idx = np.argmax(fitness)
        next_pop = [pop[best_idx], pop[best_idx].copy()] # Keep top 2
        
        while len(next_pop) < POP_SIZE:
            # Tournament
            i1, i2 = np.random.choice(POP_SIZE, 2, replace=False)
            p1 = pop[i1] if fitness[i1] > fitness[i2] else pop[i2]
            i1, i2 = np.random.choice(POP_SIZE, 2, replace=False)
            p2 = pop[i1] if fitness[i1] > fitness[i2] else pop[i2]
            
            # Crossover
            child = np.concatenate([p1[:2], p2[2:]]) if np.random.rand() < 0.6 else p1.copy()
                
            # Mutation
            for i in range(5):
                if np.random.rand() < 0.25:
                    child[i] += np.random.normal(0, (bounds[i, 1] - bounds[i, 0]) * 0.1)
                    child[i] = np.clip(child[i], bounds[i, 0], bounds[i, 1])
                    
            next_pop.append(child)
        pop = np.array(next_pop)
        
    fitness = np.array([evaluate(ind) for ind in pop])
    best_params = pop[np.argmax(fitness)]
    
    print(f"✅ Optimization Finished in {time.time()-start_time:.1f}s! Best Sharpe: {np.max(fitness):.2f}")
    return { 'f': round(best_params[0], 2), 'g': round(best_params[1], 2), 'u': round(best_params[2], 2),
             'sl': round(best_params[3], 2), 'tsl': round(best_params[4], 2) }

# ==========================================
# 3. WEB VISUALIZATION & REPORTING
# ==========================================
def generate_html_report(df, params, final_balance, roi, sharpe):
    print(f"Generating charts for web display...")
    plt.figure(figsize=(16, 12))
    df_2d = df.tail(48).copy()
    
    # --- PLOT 1: Candlesticks & States (48H) ---
    ax1 = plt.subplot(2, 1, 1)
    up, down = df_2d[df_2d['close'] >= df_2d['open']], df_2d[df_2d['close'] < df_2d['open']]
    width = 0.03 
    
    ax1.bar(up['timestamp'], up['close'] - up['open'], bottom=up['open'], color='green', width=width, zorder=3)
    ax1.vlines(up['timestamp'], up['low'], up['high'], color='green', linewidth=1, zorder=3)
    ax1.bar(down['timestamp'], down['open'] - down['close'], bottom=down['close'], color='red', width=width, zorder=3)
    ax1.vlines(down['timestamp'], down['low'], down['high'], color='red', linewidth=1, zorder=3)

    y_max, y_min = df_2d['high'].max() * 1.01, df_2d['low'].min() * 0.99
    ax1.fill_between(df_2d['timestamp'], y_max, y_min, where=(df_2d['position'] == 1), color='green', alpha=0.15, label='State 1: Long', zorder=1)
    ax1.fill_between(df_2d['timestamp'], y_max, y_min, where=(df_2d['position'] == -1), color='red', alpha=0.15, label='State 1: Short', zorder=1)
    ax1.fill_between(df_2d['timestamp'], y_max, y_min, where=(df_2d['state'] == 2), color='gray', alpha=0.10, label='State 2: Flat', zorder=1)

    sl_data, tsl_data = df_2d[df_2d['sl_hit'] == True], df_2d[df_2d['tsl_hit'] == True]
    if not sl_data.empty: ax1.scatter(sl_data['timestamp'], sl_data['exit_price'], marker='X', color='black', s=150, label='Initial SL', zorder=5)
    if not tsl_data.empty: ax1.scatter(tsl_data['timestamp'], tsl_data['exit_price'], marker='o', color='darkorange', s=120, label='Trailing SL', zorder=5)

    ax1.set_ylim(y_min, y_max)
    ax1.set_xlim(df_2d['timestamp'].min() - pd.Timedelta(hours=1), df_2d['timestamp'].max() + pd.Timedelta(hours=1))
    ax1.set_title(f"LAST 48 HOURS PRICE ({SYMBOL}) | f: {params['f']}% | g: {params['g']}% | u: {params['u']}%")
    ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3, zorder=0)
    
    # --- PLOT 2: Equity Curve (48H) ---
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    start_balance_2d, end_balance_2d = df_2d['equity'].iloc[0], df_2d['equity'].iloc[-1]
    roi_2d = ((end_balance_2d - start_balance_2d) / start_balance_2d) * 100
    
    ax2.plot(df_2d['timestamp'], df_2d['equity'], color='#2980b9', linewidth=2.5, label='Account Balance (USDT)')
    ax2.fill_between(df_2d['timestamp'], df_2d['equity'], df_2d['equity'].min() * 0.999, color='#2980b9', alpha=0.1)
    ax2.set_title(f"LAST 48 HOURS RETURNS | 2-Day ROI: {roi_2d:.2f}% | PnL: ${(end_balance_2d - start_balance_2d):.2f}")
    ax2.set_ylabel('Balance (USDT)'); ax2.legend(loc='upper left'); ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    
    plt.xticks(rotation=45); plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0); plt.close()
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hourly Reversal Backtest</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f4f9; color: #333; text-align: center; padding: 20px; }}
            .form-container {{ background: #2c3e50; color: white; border-radius: 8px; padding: 20px; margin: 0 auto 20px; width: 85%; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .form-container input {{ margin: 0 5px; padding: 5px; width: 60px; text-align: center; border-radius: 4px; border: none; font-size: 15px;}}
            .btn-run {{ padding: 10px 20px; background-color: #27ae60; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 16px; margin: 15px 10px; width: 180px;}}
            .btn-run:hover {{ background-color: #219150; }}
            .btn-opt {{ padding: 10px 20px; background-color: #f39c12; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 16px; margin: 15px 10px; width: 180px;}}
            .btn-opt:hover {{ background-color: #e67e22; }}
            .stats-container {{ background: #fff; border-radius: 8px; padding: 20px; margin: 0 auto 20px; width: 85%; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: flex; justify-content: space-around; }}
            .stat-box {{ margin: 0 10px; }}
            h1 {{ color: #2c3e50; margin-bottom: 5px;}}
            p.subtitle {{ color: #7f8c8d; margin-top: 0; margin-bottom: 20px; }}
            .value {{ font-size: 22px; font-weight: bold; color: #2980b9; margin-top: 5px; }}
            .chart-container img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px;}}
            .input-group {{ display: inline-block; margin: 5px 15px; text-align: right;}}
        </style>
    </head>
    <body>
        <h1>Advanced Hourly Reversal ({SYMBOL})</h1>
        <p class="subtitle">State 1/2 Logic | 100% Exposure | Genetic Optimizer</p>
        
        <div class="form-container">
            <form method="POST">
                <div class="input-group"><label>Wick Size > (f): <input type="number" step="0.1" name="f" value="{params['f']}"> %</label></div>
                <div class="input-group"><label>Wick/Body > (g): <input type="number" step="1" name="g" value="{params['g']}"> %</label></div>
                <div class="input-group"><label>Profit Threshold (u): <input type="number" step="0.1" name="u" value="{params['u']}"> %</label></div>
                <br>
                <div class="input-group"><label>Initial SL: <input type="number" step="0.1" name="sl" value="{params['sl']}"> %</label></div>
                <div class="input-group"><label>Trailing Act/Dist (c): <input type="number" step="0.1" name="tsl" value="{params['tsl']}"> %</label></div>
                <br>
                <button type="submit" name="action" value="run" class="btn-run">Run Backtest</button>
                <button type="submit" name="action" value="optimize" class="btn-opt">⚡ Optimize GA</button>
            </form>
        </div>

        <div class="stats-container">
            <div class="stat-box"><div>Starting Balance</div><div class="value">${STARTING_BALANCE:,.2f}</div></div>
            <div class="stat-box"><div>Final Balance (30D)</div><div class="value">${final_balance:,.2f}</div></div>
            <div class="stat-box"><div>30-Day Net ROI</div><div class="value" style="color:{'#27ae60' if roi >= 0 else '#c0392b'};">{roi:.2f}%</div></div>
            <div class="stat-box"><div>30-Day Sharpe Ratio</div><div class="value">{sharpe:.2f}</div></div>
        </div>
        
        <div class="chart-container">
            <img src="data:image/png;base64,{image_base64}" alt="Backtest Chart">
        </div>
    </body>
    </html>
    """
    return html

def execute_run(params):
    df_run = GLOBAL_DF.copy()
    
    sl, tsl, f, g, u = params['sl']/100.0, params['tsl']/100.0, params['f']/100.0, params['g']/100.0, params['u']/100.0
    final_balance, sls, tsls, positions, exit_prices, states, equity = run_backtest(df_run, sl, tsl, f, g, u)
    
    df_run['sl_hit'], df_run['tsl_hit'], df_run['position'] = sls, tsls, positions
    df_run['exit_price'], df_run['state'], df_run['equity'] = exit_prices, states, equity
    
    roi = ((final_balance - STARTING_BALANCE) / STARTING_BALANCE) * 100
    
    # Calculate Display Sharpe
    returns = np.diff(equity) / equity[:-1]
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(365*24)) if np.std(returns) > 0 else 0
    
    return generate_html_report(df_run, params, final_balance, roi, sharpe)

class BacktestServer(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        default_params = {'f': 1.0, 'g': 150.0, 'u': 0.5, 'sl': 2.0, 'tsl': 2.0}
        self.send_response(200); self.send_header('Content-type', 'text/html'); self.end_headers()
        self.wfile.write(execute_run(default_params).encode('utf-8'))

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        parsed = urllib.parse.parse_qs(self.rfile.read(length).decode('utf-8'))

        action = parsed.get('action', ['run'])[0]
        
        if action == 'optimize':
            params = run_ga_optimization(GLOBAL_DF)
        else:
            params = {
                'f': float(parsed.get('f', ['1.0'])[0]), 'g': float(parsed.get('g', ['150.0'])[0]),
                'u': float(parsed.get('u', ['0.5'])[0]), 'sl': float(parsed.get('sl', ['2.0'])[0]),
                'tsl': float(parsed.get('tsl', ['2.0'])[0])
            }
        
        self.send_response(200); self.send_header('Content-type', 'text/html'); self.end_headers()
        self.wfile.write(execute_run(params).encode('utf-8'))

if __name__ == "__main__":
    GLOBAL_DF = fetch_binance_data(SYMBOL, TIMEFRAME, DAYS_BACK)
    handler = BacktestServer
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"\n✅ Web server running! Open browser to: http://{get_local_ip()}:{PORT}  (or localhost:{PORT})")
        print("Press Ctrl+C to stop.")
        try: httpd.serve_forever()
        except KeyboardInterrupt: print("\nShutting down.")