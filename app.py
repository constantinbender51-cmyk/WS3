import http.server
import socketserver
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

# Global Storage for Pre-computed HTML
CACHED_HTML = b"<h3>Initializing... check back in 10 seconds.</h3>"

# ==========================================
# 1. Data & Processing Core
# ==========================================
def fetch_1yr_data():
    print("--- Fetching 1 Year of Data ---")
    base_url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
    
    all_data = []
    current_start = start_time
    
    while True:
        params = {'symbol': 'ETHUSDT', 'interval': '1h', 'limit': limit, 'startTime': current_start, 'endTime': end_time}
        try:
            resp = requests.get(base_url, params=params).json()
            if not resp or isinstance(resp, dict): break
            all_data.extend(resp)
            current_start = resp[-1][0] + 1
            if len(resp) < limit or current_start >= end_time: break
        except Exception as e:
            print(f"Fetch Error: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=['op_t', 'open', 'high', 'low', 'close', 'vol', 'cl_t', 'qv', 'nt', 'tb', 'tq', 'ig'])
    cols = ['open', 'high', 'low', 'close']
    df[cols] = df[cols].astype(float)
    df['timestamp'] = pd.to_datetime(df['op_t'], unit='ms')
    print(f"Loaded {len(df)} candles.")
    return df

def get_pnl(df, sl, tp):
    if df.empty: return np.array([])
    # Vectorized PnL Calculation
    o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    
    # Long
    l_win = (h >= o * (1 + tp)) & ~(l <= o * (1 - sl))
    l_loss = l <= o * (1 - sl)
    l_ret = np.where(l_loss, -sl, np.where(l_win, tp, (c - o)/o))
    
    # Short
    s_win = (l <= o * (1 - tp)) & ~(h >= o * (1 + sl))
    s_loss = h >= o * (1 + sl)
    s_ret = np.where(s_loss, -sl, np.where(s_win, tp, (o - c)/o))
    
    return l_ret + s_ret

def optimize_grid(df):
    best_sharpe = -999
    best_params = (0.01, 0.01)
    # Grid: 0.5% to 5.0%
    search_space = np.linspace(0.005, 0.05, 15)
    
    for sl in search_space:
        for tp in search_space:
            returns = get_pnl(df, sl, tp)
            if np.std(returns) == 0: continue
            # Annualized Sharpe
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(24*365)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (sl, tp)
    return best_params

def run_analysis():
    print("--- Starting Walk-Forward Analysis ---")
    df = fetch_1yr_data()
    df['month'] = df['timestamp'].dt.to_period('M')
    months = df['month'].unique()
    
    equity = []
    results_table = []
    trade_log = []
    
    # Need at least 2 months
    for i in range(len(months) - 1):
        train_m, test_m = months[i], months[i+1]
        print(f"Optimizing on {train_m} -> Testing on {test_m}")
        
        # 1. Train
        train_data = df[df['month'] == train_m]
        best_sl, best_tp = optimize_grid(train_data)
        
        # 2. Test
        test_data = df[df['month'] == test_m]
        returns = get_pnl(test_data, best_sl, best_tp)
        
        # 3. Log
        monthly_sum = np.sum(returns)
        results_table.append({
            'period': str(test_m),
            'sl': f"{best_sl*100:.1f}%", 
            'tp': f"{best_tp*100:.1f}%", 
            'net': monthly_sum
        })
        equity.extend(returns)
        
        # 4. Dense Logging (Vol Events > 0.5% movement)
        mask = np.abs(returns) > 0.005
        idxs = np.where(mask)[0]
        for idx in idxs:
            trade_log.append({
                'time': test_data.iloc[idx]['timestamp'],
                'price': test_data.iloc[idx]['close'],
                'pnl': returns[idx],
                'params': f"{best_sl*100:.1f}/{best_tp*100:.1f}"
            })
            
    return equity, results_table, trade_log

# ==========================================
# 2. HTML Generation
# ==========================================
def generate_report():
    equity, table_data, trades = run_analysis()
    
    # Chart
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(equity), color='#2980b9', linewidth=2)
    plt.title('Walk-Forward Equity Curve (1 Year)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # HTML Construction
    rows = ""
    for r in table_data:
        color = "green" if r['net'] > 0 else "red"
        rows += f"<tr><td>{r['period']}</td><td>SL {r['sl']} / TP {r['tp']}</td><td style='color:{color}'><b>{r['net']*100:.2f}%</b></td></tr>"
        
    log_rows = ""
    for t in trades[-20:]: # Last 20 major events
        c = "green" if t['pnl'] > 0 else "red"
        log_rows += f"<tr><td>{t['time']}</td><td>{t['price']:.2f}</td><td>{t['params']}</td><td style='color:{c}'>{t['pnl']*100:.2f}%</td></tr>"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Walk-Forward Report</title>
        <style>
            body {{ font-family: 'Segoe UI', monospace; background: #f0f2f5; padding: 20px; max-width: 900px; margin: auto; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th {{ text-align: left; background: #eee; padding: 10px; }}
            td {{ padding: 10px; border-bottom: 1px solid #eee; }}
            h2 {{ margin-top: 0; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Strategy Performance</h2>
            <img src="data:image/png;base64,{img}" style="width:100%">
        </div>
        
        <div class="card">
            <h2>Monthly Walk-Forward Results</h2>
            <table>
                <tr><th>Month</th><th>Optimized Params (Prev Month)</th><th>Net Return</th></tr>
                {rows}
            </table>
        </div>
        
        <div class="card">
            <h2>Recent High-Volatility Events</h2>
            <table>
                <tr><th>Time</th><th>Price</th><th>Active Params</th><th>PnL</th></tr>
                {log_rows}
            </table>
        </div>
    </body>
    </html>
    """
    return html.encode('utf-8')

# ==========================================
# 3. Server
# ==========================================
class FastHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(CACHED_HTML)

if __name__ == "__main__":
    # PRE-COMPUTE ON STARTUP
    try:
        CACHED_HTML = generate_report()
        print("--- Report Generated. Server Ready. ---")
    except Exception as e:
        print(f"Startup Failed: {e}")
        CACHED_HTML = f"<h1>Error: {e}</h1>".encode('utf-8')

    PORT = 8080
    print(f"Serving on http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), FastHandler) as httpd:
        httpd.serve_forever()
