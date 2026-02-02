import http.server
import socketserver
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

# Global Cache
CACHED_REPORT = b"<h3>Processing... Refresh in 30 seconds.</h3>"

# ==========================================
# 1. Data Engine
# ==========================================
def fetch_deep_data():
    print("--- Fetching Deep History (Starting 2021 for SMA Warmup) ---")
    base_url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    
    # Start: Jan 1, 2021 (Provides 1 year warmup for 2022 analysis)
    start_ts = int(datetime(2021, 1, 1).timestamp() * 1000)
    end_ts = int(datetime.now().timestamp() * 1000)
    
    all_data = []
    current = start_ts
    
    while True:
        params = {
            'symbol': 'ETHUSDT', 'interval': '1h', 'limit': limit,
            'startTime': current, 'endTime': end_ts
        }
        try:
            resp = requests.get(base_url, params=params).json()
            if not resp or isinstance(resp, dict): break
            all_data.extend(resp)
            current = resp[-1][0] + 1
            if len(resp) < limit or current >= end_ts: break
        except Exception as e:
            print(f"Fetch Error: {e}")
            break
    
    df = pd.DataFrame(all_data, columns=[
        'op_t', 'open', 'high', 'low', 'close', 'vol', 
        'cl_t', 'qv', 'nt', 'tb', 'tq', 'ig'
    ])
    
    cols = ['open', 'high', 'low', 'close']
    df[cols] = df[cols].astype(float)
    df['timestamp'] = pd.to_datetime(df['op_t'], unit='ms')
    
    print(f"Total Candles: {len(df)}")
    return df

# ==========================================
# 2. Indicator & Vectorized Logic
# ==========================================
def apply_indicators(df):
    # 365 Days * 24 Hours = 8760 periods
    df['sma_365d'] = df['close'].rolling(window=8760).mean()
    
    # Filter: Start Analysis from 2022 (Drop 2021 warmup)
    mask_2022 = df['timestamp'] >= datetime(2022, 1, 1)
    return df[mask_2022].copy()

def get_pnl(df, sl, tp):
    if df.empty: return np.array([])
    
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    
    # Long Logic
    l_tp_price = o * (1 + tp)
    l_sl_price = o * (1 - sl)
    
    # Check Long Hits
    l_hit_sl = l <= l_sl_price
    # Conservative: SL hit checks low. TP hit checks high AND assumes SL not hit first.
    l_hit_tp = (h >= l_tp_price) & (~l_hit_sl) 
    
    l_pnl = np.where(l_hit_sl, -sl, np.where(l_hit_tp, tp, (c - o)/o))
    
    # Short Logic
    s_tp_price = o * (1 - tp)
    s_sl_price = o * (1 + sl)
    
    # Check Short Hits
    s_hit_sl = h >= s_sl_price
    s_hit_tp = (l <= s_tp_price) & (~s_hit_sl)
    
    s_pnl = np.where(s_hit_sl, -sl, np.where(s_hit_tp, tp, (o - c)/o))
    
    return l_pnl + s_pnl

# ==========================================
# 3. Regime Optimization
# ==========================================
def optimize_regime(df, name):
    print(f"Optimizing {name} Regime ({len(df)} candles)...")
    best_sharpe = -999
    best_params = (0, 0)
    best_curve = []
    
    # Grid: 0.5% to 5.0%
    grid = np.linspace(0.005, 0.05, 20)
    
    for sl in grid:
        for tp in grid:
            returns = get_pnl(df, sl, tp)
            if len(returns) == 0 or np.std(returns) == 0: continue
            
            # Annualized Sharpe (Hourly)
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(24*365)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (sl, tp)
                best_curve = np.cumsum(returns)
                
    return best_params, best_sharpe, best_curve

# ==========================================
# 4. Execution & Report
# ==========================================
def run_full_analysis():
    # 1. Prepare Data
    raw_df = fetch_deep_data()
    df = apply_indicators(raw_df)
    
    # 2. Split Regimes
    # Above SMA (Bullish Bias)
    df_above = df[df['close'] > df['sma_365d']].copy()
    
    # Below SMA (Bearish Bias)
    df_below = df[df['close'] < df['sma_365d']].copy()
    
    # 3. Optimize Independently
    (ab_sl, ab_tp), ab_sharpe, ab_curve = optimize_regime(df_above, "Above SMA")
    (be_sl, be_tp), be_sharpe, be_curve = optimize_regime(df_below, "Below SMA")
    
    # 4. Generate Visualization
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Price vs SMA
    plt.subplot(3, 1, 1)
    plt.plot(df['timestamp'], df['close'], label='Price', color='black', alpha=0.6, linewidth=0.8)
    plt.plot(df['timestamp'], df['sma_365d'], label='365-Day SMA', color='orange', linewidth=1.5)
    plt.title('Regime Split: Price vs 365-Day SMA (2022-Present)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Above SMA Equity
    plt.subplot(3, 1, 2)
    if len(ab_curve) > 0:
        plt.plot(ab_curve, color='green', label=f'Above SMA (SL={ab_sl*100:.1f}% TP={ab_tp*100:.1f}%)')
    plt.title(f'Performance: Above SMA (Sharpe: {ab_sharpe:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Below SMA Equity
    plt.subplot(3, 1, 3)
    if len(be_curve) > 0:
        plt.plot(be_curve, color='red', label=f'Below SMA (SL={be_sl*100:.1f}% TP={be_tp*100:.1f}%)')
    plt.title(f'Performance: Below SMA (Sharpe: {be_sharpe:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # 5. Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: monospace; background: #222; color: #eee; padding: 20px; }}
            .card {{ background: #333; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th {{ background: #444; text-align: left; padding: 8px; }}
            td {{ border-bottom: 1px solid #555; padding: 8px; }}
            .highlight {{ color: #00ff00; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Trend Regime Optimization (2022 - Present)</h1>
        
        <div class="card">
            <h2>Optimal Parameters</h2>
            <table>
                <tr>
                    <th>Regime</th>
                    <th>Condition</th>
                    <th>Optimal SL</th>
                    <th>Optimal TP</th>
                    <th>Sharpe Ratio</th>
                </tr>
                <tr>
                    <td><b>Bull Market</b></td>
                    <td>Price > 365 SMA</td>
                    <td class="highlight">{ab_sl*100:.2f}%</td>
                    <td class="highlight">{ab_tp*100:.2f}%</td>
                    <td>{ab_sharpe:.4f}</td>
                </tr>
                <tr>
                    <td><b>Bear Market</b></td>
                    <td>Price < 365 SMA</td>
                    <td class="highlight">{be_sl*100:.2f}%</td>
                    <td class="highlight">{be_tp*100:.2f}%</td>
                    <td>{be_sharpe:.4f}</td>
                </tr>
            </table>
        </div>

        <div class="card">
            <h2>Visual Analysis</h2>
            <img src="data:image/png;base64,{img}" style="width: 100%; border: 1px solid #555;">
        </div>
    </body>
    </html>
    """
    return html.encode('utf-8')

# ==========================================
# 5. Server Handler
# ==========================================
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(CACHED_REPORT)

if __name__ == "__main__":
    try:
        CACHED_REPORT = run_full_analysis()
        print("--- Analysis Complete. Server Ready. ---")
    except Exception as e:
        print(f"Error: {e}")
        CACHED_REPORT = f"<h1>Error: {e}</h1>".encode('utf-8')
        
    PORT = 8080
    print(f"Serving on http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()
