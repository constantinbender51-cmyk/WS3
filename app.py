import http.server
import socketserver
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

# ==========================================
# 1. robust Data Fetching (Pagination)
# ==========================================
def fetch_1yr_data():
    base_url = "https://api.binance.com/api/v3/klines"
    symbol = 'ETHUSDT'
    interval = '1h'
    limit = 1000
    
    # End time = Now. Start time = 365 days ago
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
    
    all_data = []
    current_start = start_time
    
    print("Fetching 1 year of data...")
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
            'startTime': current_start,
            'endTime': end_time
        }
        try:
            resp = requests.get(base_url, params=params).json()
            if not resp or isinstance(resp, dict): # Handle error or empty
                break
            
            all_data.extend(resp)
            
            # Update cursor
            last_timestamp = resp[-1][0]
            current_start = last_timestamp + 1
            
            if len(resp) < limit or current_start >= end_time:
                break
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'q_vol', 'num_trades', 'tbb_base', 'tbb_quote', 'ignore'
    ])
    
    cols = ['open', 'high', 'low', 'close']
    df[cols] = df[cols].astype(float)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    return df

# ==========================================
# 2. Vectorized Backtest Core
# ==========================================
def calculate_pnl(df, sl, tp):
    if df.empty: return 0.0, pd.DataFrame()
    
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values
    timestamps = df['timestamp'].values
    
    # --- Logic ---
    # Long
    l_tp_price = open_arr * (1 + tp)
    l_sl_price = open_arr * (1 - sl)
    l_hit_sl = low_arr <= l_sl_price
    l_hit_tp = (high_arr >= l_tp_price) & (~l_hit_sl) # Conservative: SL hits first
    
    l_pnl = (close_arr - open_arr) / open_arr # Default
    l_pnl = np.where(l_hit_sl, -sl, l_pnl)
    l_pnl = np.where(l_hit_tp, tp, l_pnl)
    
    # Short
    s_tp_price = open_arr * (1 - tp)
    s_sl_price = open_arr * (1 + sl)
    s_hit_sl = high_arr >= s_sl_price
    s_hit_tp = (low_arr <= s_tp_price) & (~s_hit_sl)
    
    s_pnl = (open_arr - close_arr) / open_arr # Default
    s_pnl = np.where(s_hit_sl, -sl, s_pnl)
    s_pnl = np.where(s_hit_tp, tp, s_pnl)
    
    total_pnl = l_pnl + s_pnl
    
    return total_pnl

# ==========================================
# 3. Optimization (Grid Search)
# ==========================================
def optimize_slice(df):
    best_sharpe = -9999
    best_params = (0.01, 0.01) # Default fallback
    
    # Search Space: 0.5% to 5.0%
    # Coarser grid for speed
    rng = np.linspace(0.005, 0.05, 20) 
    
    for sl in rng:
        for tp in rng:
            pnl = calculate_pnl(df, sl, tp)
            if np.std(pnl) == 0: continue
            
            # Annualized Sharpe (Hourly)
            sharpe = (np.mean(pnl) / np.std(pnl)) * np.sqrt(24*365)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (sl, tp)
                
    return best_params

# ==========================================
# 4. Walk-Forward Engine
# ==========================================
def run_walk_forward():
    df = fetch_1yr_data()
    
    # Split by Month
    df['month'] = df['timestamp'].dt.to_period('M')
    months = df['month'].unique()
    
    results = []
    equity_curve = []
    trade_log = []
    
    # Need at least 2 months (1 to train, 1 to test)
    for i in range(len(months) - 1):
        train_month = months[i]
        test_month = months[i+1]
        
        # 1. Optimize on Train
        train_df = df[df['month'] == train_month].copy()
        best_sl, best_tp = optimize_slice(train_df)
        
        # 2. Test on Next Month
        test_df = df[df['month'] == test_month].copy()
        pnl_arr = calculate_pnl(test_df, best_sl, best_tp)
        
        # 3. Record Data
        monthly_return = np.sum(pnl_arr)
        results.append({
            'Month': str(test_month),
            'Used SL': f"{best_sl*100:.2f}%",
            'Used TP': f"{best_tp*100:.2f}%",
            'PnL': f"{monthly_return*100:.2f}%",
            'Raw_PnL': monthly_return
        })
        
        # Build Equity Curve segment
        equity_curve.extend(pnl_arr)
        
        # Build Trade Log (Only non-zero trades for density)
        # Re-calc hits for logging
        open_arr = test_df['open'].values
        low_arr = test_df['low'].values
        high_arr = test_df['high'].values
        
        # Identify interesting candles (High Volatility)
        # Logic: If abs(pnl) > 0.001 (approx 0.1% change)
        mask = np.abs(pnl_arr) > 0.0001
        
        indices = np.where(mask)[0]
        for idx in indices:
            row_time = test_df.iloc[idx]['timestamp']
            row_pnl = pnl_arr[idx]
            trade_log.append({
                'Time': row_time,
                'Type': 'Vol Event',
                'Price': test_df.iloc[idx]['close'],
                'PnL': row_pnl,
                'SL': best_sl,
                'TP': best_tp
            })
            
    return df, equity_curve, results, trade_log

# ==========================================
# 5. Rendering
# ==========================================
class ReportHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            df, equity_stream, monthly_results, trades = run_walk_forward()
            
            # --- Plotting ---
            cum_equity = np.cumsum(equity_stream)
            
            plt.figure(figsize=(12, 6))
            plt.plot(cum_equity, label='Walk-Forward Equity', color='blue')
            plt.title('Walk-Forward Analysis (Train Month N, Trade Month N+1)')
            plt.xlabel('Hours (Cumulative)')
            plt.ylabel('Return')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            # --- HTML ---
            html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: monospace; background: #f4f4f4; padding: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; background: white; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #333; color: white; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .pos {{ color: green; }}
                    .neg {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>12-Month Walk-Forward Analysis</h1>
                
                <div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                    <h3>Equity Curve</h3>
                    <img src="data:image/png;base64,{img_b64}" style="width: 100%; max-width: 1000px;">
                </div>

                <h3>Monthly Performance (Optimized on Prev Month)</h3>
                <table>
                    <tr><th>Month</th><th>Params (SL / TP)</th><th>Net Return</th></tr>
                    {''.join([f"<tr><td>{r['Month']}</td><td>SL:{r['Used SL']} TP:{r['Used TP']}</td><td class='{'pos' if r['Raw_PnL'] > 0 else 'neg'}'>{r['PnL']}</td></tr>" for r in monthly_results])}
                </table>
                
                <h3>Significant Trade Events (Sample)</h3>
                <table>
                    <tr><th>Time</th><th>Price</th><th>Params</th><th>Net PnL</th></tr>
                    {''.join([f"<tr><td>{t['Time']}</td><td>{t['Price']:.2f}</td><td>SL:{t['SL']*100:.1f}% TP:{t['TP']*100:.1f}%</td><td class='{'pos' if t['PnL'] > 0 else 'neg'}'>{t['PnL']*100:.2f}%</td></tr>" for t in trades[-50:]])} 
                </table>
                <p><i>*Showing last 50 events for brevity.</i></p>
            </body>
            </html>
            """
            
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
            
        except Exception as e:
            self.send_response(500)
            self.wfile.write(str(e).encode('utf-8'))

if __name__ == "__main__":
    PORT = 8080
    print(f"Serving Walk-Forward Analysis on port {PORT}...")
    with socketserver.TCPServer(("", PORT), ReportHandler) as httpd:
        httpd.serve_forever()
