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
# 1. Fetch Data
# ==========================================
def get_binance_data():
    base_url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': 'ETHUSDT', 'interval': '1h', 'limit': 1000}
    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()
    except Exception as e:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'q_vol', 'num_trades', 'tbb_base', 'tbb_quote', 'ignore'
    ])
    
    cols = ['open', 'high', 'low', 'close']
    df[cols] = df[cols].astype(float)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    
    # Filter last 30 days
    cutoff = datetime.now() - timedelta(days=30)
    df = df[df['timestamp'] > cutoff].reset_index(drop=True)
    return df

# ==========================================
# 2. Logic (Strict Re-implementation)
# ==========================================
def get_triggers(df):
    SL = 0.025 # 2.5%
    TP = 0.026 # 2.6%
    
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    
    # --- LONG LOGIC ---
    l_tp_price = open_arr * (1 + TP)
    l_sl_price = open_arr * (1 - SL)
    
    # SL Hit?
    l_hit_sl = low_arr <= l_sl_price
    # TP Hit? (Only if High reached TP AND SL was not hit first)
    # Conservative assumption: If both happen in one candle, SL happens first.
    # However, for the "Green Spike" to exist, the optimizer likely found candles 
    # where High > TP but Low > SL (Clean Win) OR it exploited a gap.
    # Let's map strict logic:
    l_hit_tp = (high_arr >= l_tp_price) & (~l_hit_sl)
    
    # --- SHORT LOGIC ---
    s_tp_price = open_arr * (1 - TP)
    s_sl_price = open_arr * (1 + SL)
    
    # SL Hit?
    s_hit_sl = high_arr >= s_sl_price
    # TP Hit? (Only if Low reached TP AND SL was not hit first)
    s_hit_tp = (low_arr <= s_tp_price) & (~s_hit_sl)
    
    return l_hit_tp, l_hit_sl, s_hit_tp, s_hit_sl

# ==========================================
# 3. Server & Plotting
# ==========================================
class TriggerPlotHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        df = get_binance_data()
        if df.empty:
            self.send_response(500)
            self.wfile.write(b"No Data")
            return
            
        l_tp, l_sl, s_tp, s_sl = get_triggers(df)
        
        # Setup Plot
        plt.figure(figsize=(14, 8))
        
        # 1. Price Line
        plt.plot(df['timestamp'], df['close'], color='black', alpha=0.4, linewidth=1, label='Close Price')
        
        # 2. Markers
        # Use boolean indexing to plot only the points where triggers occurred
        
        # Long TP (Green Up Triangle)
        if l_tp.any():
            plt.scatter(df.loc[l_tp, 'timestamp'], df.loc[l_tp, 'high'], 
                        color='green', marker='^', s=80, label='Long TP (+2.6%)', zorder=3)
            
        # Long SL (Red Down Triangle)
        if l_sl.any():
            plt.scatter(df.loc[l_sl, 'timestamp'], df.loc[l_sl, 'low'], 
                        color='red', marker='v', s=80, label='Long SL (-2.5%)', zorder=3)
            
        # Short TP (Cyan Down Triangle)
        if s_tp.any():
            plt.scatter(df.loc[s_tp, 'timestamp'], df.loc[s_tp, 'low'], 
                        color='cyan', marker='v', s=80, label='Short TP (+2.6%)', zorder=3)
            
        # Short SL (Orange Up Triangle)
        if s_sl.any():
            plt.scatter(df.loc[s_sl, 'timestamp'], df.loc[s_sl, 'high'], 
                        color='orange', marker='^', s=80, label='Short SL (-2.5%)', zorder=3)

        plt.title('ETH/USDT Trigger Verification (SL=2.5%, TP=2.6%)')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        html = f"""
        <html>
        <body style="background: #222; color: #eee; text-align: center;">
            <h2>Trigger Visualization</h2>
            <img src="data:image/png;base64,{img_b64}" style="max-width: 95%; border: 1px solid #555;">
        </body>
        </html>
        """
        
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

if __name__ == "__main__":
    PORT = 8080
    print(f"Serving on port {PORT}...")
    server = socketserver.TCPServer(("", PORT), TriggerPlotHandler)
    server.serve_forever()
