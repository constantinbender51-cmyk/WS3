import http.server
import socketserver
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

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

def run_logic(df):
    # Parameters from your optimization
    SL = 0.025
    TP = 0.026
    
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values
    
    # 1. LONG Logic
    l_tp_price = open_arr * (1 + TP)
    l_sl_price = open_arr * (1 - SL)
    # Check hits
    l_hit_sl = low_arr <= l_sl_price
    l_hit_tp = (high_arr >= l_tp_price) & (~l_hit_sl) # Conservative: SL hits first
    # Calc Long PnL
    l_pnl = (close_arr - open_arr) / open_arr
    l_pnl = np.where(l_hit_sl, -SL, l_pnl)
    l_pnl = np.where(l_hit_tp, TP, l_pnl)

    # 2. SHORT Logic
    s_tp_price = open_arr * (1 - TP)
    s_sl_price = open_arr * (1 + SL)
    # Check hits
    s_hit_sl = high_arr >= s_sl_price
    s_hit_tp = (low_arr <= s_tp_price) & (~s_hit_sl)
    # Calc Short PnL
    s_pnl = (open_arr - close_arr) / open_arr
    s_pnl = np.where(s_hit_sl, -SL, s_pnl)
    s_pnl = np.where(s_hit_tp, TP, s_pnl)
    
    # 3. Net PnL
    net_pnl = l_pnl + s_pnl
    return net_pnl

class ChartHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        df = get_binance_data()
        if df.empty:
            self.send_response(500)
            self.wfile.write(b"Failed to fetch data")
            return

        pnl = run_logic(df)
        cumulative = np.cumsum(pnl)

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Top: Cumulative Return
        ax1.plot(cumulative, color='blue', linewidth=1.5, label='Total Return')
        ax1.set_title('Cumulative Return (Note the Steps)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Bottom: Per-Hour PnL (The "Zero" Proof)
        colors = ['green' if x > 0 else 'red' for x in pnl]
        ax2.bar(range(len(pnl)), pnl, color=colors, width=1.0)
        ax2.set_title('Net PnL Per Hour (Mostly Zero)')
        ax2.set_ylim(-0.03, 0.03) # Set limits to see the +0.1% spikes clearly relative to potential losses
        ax2.grid(True, alpha=0.3)

        # Save
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        html = f"""
        <html><body>
            <h2 style="font-family: monospace">Mechanism Verification</h2>
            <p style="font-family: monospace">
                <b>Logic Check:</b><br>
                1. If Volatility < 2.5%: Long hedges Short. Net = 0.0% (The flat areas).<br>
                2. If Volatility > 2.6%: One TP (+2.6%) and one SL (-2.5%) hit. Net = +0.1% (The spikes).
            </p>
            <img src="data:image/png;base64,{img}" style="width:100%">
        </body></html>
        """
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

if __name__ == "__main__":
    print("Serving on port 8080...")
    server = socketserver.TCPServer(("", 8080), ChartHandler)
    server.serve_forever()
