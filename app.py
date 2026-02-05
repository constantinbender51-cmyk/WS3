import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import http.server
import socketserver
import numpy as np

# 1. Fetch Data
def fetch_binance_history(symbol, start_date_str):
    exchange = ccxt.binance({'enableRateLimit': True})
    timeframe = '1d'
    since = exchange.parse8601(f'{start_date_str}T00:00:00Z')
    all_ohlcv = []
    
    print(f"Fetching {symbol}...")
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 86400000 
            if since > exchange.milliseconds():
                break
        except Exception as e:
            print(f"Error: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df['close']

symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT']
start_date = '2018-01-01'

data_frames = {}
for sym in symbols:
    data_frames[sym] = fetch_binance_history(sym, start_date)

# Align Data
data = pd.DataFrame(data_frames).sort_index()
data.columns = ['BTC', 'ETH', 'XRP', 'SOL']
data = data.ffill()

# 2. Strategy Logic
btc = data['BTC']
sma_365 = btc.rolling(window=365).mean()

# Signal Generation
# Signal 1: Long BTC, Short Others
# Signal -1: Short BTC, Long Others
raw_signal = np.where(sma_365 > btc, 1, -1)
signal_lagged = pd.Series(raw_signal, index=data.index).shift(1)

# Returns
daily_rets = data.pct_change()
btc_ret = daily_rets['BTC']

# Basket Return Construction
# Size of BTC (1.0) = Size of Others Combined (1.0)
# 'mean(axis=1)' calculates the return of an equal-weighted basket summing to 1.0
# Handles NaN dynamically (e.g., before SOL listing, basket is 50% ETH / 50% XRP)
others_ret = daily_rets[['ETH', 'XRP', 'SOL']].mean(axis=1)

# Strategy Calculation
# If Signal=1:  (1.0 * BTC_Ret) - (1.0 * Others_Ret)
# If Signal=-1: (-1.0 * BTC_Ret) + (1.0 * Others_Ret)
strat_ret = signal_lagged * (btc_ret - others_ret)
strat_cum = (1 + strat_ret.fillna(0)).cumprod()

# 3. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Price vs SMA
ax1.plot(btc.index, btc, label='BTC Price', color='black', alpha=0.6)
ax1.plot(sma_365.index, sma_365, label='SMA 365', color='orange', linewidth=2)
ax1.set_yscale('log')
ax1.set_title('Binance BTC/USDT vs SMA 365')
ax1.legend()
ax1.grid(True, which="both", alpha=0.3)

# PnL
ax2.plot(strat_cum.index, strat_cum, label='Dollar Neutral (1 BTC : 1 Basket)', color='green')
ax2.set_title('Cumulative PnL')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
img_b64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close(fig)

# 4. Serve
PORT = 8080
html = f"""
<!DOCTYPE html>
<html>
<head><title>Strategy Output</title></head>
<body style="font-family: monospace; background: #eee; padding: 20px;">
    <h2>BTC vs Alts Mean Reversion</h2>
    <div>
        <b>Weighting:</b> 100% BTC vs 100% Alts Basket (Equal Weight)<br>
        <b>Signal:</b> SMA365 > BTC ? Long BTC/Short Alts : Short BTC/Long Alts
    </div>
    <br>
    <img src="data:image/png;base64,{img_b64}" style="border: 2px solid #555;">
</body>
</html>
"""

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

print(f"Serving on port {PORT}...")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
