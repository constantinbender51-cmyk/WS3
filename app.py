import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import http.server
import socketserver

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

symbols = ['BTC/USDT', 'XRP/USDT']
start_date = '2018-01-01'

data_frames = {}
for sym in symbols:
    data_frames[sym] = fetch_binance_history(sym, start_date)

# Align Data
data = pd.DataFrame(data_frames).sort_index()
data.columns = ['BTC', 'XRP']
data = data.dropna() # Drop rows where XRP didn't exist yet if applicable (2018 ok)

# 2. Strategy Logic
# Static: Long BTC, Short XRP
daily_rets = data.pct_change()

# Strategy Return = BTC Return - XRP Return
strat_ret = daily_rets['BTC'] - daily_rets['XRP']
strat_cum = (1 + strat_ret.fillna(0)).cumprod()

# 3. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top: Normalized Prices (to compare relative performance)
norm_btc = data['BTC'] / data['BTC'].iloc[0]
norm_xrp = data['XRP'] / data['XRP'].iloc[0]

ax1.plot(norm_btc.index, norm_btc, label='BTC Normalized', color='orange')
ax1.plot(norm_xrp.index, norm_xrp, label='XRP Normalized', color='blue')
ax1.set_yscale('log')
ax1.set_title('Normalized Performance (Base=1.0)')
ax1.legend()
ax1.grid(True, which="both", alpha=0.3)

# Bottom: PnL
ax2.plot(strat_cum.index, strat_cum, label='Long BTC / Short XRP', color='green')
ax2.set_title('Cumulative Strategy PnL')
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
<head><title>Long BTC Short XRP</title></head>
<body style="font-family: monospace; background: #eee; padding: 20px;">
    <h2>Static Pair Trade: Long BTC / Short XRP</h2>
    <div>
        <b>Logic:</b> Daily Rebalance 100% Long BTC vs 100% Short XRP.<br>
        <b>Formula:</b> PnL = BTC_Change - XRP_Change
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
