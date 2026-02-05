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
from datetime import datetime

# 1. Fetch Data (Binance via CCXT)
def fetch_binance_history(symbol, start_date_str):
    exchange = ccxt.binance({'enableRateLimit': True})
    timeframe = '1d'
    since = exchange.parse8601(f'{start_date_str}T00:00:00Z')
    all_ohlcv = []
    
    print(f"Fetching {symbol} from {start_date_str}...")
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 86400000  # Advance by 1 day in ms
            if since > exchange.milliseconds():
                break
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df['close']

# Symbols map
symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT']
start_date = '2018-01-01'

# Dict to hold series
closes = {}
for sym in symbols:
    closes[sym] = fetch_binance_history(sym, start_date)

# Combine and align (Outer join to keep index integrity, then ffill)
data = pd.DataFrame(closes).sort_index()
# Rename cols for easier access
data.columns = ['BTC', 'ETH', 'XRP', 'SOL']
data = data.ffill()

# 2. Strategy Logic
btc_price = data['BTC']
sma_365 = btc_price.rolling(window=365).mean()

# Signal Logic
# SMA > Price -> Signal = 1 (Long BTC, Short Others)
# SMA < Price -> Signal = -1 (Short BTC, Long Others)
raw_signal = np.where(sma_365 > btc_price, 1, -1)
signal_series = pd.Series(raw_signal, index=data.index).shift(1)

# Returns
daily_rets = data.pct_change()
btc_ret = daily_rets['BTC']

# "Others" return: Equal weight of available alts (ignores NaNs for pre-listing dates)
# This dynamically handles when XRP or SOL didn't exist yet
others_ret = daily_rets[['ETH', 'XRP', 'SOL']].mean(axis=1)

# Strategy PnL
# If Signal 1: +1*BTC, -1*Others
# If Signal -1: -1*BTC, +1*Others
# Result: Signal * (BTC - Others)
strat_ret = signal_series * (btc_ret - others_ret)
strat_cum = (1 + strat_ret.fillna(0)).cumprod()

# 3. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Price vs SMA
ax1.plot(btc_price.index, btc_price, label='BTC/USDT', color='black', alpha=0.6)
ax1.plot(sma_365.index, sma_365, label='SMA 365', color='orange', linewidth=2)
ax1.set_yscale('log')
ax1.set_title('Binance BTC/USDT vs SMA 365')
ax1.legend()
ax1.grid(True, which="both", alpha=0.3)

# PnL
ax2.plot(strat_cum.index, strat_cum, label='Strategy PnL', color='green')
ax2.set_title('Cumulative PnL (Base 1.0)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save to buffer
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close(fig)

# 4. Server
PORT = 8080
html = f"""
<!DOCTYPE html>
<html>
<head><title>Binance Strategy</title></head>
<body style="font-family: monospace; background: #f0f0f0; padding: 20px;">
    <h2>Source: Binance | Start: {start_date}</h2>
    <div>
        <b>Logic:</b><br>
        IF SMA_365 > BTC_PRICE: LONG BTC, SHORT BASKET(ETH, XRP, SOL)<br>
        ELSE: SHORT BTC, LONG BASKET
    </div>
    <br>
    <img src="data:image/png;base64,{img_base64}" style="border: 1px solid #ccc;">
</body>
</html>
"""

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

print(f"Serving on port {PORT}")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
