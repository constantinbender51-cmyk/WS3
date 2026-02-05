import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import http.server
import socketserver
import numpy as np

# 1. Fetch Data
tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD']
start_date = '2018-01-01'
print(f"Fetching data for {tickers} from {start_date}...")
data = yf.download(tickers, start=start_date, progress=False)['Close']

# Forward fill to handle any potential missing daily prints (though crypto is 24/7)
data = data.ffill()

# 2. Strategy Logic
# BTC SMA 365
btc_price = data['BTC-USD']
sma_365 = btc_price.rolling(window=365).mean()

# Signal: 
# SMA > Price -> Long BTC, Short Others (Signal = 1)
# SMA < Price -> Short BTC, Long Others (Signal = -1)
# Signal is lagged by 1 day to avoid lookahead bias (trade next day based on today's close)
signal = np.where(sma_365 > btc_price, 1, -1)
signal_series = pd.Series(signal, index=data.index).shift(1)

# Returns
daily_rets = data.pct_change()

# Components
btc_ret = daily_rets['BTC-USD']
others_ret = daily_rets[['ETH-USD', 'XRP-USD', 'SOL-USD']].mean(axis=1) # Equal weight basket of available alts

# Strategy Return
# Signal=1: Long BTC (+1 * btc), Short Others (-1 * others) -> BTC - Others
# Signal=-1: Short BTC (-1 * btc), Long Others (-(-1) * others) -> Others - BTC
# Formula: Signal * (BTC - Others)
strat_ret = signal_series * (btc_ret - others_ret)

# Cumulative PnL
strat_cum = (1 + strat_ret.fillna(0)).cumprod()

# 3. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top: Price vs SMA
ax1.plot(btc_price.index, btc_price, label='BTC Price', color='black', alpha=0.6)
ax1.plot(sma_365.index, sma_365, label='SMA 365', color='orange', linewidth=2)
ax1.set_yscale('log')
ax1.set_title('BTC Price vs 365 SMA (Log Scale)')
ax1.legend()
ax1.grid(True, which="both", ls="-", alpha=0.2)

# Bottom: PnL
ax2.plot(strat_cum.index, strat_cum, label='Long BTC / Short Alts (Mean Reversion)', color='green')
ax2.set_title('Cumulative Strategy PnL (Initial=1.0)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()

# Save to buffer
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close(fig)

# 4. Serve on 8080
PORT = 8080
html_content = f"""
<!DOCTYPE html>
<html>
<head><title>Strategy Output</title></head>
<body>
    <h1>BTC vs Alts Mean Reversion Strategy</h1>
    <p><b>Logic:</b> If SMA365 > BTC Price: Long BTC, Short Alts. Else: Short BTC, Long Alts.</p>
    <img src="data:image/png;base64,{img_base64}" alt="Strategy Plot">
</body>
</html>
"""

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))

print(f"Serving plot on port {PORT}...")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
