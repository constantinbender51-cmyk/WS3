import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import sawtooth
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import time

# 1. Fetch Data
def fetch_data():
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1d'
    since = exchange.parse8601('2018-01-01T00:00:00Z')
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 86400000  # Advance by 1 day
            if since > exchange.milliseconds():
                break
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    # Remove duplicates if any
    df = df[~df.index.duplicated(keep='first')]
    return df

# 2. Analysis
df = fetch_data()
df['t'] = np.arange(len(df))

# SMA and Offset
# SMA 1460
df['sma'] = df['close'].rolling(window=1460).mean()
# Offset -1460 (Shift backwards/upwards)
df['sma_shifted'] = df['sma'].shift(-1460)

# Fit straight line to the shifted SMA
# Drop NaNs created by rolling and shifting to fit
fit_data = df.dropna(subset=['sma_shifted'])

if not fit_data.empty:
    coeffs = np.polyfit(fit_data['t'], fit_data['sma_shifted'], 1)
    slope, intercept = coeffs
    df['linear_trend'] = slope * df['t'] + intercept
else:
    # Fallback if insufficient data (e.g., < 2920 days history for valid shift)
    # Using Close for fallback to prevent crash, though logic dictates we need specific history length
    coeffs = np.polyfit(df['t'], df['close'], 1) 
    df['linear_trend'] = coeffs[0] * df['t'] + coeffs[1]

# 3. Residuals (OHLC Close - Line)
df['residuals'] = df['close'] - df['linear_trend']

# 4. Saw Wave Fit
# Lambda = 1460 days
def saw_wave(t, amplitude, phase, offset):
    # period = 1460
    return offset + amplitude * sawtooth(2 * np.pi * (t - phase) / 1460)

# Fit to residuals
p0 = [df['residuals'].std(), 0, 0]
try:
    popt, _ = curve_fit(saw_wave, df['t'], df['residuals'], p0=p0)
    df['saw_fit'] = saw_wave(df['t'], *popt)
except:
    df['saw_fit'] = np.zeros(len(df))

# 5. Serve Plot
class PlotHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Price and Trends
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['close'], label='BTC Close', alpha=0.5)
        plt.plot(df.index, df['linear_trend'], label='Linear Trend', color='red', linestyle='--')
        plt.plot(df.index, df['sma_shifted'], label='SMA(1460) Shift(-1460)', color='orange', alpha=0.7)
        plt.title('BTC/USDT: Close vs Linear Trend on Shifted SMA')
        plt.legend()
        plt.grid(True)

        # Subplot 2: Residuals and Saw Wave
        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['residuals'], label='Residuals (Close - Trend)', alpha=0.5)
        plt.plot(df.index, df['saw_fit'], label=f'Saw Fit ($\lambda=1460$)', color='green', linewidth=2)
        plt.title('Detrended Residuals vs 1460-Day Sawtooth Fit')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.end_headers()
        self.wfile.write(buf.getvalue())

def run(server_class=HTTPServer, handler_class=PlotHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Serving plot on port {port}...')
    httpd.serve_forever()

if __name__ == "__main__":
    run()
