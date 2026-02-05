import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from http.server import BaseHTTPRequestHandler, HTTPServer
import io

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
            since = ohlcv[-1][0] + 86400000 
            if since > exchange.milliseconds():
                break
        except Exception:
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

# Analysis
df = fetch_data()
df['t'] = np.arange(len(df))

# SMA 1460 and Offset
df['sma'] = df['close'].rolling(window=1460).mean()
df['sma_shifted'] = df['sma'].shift(-1460)

# Linear Fit to Shifted SMA
fit_data = df.dropna(subset=['sma_shifted'])
if not fit_data.empty:
    coeffs = np.polyfit(fit_data['t'], fit_data['sma_shifted'], 1)
    df['linear_trend'] = coeffs[0] * df['t'] + coeffs[1]
else:
    coeffs = np.polyfit(df['t'], df['close'], 1) 
    df['linear_trend'] = coeffs[0] * df['t'] + coeffs[1]

# Residuals
df['residuals'] = df['close'] - df['linear_trend']

# Sine Wave Fit (Period 1460)
def sine_wave(t, amplitude, phase, offset):
    return offset + amplitude * np.sin(2 * np.pi * (t - phase) / 1460)

p0 = [df['residuals'].std(), 0, 0]
try:
    popt, _ = curve_fit(sine_wave, df['t'], df['residuals'], p0=p0)
    df['sine_fit'] = sine_wave(df['t'], *popt)
except:
    df['sine_fit'] = np.zeros(len(df))

# Plot Server
class PlotHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['close'], label='BTC Close', alpha=0.5)
        plt.plot(df.index, df['linear_trend'], label='Linear Trend', color='red', linestyle='--')
        plt.plot(df.index, df['sma_shifted'], label='SMA(1460) Shift(-1460)', color='orange', alpha=0.7)
        plt.title('BTC/USDT: Close vs Linear Trend on Shifted SMA')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['residuals'], label='Residuals (Close - Trend)', alpha=0.5)
        plt.plot(df.index, df['sine_fit'], label=f'Sine Fit ($\lambda=1460$)', color='green', linewidth=2)
        plt.title('Detrended Residuals vs 1460-Day Sine Wave Fit')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.end_headers()
        self.wfile.write(buf.getvalue())

def run(port=8080):
    server = HTTPServer(('', port), PlotHandler)
    print(f'Serving plot on port {port}...')
    server.serve_forever()

if __name__ == "__main__":
    run()
