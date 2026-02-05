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

# SMA 730 and Offset -730
df['sma'] = df['close'].rolling(window=730).mean()
df['sma_shifted'] = df['sma'].shift(-730)

# Linear Fit to Shifted SMA
fit_data = df.dropna(subset=['sma_shifted'])
if not fit_data.empty:
    coeffs = np.polyfit(fit_data['t'], fit_data['sma_shifted'], 1)
    df['linear_trend'] = coeffs[0] * df['t'] + coeffs[1]
else:
    # Fallback if insufficient data
    coeffs = np.polyfit(df['t'], df['close'], 1) 
    df['linear_trend'] = coeffs[0] * df['t'] + coeffs[1]

# Residuals
df['residuals'] = df['close'] - df['linear_trend']

# Composite Wave Fit
# Periods: 730, 365, 182.5 (6m), 91.25 (3m), 42 (6w), 21 (3w)
periods = [730, 365, 182.5, 91.25, 42, 21]

def composite_wave(t, offset, *params):
    # params structure: [amp1, phase1, amp2, phase2, ...]
    y = np.full_like(t, offset, dtype=float)
    for i, period in enumerate(periods):
        amp = params[i*2]
        phase = params[i*2 + 1]
        # Using Sine for harmonic summation
        y += amp * np.sin(2 * np.pi * (t - phase) / period)
    return y

# Initial guess: Offset=0, Amps=std/6, Phases=0
p0 = [0] + [df['residuals'].std() / len(periods), 0] * len(periods)

try:
    # Bounds to help convergence (Amplitude positive, Phase unconstrained)
    # This is a complex fit, might need higher max_nfev
    popt, _ = curve_fit(composite_wave, df['t'], df['residuals'], p0=p0, maxfev=10000)
    df['composite_fit'] = composite_wave(df['t'], *popt)
except Exception as e:
    print(f"Fit failed: {e}")
    df['composite_fit'] = np.zeros(len(df))

# Plot Server
class PlotHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['close'], label='BTC Close', alpha=0.5)
        plt.plot(df.index, df['linear_trend'], label='Linear Trend', color='red', linestyle='--')
        plt.plot(df.index, df['sma_shifted'], label='SMA(730) Shift(-730)', color='orange', alpha=0.7)
        plt.title('BTC/USDT: Close vs Linear Trend on SMA(730) Shift(-730)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['residuals'], label='Residuals', alpha=0.4)
        plt.plot(df.index, df['composite_fit'], label='Composite Wave Fit (730, 365, 6m, 3m, 6w, 3w)', color='green', linewidth=1.5)
        plt.title('Residuals vs Multi-Period Harmonic Fit')
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
