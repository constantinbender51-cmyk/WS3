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

df['residuals'] = df['close'] - df['linear_trend']

# Multi-Wave Fit (Triangle)
# Periods: 1460, 730, 365, 182.5 (6mo), 91.25 (3mo), 42 (6wk), 21 (3wk)
def triangle_wave_component(t, amp, phase, period):
    freq = 2 * np.pi / period
    return amp * (2 / np.pi) * np.arcsin(np.sin(freq * (t - phase)))

def composite_wave(t, off, 
                   a1, p1, 
                   a2, p2, 
                   a3, p3, 
                   a4, p4, 
                   a5, p5, 
                   a6, p6, 
                   a7, p7):
    w1 = triangle_wave_component(t, a1, p1, 1460)
    w2 = triangle_wave_component(t, a2, p2, 730)
    w3 = triangle_wave_component(t, a3, p3, 365)
    w4 = triangle_wave_component(t, a4, p4, 182.5) # 6 months
    w5 = triangle_wave_component(t, a5, p5, 91.25) # 3 months
    w6 = triangle_wave_component(t, a6, p6, 42)    # 6 weeks
    w7 = triangle_wave_component(t, a7, p7, 21)    # 3 weeks
    return off + w1 + w2 + w3 + w4 + w5 + w6 + w7

# Initial guesses: std of residuals for amps, 0 for phases
std = df['residuals'].std()
p0 = [0, 
      std, 0, 
      std/2, 0, 
      std/3, 0, 
      std/4, 0, 
      std/5, 0, 
      std/6, 0, 
      std/7, 0]

try:
    popt, _ = curve_fit(composite_wave, df['t'], df['residuals'], p0=p0, maxfev=10000)
    df['composite_fit'] = composite_wave(df['t'], *popt)
except Exception as e:
    print(f"Fit failed: {e}")
    df['composite_fit'] = np.zeros(len(df))

class PlotHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['close'], label='BTC Close', alpha=0.4, color='gray')
        plt.plot(df.index, df['linear_trend'] + df['composite_fit'], label='Trend + Composite Waves', color='blue', linewidth=1)
        plt.title('BTC/USDT: Price vs Composite Triangle Model')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['residuals'], label='Residuals', alpha=0.3, color='gray')
        plt.plot(df.index, df['composite_fit'], label='Composite Fit (1460, 730, 365, 6m, 3m, 6w, 3w)', color='green', linewidth=1.5)
        plt.title('Detrended Residuals vs Multi-Frequency Triangle Fit')
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
