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

# Composite Triangle Wave Fit (1460d + 730d)
def triangle_func(t, period, phase):
    frequency = 2 * np.pi / period
    return (2 / np.pi) * np.arcsin(np.sin(frequency * (t - phase)))

def composite_wave(t, a1, p1, a2, p2, offset):
    w1 = a1 * triangle_func(t, 1460, p1)
    w2 = a2 * triangle_func(t, 730, p2)
    return offset + w1 + w2

# Fit
# Initial guess: split variance between amplitudes, zero phase
std_res = df['residuals'].std()
p0 = [std_res, 0, std_res/2, 0, 0] 

try:
    popt, _ = curve_fit(composite_wave, df['t'], df['residuals'], p0=p0)
    df['composite_fit'] = composite_wave(df['t'], *popt)
    df['w1460'] = popt[0] * triangle_func(df['t'], 1460, popt[1])
    df['w730'] = popt[2] * triangle_func(df['t'], 730, popt[3]) + popt[4] # Add offset to one component for viz
except Exception as e:
    print(f"Fit failed: {e}")
    df['composite_fit'] = np.zeros(len(df))
    df['w1460'] = np.zeros(len(df))
    df['w730'] = np.zeros(len(df))

# Plot Server
class PlotHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df['close'], label='BTC Close', alpha=0.5)
        plt.plot(df.index, df['linear_trend'], label='Linear Trend', color='red', linestyle='--')
        plt.plot(df.index, df['sma_shifted'], label='SMA(1460) Shift(-1460)', color='orange', alpha=0.7)
        plt.title('BTC/USDT: Close vs Linear Trend')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['residuals'], label='Residuals', alpha=0.4, color='gray')
        plt.plot(df.index, df['composite_fit'], label='Composite Fit (1460d + 730d)', color='blue', linewidth=2)
        plt.title('Residuals vs Composite Triangle Wave Fit')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(df.index, df['w1460'], label='1460d Component', color='green', linestyle='--')
        plt.plot(df.index, df['w730'], label='730d Component', color='purple', linestyle='--')
        plt.title('Wave Components')
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
