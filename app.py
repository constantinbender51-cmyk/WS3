import requests
import pandas as pd
import numpy as np
import io
from http.server import BaseHTTPRequestHandler, HTTPServer
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.optimize import curve_fit
import time

def fetch_binance_data(symbol="ETHUSDT", interval="1d", start_year=2018):
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(pd.Timestamp(f"{start_year}-01-01").timestamp() * 1000)
    all_data = []
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "limit": 1000
        }
        
        try:
            resp = requests.get(base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            if not data:
                break
                
            all_data.extend(data)
            start_ts = data[-1][0] + 1 
            
            if len(data) < 1000:
                break
                
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Fetch error: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades", 
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    cols = ["open", "high", "low", "close"]
    df[cols] = df[cols].astype(float)
    return df

def process_data(df):
    # --- 1. Linear Trend (1460 Logic) ---
    df["sma_1460"] = df["close"].rolling(window=1460).mean()
    df["sma_1460_shifted"] = df["sma_1460"].shift(-1460)
    
    valid_mask = ~np.isnan(df["sma_1460_shifted"])
    x_valid = np.arange(len(df))[valid_mask]
    y_valid = df.loc[valid_mask, "sma_1460_shifted"].values
    
    if len(x_valid) > 0:
        slope, intercept = np.polyfit(x_valid, y_valid, 1)
    else:
        slope, intercept = 0, 0
    
    x_full = np.arange(len(df))
    trend_line = slope * x_full + intercept
    
    # --- 2. 365 SMA Logic ---
    df["sma_365"] = df["close"].rolling(window=365).mean()
    # "Shifted 365 days" usually implies forward projection or alignment
    # Assuming shift(-365) to match the previous logic of projecting backwards/fitting
    df["sma_365_shifted"] = df["sma_365"].shift(-365)

    # --- 3. Detrend ---
    df_detrended = df[["open", "high", "low", "close"]].subtract(trend_line, axis=0)
    df_detrended["open_time"] = df["open_time"]
    y_detrended = df_detrended["close"].values

    # --- 4. Square Wave Fits ---
    
    def sine_func(x, A, phi, C, omega):
        return A * np.sin(omega * x + phi) + C

    def fit_square_wave(period, label_prefix):
        omega = 2 * np.pi / period
        p0 = [np.std(y_detrended), 0, np.mean(y_detrended)]
        
        try:
            # A. Fit Sine to get parameters
            # Use a lambda wrapper to fix omega for curve_fit
            fit_func = lambda x, A, phi, C: sine_func(x, A, phi, C, omega)
            
            popt, _ = curve_fit(fit_func, x_full, y_detrended, p0=p0)
            A_sine, phi_opt, C_sine = popt
            
            # B. Construct Square Wave
            # Peak Aligned: Drop at Peak (Cosine Logic)
            sq_basis = np.sign(np.cos(omega * x_full + phi_opt))
            
            # Force Amplitude
            fitted_wave = np.abs(A_sine) * sq_basis + C_sine
            info_str = f"{label_prefix} Sq (A={abs(A_sine):.0f})"
            return fitted_wave, info_str, abs(A_sine)
            
        except Exception as e:
            print(f"Fitting error {label_prefix}: {e}")
            return np.zeros_like(x_full), "Fit Failed", 0

    # Fit 1460 Square
    wave_1460, str_1460, amp_1460 = fit_square_wave(1460, "1460")
    
    # Fit 365 Square
    wave_365, str_365, amp_365 = fit_square_wave(365, "365")

    return df, trend_line, df_detrended, (slope, intercept), (wave_1460, str_1460), (wave_365, str_365)

class PlotHandler(BaseHTTPRequestHandler):
    data_context = None 
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            
            df, trend, df_detrended, lin_params, (w1460, s1460), (w365, s365) = self.data_context
            
            fig = Figure(figsize=(12, 12), dpi=100)
            (ax1, ax2) = fig.subplots(2, 1, sharex=True)
            
            # Upper Plot
            ax1.set_title(f"ETH/USDT | Trend Fit (1460 SMA): y = {lin_params[0]:.2f}x + {lin_params[1]:.2f}")
            ax1.plot(df["open_time"], df["close"], label="Close Price", linewidth=1, color='blue', alpha=0.4)
            ax1.plot(df["open_time"], df["sma_1460_shifted"], label="1460 SMA (Shift -1460)", color='orange', linewidth=1.5, linestyle=":")
            ax1.plot(df["open_time"], df["sma_365_shifted"], label="365 SMA (Shift -365)", color='cyan', linewidth=1.5)
            ax1.plot(df["open_time"], trend, label="Linear Trend", color='red', linestyle="--")
            ax1.legend(loc="upper left", fontsize='small')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylabel("Price (USDT)")
            
            # Lower Plot
            ax2.set_title(f"Deduced OHLC | {s1460} & {s365}")
            ax2.plot(df_detrended["open_time"], df_detrended["close"], label="Detrended Close", linewidth=1, color='green', alpha=0.5)
            
            # Plot Square Waves
            ax2.plot(df_detrended["open_time"], w1460, label="Sq 1460 (Peak Aligned)", color='magenta', linewidth=2, alpha=0.8)
            ax2.plot(df_detrended["open_time"], w365, label="Sq 365 (Peak Aligned)", color='purple', linewidth=1.5, alpha=0.9)
            
            # Shading (using 1460 as primary cycle for background)
            y_min, y_max = ax2.get_ylim()
            center = w1460.mean()
            ax2.fill_between(df_detrended["open_time"], y_min, y_max, where=(w1460 >= center), color='green', alpha=0.05)
            ax2.fill_between(df_detrended["open_time"], y_min, y_max, where=(w1460 < center), color='red', alpha=0.05)
            
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.legend(loc="upper left", fontsize='small')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylabel("Deviation")
            
            fig.autofmt_xdate()
            
            output = io.BytesIO()
            FigureCanvasAgg(fig).print_png(output)
            self.wfile.write(output.getvalue())
        else:
            self.send_error(404)

if __name__ == "__main__":
    print("Fetching data...")
    df = fetch_binance_data()
    
    print(f"Data fetched: {len(df)} records. Processing...")
    processed_data = process_data(df)
    
    PlotHandler.data_context = processed_data
    
    port = 8080
    server_address = ('', port)
    httpd = HTTPServer(server_address, PlotHandler)
    
    print(f"Serving plot at http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()
