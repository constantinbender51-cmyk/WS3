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
    # 1. Calculate 1460 SMA and Shift for Linear Trend
    df["sma_1460"] = df["close"].rolling(window=1460).mean()
    df["sma_1460_shifted"] = df["sma_1460"].shift(-1460)
    
    # Fit line to shifted SMA
    valid_mask = ~np.isnan(df["sma_1460_shifted"])
    x_valid = np.arange(len(df))[valid_mask]
    y_valid = df.loc[valid_mask, "sma_1460_shifted"].values
    
    if len(x_valid) > 0:
        slope, intercept = np.polyfit(x_valid, y_valid, 1)
    else:
        slope, intercept = 0, 0
    
    x_full = np.arange(len(df))
    trend_line = slope * x_full + intercept
    
    # 2. Detrend
    df_detrended = df[["open", "high", "low", "close"]].subtract(trend_line, axis=0)
    df_detrended["open_time"] = df["open_time"]
    
    # 3. Custom Square Wave: "Start of down period to peak of 4 year cycle"
    # Interpreting "Down Period to Peak" as the boundaries of the cycle phases.
    # Standard 4 year cycle ~ 1460 days.
    # To align phases:
    # We maintain the lambda=1460.
    # Instead of a symmetric 50/50 square wave, this request implies a specific alignment 
    # but likely still a square logic (High/Low).
    
    # We will fit a standard square wave first to find the dominant High/Low structure,
    # effectively capturing the "Bull" (Peak) and "Bear" (Down) phases.
    
    fixed_period = 1460
    omega = 2 * np.pi / fixed_period
    y_detrended = df_detrended["close"].values
    
    def sine_func(x, A, phi, C):
        return A * np.sin(omega * x + phi) + C
    
    p0 = [np.std(y_detrended), 0, np.mean(y_detrended)]
    
    try:
        # A. Get Phase from Sine Fit
        popt, _ = curve_fit(sine_func, x_full, y_detrended, p0=p0)
        phi_opt = popt[1]
        
        # B. Create Square Basis (-1, 1)
        # Note: If we want to strictly define "Down to Peak", we are essentially 
        # modeling the binary state of the cycle.
        sq_basis = np.sign(np.sin(omega * x_full + phi_opt))
        
        # C. Linear Regression to find best Amplitude & Offset
        A_sq, C_sq = np.polyfit(sq_basis, y_detrended, 1)
        
        fitted_wave = A_sq * sq_basis + C_sq
        wave_str = f"Sq Fit (λ=1460): A={A_sq:.2f}, φ={phi_opt:.2f}"
        
    except Exception as e:
        print(f"Fitting error: {e}")
        fitted_wave = np.zeros_like(x_full)
        wave_str = "Fit Failed"

    return df, trend_line, df_detrended, (slope, intercept), fitted_wave, wave_str

class PlotHandler(BaseHTTPRequestHandler):
    data_context = None 
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            
            df, trend, df_detrended, lin_params, fitted_wave, wave_str = self.data_context
            
            fig = Figure(figsize=(12, 10), dpi=100)
            (ax1, ax2) = fig.subplots(2, 1, sharex=True)
            
            # Upper Plot
            ax1.set_title(f"ETH/USDT | Fit to 1460 SMA (Shift -1460): y = {lin_params[0]:.4f}x + {lin_params[1]:.2f}")
            ax1.plot(df["open_time"], df["close"], label="Close Price", linewidth=1, color='blue', alpha=0.5)
            ax1.plot(df["open_time"], df["sma_1460_shifted"], label="1460 SMA (Shift -1460)", color='orange', linewidth=1.5)
            ax1.plot(df["open_time"], trend, label="Linear Trend", color='red', linestyle="--")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylabel("Price (USDT)")
            
            # Lower Plot
            ax2.set_title(f"Deduced OHLC & {wave_str}")
            ax2.plot(df_detrended["open_time"], df_detrended["close"], label="Detrended Close", linewidth=1, color='green', alpha=0.6)
            ax2.plot(df_detrended["open_time"], fitted_wave, label="Square Wave (Cycle)", color='magenta', linewidth=2)
            
            # Shade regions based on square wave state
            # High state = Peak/Bull, Low state = Down/Bear
            y_min, y_max = ax2.get_ylim()
            ax2.fill_between(df_detrended["open_time"], y_min, y_max, where=(fitted_wave > fitted_wave.mean()), color='green', alpha=0.1, label="Up Phase")
            ax2.fill_between(df_detrended["open_time"], y_min, y_max, where=(fitted_wave <= fitted_wave.mean()), color='red', alpha=0.1, label="Down Phase")
            
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.legend(loc="upper left")
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
