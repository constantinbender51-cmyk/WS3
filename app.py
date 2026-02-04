import requests
import pandas as pd
import numpy as np
import io
from http.server import BaseHTTPRequestHandler, HTTPServer
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
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
    x = np.arange(len(df))
    y = df["close"].values
    
    slope, intercept = np.polyfit(x, y, 1)
    trend_line = slope * x + intercept
    
    # 730 SMA and Shift -730
    df["sma_730"] = df["close"].rolling(window=730).mean()
    df["sma_730_shifted"] = df["sma_730"].shift(-730)

    df_detrended = df[["open", "high", "low", "close"]].subtract(trend_line, axis=0)
    df_detrended["open_time"] = df["open_time"]
    
    return df, trend_line, df_detrended, (slope, intercept)

class PlotHandler(BaseHTTPRequestHandler):
    data_context = None 
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            
            df, trend, df_detrended, params = self.data_context
            
            fig = Figure(figsize=(12, 10), dpi=100)
            (ax1, ax2) = fig.subplots(2, 1, sharex=True)
            
            ax1.set_title(f"ETH/USDT (2018-Present) | Fit: y = {params[0]:.4f}x + {params[1]:.2f}")
            ax1.plot(df["open_time"], df["close"], label="Close Price", linewidth=1, color='blue')
            ax1.plot(df["open_time"], trend, label="Linear Trend", color='red', linestyle="--")
            
            # Plot SMA
            ax1.plot(df["open_time"], df["sma_730_shifted"], label="730 SMA (Shift -730)", color='orange', linewidth=1.5)

            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylabel("Price (USDT)")
            
            ax2.set_title("Deduced OHLC (Detrended Residuals)")
            ax2.plot(df_detrended["open_time"], df_detrended["close"], label="Detrended Close", linewidth=1, color='green')
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylabel("Deviation from Trend")
            
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
