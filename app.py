import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from http.server import SimpleHTTPRequestHandler
import socketserver
import sys

# --- Configuration ---
PORT = 8080
TIMESPAN = "8years"  # Fetches full history

def fetch_data():
    """Fetches P (Market Cap) and Q (Hashrate)."""
    print("--- Fetching Raw Data ---")
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    def get_json(chart):
        url = f"https://api.blockchain.info/charts/{chart}?timespan={TIMESPAN}&format=json&sampled=true"
        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            df = pd.DataFrame(r.json()['values'])
            df['x'] = pd.to_datetime(df['x'], unit='s')
            return df.set_index('x').resample('D').mean().interpolate()
        except Exception as e:
            print(f"Error fetching {chart}: {e}")
            sys.exit(1)

    df = pd.merge(get_json("hash-rate"), get_json("market-cap"), left_index=True, right_index=True)
    df.columns = ['Hashrate_TH', 'MarketCap_USD']
    
    # Q = Hashrate (Units) in EH/s
    # P = Market Cap (Price) in Billions
    df['Q'] = df['Hashrate_TH'] / 1_000_000
    df['P'] = df['MarketCap_USD'] / 1_000_000_000
    return df

def construct_demand_curve(df):
    """
    Constructs DEMAND by binning the Price (Y-axis).
    For every Price level, we find the average Quantity demanded.
    """
    # 1. Create Price Bins (e.g., every $50 Billion)
    min_p, max_p = df['P'].min(), df['P'].max()
    p_bins = np.arange(min_p, max_p + 50, 50)
    
    df['P_Bin'] = pd.cut(df['P'], p_bins)
    
    # 2. Group by Price Bin -> Calculate Avg Quantity
    # We want Q as a function of P
    demand_data = df.groupby('P_Bin', observed=True)['Q'].mean().reset_index()
    
    # We also need the midpoint of the P_Bin for plotting
    demand_data['P_Mid'] = demand_data['P_Bin'].apply(lambda x: x.mid)
    
    return demand_data.dropna()

def construct_supply_curve(df):
    """
    Constructs SUPPLY by binning the Quantity (X-axis).
    For every Quantity level, we find the average Price supplied.
    """
    # 1. Create Quantity Bins (e.g., every 10 EH/s)
    min_q, max_q = df['Q'].min(), df['Q'].max()
    q_bins = np.arange(min_q, max_q + 10, 10)
    
    df['Q_Bin'] = pd.cut(df['Q'], q_bins)
    
    # 2. Group by Quantity Bin -> Calculate Avg Price
    # We want P as a function of Q
    supply_data = df.groupby('Q_Bin', observed=True)['P'].mean().reset_index()
    
    # Midpoint of Q_Bin for plotting
    supply_data['Q_Mid'] = supply_data['Q_Bin'].apply(lambda x: x.mid)
    
    return supply_data.dropna()

def smooth_and_plot(df, demand, supply):
    plt.figure(figsize=(10, 7))
    
    # 1. Plot Raw Data (Background)
    plt.scatter(df['Q'], df['P'], color='lightgray', s=10, alpha=0.5, label='Raw Data Points')

    # 2. Plot DEMAND Line (Binning Y to find X)
    # X = Avg Q, Y = P Midpoint
    # Smoothing
    try:
        y_d = demand['P_Mid'].values
        x_d = demand['Q'].values
        # Sort by Y for interpolation
        idx = np.argsort(y_d)
        y_d, x_d = y_d[idx], x_d[idx]
        
        spl_d = make_interp_spline(y_d, x_d, k=3)
        y_smooth = np.linspace(y_d.min(), y_d.max(), 300)
        x_smooth = spl_d(y_smooth)
        plt.plot(x_smooth, y_smooth, 'b-', linewidth=3, label='Demand Line (Avg Q at Price P)')
    except:
        plt.plot(demand['Q'], demand['P_Mid'], 'b-', linewidth=3, label='Demand Line')

    # 3. Plot SUPPLY Line (Binning X to find Y)
    # X = Q Midpoint, Y = Avg P
    # Smoothing
    try:
        x_s = supply['Q_Mid'].values
        y_s = supply['P'].values
        # Sort by X
        idx = np.argsort(x_s)
        x_s, y_s = x_s[idx], y_s[idx]
        
        spl_s = make_interp_spline(x_s, y_s, k=3)
        x_smooth = np.linspace(x_s.min(), x_s.max(), 300)
        y_smooth = spl_s(x_smooth)
        plt.plot(x_smooth, y_smooth, 'r-', linewidth=3, label='Supply Line (Avg P at Quantity Q)')
    except:
        plt.plot(supply['Q_Mid'], supply['P'], 'r-', linewidth=3, label='Supply Line')

    # Formatting
    plt.title('Bitcoin Supply & Demand (Constructed from Data)', fontsize=14)
    plt.xlabel('Quantity: Hashrate (EH/s)', fontsize=12)
    plt.ylabel('Price: Market Cap ($ Billions)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig("btc_sd_plot.png")
    print("Plot saved to btc_sd_plot.png")

if __name__ == "__main__":
    df = fetch_data()
    
    # Construct Lines
    d_curve = construct_demand_curve(df.copy())
    s_curve = construct_supply_curve(df.copy())
    
    # Plot
    smooth_and_plot(df, d_curve, s_curve)
    
    # Serve
    with open("index.html", "w") as f:
        f.write("<html><body><h1>Bitcoin Supply/Demand Construction</h1><img src='btc_sd_plot.png'></body></html>")
    
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), SimpleHTTPRequestHandler) as httpd:
        print(f"\nServing at http://localhost:{PORT}")
        httpd.serve_forever()
