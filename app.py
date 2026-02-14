import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from http.server import SimpleHTTPRequestHandler
import socketserver
import threading
import time
import os

# --- Configuration ---
PORT = 8080
TIMESPAM = "8years"  # Blockchain.com API format
PLOT_FILENAME = "market_cap_vs_hashrate.png"

def fetch_blockchain_data(chart_name):
    """Fetches JSON data from Blockchain.com public charts API."""
    url = f"https://api.blockchain.info/charts/{chart_name}?timespan={TIMESPAM}&format=json"
    print(f"Fetching {chart_name} data from {url}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['values'])
        df['x'] = pd.to_datetime(df['x'], unit='s')
        df.columns = ['Date', chart_name]
        return df.set_index('Date')
        
    except Exception as e:
        print(f"Error fetching {chart_name}: {e}")
        return None

def process_and_plot():
    # 1. Fetch Data
    df_hash = fetch_blockchain_data("hash-rate")
    df_cap = fetch_blockchain_data("market-cap")

    if df_hash is None or df_cap is None:
        print("Failed to retrieve data. Exiting.")
        return

    # 2. Merge Data (Align by Date)
    # Resample to daily to ensure alignment, then interpolate missing values
    df_hash = df_hash.resample('D').mean().interpolate()
    df_cap = df_cap.resample('D').mean().interpolate()
    
    # Inner join to keep only overlapping dates
    df = pd.merge(df_hash, df_cap, left_index=True, right_index=True)
    
    # Rename columns for clarity
    df.columns = ['Hashrate_THs', 'MarketCap_USD']
    
    # Convert Hashrate to Exahash (EH/s) for readability (1 EH/s = 1,000,000 TH/s)
    df['Hashrate_EH'] = df['Hashrate_THs'] / 1_000_000
    
    # Convert Market Cap to Billions ($B)
    df['MarketCap_B'] = df['MarketCap_USD'] / 1_000_000_000

    # Create 'Year' column for color coding
    df['Year'] = df.index.year

    # 3. Plotting
    print("Generating plot...")
    plt.figure(figsize=(12, 8))
    
    # Scatter plot: X=Hashrate (Security Supply), Y=Market Cap (Demand)
    scatter = plt.scatter(
        df['Hashrate_EH'], 
        df['MarketCap_B'], 
        c=df['Year'], 
        cmap='viridis', 
        alpha=0.7, 
        edgecolors='k',
        s=30
    )
    
    # Add trend line (Polynomial fit just to show direction)
    z = pd.np.polyfit(df['Hashrate_EH'], df['MarketCap_B'], 2)
    p = pd.np.poly1d(z)
    plt.plot(df['Hashrate_EH'], p(df['Hashrate_EH']), "r--", alpha=0.5, label="Equilibrium Trend")

    # Labels and Style
    plt.title(f'Bitcoin Security-Utility Equilibrium (8 Years)\nSupply (Hashrate) vs Demand (Market Cap)', fontsize=14)
    plt.xlabel('Security Supply (Hashrate in EH/s)', fontsize=12)
    plt.ylabel('Network Demand (Market Cap in $ Billions)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Year')
    
    # Annotate the start and end
    start_row = df.iloc[0]
    end_row = df.iloc[-1]
    plt.annotate('Start', (start_row['Hashrate_EH'], start_row['MarketCap_B']), xytext=(10, 10), textcoords='offset points')
    plt.annotate('Now', (end_row['Hashrate_EH'], end_row['MarketCap_B']), xytext=(-20, 20), textcoords='offset points', fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOT_FILENAME)
    print(f"Plot saved to {PLOT_FILENAME}")

def run_server():
    """Serves the current directory on PORT."""
    handler = SimpleHTTPRequestHandler
    try:
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            print(f"\nServing plot at http://localhost:{PORT}/{PLOT_FILENAME}")
            print("Press Ctrl+C to stop.")
            httpd.serve_forever()
    except OSError as e:
        print(f"Error starting server on port {PORT}: {e}")
        print("Try changing the PORT variable at the top of the script.")

if __name__ == "__main__":
    process_and_plot()
    # Check if plot exists before starting server
    if os.path.exists(PLOT_FILENAME):
        run_server()
