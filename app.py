import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from http.server import SimpleHTTPRequestHandler
import socketserver
import os
import sys

# --- Configuration ---
PORT = 8080
TIMESPAN = "8years"

def fetch_and_plot_hard_data():
    """Fetches real data and plots the historical scatter."""
    print("--- Generating Plot 1: Hard Data (8 Years) ---")
    
    def get_data(chart):
        url = f"https://api.blockchain.info/charts/{chart}?timespan={TIMESPAN}&format=json&sampled=true"
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            df = pd.DataFrame(r.json()['values'])
            df['x'] = pd.to_datetime(df['x'], unit='s')
            df.columns = ['Date', chart]
            return df.set_index('Date').resample('D').mean().interpolate()
        except Exception as e:
            print(f"Error fetching {chart}: {e}")
            return None

    df_hash = get_data("hash-rate")
    df_cap = get_data("market-cap")

    if df_hash is not None and df_cap is not None:
        # Merge and Clean
        df = pd.merge(df_hash, df_cap, left_index=True, right_index=True)
        df.columns = ['Hashrate_TH', 'MarketCap_USD']
        
        # Convert units
        df['Q'] = df['Hashrate_TH'] / 1_000_000  # EH/s
        df['P'] = df['MarketCap_USD'] / 1_000_000_000  # Billions
        df['Year'] = df.index.year

        # PLOT 1: The Scatter
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df['Q'], df['P'], c=df['Year'], cmap='plasma', alpha=0.6, edgecolors='k', s=40)
        
        # Add Trendline
        z = np.polyfit(df['Q'], df['P'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(df['Q'].min(), df['Q'].max(), 100)
        plt.plot(x_trend, p(x_trend), "k--", alpha=0.5, label="Historical Trend")

        plt.title('Plot 1: Real-World Equilibrium Path (2018-2026)\nHashrate (Supply) vs Market Cap (Demand)')
        plt.xlabel('Quantity: Security Supply (EH/s)')
        plt.ylabel('Price: Network Value ($ Billions)')
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Year')
        plt.legend()
        plt.tight_layout()
        plt.savefig("current_data.png")
        print("Saved current_data.png")
    else:
        print("Skipping Plot 1 due to API error.")

def plot_theoretical_curves():
    """Plots the constructed Supply & Demand curves based on our economic model."""
    print("--- Generating Plot 2: Theoretical Supply & Demand ---")
    
    # 1. Create the X-Axis (Hashrate from 0 to 1200 EH/s)
    Q = np.linspace(0, 1200, 500)
    
    # 2. Construct Supply Curve (S) - Marginal Cost
    # Formula: Quadratic cost scaling (Difficulty + Energy Hardware constraints)
    # At 600 EH/s, Cost is roughly $1.2T. At 1000 EH/s, cost skyrockets.
    S = 0.003 * Q**2 + 0.5 * Q + 100 
    
    # 3. Construct Demand Curve (D) - Marginal Utility
    # Formula: Sigmoid (S-Curve).
    # Utility rises fast then flattens (Diminishing Returns) around 700 EH/s.
    L = 2500  # Max Utility ($2.5T)
    k = 0.012 # Steepness
    x0 = 350  # Midpoint (EH/s)
    D = L / (1 + np.exp(-k * (Q - x0)))

    # PLOT 2
    plt.figure(figsize=(10, 6))
    
    # Plot Curves
    plt.plot(Q, S, 'r-', linewidth=3, label='Supply (Marginal Cost of Mining)')
    plt.plot(Q, D, 'b-', linewidth=3, label='Demand (Utility of Persistence)')
    
    # Fill "Profit" and "Loss" zones
    plt.fill_between(Q, S, D, where=(D > S), color='green', alpha=0.1, label='Profit Zone (Miner Entry)')
    plt.fill_between(Q, S, D, where=(S > D), color='red', alpha=0.1, label='Loss Zone (Miner Capitulation)')

    # Find Intersection (Equilibrium)
    idx = np.argwhere(np.diff(np.sign(S - D))).flatten()
    if len(idx) > 0:
        eq_Q = Q[idx[0]]
        eq_P = S[idx[0]]
        plt.plot(eq_Q, eq_P, 'ko', markersize=10, label=f'Equilibrium ({int(eq_Q)} EH/s)')
        plt.annotate(f'Stable State\n{int(eq_Q)} EH/s', (eq_Q, eq_P), xytext=(eq_Q+50, eq_P-400),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    # Mark the 2026 "Crash" Point (The Hook)
    # We are currently at High Supply (1000 EH) but Demand dropped.
    current_Q = 1000
    current_P_demand = L / (1 + np.exp(-k * (current_Q - x0))) # On the blue line
    # Actually market is below blue line in panic, but let's show the gap
    
    plt.axvline(x=1000, color='gray', linestyle='--', alpha=0.5)
    plt.text(1000, 100, "Zettahash Barrier\n(Feb 2026)", rotation=90, verticalalignment='bottom')

    plt.title('Plot 2: The Security-Utility Equilibrium Model\nWhy "More Security" Stops Increasing Price')
    plt.xlabel('Quantity: Security Supply (EH/s)')
    plt.ylabel('Price: Network Value ($ Billions)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 3000)
    plt.xlim(0, 1200)
    plt.tight_layout()
    plt.savefig("supply_demand_model.png")
    print("Saved supply_demand_model.png")

def create_index_html():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bitcoin Economic Analysis</title>
        <style>
            body { font-family: sans-serif; text-align: center; background: #f4f4f9; padding: 20px; }
            .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
            .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; }
            h1 { color: #333; }
            h2 { color: #555; }
            p { max-width: 600px; margin: 10px auto; color: #666; }
        </style>
    </head>
    <body>
        <h1>Bitcoin Network Operation: Supply & Demand</h1>
        <div class="container">
            <div class="card">
                <h2>1. The Hard Data (8 Years)</h2>
                <img src="current_data.png" alt="Historical Data">
                <p>Real-time data showing the migration of the network state. Note the vertical spikes (Demand Shocks) and horizontal drifts (Supply Catch-up).</p>
            </div>
            <div class="card">
                <h2>2. The Economic Model</h2>
                <img src="supply_demand_model.png" alt="Theoretical Model">
                <p>The <b>Supply Curve (Red)</b> represents the cost to produce security. The <b>Demand Curve (Blue)</b> represents the utility of that security. <br>The 'Profit Zone' is where miners rush in. The 'Loss Zone' is where we are now (Oversupply).</p>
            </div>
        </div>
    </body>
    </html>
    """
    with open("index.html", "w") as f:
        f.write(html)
    print("Generated index.html")

def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    handler = SimpleHTTPRequestHandler
    try:
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            print(f"\n✅ DASHBOARD LIVE: http://localhost:{PORT}")
            print("Press Ctrl+C to stop.")
            httpd.serve_forever()
    except OSError as e:
        print(f"Error on port {PORT}: {e}")

if __name__ == "__main__":
    fetch_and_plot_hard_data()
    plot_theoretical_curves()
    create_index_html()
    run_server()
