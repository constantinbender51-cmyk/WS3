import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import http.server
import socketserver
from datetime import datetime
import io
import base64

# Configuration
URL = "https://workspace-production-9fae.up.railway.app/history"
PORT = 8080

def get_stats_and_plot():
    # 1. Fetch and Filter Data
    response = requests.get(URL)
    raw_data = response.json()
    df = pd.DataFrame(raw_data)
    
    # Sort chronologically for PnL curve
    df['dt'] = pd.to_datetime(df['time'])
    df = df.sort_values('dt')

    # Calculate 1% Win Rate based on RAW data
    one_percent_wins = len(df[df['pnl'] >= 0.01])
    one_percent_wr = (one_percent_wins / len(df)) * 100 if len(df) > 0 else 0

    # Apply filter: Ignore all PnLs where |pnl| < 1% (0.01)
    df_filtered = df[df['pnl'].abs() >= 0.01].copy()
    pnl = df_filtered['pnl']

    # 2. Calculate Statistical Metrics (Filtered)
    metrics = {
        "Total Raw Trades": len(df),
        "Significant Trades (>=1%)": len(pnl),
        "1% Win Rate (Overall)": f"{one_percent_wr:.2f}%",
        "Mean PnL (Filtered)": f"{pnl.mean():.4f}",
        "Std Dev (Filtered)": f"{pnl.std():.4f}",
        "Skewness": f"{pnl.skew():.4f}",
        "Kurtosis": f"{pnl.kurtosis():.4f}",
        "Max Drawdown": f"{(df['pnl'].cumsum().cummax() - df['pnl'].cumsum()).max():.4f}"
    }

    # 3. Generate Plot (Filtered Cumulative PnL)
    plt.figure(figsize=(10, 5))
    cumulative_pnl = pnl.cumsum()
    plt.plot(df_filtered['dt'], cumulative_pnl, marker='o', linestyle='-', markersize=3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    plt.title('Cumulative PnL (Trades >= 1% Absolute)')
    plt.grid(True, alpha=0.3)
    
    # Save plot to buffer to serve as base64 (no local file needed)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return metrics, img_data

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            metrics, img_str = get_stats_and_plot()
            
            # Build HTML table for metrics
            rows = "".join([f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in metrics.items()])
            
            html = f"""
            <html>
            <head><style>
                body {{ font-family: sans-serif; margin: 40px; background: #f4f4f9; }}
                table {{ border-collapse: collapse; width: 400px; margin-bottom: 20px; background: white; }}
                td, th {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .container {{ display: flex; flex-direction: column; align-items: center; }}
            </style></head>
            <body>
                <div class="container">
                    <h1>PnL Statistical Distribution</h1>
                    <table>{rows}</table>
                    <img src="data:image/png;base64,{img_str}">
                </div>
            </body>
            </html>
            """
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            self.send_error(404)

print(f"Server starting at http://localhost:{PORT}")
with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
    httpd.serve_forever()
