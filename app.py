import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import http.server
import socketserver
from datetime import datetime

# 1. Fetch Data
URL = "https://workspace-production-9fae.up.railway.app/history"
response = requests.get(URL)
data = response.json()

# 2. Process Data (FIXED)
# Convert string time to datetime objects for proper sorting
for item in data:
    item['dt'] = datetime.strptime(item['time'], "%Y-%m-%dT%H:%M:%S")

# SORT the data by time (Oldest -> Newest) so the line doesn't scribble
data.sort(key=lambda x: x['dt'])

dates = [item['dt'] for item in data]
pnls = [item['pnl'] for item in data]
cumulative_pnl = [sum(pnls[:i+1]) for i in range(len(pnls))]

# 3. Generate Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(dates, cumulative_pnl, marker='o', linestyle='-', color='b', markersize=4)

ax.set_title('Cumulative PnL Over Time')
ax.set_xlabel('Time')
ax.set_ylabel('PnL')
ax.grid(True)

# Format the X-axis to handle dates beautifully (prevents text overlap)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
fig.autofmt_xdate() # Rotates and aligns ticks

# Save plot
plot_filename = "pnl_plot.png"
plt.savefig(plot_filename)
plt.close()

# 4. Serve on Port 8080
PORT = 8080
class PlotHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            # Added a meta refresh to auto-reload the page if you re-run the script
            self.wfile.write(f'''
                <html>
                    <head><title>PnL Plot</title></head>
                    <body>
                        <h1>PnL History Plot</h1>
                        <img src="{plot_filename}" style="max-width:100%">
                    </body>
                </html>
            '''.encode())
        else:
            super().do_GET()

print(f"Serving PnL plot at http://localhost:{PORT}")
with socketserver.TCPServer(("", PORT), PlotHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
