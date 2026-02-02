import http.server
import socketserver
import requests
import pandas as pd
import numpy as np
import io

def get_data_and_audit():
    # 1. Fetch
    url = "https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=1h&limit=1000"
    try:
        df = pd.DataFrame(requests.get(url).json(), columns=[
            'time', 'open', 'high', 'low', 'close', 'vol', 'ct', 'qv', 'nt', 'tb', 'tq', 'ig'
        ])
    except:
        return "<h3>API Error</h3>"

    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float})
    df['time'] = pd.to_datetime(df['time'], unit='ms')

    # 2. Logic (SL=2.5%, TP=2.6%)
    SL, TP = 0.025, 0.026
    
    # Short TP Trigger: Low <= Open * (1 - TP)
    short_tp_price = df['open'] * (1 - TP)
    short_wins = df[df['low'] <= short_tp_price].copy()
    
    if short_wins.empty:
        return "<h3>No Short TPs found in last 1000 candles.</h3>"

    # 3. Check Long SL on those exact candles
    # Long SL Trigger: Low <= Open * (1 - SL)
    long_sl_price = short_wins['open'] * (1 - SL)
    short_wins['long_sl_hit'] = short_wins['low'] <= long_sl_price
    
    # Formatter
    html = "<h3>Audit: Short TP Events vs Long SL</h3>"
    html += "<table border='1' style='border-collapse:collapse; text-align:center'>"
    html += "<tr><th>Time</th><th>Open</th><th>Low</th><th>Short TP (<2.6%)</th><th>Long SL (<2.5%)</th><th>Status</th></tr>"
    
    for _, row in short_wins.iterrows():
        status = "<b>OK</b>" if row['long_sl_hit'] else "<b style='color:red'>IMPOSSIBLE ANOMALY</b>"
        html += f"""<tr>
            <td>{row['time']}</td>
            <td>{row['open']:.2f}</td>
            <td>{row['low']:.2f}</td>
            <td>{row['open']*(1-TP):.2f}</td>
            <td>{row['open']*(1-SL):.2f}</td>
            <td>{status}</td>
        </tr>"""
    html += "</table>"
    
    return html

class AuditHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(get_data_and_audit().encode('utf-8'))

if __name__ == "__main__":
    print("Serving audit on port 8080...")
    socketserver.TCPServer(("", 8080), AuditHandler).serve_forever()
