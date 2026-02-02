import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import time
from datetime import datetime, timedelta

def fetch_data():
    exchange = ccxt.binance({'enableRateLimit': True})
    symbol = 'ETH/USDT'
    timeframe = '1h'
    since = exchange.parse8601('2021-01-01T00:00:00Z')
    limit = 1000
    all_ohlcv = []

    print(f"[SYSTEM] Fetching {symbol} {timeframe} data from {exchange.iso8601(since)}...")
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            # Check if we reached near present time
            if since > exchange.milliseconds() - 3600000:
                break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(5)
            continue

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def run_backtest(df):
    print("[SYSTEM] Calculating indicators and executing backtest logic...")
    
    # 365 days * 24 hours = 8760 hours
    sma_window = 365 * 24
    df['sma'] = df['close'].rolling(window=sma_window).mean()
    
    # Drop initial data without SMA
    df.dropna(subset=['sma'], inplace=True)
    
    # Convert to numpy for performance
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    smas = df['sma'].values
    times = df.index
    
    n = len(df)
    equity_curve = [0.0]
    
    # Active trades list: [entry_price, tp_price, sl_price, direction (1/-1), entry_index]
    active_trades = []
    
    for i in range(n - 1):
        current_close = closes[i]
        current_sma = smas[i]
        
        # Determine TP/SL percentages
        if current_close > current_sma:
            tp_pct = 0.04
            sl_pct = 0.005
        else:
            tp_pct = 0.04
            sl_pct = 0.03
            
        # Create Long
        long_tp = current_close * (1 + tp_pct)
        long_sl = current_close * (1 - sl_pct)
        active_trades.append([current_close, long_tp, long_sl, 1, i])
        
        # Create Short
        short_tp = current_close * (1 - tp_pct)
        short_sl = current_close * (1 + sl_pct)
        active_trades.append([current_close, short_tp, short_sl, -1, i])
        
        # Process Active Trades against Next Candle (i+1)
        next_high = highs[i+1]
        next_low = lows[i+1]
        
        remaining_trades = []
        hourly_pnl = 0.0
        
        for trade in active_trades:
            entry, tp, sl, direction, entry_idx = trade
            
            # Check conditions
            hit_tp = False
            hit_sl = False
            pnl = 0.0
            
            if direction == 1: # Long
                if next_low <= sl:
                    hit_sl = True
                    pnl = -sl_pct
                elif next_high >= tp:
                    hit_tp = True
                    pnl = tp_pct
            else: # Short
                if next_high >= sl:
                    hit_sl = True
                    pnl = -sl_pct 
                elif next_low <= tp:
                    hit_tp = True
                    pnl = tp_pct
            
            if hit_sl or hit_tp:
                hourly_pnl += pnl
            else:
                remaining_trades.append(trade)
                
        active_trades = remaining_trades
        equity_curve.append(equity_curve[-1] + hourly_pnl)

    # Return full times array to match equity_curve length (n)
    return times, equity_curve

class PlotHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # Generate plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.server.times, self.server.equity, label='Cumulative PnL (Uncompounded)')
        plt.title(f'ETH/USDT Hourly Straddle Strategy\n(Above SMA: TP 4% SL 0.5% | Below SMA: TP 4% SL 3%)')
        plt.xlabel('Date')
        plt.ylabel('Return (R)')
        plt.legend()
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        import base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        html = f"""
        <html>
        <head><title>Strategy Backtest</title></head>
        <body style="background-color: #1e1e1e; color: #d4d4d4; font-family: monospace;">
            <div style="text-align: center; padding: 20px;">
                <h1>ETH/USDT 1H Strategy Backtest</h1>
                <img src="data:image/png;base64,{img_str}" style="border: 1px solid #555;"/>
            </div>
        </body>
        </html>
        """
        self.wfile.write(html.encode('utf-8'))

def main():
    # 1. Fetch
    df = fetch_data()
    
    # 2. Backtest
    times, equity = run_backtest(df)
    
    # 3. Serve
    port = 8080
    server_address = ('', port)
    httpd = HTTPServer(server_address, PlotHandler)
    
    # Inject data into server instance for handler access
    httpd.times = times
    httpd.equity = equity
    
    print(f"[SYSTEM] Serving plot at http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Server stopped.")

if __name__ == "__main__":
    main()
