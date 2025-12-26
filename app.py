import os
import pandas as pd
import yfinance as yf
import requests
import numpy as np
import logging
import time
import threading
from datetime import datetime, timedelta
from io import StringIO
from flask import Flask, render_template_string, jsonify

# ==========================================
# CONFIGURATION
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# SIMULATION PARAMETERS
DEV_STOCK_LIMIT = 60  # Set to None for full S&P 500
TOP_N = 15            # Number of Longs and Shorts (Total portfolio = 2 * TOP_N)
YEARS_HISTORY = 4
INITIAL_CAPITAL = 100000.0

# GLOBAL STORAGE FOR SERVER
BACKTEST_RESULTS = None

# ==========================================
# 1. RISK & POSITION MANAGEMENT
# ==========================================
class Position:
    def __init__(self, ticker, entry_price, side='long', weight=0.0):
        self.ticker = ticker
        self.entry_price = entry_price
        self.side = side 
        self.weight = weight 
        
        self.is_active = True
        self.is_trailing = False
        self.frozen_price = None 
        self.extreme_price = entry_price 

    def update(self, current_price):
        if not self.is_active:
            return self.frozen_price

        if self.side == 'long':
            if current_price > self.extreme_price: self.extreme_price = current_price
            if current_price < (self.entry_price * 0.80):
                self.is_active, self.frozen_price = False, self.entry_price * 0.80
                return self.frozen_price
            if not self.is_trailing and current_price > (self.entry_price * 1.20):
                self.is_trailing = True
            if self.is_trailing and current_price < (self.extreme_price * 0.90):
                self.is_active, self.frozen_price = False, self.extreme_price * 0.90
                return self.frozen_price
        elif self.side == 'short':
            if current_price < self.extreme_price: self.extreme_price = current_price
            if current_price > (self.entry_price * 1.20):
                self.is_active, self.frozen_price = False, self.entry_price * 1.20 
                return self.frozen_price
            if not self.is_trailing and current_price < (self.entry_price * 0.80):
                self.is_trailing = True
            if self.is_trailing and current_price > (self.extreme_price * 1.10):
                self.is_active, self.frozen_price = False, self.extreme_price * 1.10
                return self.frozen_price
        return current_price

    def get_pct_return(self, current_price):
        eff_price = self.frozen_price if not self.is_active else current_price
        if self.side == 'long':
            return (eff_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - eff_price) / self.entry_price

# ==========================================
# 2. DATA ENGINES
# ==========================================
class FredHistoricalEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

    def fetch(self, sid):
        if not self.api_key: return pd.DataFrame()
        try:
            r = requests.get(self.base_url, params={"series_id": sid, "api_key": self.api_key, "file_type": "json"})
            df = pd.DataFrame(r.json()['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return df[['date', 'value']].dropna()
        except: return pd.DataFrame()

    def get_regimes(self):
        logger.info("Building Economic Regimes...")
        start = (datetime.now() - timedelta(days=365*9)).strftime('%Y-%m-%d')
        rate = self.fetch("FEDFUNDS").rename(columns={'value': 'ir'})
        bs = self.fetch("WALCL").rename(columns={'value': 'bs'})
        if rate.empty or bs.empty: return pd.DataFrame()
        df = pd.merge_asof(bs.sort_values('date'), rate.sort_values('date'), on='date')
        df = df.set_index('date')
        df['ir_avg'] = df['ir'].rolling(window=260, min_periods=50).mean()
        df['bs_avg'] = df['bs'].rolling(window=52, min_periods=20).mean()
        v = df['ir_avg'].notna() & df['bs_avg'].notna()
        hr = df['ir'] > df['ir_avg']
        hb = df['bs'] > df['bs_avg']
        df['regime'] = "None"
        df.loc[v & (~hr) & hb, 'regime'] = "A"
        df.loc[v & (~hr) & (~hb), 'regime'] = "B"
        df.loc[v & hr & hb, 'regime'] = "C"
        df.loc[v & hr & (~hb), 'regime'] = "D"
        return df[df.index >= (datetime.now() - timedelta(days=365*YEARS_HISTORY))]

class StockEngine:
    def process(self, ticker, dates):
        try:
            s = yf.Ticker(ticker)
            h = s.history(period="5y", interval="1wk")
            if h.empty: return None
            h = h[['Close']].rename(columns={'Close': 'price'})
            h.index = h.index.tz_localize(None)
            f = s.quarterly_financials.T
            if f.empty: return None
            f.columns = f.columns.str.strip()
            f = f.apply(pd.to_numeric, errors='coerce').sort_index()
            fd = pd.DataFrame(index=f.index)
            if 'Total Revenue' in f.columns: fd['growth'] = f['Total Revenue'].pct_change(periods=4)
            if 'Operating Income' in f.columns and 'Total Revenue' in f.columns:
                fd['margin'] = f['Operating Income'] / f['Total Revenue']
            if 'Basic EPS' in f.columns: fd['eps'] = f['Basic EPS'].rolling(window=4).sum()
            fd.index = pd.to_datetime(fd.index).tz_localize(None)
            df = h.join(fd.reindex(h.index, method='ffill'))
            if 'eps' in df.columns and 'margin' in df.columns:
                df['profit'] = df['margin'] * (df['eps'] / df['price'])
            df['ticker'] = ticker
            return df.reindex(dates, method='ffill')
        except: return None

# ==========================================
# 3. BACKTEST EXECUTION
# ==========================================
def run_backtest():
    global BACKTEST_RESULTS
    fred = FredHistoricalEngine(FRED_API_KEY)
    econ = fred.get_regimes()
    if econ.empty: 
        logger.error("No Economic Data. Check FRED Key.")
        return

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    ts = pd.read_html(StringIO(requests.get(url, headers={"User-Agent":"Mozilla"}).text))[0]['Symbol'].tolist()
    ts = [t.replace('.','-') for t in ts][:DEV_STOCK_LIMIT]

    cache = {}
    logger.info(f"Loading {len(ts)} stocks...")
    for t in ts:
        d = StockEngine().process(t, econ.index)
        if d is not None: cache[t] = d

    val = INITIAL_CAPITAL
    ps = []
    cur_reg = None
    hist = []
    
    for date in econ.index.sort_values():
        reg = econ.loc[date, 'regime']
        if reg != cur_reg:
            if ps:
                pnl = sum(p.get_pct_return(cache[p.ticker].loc[date, 'price']) * p.weight for p in ps)
                val *= (1 + pnl)
            cur_reg, ps = reg, []
            snaps = []
            for t, d in cache.items():
                try:
                    row = d.loc[date]
                    if pd.notna(row['price']):
                        snaps.append({'t': t, 'p': row['price'], 'g': row.get('growth', 0), 'pr': row.get('profit', 0)})
                except: pass
            sdf = pd.DataFrame(snaps).dropna()
            if sdf.empty: continue
            sdf['zg'] = (sdf['g'] - sdf['g'].mean()) / (sdf['g'].std() + 1e-6)
            sdf['zp'] = (sdf['pr'] - sdf['pr'].mean()) / (sdf['pr'].std() + 1e-6)
            if reg == "A": sdf['score'] = sdf['zg']
            elif reg == "D": sdf['score'] = sdf['zp']
            else: sdf['score'] = sdf['zg'] + sdf['zp']
            sdf = sdf.sort_values('score', ascending=False)
            w = 0.5 / TOP_N
            for _, r in sdf.head(TOP_N).iterrows(): ps.append(Position(r['t'], r['p'], 'long', w))
            for _, r in sdf.tail(TOP_N).iterrows(): ps.append(Position(r['t'], r['p'], 'short', w))

        pnl_now = sum(p.get_pct_return(cache[p.ticker].loc[date, 'price']) * p.weight for p in ps) if ps else 0
        hist.append({'date': date.strftime('%Y-%m-%d'), 'val': round(val * (1 + pnl_now), 2), 'reg': reg})

    BACKTEST_RESULTS = pd.DataFrame(hist)
    logger.info("Backtest Complete.")

# ==========================================
# 4. WEB SERVER
# ==========================================
app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Regime Backtest Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; background: #f5f7f9; color: #333; }
        .container { max-width: 1000px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { margin-top: 0; color: #1a202c; }
        .stats { display: flex; gap: 20px; margin-bottom: 30px; }
        .stat-card { flex: 1; padding: 20px; background: #edf2f7; border-radius: 8px; }
        .stat-label { font-size: 12px; text-transform: uppercase; color: #718096; letter-spacing: 0.05em; }
        .stat-value { font-size: 24px; font-weight: bold; color: #2d3748; }
        .regime-A { color: #48bb78; } .regime-B { color: #4299e1; } .regime-C { color: #ed8936; } .regime-D { color: #f56565; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Portfolio Performance</h1>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Initial Capital</div>
                <div class="stat-value">${{ "{:,.2f}".format(initial_cap) }}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Final Value</div>
                <div class="stat-value">${{ "{:,.2f}".format(final_val) }}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Return</div>
                <div class="stat-value">{{ "{:.2f}%".format(total_ret) }}</div>
            </div>
        </div>
        <canvas id="mainChart" width="800" height="400"></canvas>
    </div>

    <script>
        const rawData = {{ chart_data | safe }};
        const ctx = document.getElementById('mainChart').getContext('2d');
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: rawData.map(d => d.date),
                datasets: [{
                    label: 'Portfolio Value ($)',
                    data: rawData.map(d => d.val),
                    borderColor: '#3182ce',
                    backgroundColor: 'rgba(49, 130, 206, 0.1)',
                    fill: true,
                    tension: 0.2,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { grid: { display: false } },
                    y: { beginAtZero: false }
                }
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    if BACKTEST_RESULTS is None:
        return "Backtest still running, please refresh in a moment..."
    
    final_val = BACKTEST_RESULTS['val'].iloc[-1]
    total_ret = ((final_val / INITIAL_CAPITAL) - 1) * 100
    
    return render_template_string(
        DASHBOARD_HTML,
        initial_cap=INITIAL_CAPITAL,
        final_val=final_val,
        total_ret=total_ret,
        chart_data=BACKTEST_RESULTS.to_json(orient='records')
    )

def start_server():
    app.run(host='0.0.0.0', port=8080)

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # Start server in a background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    logger.info("Backtest starting... dashboard will be available at http://localhost:8080")
    
    # Run simulation
    run_backtest()
    
    # Keep the main thread alive for the server
    while True:
        time.sleep(1)