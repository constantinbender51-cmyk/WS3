import os
import pandas as pd
import yfinance as yf
import requests
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from io import StringIO

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

# ==========================================
# 1. RISK & POSITION MANAGEMENT
# ==========================================
class Position:
    def __init__(self, ticker, entry_price, side='long', weight=0.0):
        self.ticker = ticker
        self.entry_price = entry_price
        self.side = side # 'long' or 'short'
        self.weight = weight 
        
        self.is_active = True
        self.is_trailing = False
        self.frozen_price = None 
        self.extreme_price = entry_price 

    def update(self, current_price):
        if not self.is_active:
            return self.frozen_price

        # --- LONG LOGIC ---
        if self.side == 'long':
            if current_price > self.extreme_price: self.extreme_price = current_price
            
            # Static Stop (-20%)
            if current_price < (self.entry_price * 0.80):
                self.is_active, self.frozen_price = False, self.entry_price * 0.80
                return self.frozen_price
            
            # Trailing Activation (+20%)
            if not self.is_trailing and current_price > (self.entry_price * 1.20):
                self.is_trailing = True
            
            # Trailing Stop (-10% from Peak)
            if self.is_trailing and current_price < (self.extreme_price * 0.90):
                self.is_active, self.frozen_price = False, self.extreme_price * 0.90
                return self.frozen_price

        # --- SHORT LOGIC ---
        elif self.side == 'short':
            if current_price < self.extreme_price: self.extreme_price = current_price
                
            # Static Stop (+20% price increase = 20% loss)
            if current_price > (self.entry_price * 1.20):
                self.is_active, self.frozen_price = False, self.entry_price * 1.20 
                return self.frozen_price

            # Trailing Activation (20% price drop = 20% profit)
            if not self.is_trailing and current_price < (self.entry_price * 0.80):
                self.is_trailing = True
                
            # Trailing Stop (+10% price increase from Trough)
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
        df.loc[v & (~hr) & hb, 'regime'] = "A" # Low Rate / High Liq
        df.loc[v & (~hr) & (~hb), 'regime'] = "B" # Low Rate / Low Liq
        df.loc[v & hr & hb, 'regime'] = "C" # High Rate / High Liq
        df.loc[v & hr & (~hb), 'regime'] = "D" # High Rate / Low Liq
        
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
def backtest():
    fred = FredHistoricalEngine(FRED_API_KEY)
    econ = fred.get_regimes()
    if econ.empty: 
        logger.error("No Economic Data. Check FRED Key.")
        return

    # Fetch tickers
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    ts = pd.read_html(StringIO(requests.get(url, headers={"User-Agent":"Mozilla"}).text))[0]['Symbol'].tolist()
    ts = [t.replace('.','-') for t in ts][:DEV_STOCK_LIMIT]

    cache = {}
    time.sleep(0.2)
    print(f"Loading {len(ts)} stocks...")
    for t in ts:
        d = StockEngine().process(t, econ.index)
        if d is not None: cache[t] = d

    # Loop State
    val = INITIAL_CAPITAL
    ps = []
    cur_reg = None
    hist = []
    
    for date in econ.index.sort_values():
        reg = econ.loc[date, 'regime']
        
        # REBALANCE Logic
        if reg != cur_reg:
            logger.info(f"[{date.date()}] Regime: {reg}")
            
            # Close existing
            if ps:
                pnl = sum(p.get_pct_return(cache[p.ticker].loc[date, 'price']) * p.weight for p in ps)
                val *= (1 + pnl)
            
            cur_reg, ps = reg, []
            
            # Rank stocks for new regime
            snaps = []
            for t, d in cache.items():
                try:
                    row = d.loc[date]
                    if pd.notna(row['price']):
                        snaps.append({'t': t, 'p': row['price'], 'g': row.get('growth', 0), 'pr': row.get('profit', 0)})
                except: pass
            
            sdf = pd.DataFrame(snaps).dropna()
            if sdf.empty: continue

            # MULTI-MATHEMATICAL SCORING (Z-Score Summation)
            # Standardize metrics so they can be added fairly
            sdf['zg'] = (sdf['g'] - sdf['g'].mean()) / sdf['g'].std()
            sdf['zp'] = (sdf['pr'] - sdf['pr'].mean()) / sdf['pr'].std()
            
            if reg == "A": sdf['score'] = sdf['zg'] # Growth only
            elif reg == "D": sdf['score'] = sdf['zp'] # Profit only
            else: sdf['score'] = sdf['zg'] + sdf['zp'] # Combined Factor Score
            
            sdf = sdf.sort_values('score', ascending=False)
            
            w = 0.5 / TOP_N
            for _, r in sdf.head(TOP_N).iterrows(): ps.append(Position(r['t'], r['p'], 'long', w))
            for _, r in sdf.tail(TOP_N).iterrows(): ps.append(Position(r['t'], r['p'], 'short', w))

        # TRACKING Logic
        pnl_now = sum(p.get_pct_return(cache[p.ticker].loc[date, 'price']) * p.weight for p in ps) if ps else 0
        hist.append({'date': date, 'val': val * (1 + pnl_now), 'reg': reg})

    # Results
    rdf = pd.DataFrame(hist).set_index('date')
    time.sleep(0.2)
    print("\n" + "="*30 + "\nRESULTS\n" + "="*30)
    print(f"Final Value: ${rdf['val'].iloc[-1]:,.2f}")
    print(f"Total Return: {((rdf['val'].iloc[-1]/INITIAL_CAPITAL)-1)*100:.2f}%")
    
    # Save results
    rdf.to_csv("regime_backtest_results.csv")
    time.sleep(0.2)
    print("\nSample Curve (Last 10 weeks):")
    print(rdf['val'].tail(10))

if __name__ == "__main__":
    backtest()