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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
DEV_STOCK_LIMIT = 10 
YEARS_HISTORY = 4

# Fetch Tickers
try:
    logger.info("Fetching S&P 500 tickers...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(response.text))
    TICKERS = [t.replace('.', '-') for t in tables[0]['Symbol'].tolist()]
    if DEV_STOCK_LIMIT:
        TICKERS = TICKERS[:DEV_STOCK_LIMIT]
except Exception as e:
    logger.error(f"Ticker fetch failed: {e}")
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

# ==========================================
# 1. FRED ENGINE
# ==========================================
class FredHistoricalEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

    def fetch_series(self, series_id, start_date=None):
        if not self.api_key:
            dates = pd.date_range(end=datetime.now(), periods=52*10, freq='W-FRI')
            return pd.DataFrame({'date': dates, 'value': np.random.uniform(2, 5, len(dates))})
        
        params = {"series_id": series_id, "api_key": self.api_key, "file_type": "json", "sort_order": "asc"}
        if start_date: params["observation_start"] = start_date
        
        try:
            r = requests.get(self.base_url, params=params)
            data = r.json().get("observations", [])
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return df[['date', 'value']].dropna().sort_values('date')
        except: return pd.DataFrame()

    def get_economic_regime(self):
        start_date = (datetime.now() - timedelta(days=365*9)).strftime('%Y-%m-%d')
        df_rate = self.fetch_series("FEDFUNDS", start_date).rename(columns={'value': 'interest_rate'})
        df_bs = self.fetch_series("WALCL", start_date).rename(columns={'value': 'balance_sheet'})
        
        if df_rate.empty or df_bs.empty: return pd.DataFrame()

        df = pd.merge_asof(df_bs, df_rate, on='date', direction='backward').set_index('date')
        df['avg_rate_5y'] = df['interest_rate'].rolling(window=260, min_periods=50).mean()
        df['avg_bs_1y'] = df['balance_sheet'].rolling(window=52, min_periods=20).mean()

        valid = df['avg_rate_5y'].notna() & df['avg_bs_1y'].notna()
        high_rate = df['interest_rate'] > df['avg_rate_5y']
        high_bs = df['balance_sheet'] > df['avg_bs_1y']

        df['category_id'] = 0
        df.loc[valid & high_rate & high_bs, 'category_id'] = 1
        df.loc[valid & high_rate & (~high_bs), 'category_id'] = 2
        df.loc[valid & (~high_rate) & high_bs, 'category_id'] = 3
        df.loc[valid & (~high_rate) & (~high_bs), 'category_id'] = 4
        
        names = {1: "High Rate / Liq Rising", 2: "High Rate / Liq Falling", 
                 3: "Low Rate / Liq Rising", 4: "Low Rate / Liq Falling", 0: "N/A"}
        df['category_name'] = df['category_id'].map(names)
        
        return df[df.index >= (datetime.now() - timedelta(days=365*YEARS_HISTORY))]

# ==========================================
# 2. STOCK ENGINE
# ==========================================
class StockHistoricalEngine:
    def process_ticker(self, ticker, economic_df):
        try:
            stock = yf.Ticker(ticker)
            ohlc = stock.history(period="5y", interval="1wk")
            if ohlc.empty: return None
            
            ohlc = ohlc[['Close']].rename(columns={'Close': 'price'})
            ohlc.index = ohlc.index.tz_localize(None)
            
            fin = stock.quarterly_financials.T
            if fin.empty: return None
            
            fin.columns = fin.columns.str.strip()
            fin = fin.apply(pd.to_numeric, errors='coerce')
            fin = fin.sort_index()
            
            fund_data = pd.DataFrame(index=fin.index)
            
            # Growth Metric
            if 'Total Revenue' in fin.columns:
                fund_data['rev_growth'] = fin['Total Revenue'].pct_change(periods=4)
            
            # Profitability Metric Components
            if 'Operating Income' in fin.columns and 'Total Revenue' in fin.columns:
                fund_data['op_margin'] = fin['Operating Income'] / fin['Total Revenue']
            
            if 'Basic EPS' in fin.columns:
                fund_data['eps_ttm'] = fin['Basic EPS'].rolling(window=4).sum()

            fund_data.index = pd.to_datetime(fund_data.index).tz_localize(None)
            aligned = fund_data.reindex(ohlc.index, method='ffill')
            
            df = ohlc.join(aligned)
            
            # Calculate Combined Profitability Score: (Margin * Earnings Yield)
            # Earnings Yield = EPS / Price
            if 'eps_ttm' in df.columns and 'op_margin' in df.columns:
                df['profit_score'] = df['op_margin'] * (df['eps_ttm'] / df['price'])

            # Merge Economics
            econ_reset = economic_df[['category_id', 'category_name']].reset_index()
            df_reset = df.reset_index()
            
            df_final = pd.merge_asof(df_reset.sort_values('date'), 
                                     econ_reset.sort_values('date'), 
                                     on='date', direction='backward')
            
            df_final['weekly_return'] = df_final['price'].pct_change()
            df_final['ticker'] = ticker
            
            # TRIMMING: Start only when rev_growth is available
            df_final = df_final.dropna(subset=['rev_growth', 'category_id']).set_index('date')
            
            return df_final

        except Exception as e:
            return None

# ==========================================
# 3. RUNNER
# ==========================================
def generate_analysis():
    fred = FredHistoricalEngine(FRED_API_KEY)
    economy_df = fred.get_economic_regime()
    if economy_df.empty: return

    engine = StockHistoricalEngine()
    all_data = []

    time.sleep(0.2)
    print("\n---------------------------------------------------")
    print(" 📦 PROCESSING STOCKS (Syncing to Revenue Growth Start)")
    print("---------------------------------------------------")

    for t in TICKERS:
        df = engine.process_ticker(t, economy_df)
        if df is not None and not df.empty:
            all_data.append(df)
            time.sleep(0.2)
            print(f"Stock: {t}")
            time.sleep(0.2)
            print(f"Count: {len(df)}")
            time.sleep(0.2)
            print(f"Timeframe: weekly")
            time.sleep(0.2)
            print(f"Data Starts: {df.index.min().date()}")
            time.sleep(0.2)
            print("-" * 15)

    if not all_data: return
    master_df = pd.concat(all_data)

    time.sleep(0.2)
    print("\n📊 REGIME ANALYSIS (Growth & Combined Profitability Score)")
    
    for cat_id in sorted(master_df['category_id'].unique()):
        if cat_id == 0: continue
        sub = master_df[master_df['category_id'] == cat_id]
        cat_name = sub['category_name'].iloc[0]
        
        time.sleep(0.2)
        print(f"=== {cat_name} (n={len(sub)}) ===")
        time.sleep(0.2)
        print(f"Market Avg Weekly Return: {sub['weekly_return'].mean()*100:.2f}%")
        
        # Growth Impact
        median_g = sub['rev_growth'].median()
        high_g = sub[sub['rev_growth'] > median_g]['weekly_return'].mean()
        time.sleep(0.2)
        print(f"  > High Rev Growth Returns: {high_g*100:.3f}%")
        
        # Profitability Score Impact
        if 'profit_score' in sub.columns:
            median_p = sub['profit_score'].median()
            high_p = sub[sub['profit_score'] > median_p]['weekly_return'].mean()
            time.sleep(0.2)
            print(f"  > High Profitability Score Returns: {high_p*100:.3f}%")
        print("")

if __name__ == "__main__":
    generate_analysis()