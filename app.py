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
# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# DEVELOPMENT LIMIT: Set to an integer (e.g., 5 or 10) to limit stocks processed.
# Set to None to run the full list.
DEV_STOCK_LIMIT = 10 

# List of tickers to analyze. 
TICKERS = []

# Fetch S&P 500 tickers dynamically
try:
    logger.info("Fetching S&P 500 tickers from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    tables = pd.read_html(StringIO(response.text))
    sp500 = tables[0]
    TICKERS = [t.replace('.', '-') for t in sp500['Symbol'].tolist()]
    logger.info(f"Successfully fetched {len(TICKERS)} tickers.")
except Exception as e:
    logger.error(f"Error fetching S&P 500 list: {e}")
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

if DEV_STOCK_LIMIT:
    logger.warning(f"⚠️ DEV MODE: Limiting analysis to first {DEV_STOCK_LIMIT} stocks.")
    TICKERS = TICKERS[:DEV_STOCK_LIMIT]

YEARS_HISTORY = 4

# ==========================================
# 1. FRED HISTORICAL ENGINE
# ==========================================
class FredHistoricalEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

    def fetch_series(self, series_id, start_date=None):
        """Fetches data from FRED or mocks it if no key provided."""
        if not self.api_key:
            logger.warning(f"⚠️ No FRED Key provided. Generating mock data for {series_id}...")
            dates = pd.date_range(end=datetime.now(), periods=52*10, freq='W-FRI')
            return pd.DataFrame({'value': np.random.uniform(2, 5, len(dates))}, index=dates)

        params = {
            "series_id": series_id, "api_key": self.api_key,
            "file_type": "json", "sort_order": "asc"
        }
        if start_date:
            params["observation_start"] = start_date

        try:
            logger.info(f"Fetching {series_id} from FRED...")
            response = requests.get(self.base_url, params=params)
            
            if response.status_code != 200:
                logger.error(f"FRED API Error {response.status_code} for {series_id}: {response.text[:200]}")
            
            response.raise_for_status()
            data = response.json().get("observations", [])
            
            if not data:
                logger.warning(f"FRED returned no observations for {series_id}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            df = df[['date', 'value']].dropna().sort_values('date')
            logger.info(f"Successfully fetched {len(df)} rows for {series_id}")
            return df
        except Exception as e:
            logger.error(f"Exception fetching {series_id}: {e}", exc_info=True)
            return pd.DataFrame()

    def get_economic_regime(self):
        logger.info("📊 Building Historical Economic Regime...")
        
        start_date = (datetime.now() - timedelta(days=365*9)).strftime('%Y-%m-%d')
        
        df_rate = self.fetch_series("FEDFUNDS", start_date) 
        df_bs = self.fetch_series("WALCL", start_date) 

        if df_rate.empty or df_bs.empty:
            logger.error("❌ Critical: Missing FRED data. Cannot build economic regime.")
            return pd.DataFrame()

        df_rate = df_rate.rename(columns={'value': 'interest_rate'})
        df_bs = df_bs.rename(columns={'value': 'balance_sheet'})

        logger.info("Merging datasets using merge_asof...")
        df = pd.merge_asof(
            df_bs, 
            df_rate, 
            on='date', 
            direction='backward'
        )

        df = df.set_index('date').sort_index()

        df['avg_rate_5y'] = df['interest_rate'].rolling(window=260, min_periods=50).mean()
        df['avg_bs_1y'] = df['balance_sheet'].rolling(window=52, min_periods=20).mean()

        df['category_id'] = 0
        df['category_name'] = "Insuff Data"

        valid = df['avg_rate_5y'].notna() & df['avg_bs_1y'].notna()
        high_rate = df['interest_rate'] > df['avg_rate_5y']
        high_bs = df['balance_sheet'] > df['avg_bs_1y']

        c1 = valid & high_rate & high_bs
        df.loc[c1, 'category_id'] = 1
        df.loc[c1, 'category_name'] = "High Rate / Liquidity Rising"

        c2 = valid & high_rate & (~high_bs)
        df.loc[c2, 'category_id'] = 2
        df.loc[c2, 'category_name'] = "High Rate / Liquidity Falling"

        c3 = valid & (~high_rate) & high_bs
        df.loc[c3, 'category_id'] = 3
        df.loc[c3, 'category_name'] = "Low Rate / Liquidity Rising"

        c4 = valid & (~high_rate) & (~high_bs)
        df.loc[c4, 'category_id'] = 4
        df.loc[c4, 'category_name'] = "Low Rate / Liquidity Falling"

        cutoff = datetime.now() - timedelta(days=365*YEARS_HISTORY)
        result = df[df.index >= cutoff]
        
        logger.info(f"Economic Regime built successfully. Rows: {len(result)}")
        return result

# ==========================================
# 2. STOCK DATA ENGINE (YFINANCE)
# ==========================================
class StockHistoricalEngine:
    def process_ticker(self, ticker, economic_df):
        try:
            stock = yf.Ticker(ticker)
            ohlc = stock.history(period="5y", interval="1wk")
            if ohlc.empty: 
                logger.warning(f"No price data found for {ticker}")
                return None
            
            ohlc = ohlc[['Close', 'Volume']].copy()
            ohlc.columns = ['price', 'volume']
            ohlc.index = ohlc.index.tz_localize(None) 
            ohlc.index.name = 'date' 

            fin = stock.quarterly_financials.T 
            
            if fin.empty:
                fund_data = pd.DataFrame(index=ohlc.index) 
                logger.debug(f"No fundamental data for {ticker}. Using price only.")
            else:
                fin.columns = fin.columns.str.strip() 
                fin = fin.apply(pd.to_numeric, errors='coerce') 
                
                fin = fin.sort_index()
                fin = fin[~fin.index.duplicated(keep='last')]
                fund_data = pd.DataFrame(index=fin.index)
                
                if 'Total Revenue' in fin.columns and 'Cost Of Revenue' in fin.columns:
                    fund_data['gross_margin'] = (fin['Total Revenue'] - fin['Cost Of Revenue']) / fin['Total Revenue']
                elif 'Gross Profit' in fin.columns and 'Total Revenue' in fin.columns:
                     fund_data['gross_margin'] = fin['Gross Profit'] / fin['Total Revenue']
                
                if 'Operating Income' in fin.columns and 'Total Revenue' in fin.columns:
                    fund_data['operating_margin'] = fin['Operating Income'] / fin['Total Revenue']
                    
                if 'Net Income' in fin.columns and 'Total Revenue' in fin.columns:
                    fund_data['net_margin'] = fin['Net Income'] / fin['Total Revenue']

                if 'Total Revenue' in fin.columns:
                    fund_data['revenue_growth_yoy'] = fin['Total Revenue'].pct_change(periods=4) 

                fund_data.index = pd.to_datetime(fund_data.index).tz_localize(None)
                fund_data = fund_data.sort_index()

            aligned_funds = fund_data.reindex(ohlc.index, method='ffill')
            df = ohlc.join(aligned_funds)

            df = df.sort_index()
            economic_df = economic_df.sort_index()
            
            df_reset = df.reset_index()
            econ_reset = economic_df[['category_id', 'category_name']].reset_index()
            
            if 'date' not in df_reset.columns:
                df_reset = df_reset.rename(columns={'index': 'date'})
            if 'date' not in econ_reset.columns:
                econ_reset = econ_reset.rename(columns={'index': 'date'})

            df_merged = pd.merge_asof(
                df_reset,
                econ_reset,
                on='date',
                direction='backward',
                tolerance=pd.Timedelta(days=14) 
            )
            
            df_merged = df_merged.set_index('date')
            df_merged['weekly_return'] = df_merged['price'].pct_change()
            df_merged['ticker'] = ticker
            
            df_final = df_merged.dropna(subset=['category_id'])
            return df_final

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}") 
            return None

# ==========================================
# 3. ANALYSIS & REPORTING
# ==========================================
def generate_analysis():
    logger.info("Starting Market Regime Analysis...")
    
    fred = FredHistoricalEngine(FRED_API_KEY)
    economy_df = fred.get_economic_regime()
    
    if economy_df.empty:
        logger.error("❌ Failed to generate economic data. Check logs above for API errors.")
        return

    logger.info(f"✅ Economic Regime Built ({len(economy_df)} weeks)")
    
    time.sleep(0.2)
    print("\n---------------------------------------------------")
    time.sleep(0.2)
    print(" FULL ECONOMIC REGIME DATA (ALL WEEKS)")
    time.sleep(0.2)
    print("---------------------------------------------------")
    time.sleep(0.2)
    print(economy_df.to_string())
    time.sleep(0.2)
    print("\n---------------------------------------------------\n")

    engine = StockHistoricalEngine()
    all_data = []

    logger.info(f"📥 Processing {len(TICKERS)} stocks...")
    for i, t in enumerate(TICKERS):
        if (i+1) % 5 == 0: logger.info(f"  > Processed [{i+1}/{len(TICKERS)}] stocks...")
        df = engine.process_ticker(t, economy_df)
        if df is not None and not df.empty:
            all_data.append(df)
    
    if not all_data:
        logger.error("❌ No stock data processed. (All DataFrames were empty)")
        return

    master_df = pd.concat(all_data)
    
    first_ticker = TICKERS[0]
    first_stock_data = master_df[master_df['ticker'] == first_ticker]
    
    time.sleep(0.2)
    print("\n---------------------------------------------------")
    time.sleep(0.2)
    print(f" FULL DATA FOR TICKER: {first_ticker}")
    time.sleep(0.2)
    print("---------------------------------------------------")
    cols = ['price', 'weekly_return', 'category_name', 'gross_margin', 'operating_margin', 'revenue_growth_yoy']
    valid_cols = [c for c in cols if c in first_stock_data.columns]
    
    time.sleep(0.2)
    print("--- HEAD (First 5 Rows) ---")
    time.sleep(0.2)
    print(first_stock_data[valid_cols].head().to_string())
    time.sleep(0.2)
    print("\n--- TAIL (Last 5 Rows) ---")
    time.sleep(0.2)
    print(first_stock_data[valid_cols].tail().to_string())
    time.sleep(0.2)
    print("\n---------------------------------------------------\n")

    time.sleep(0.2)
    print("\n📊 ANALYSIS RESULTS BY REGIME")
    time.sleep(0.2)
    print("Metrics: Average Weekly Return per category based on Fundamental traits\n")

    categories = master_df['category_id'].unique()
    
    for cat_id in sorted(categories):
        if cat_id == 0: continue
        
        subset = master_df[master_df['category_id'] == cat_id]
        cat_name = subset['category_name'].iloc[0]
        weeks_count = subset.index.nunique()
        
        time.sleep(0.2)
        print(f"=== {cat_name} (n={weeks_count} weeks) ===")
        time.sleep(0.2)
        print(f"Avg Weekly Market Return: {subset['weekly_return'].mean()*100:.2f}%")
        
        if 'gross_margin' in subset.columns and subset['gross_margin'].notna().sum() > 10:
            median_gm = subset['gross_margin'].median()
            high_gm = subset[subset['gross_margin'] > median_gm]['weekly_return'].mean()
            low_gm = subset[subset['gross_margin'] <= median_gm]['weekly_return'].mean()
            
            time.sleep(0.2)
            print(f"  > High Gross Margin Stocks Return: {high_gm*100:.3f}%")
            time.sleep(0.2)
            print(f"  > Low Gross Margin Stocks Return:  {low_gm*100:.3f}%")
            time.sleep(0.2)
            print(f"  > Spread: {(high_gm - low_gm)*100:.3f}%")
        
        if 'operating_margin' in subset.columns and subset['operating_margin'].notna().sum() > 10:
            median_om = subset['operating_margin'].median()
            high_om = subset[subset['operating_margin'] > median_om]['weekly_return'].mean()
            low_om = subset[subset['operating_margin'] <= median_om]['weekly_return'].mean()
            
            time.sleep(0.2)
            print(f"  > High Operating Margin Stocks Return: {high_om*100:.3f}%")
            time.sleep(0.2)
            print(f"  > Low Operating Margin Stocks Return:  {low_om*100:.3f}%")

        time.sleep(0.2)
        print("")

    filename = "market_regime_analysis.csv"
    master_df.to_csv(filename)
    logger.info(f"✅ Full analysis saved to {filename}")

if __name__ == "__main__":
    generate_analysis()