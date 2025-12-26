import os
import pandas as pd
import yfinance as yf
import requests
import numpy as np
import logging
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
            
            # Note: We do NOT set index here anymore, to facilitate merge_asof later
            df = df[['date', 'value']].dropna().sort_values('date')
            logger.info(f"Successfully fetched {len(df)} rows for {series_id}")
            return df
        except Exception as e:
            logger.error(f"Exception fetching {series_id}: {e}", exc_info=True)
            return pd.DataFrame()

    def get_economic_regime(self):
        logger.info("📊 Building Historical Economic Regime...")
        
        # We need 9 years of data to calculate a 5-year rolling average for the last 4 years
        start_date = (datetime.now() - timedelta(days=365*9)).strftime('%Y-%m-%d')
        
        # 1. Fetch Data
        df_rate = self.fetch_series("FEDFUNDS", start_date) # Monthly
        df_bs = self.fetch_series("WALCL", start_date) # Weekly

        if df_rate.empty or df_bs.empty:
            logger.error("❌ Critical: Missing FRED data. Cannot build economic regime.")
            return pd.DataFrame()

        # Rename columns for clarity before merge
        df_rate = df_rate.rename(columns={'value': 'interest_rate'})
        df_bs = df_bs.rename(columns={'value': 'balance_sheet'})

        # 2. Merge using merge_asof (The "Exact Logic" from original file)
        # This matches each Weekly Balance Sheet entry with the last known Monthly Rate
        # This is robust against misaligned dates (Wed vs Fri)
        logger.info("Merging datasets using merge_asof...")
        df = pd.merge_asof(
            df_bs, 
            df_rate, 
            on='date', 
            direction='backward'
        )

        # Set index to date after merge
        df = df.set_index('date').sort_index()

        # 3. Calculate Logic (Rolling Averages)
        # 5 Year Average for Rates (approx 260 weeks)
        df['avg_rate_5y'] = df['interest_rate'].rolling(window=260, min_periods=50).mean()
        
        # 1 Year Average for Balance Sheet (approx 52 weeks)
        df['avg_bs_1y'] = df['balance_sheet'].rolling(window=52, min_periods=20).mean()

        # 4. Determine Categories (Vectorized Approach)
        df['category_id'] = 0
        df['category_name'] = "Insuff Data"

        valid = df['avg_rate_5y'].notna() & df['avg_bs_1y'].notna()

        high_rate = df['interest_rate'] > df['avg_rate_5y']
        high_bs = df['balance_sheet'] > df['avg_bs_1y']

        # Case 1: High Rate / High Liquidity
        c1 = valid & high_rate & high_bs
        df.loc[c1, 'category_id'] = 1
        df.loc[c1, 'category_name'] = "High Rate / Liquidity Rising"

        # Case 2: High Rate / Low Liquidity
        c2 = valid & high_rate & (~high_bs)
        df.loc[c2, 'category_id'] = 2
        df.loc[c2, 'category_name'] = "High Rate / Liquidity Falling"

        # Case 3: Low Rate / High Liquidity
        c3 = valid & (~high_rate) & high_bs
        df.loc[c3, 'category_id'] = 3
        df.loc[c3, 'category_name'] = "Low Rate / Liquidity Rising"

        # Case 4: Low Rate / Low Liquidity
        c4 = valid & (~high_rate) & (~high_bs)
        df.loc[c4, 'category_id'] = 4
        df.loc[c4, 'category_name'] = "Low Rate / Liquidity Falling"

        # Trim to requested history (last 4 years)
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
            # logger.info(f"Processing {ticker}...") 
            stock = yf.Ticker(ticker)
            
            # 1. Get Price Data (Weekly)
            ohlc = stock.history(period="5y", interval="1wk")
            if ohlc.empty: 
                logger.warning(f"No price data found for {ticker}")
                return None
            
            ohlc = ohlc[['Close', 'Volume']].copy()
            ohlc.columns = ['price', 'volume']
            ohlc.index = ohlc.index.tz_localize(None) 

            # 2. Get Fundamentals (Quarterly)
            fin = stock.quarterly_financials.T 
            
            if fin.empty:
                fund_data = pd.DataFrame(index=ohlc.index) 
                logger.debug(f"No fundamental data for {ticker}. Using price only.")
            else:
                fin = fin.sort_index()
                fin = fin[~fin.index.duplicated(keep='last')]
                fund_data = pd.DataFrame(index=fin.index)
            
                # --- PROFITABILITY & MARGINS ---
                # Gross Margin
                if 'Total Revenue' in fin.columns and 'Cost Of Revenue' in fin.columns:
                    fund_data['gross_margin'] = (fin['Total Revenue'] - fin['Cost Of Revenue']) / fin['Total Revenue']
                elif 'Gross Profit' in fin.columns and 'Total Revenue' in fin.columns:
                     fund_data['gross_margin'] = fin['Gross Profit'] / fin['Total Revenue']
                
                # Operating Margin
                if 'Operating Income' in fin.columns and 'Total Revenue' in fin.columns:
                    fund_data['operating_margin'] = fin['Operating Income'] / fin['Total Revenue']
                    
                # Net Margin
                if 'Net Income' in fin.columns and 'Total Revenue' in fin.columns:
                    fund_data['net_margin'] = fin['Net Income'] / fin['Total Revenue']

                # --- GROWTH PROXIES ---
                # YoY Revenue Growth
                if 'Total Revenue' in fin.columns:
                    fund_data['revenue_growth_yoy'] = fin['Total Revenue'].pct_change(periods=4) 

                fund_data.index = pd.to_datetime(fund_data.index).tz_localize(None)
                fund_data = fund_data.sort_index()

            # 3. Merge Price and Fundamentals
            aligned_funds = fund_data.reindex(ohlc.index, method='ffill')
            df = ohlc.join(aligned_funds)

            # 4. Merge with Economic Data
            df = df.join(economic_df[['category_id', 'category_name']])

            # 5. Calculate Weekly Returns
            df['weekly_return'] = df['price'].pct_change()
            df['ticker'] = ticker
            
            df = df.dropna(subset=['category_id'])
            return df

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}") 
            return None

# ==========================================
# 3. ANALYSIS & REPORTING
# ==========================================
def generate_analysis():
    logger.info("Starting Market Regime Analysis...")
    
    # 1. Prepare Economy
    fred = FredHistoricalEngine(FRED_API_KEY)
    economy_df = fred.get_economic_regime()
    
    if economy_df.empty:
        logger.error("❌ Failed to generate economic data. Check logs above for API errors.")
        return

    logger.info(f"✅ Economic Regime Built ({len(economy_df)} weeks)")
    print("\nRegime Distribution:")
    print(economy_df['category_name'].value_counts())
    print("\n---------------------------------------------------\n")

    # 2. Process Stocks
    engine = StockHistoricalEngine()
    all_data = []

    logger.info(f"📥 Processing {len(TICKERS)} stocks...")
    for i, t in enumerate(TICKERS):
        if (i+1) % 5 == 0: logger.info(f"  > Processed [{i+1}/{len(TICKERS)}] stocks...")
        df = engine.process_ticker(t, economy_df)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        logger.error("❌ No stock data processed.")
        return

    master_df = pd.concat(all_data)
    
    # 3. Group by Regime and Calculate Metrics
    
    print("\n📊 ANALYSIS RESULTS BY REGIME")
    print("Metrics: Average Weekly Return per category based on Fundamental traits\n")

    categories = master_df['category_id'].unique()
    
    for cat_id in sorted(categories):
        if cat_id == 0: continue
        
        subset = master_df[master_df['category_id'] == cat_id]
        cat_name = subset['category_name'].iloc[0]
        weeks_count = subset.index.nunique()
        
        print(f"=== {cat_name} (n={weeks_count} weeks) ===")
        print(f"Avg Weekly Market Return: {subset['weekly_return'].mean()*100:.2f}%")
        
        # Check Gross Margin Impact
        if 'gross_margin' in subset.columns and subset['gross_margin'].notna().sum() > 10:
            median_gm = subset['gross_margin'].median()
            high_gm = subset[subset['gross_margin'] > median_gm]['weekly_return'].mean()
            low_gm = subset[subset['gross_margin'] <= median_gm]['weekly_return'].mean()
            print(f"  > High Gross Margin Stocks Return: {high_gm*100:.3f}%")
            print(f"  > Low Gross Margin Stocks Return:  {low_gm*100:.3f}%")
            print(f"  > Spread: {(high_gm - low_gm)*100:.3f}%")
        
        # Check Operating Margin Impact
        if 'operating_margin' in subset.columns and subset['operating_margin'].notna().sum() > 10:
            median_om = subset['operating_margin'].median()
            high_om = subset[subset['operating_margin'] > median_om]['weekly_return'].mean()
            low_om = subset[subset['operating_margin'] <= median_om]['weekly_return'].mean()
            print(f"  > High Operating Margin Stocks Return: {high_om*100:.3f}%")
            print(f"  > Low Operating Margin Stocks Return:  {low_om*100:.3f}%")

        print("")

    # 4. Export Raw Data
    filename = "market_regime_analysis.csv"
    master_df.to_csv(filename)
    logger.info(f"✅ Full analysis saved to {filename}")

if __name__ == "__main__":
    generate_analysis()