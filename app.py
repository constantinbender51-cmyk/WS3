import os
import pandas as pd
import yfinance as yf
import requests
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
# If empty, the script will mock FRED data for testing purposes
# Get a key here: https://fred.stlouisfed.org/docs/api/api_key.html

# List of tickers to analyze. 
# Full S&P 500 takes a long time. specific list for demo:
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", 
    "JPM", "V", "JNJ", "WMT", "PG", "XOM", "MA", "HD",
    "CVX", "MRK", "ABBV", "KO", "PEP"
]
# To use full S&P 500, uncomment below (WARNING: Takes 20+ mins):
# import pandas_datareader as pdr
# TICKERS = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()
# TICKERS = [t.replace('.', '-') for t in TICKERS]

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
            print(f"⚠️ No FRED Key. Generating mock data for {series_id}...")
            dates = pd.date_range(end=datetime.now(), periods=52*10, freq='W-FRI')
            return pd.DataFrame({'value': np.random.uniform(2, 5, len(dates))}, index=dates)

        params = {
            "series_id": series_id, "api_key": self.api_key,
            "file_type": "json", "sort_order": "asc"
        }
        if start_date:
            params["observation_start"] = start_date

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json().get("observations", [])
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna().set_index('date').sort_index()
            return df
        except Exception as e:
            print(f"Error fetching {series_id}: {e}")
            return pd.DataFrame()

    def get_economic_regime(self):
        print("📊 Building Historical Economic Regime...")
        
        # We need 9 years of data to calculate a 5-year rolling average for the last 4 years
        start_date = (datetime.now() - timedelta(days=365*9)).strftime('%Y-%m-%d')
        
        # 1. Fetch Data
        fed_funds = self.fetch_series("FEDFUNDS", start_date) # Monthly
        balance_sheet = self.fetch_series("WALCL", start_date) # Weekly

        # 2. Resample to Weekly (Friday) to align both
        # Forward fill monthly rates to weekly
        df = pd.DataFrame(index=balance_sheet.index)
        df = df.join(balance_sheet.rename(columns={'value': 'balance_sheet'}))
        df = df.join(fed_funds.rename(columns={'value': 'interest_rate'}).resample('W-FRI').ffill())
        
        df = df.ffill().dropna()

        # 3. Calculate Logic (Rolling Averages)
        # 5 Year Average for Rates (approx 260 weeks)
        df['avg_rate_5y'] = df['interest_rate'].rolling(window=260, min_periods=50).mean()
        
        # 1 Year Average for Balance Sheet (approx 52 weeks)
        df['avg_bs_1y'] = df['balance_sheet'].rolling(window=52, min_periods=20).mean()

        # 4. Determine Categories
        def classify(row):
            if pd.isna(row['avg_rate_5y']) or pd.isna(row['avg_bs_1y']):
                return 0, "Insuff Data"
            
            high_rate = row['interest_rate'] > row['avg_rate_5y']
            high_bs = row['balance_sheet'] > row['avg_bs_1y']

            if high_rate and high_bs: return 1, "High Rate / Liquidity Rising"
            if high_rate and not high_bs: return 2, "High Rate / Liquidity Falling"
            if not high_rate and high_bs: return 3, "Low Rate / Liquidity Rising"
            if not high_rate and not high_bs: return 4, "Low Rate / Liquidity Falling"

        df[['category_id', 'category_name']] = df.apply(
            lambda x: pd.Series(classify(x)), axis=1
        )

        # Trim to requested history (last 4 years)
        cutoff = datetime.now() - timedelta(days=365*YEARS_HISTORY)
        return df[df.index >= cutoff]

# ==========================================
# 2. STOCK DATA ENGINE (YFINANCE)
# ==========================================
class StockHistoricalEngine:
    def process_ticker(self, ticker, economic_df):
        try:
            stock = yf.Ticker(ticker)
            
            # 1. Get Price Data (Weekly)
            # We fetch 5y to ensure we have enough for the 4y window
            ohlc = stock.history(period="5y", interval="1wk")
            if ohlc.empty: return None
            
            # Clean columns
            ohlc = ohlc[['Close', 'Volume']].copy()
            ohlc.columns = ['price', 'volume']
            ohlc.index = ohlc.index.tz_localize(None) # Remove timezone for merging

            # 2. Get Fundamentals (Quarterly)
            # yfinance returns these with columns as dates
            fin = stock.quarterly_financials.T 
            bs = stock.quarterly_balance_sheet.T
            
            # Note: Free yfinance usually provides limited history (last 4-5 quarters).
            # We will calculate what we can and fill the rest.
            
            fund_data = pd.DataFrame(index=fin.index)
            
            # --- PROFITABILITY & MARGINS ---
            # Gross Margin = (Total Revenue - Cost of Revenue) / Total Revenue
            if 'Total Revenue' in fin.columns and 'Cost Of Revenue' in fin.columns:
                fund_data['gross_margin'] = (fin['Total Revenue'] - fin['Cost Of Revenue']) / fin['Total Revenue']
            elif 'Gross Profit' in fin.columns and 'Total Revenue' in fin.columns:
                 fund_data['gross_margin'] = fin['Gross Profit'] / fin['Total Revenue']
            
            # Operating Margin = Operating Income / Total Revenue
            if 'Operating Income' in fin.columns and 'Total Revenue' in fin.columns:
                fund_data['operating_margin'] = fin['Operating Income'] / fin['Total Revenue']
                
            # Net Margin = Net Income / Total Revenue
            if 'Net Income' in fin.columns and 'Total Revenue' in fin.columns:
                fund_data['net_margin'] = fin['Net Income'] / fin['Total Revenue']

            # --- GROWTH PROXIES ---
            # YoY Revenue Growth
            if 'Total Revenue' in fin.columns:
                fund_data['revenue_growth_yoy'] = fin['Total Revenue'].pct_change(periods=4) # 4 quarters ago

            # Ensure index is datetime
            fund_data.index = pd.to_datetime(fund_data.index).tz_localize(None)
            fund_data = fund_data.sort_index()

            # 3. Merge Price and Fundamentals
            # We reindex fundamentals to the weekly price index and forward fill
            # Meaning: The margins from Q1 report apply to all weeks until Q2 report comes out
            aligned_funds = fund_data.reindex(ohlc.index, method='ffill')
            
            df = ohlc.join(aligned_funds)

            # 4. Merge with Economic Data
            # Align economic data to stock dates
            df = df.join(economic_df[['category_id', 'category_name']])

            # 5. Calculate Weekly Returns
            df['weekly_return'] = df['price'].pct_change()
            
            df['ticker'] = ticker
            
            # Drop rows where we don't have an economic category (older than 4 years)
            df = df.dropna(subset=['category_id'])
            
            return df

        except Exception as e:
            # print(f"Error processing {ticker}: {e}") # Uncomment to debug
            return None

# ==========================================
# 3. ANALYSIS & REPORTING
# ==========================================
def generate_analysis():
    # 1. Prepare Economy
    fred = FredHistoricalEngine(FRED_API_KEY)
    economy_df = fred.get_economic_regime()
    
    if economy_df.empty:
        print("❌ Failed to generate economic data.")
        return

    print(f"✅ Economic Regime Built ({len(economy_df)} weeks)")
    print(economy_df['category_name'].value_counts())
    print("\n---------------------------------------------------\n")

    # 2. Process Stocks
    engine = StockHistoricalEngine()
    all_data = []

    print(f"📥 Processing {len(TICKERS)} stocks...")
    for t in TICKERS:
        df = engine.process_ticker(t, economy_df)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        print("❌ No stock data processed.")
        return

    master_df = pd.concat(all_data)
    
    # 3. Group by Regime and Calculate Metrics
    # We want to know: Under Regime X, what correlated with high returns?
    
    print("📊 ANALYSIS RESULTS BY REGIME")
    print("Metrics: Average Weekly Return per category based on Fundamental traits\n")

    categories = master_df['category_id'].unique()
    
    for cat_id in sorted(categories):
        if cat_id == 0: continue
        
        subset = master_df[master_df['category_id'] == cat_id]
        cat_name = subset['category_name'].iloc[0]
        weeks_count = subset.index.nunique()
        
        print(f"=== {cat_name} (n={weeks_count} weeks) ===")
        print(f"Avg Weekly Market Return: {subset['weekly_return'].mean()*100:.2f}%")
        
        # Split into High vs Low Margin companies for this period
        # Note: We use median of the *subset* to determine high/low relative to peers at that time
        
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
    print(f"✅ Full analysis saved to {filename}")

if __name__ == "__main__":
    generate_analysis()