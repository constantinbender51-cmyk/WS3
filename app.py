import os
import time
import threading
import requests
import pandas as pd
import webbrowser
from datetime import timedelta
from flask import Flask, render_template, jsonify
from io import StringIO

# ==========================================
# CONFIGURATION
# ==========================================
# ---------------------------------------------------------
# ⚠️ INSERT YOUR API KEYS HERE
# ---------------------------------------------------------
FRED_API_KEY = "8005f92c424c0503df32084af3e66daf"        # Get from: https://fred.stlouisfed.org/docs/api/api_key.html
FINNHUB_API_KEY = "d4g6br9r01qm5b344ro0d4g6br9r01qm5b344rog"     # Get from: https://finnhub.io/
# ---------------------------------------------------------

# LIMIT for testing (Set to None to fetch ALL S&P 500 stocks)
# Fetching 500 stocks takes ~10 mins on free tier (60 calls/min)
STOCK_FETCH_LIMIT = 50 

BASE_URL_FRED = "https://api.stlouisfed.org/fred/series/observations"
BASE_URL_FINNHUB = "https://finnhub.io/api/v1"

# ==========================================
# 1. FRED DATA ENGINE
# ==========================================
class FredEngine:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_series(self, series_id):
        """Fetches data from FRED and returns a clean DataFrame."""
        if not self.api_key: return pd.DataFrame()
        
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "asc"
        }
        try:
            response = requests.get(BASE_URL_FRED, params=params)
            response.raise_for_status()
            data = response.json().get("observations", [])
            
            df = pd.DataFrame(data)
            if df.empty: return pd.DataFrame()

            df = df[df['value'] != '.'] 
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'])
            return df[['date', 'value']].sort_values('date')
        except Exception as e:
            print(f"Error fetching {series_id}: {e}")
            return pd.DataFrame()

    def analyze_economy(self):
        """Determines the current economic category."""
        print("📊 Analyzing Economic Data...")
        df_rate = self.fetch_series("FEDFUNDS")
        df_bs = self.fetch_series("WALCL")

        if df_rate.empty or df_bs.empty:
            return None

        df_rate.rename(columns={'value': 'interest_rate'}, inplace=True)
        df_bs.rename(columns={'value': 'balance_sheet'}, inplace=True)

        # Merge Monthly Rate into Weekly Balance Sheet
        df_merged = pd.merge_asof(
            df_bs, 
            df_rate, 
            on='date', 
            direction='backward'
        )

        df_merged['bs_change'] = df_merged['balance_sheet'].diff()
        
        # Filter Last 10 Years
        latest_date = df_merged['date'].max()
        cutoff_date = latest_date - timedelta(days=365 * 10)
        df_10y = df_merged[df_merged['date'] >= cutoff_date].dropna().copy()

        avg_rate_10y = df_10y['interest_rate'].mean()
        
        # Determine Current Status (using the latest data point)
        current = df_10y.iloc[-1]
        
        high_rate = current['interest_rate'] > avg_rate_10y
        growing_bs = current['bs_change'] >= 0
        
        if high_rate and growing_bs:
            category_id = 1
            category_name = "High Rate / Growing BS"
        elif high_rate and not growing_bs:
            category_id = 2
            category_name = "High Rate / Shrinking BS"
        elif not high_rate and growing_bs:
            category_id = 3
            category_name = "Low Rate / Growing BS"
        else: # not high_rate and not growing_bs
            category_id = 4
            category_name = "Low Rate / Shrinking BS"

        return {
            "category_id": category_id,
            "category_name": category_name,
            "current_rate": current['interest_rate'],
            "avg_rate_10y": avg_rate_10y,
            "bs_change": current['bs_change'],
            "date": current['date']
        }

# ==========================================
# 2. FINNHUB DATA ENGINE
# ==========================================
class StockEngine:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_sp500_tickers(self):
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            tables = pd.read_html(StringIO(response.text))
            df = tables[0]
            tickers = df['Symbol'].tolist()
            # Clean tickers (Wikipedia uses '.' but Finnhub often prefers '-')
            tickers = [t.replace('.', '-') for t in tickers]
            return tickers
        except Exception as e:
            print(f"Error getting tickers: {e}")
            return []

    def get_metrics(self, ticker):
        if not self.api_key: return None
        endpoint = f"{BASE_URL_FINNHUB}/stock/metric"
        params = {'symbol': ticker, 'metric': 'all', 'token': self.api_key}
        
        try:
            response = requests.get(endpoint, params=params)
            if response.status_code == 429:
                print(f"⚠️ Rate limit. Sleeping...")
                time.sleep(30)
                return self.get_metrics(ticker)
                
            if response.status_code != 200: return None

            data = response.json()
            metrics = data.get('metric', {})
            
            return {
                'Ticker': ticker,
                'PE': metrics.get('peBasicExclExtraTTM'),
                'Growth': metrics.get('epsGrowth5Y'), # Using 5Y EPS Growth
                'Beta': metrics.get('beta'),
                'Price': metrics.get('52WeekHigh'), # Approx price (using high for ref)
                'Yield': metrics.get('dividendYieldIndicatedAnnual'),
            }
        except:
            return None

# ==========================================
# 3. OPTIMIZER ENGINE
# ==========================================
def optimize_portfolio(df, category_id):
    """
    Applies the selection logic based on the user's prompt.
    
    Logic:
    - Cat 3 (Low Rate, BS Rising): Pick best Growth.
    - Cat 4 (Low Rate, BS Falling): Pick best Growth / PE.
    - Cat 1 (High Rate, BS Rising): Pick best Growth / PE.
    - Cat 2 (High Rate, BS Falling): Pick best PE.
    """
    df = df.dropna(subset=['PE', 'Growth']).copy()
    
    # Clean data: Ensure PE is positive for Ratio calculations to make sense
    df = df[df['PE'] > 0] 

    # Calculate the custom metric
    df['Growth_PE_Ratio'] = df['Growth'] / df['PE']

    if category_id == 3:
        # Strategy: Best Growth
        # Sort by Growth Descending
        df = df.sort_values(by='Growth', ascending=False)
        strategy_name = "Aggressive Growth (Market Liquidity High)"
        metric_used = "EPS Growth (5Y)"

    elif category_id == 4:
        # Strategy: Growth / PE
        # Sort by Ratio Descending
        df = df.sort_values(by='Growth_PE_Ratio', ascending=False)
        strategy_name = "Growth at Reasonable Price (Liquidity Drying)"
        metric_used = "Growth / PE Ratio"

    elif category_id == 1:
        # Strategy: Growth / PE
        # Sort by Ratio Descending
        df = df.sort_values(by='Growth_PE_Ratio', ascending=False)
        strategy_name = "Balanced Growth (High Rates, High Liquidity)"
        metric_used = "Growth / PE Ratio"

    elif category_id == 2:
        # Strategy: Best PE (Lowest PE)
        # Sort by PE Ascending
        df = df.sort_values(by='PE', ascending=True)
        strategy_name = "Deep Value (Tight Financial Conditions)"
        metric_used = "P/E Ratio"
        
    else:
        # Fallback
        df = df.sort_values(by='PE', ascending=True)
        strategy_name = "Unknown"
        metric_used = "P/E"

    return df, strategy_name, metric_used

# ==========================================
# 4. WEB SERVER (FLASK)
# ==========================================
app = Flask(__name__)
app_data = {} # Global store for the latest analysis

@app.route('/')
def dashboard():
    return render_template('index.html', data=app_data)

def run_flask():
    print("🚀 Web Server started on http://127.0.0.1:5000")
    # Disable reloader to prevent main thread interference in this script setup
    app.run(port=5000, debug=False, use_reloader=False)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("="*60)
    print(" MARKET OPTIMIZER ENGINE ")
    print("="*60)

    # 1. Check Keys
    if not FRED_API_KEY or not FINNHUB_API_KEY:
        print("❌ ERROR: API Keys missing. Please edit app.py and add your keys.")
        return

    # 2. Analyze Economy
    fred = FredEngine(FRED_API_KEY)
    economy = fred.analyze_economy()
    
    if not economy:
        print("❌ Failed to fetch Economic Data.")
        return

    print(f"✅ Economic Status: {economy['category_name']}")
    print(f"   (Rate: {economy['current_rate']}%, Avg: {economy['avg_rate_10y']:.2f}%)")

    # 3. Fetch Stocks
    print("\n📥 Fetching S&P 500 Data (This takes time due to rate limits)...")
    stock_engine = StockEngine(FINNHUB_API_KEY)
    tickers = stock_engine.get_sp500_tickers()
    
    if STOCK_FETCH_LIMIT:
        print(f"⚠️ TEST MODE: Limiting to first {STOCK_FETCH_LIMIT} tickers.")
        tickers = tickers[:STOCK_FETCH_LIMIT]

    stock_data = []
    # Using simple loop to respect rate limits (60/min)
    for i, t in enumerate(tickers):
        print(f"\r   Processing {i+1}/{len(tickers)}: {t}", end="")
        m = stock_engine.get_metrics(t)
        if m: stock_data.append(m)
        time.sleep(1.05) # Sleep to stay under 60 calls/min

    df_stocks = pd.DataFrame(stock_data)
    print("\n✅ Stock Data Fetched.")

    # 4. Optimize
    ranked_df, strategy, metric = optimize_portfolio(df_stocks, economy['category_id'])

    # 5. Prepare Data for UI
    top_5 = ranked_df.head(5).to_dict('records')
    bottom_5 = ranked_df.tail(5).to_dict('records') # Worst matches for the strategy
    
    # Convert full DF to dict for table
    full_table = ranked_df.to_dict('records')

    # Update Global Data Store
    global app_data
    app_data = {
        "economy": economy,
        "strategy": {
            "name": strategy,
            "metric": metric
        },
        "top_5": top_5,
        "bottom_5": bottom_5,
        "full_table": full_table,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 6. Start Server in Thread
    t = threading.Thread(target=run_flask)
    t.daemon = True
    t.start()

    # Open Browser
    time.sleep(1) # Give server a second to warm up
    webbrowser.open("http://127.0.0.1:5000")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == "__main__":
    main() 