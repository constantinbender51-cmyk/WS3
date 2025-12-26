import os
import time
import threading
import requests
import pandas as pd
from datetime import timedelta
from flask import Flask, render_template
from io import StringIO

# ==========================================
# CONFIGURATION f
# ==========================================
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

BASE_URL_FRED = "https://api.stlouisfed.org/fred/series/observations"
BASE_URL_FINNHUB = "https://finnhub.io/api/v1"

# 500 = Fetch all. 40 = Fast boot for testing.
STOCK_FETCH_LIMIT = 500

APP_DATA = None
IS_READY = False

# ==========================================
# 1. FRED DATA ENGINE
# ==========================================
class FredEngine:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_series(self, series_id):
        if not self.api_key: return pd.DataFrame()
        
        params = {
            "series_id": series_id, "api_key": self.api_key,
            "file_type": "json", "sort_order": "asc"
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
        print("📊 Analyzing Economic Data...")
        df_rate = self.fetch_series("FEDFUNDS")
        df_bs = self.fetch_series("WALCL")

        if df_rate.empty or df_bs.empty: return None

        df_rate.rename(columns={'value': 'interest_rate'}, inplace=True)
        df_bs.rename(columns={'value': 'balance_sheet'}, inplace=True)

        # Merge Monthly Rate into Weekly Balance Sheet
        df_merged = pd.merge_asof(df_bs, df_rate, on='date', direction='backward')
        
        # Latest Data Point
        latest_date = df_merged['date'].max()
        current = df_merged.iloc[-1]

        # --- NEW LOGIC START ---
        
        # 1. Interest Rate Logic: Compare Current to 5-Year Average
        cutoff_5y = latest_date - timedelta(days=365 * 5)
        df_5y = df_merged[df_merged['date'] >= cutoff_5y].dropna()
        avg_rate_5y = df_5y['interest_rate'].mean()
        
        high_rate = current['interest_rate'] > avg_rate_5y

        # 2. Balance Sheet Logic: Compare Current to 1-Year Average
        cutoff_1y = latest_date - timedelta(days=365 * 1)
        df_1y = df_merged[df_merged['date'] >= cutoff_1y].dropna()
        avg_bs_1y = df_1y['balance_sheet'].mean()

        # "If bs is above 1 year average it's rising or high"
        high_bs = current['balance_sheet'] > avg_bs_1y

        # --- CATEGORY MAPPING ---
        
        if high_rate and high_bs:
            # High Rate / High BS
            cat_id, name = 1, "High Rate / High Liquidity"
            
        elif high_rate and not high_bs:
            # High Rate / Low BS
            cat_id, name = 2, "High Rate / Low Liquidity"
            
        elif not high_rate and high_bs:
            # Low Rate / High BS
            cat_id, name = 3, "Low Rate / High Liquidity"
            
        else: # not high_rate and not high_bs
            # Low Rate / Low BS
            cat_id, name = 4, "Low Rate / Low Liquidity"

        return {
            "category_id": cat_id, 
            "category_name": name,
            "current_rate": current['interest_rate'],
            "avg_rate_5y": avg_rate_5y,
            "current_bs": current['balance_sheet'],
            "avg_bs_1y": avg_bs_1y,
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
            tickers = [t.replace('.', '-') for t in df['Symbol'].tolist()]
            return tickers
        except Exception as e:
            print(f"Error getting tickers: {e}")
            return []

    def get_metrics(self, ticker):
        if not self.api_key: return None
        params = {'symbol': ticker, 'metric': 'all', 'token': self.api_key}
        
        try:
            response = requests.get(f"{BASE_URL_FINNHUB}/stock/metric", params=params)
            
            if response.status_code == 429:
                print(f"⚠️ Rate limit hit for {ticker}. Retrying in 30s...")
                time.sleep(30) 
                return self.get_metrics(ticker)

            if response.status_code != 200: return None
            data = response.json()
            m = data.get('metric', {})
            return {
                'Ticker': ticker,
                'PE': m.get('peBasicExclExtraTTM'),
                'Growth': m.get('epsGrowth5Y'),
                'Beta': m.get('beta'),
                'Price': m.get('52WeekHigh'),
                'Yield': m.get('dividendYieldIndicatedAnnual'),
            }
        except: return None

# ==========================================
# 3. BACKGROUND WORKER
# ==========================================
def run_analysis_logic():
    global APP_DATA, IS_READY
    print("🚀 Starting Background Analysis...")
    
    if not FRED_API_KEY or not FINNHUB_API_KEY:
        print("❌ API Keys missing.")
        return

    # 1. Economy
    fred = FredEngine(FRED_API_KEY)
    economy = fred.analyze_economy()
    if not economy: return

    # 2. Stocks
    stock_engine = StockEngine(FINNHUB_API_KEY)
    tickers = stock_engine.get_sp500_tickers()
    
    if STOCK_FETCH_LIMIT:
        tickers = tickers[:STOCK_FETCH_LIMIT]

    stock_data = []
    print(f"📥 Fetching {len(tickers)} stocks...")
    
    for i, t in enumerate(tickers):
        if i % 10 == 0: print(f"Processing {i}/{len(tickers)}...")
        m = stock_engine.get_metrics(t)
        if m: stock_data.append(m)
        time.sleep(1.1)

    df = pd.DataFrame(stock_data)

    # 3. Optimize based on logic
    cat_id = economy['category_id']
    
    # Pre-clean
    df = df.dropna(subset=['PE', 'Growth']).copy()
    df = df[df['PE'] > 0]
    
    # Calculate Custom Ratio: Growth / PE (High is better)
    df['Growth_PE_Ratio'] = df['Growth'] / df['PE']

    if cat_id == 3:
        # Low Rate / High BS (Rising) -> Best Growth
        df = df.sort_values(by='Growth', ascending=False)
        strategy_name = "Aggressive Growth"
        metric_used = "EPS Growth (5Y)"
        
    elif cat_id == 4:
        # Low Rate / Low BS (Falling) -> Growth / PE
        df = df.sort_values(by='Growth_PE_Ratio', ascending=False)
        strategy_name = "Growth at Reasonable Price"
        metric_used = "Growth / PE Ratio"
        
    elif cat_id == 1:
        # High Rate / High BS (Rising) -> Growth / PE
        df = df.sort_values(by='Growth_PE_Ratio', ascending=False)
        strategy_name = "Balanced Growth"
        metric_used = "Growth / PE Ratio"
        
    elif cat_id == 2:
        # High Rate / Low BS (Falling) -> Lowest P/E
        df = df.sort_values(by='PE', ascending=True)
        strategy_name = "Deep Value (Safety)"
        metric_used = "P/E Ratio (Lowest)"

    APP_DATA = {
        "economy": economy,
        "strategy": {"name": strategy_name, "metric": metric_used},
        "top_5": df.head(5).to_dict('records'),
        "bottom_5": df.tail(5).to_dict('records'),
        "full_table": df.to_dict('records'),
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    IS_READY = True
    print("✅ Analysis Complete.")

# ==========================================
# 4. SERVER
# ==========================================
app = Flask(__name__)

@app.route('/')
def dashboard():
    if not IS_READY or APP_DATA is None:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta http-equiv="refresh" content="30">
            <style>
                body { font-family: monospace; display: flex; justify-content: center; align-items: center; height: 100vh; background: #f4f4f4; }
                .box { text-align: center; border: 1px solid #000; padding: 40px; background: #fff; }
            </style>
        </head>
        <body>
            <div class="box">
                <h1>ANALYZING S&P 500</h1>
                <p>Fetching full dataset (500 stocks)...</p>
                <p>This process takes approx 8-10 minutes due to API limits.</p>
                <p><strong>This page will auto-refresh.</strong></p>
            </div>
        </body>
        </html>
        """
    return render_template('index.html', data=APP_DATA)

if __name__ == "__main__":
    t = threading.Thread(target=run_analysis_logic)
    t.daemon = True
    t.start()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)