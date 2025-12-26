import os
import time
import threading
import requests
import pandas as pd
from datetime import timedelta
from flask import Flask, render_template
from io import StringIO

# ==========================================
# CONFIGURATION
# ==========================================
# Try to get keys from Environment (Railway), otherwise use empty string
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

BASE_URL_FRED = "https://api.stlouisfed.org/fred/series/observations"
BASE_URL_FINNHUB = "https://finnhub.io/api/v1"

# Limit to first 40 stocks to prevent Railway timeouts during boot
# Set to None or 500 if you have a paid plan or longer timeout
STOCK_FETCH_LIMIT = 40

# Global storage for the app
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

        df_merged = pd.merge_asof(df_bs, df_rate, on='date', direction='backward')
        df_merged['bs_change'] = df_merged['balance_sheet'].diff()
        
        latest_date = df_merged['date'].max()
        cutoff_date = latest_date - timedelta(days=365 * 10)
        df_10y = df_merged[df_merged['date'] >= cutoff_date].dropna().copy()

        avg_rate_10y = df_10y['interest_rate'].mean()
        current = df_10y.iloc[-1]
        
        high_rate = current['interest_rate'] > avg_rate_10y
        growing_bs = current['bs_change'] >= 0
        
        if high_rate and growing_bs:
            cat_id, name = 1, "High Rate / Growing BS"
        elif high_rate and not growing_bs:
            cat_id, name = 2, "High Rate / Shrinking BS"
        elif not high_rate and growing_bs:
            cat_id, name = 3, "Low Rate / Growing BS"
        else:
            cat_id, name = 4, "Low Rate / Shrinking BS"

        return {
            "category_id": cat_id, "category_name": name,
            "current_rate": current['interest_rate'],
            "avg_rate_10y": avg_rate_10y, "bs_change": current['bs_change'],
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
                time.sleep(1) # Quick sleep for rate limit
                return None
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
# 3. ANALYSIS THREAD
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
    for t in tickers:
        m = stock_engine.get_metrics(t)
        if m: stock_data.append(m)
        time.sleep(0.1) # Slight delay to be polite

    df = pd.DataFrame(stock_data)

    # 3. Optimize
    cat_id = economy['category_id']
    df = df.dropna(subset=['PE', 'Growth']).copy()
    df = df[df['PE'] > 0]
    df['Growth_PE_Ratio'] = df['Growth'] / df['PE']

    if cat_id == 3:
        df = df.sort_values(by='Growth', ascending=False)
        strategy, metric = "Aggressive Growth", "EPS Growth (5Y)"
    elif cat_id == 4:
        df = df.sort_values(by='Growth_PE_Ratio', ascending=False)
        strategy, metric = "Growth at Reasonable Price", "Growth / PE Ratio"
    elif cat_id == 1:
        df = df.sort_values(by='Growth_PE_Ratio', ascending=False)
        strategy, metric = "Balanced Growth", "Growth / PE Ratio"
    else:
        df = df.sort_values(by='PE', ascending=True)
        strategy, metric = "Deep Value", "P/E Ratio"

    APP_DATA = {
        "economy": economy,
        "strategy": {"name": strategy, "metric": metric},
        "top_5": df.head(5).to_dict('records'),
        "bottom_5": df.tail(5).to_dict('records'),
        "full_table": df.to_dict('records'),
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    IS_READY = True
    print("✅ Analysis Complete.")

# ==========================================
# 4. WEB SERVER
# ==========================================
app = Flask(__name__)

@app.route('/')
def dashboard():
    # If analysis isn't done yet, show a loading message
    if not IS_READY or APP_DATA is None:
        return """
        <div style="font-family: monospace; padding: 50px; text-align: center;">
            <h1>ANALYZING MARKET DATA...</h1>
            <p>Please refresh this page in 30-60 seconds.</p>
            <p>Fetching S&P 500 metrics and FRED data.</p>
        </div>
        """
    return render_template('index.html', data=APP_DATA)

if __name__ == "__main__":
    # Start analysis in a background thread so the server boots immediately
    # This prevents Railway from timing out while we fetch stocks
    t = threading.Thread(target=run_analysis_logic)
    t.daemon = True
    t.start()

    # Get PORT from Railway environment variable
    port = int(os.environ.get("PORT", 5000))
    
    # ⚠️ Host set to 0.0.0.0 as requested
    print(f"🚀 Starting Flask on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port)