import os
import time
import requests
import pandas as pd
from tqdm import tqdm # Professional progress bar
from io import StringIO # Added for robust string handling

# Configuration
API_KEY = "YOUR_FINNHUB_API_KEY"  # Replace this or set env variable
BASE_URL = "https://finnhub.io/api/v1"

def get_sp500_tickers():
    """
    Fetches the current S&P 500 tickers from Wikipedia.
    (Note: The S&P 500 constituent list is often a premium/paid feature on APIs 
    due to licensing, so scraping Wikipedia is the standard free 'hack').
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        # Wikipedia blocks simple scripts, so we need to set a User-Agent
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Check for HTTP errors
        
        # Use StringIO to avoid pandas FutureWarning about passing literal html strings
        tables = pd.read_html(StringIO(response.text))
        
        # The first table usually contains the tickers
        df = tables[0]
        # Wikipedia uses dots (BRK.B) but APIs usually use dashes (BRK-B) or just dots
        # Finnhub generally supports dots, but let's standardize if needed.
        tickers = df['Symbol'].tolist()
        print(f"✅ Successfully loaded {len(tickers)} tickers from S&P 500.")
        return tickers
    except Exception as e:
        print(f"❌ Error fetching tickers: {e}")
        return []

def get_financial_metrics(ticker, api_key):
    """
    Queries Finnhub's 'Basic Financials' endpoint for a specific stock.
    Returns a dictionary of key metrics.
    """
    endpoint = f"{BASE_URL}/stock/metric"
    params = {
        'symbol': ticker,
        'metric': 'all',
        'token': api_key
    }
    
    try:
        response = requests.get(endpoint, params=params)
        
        # Rate Limit Handling (HTTP 429)
        if response.status_code == 429:
            print(f"\n⚠️ Rate limit hit for {ticker}. Sleeping for 30s...")
            time.sleep(30)
            return get_financial_metrics(ticker, api_key) # Retry
            
        if response.status_code != 200:
            return None

        data = response.json()
        metrics = data.get('metric', {})
        
        # Return a simplified dictionary with the keys you care about
        # Finnhub keys can be cryptic; here we map them to readable names.
        return {
            'Ticker': ticker,
            'P/E (TTM)': metrics.get('peBasicExclExtraTTM'),
            'P/E (Normalized)': metrics.get('peNormalizedAnnual'),
            'EPS Growth (5Y)': metrics.get('epsGrowth5Y'),
            'Revenue Growth (3Y)': metrics.get('revenueGrowth3Y'),
            'Beta': metrics.get('beta'),
            '52W High': metrics.get('52WeekHigh'),
            '52W Low': metrics.get('52WeekLow'),
            'Div Yield (%)': metrics.get('dividendYieldIndicatedAnnual'),
        }

    except Exception as e:
        return None

def main():
    # 1. Setup
    if API_KEY == "YOUR_FINNHUB_API_KEY":
        print("⚠️ Please edit the script and add your Finnhub API Key.")
        return

    # 2. Get Tickers
    tickers = get_sp500_tickers()
    if not tickers:
        return

    results = []
    
    # 3. Fetch Data with Progress Bar
    print("🚀 Starting API calls to Finnhub...")
    
    # We use tqdm to show a nice progress bar
    for ticker in tqdm(tickers):
        data = get_financial_metrics(ticker, API_KEY)
        if data:
            results.append(data)
        
        # Finnhub Free Tier Limit: 60 calls / minute
        # We sleep 1.1 seconds to stay safely under the limit (approx 55 calls/min)
        time.sleep(1.1)

    # 4. Save to CSV
    df = pd.DataFrame(results)
    
    # Reorder columns for readability
    cols = ['Ticker', 'P/E (TTM)', 'EPS Growth (5Y)', 'Revenue Growth (3Y)', 
            'Beta', 'Div Yield (%)', '52W High', '52W Low']
    
    # Filter only columns that exist in our data
    final_cols = [c for c in cols if c in df.columns]
    df = df[final_cols]
    
    output_file = 'sp500_finnhub_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Done! Data for {len(df)} stocks saved to '{output_file}'")
    
    # Optional: Print the top 5 'Undervalued' Growth stocks (Low PE, High Growth)
    print("\n--- Top 5 Potential 'Growth at Value' Candidates ---")
    try:
        # Filter: Positive Growth & Positive PE
        filtered = df[(df['P/E (TTM)'] > 0) & (df['EPS Growth (5Y)'] > 10)]
        # Sort by P/E ascending
        print(filtered.sort_values(by='P/E (TTM)').head(5).to_string(index=False))
    except:
        pass

if __name__ == "__main__":
    main()