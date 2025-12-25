import yfinance as yf
import pandas as pd
import time

def get_sp500_data():
    # 1. Get the list of S&P 500 tickers from Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    sp500_tickers = tables[0]['Symbol'].tolist()

    # 2. Prepare a list to store data
    stock_data = []

    print(f"Fetching data for {len(sp500_tickers)} stocks... (This may take a few minutes)")

    # 3. Loop through tickers and fetch data
    # Note: We use a small subset ([:10]) for testing. Remove '[:10]' to get all 500.
    for ticker in sp500_tickers[:10]: 
        try:
            # Handle tickers with dots (e.g., BRK.B -> BRK-B)
            ticker_obj = yf.Ticker(ticker.replace('.', '-'))
            info = ticker_obj.info
            
            stock_data.append({
                'Ticker': ticker,
                'Name': info.get('shortName'),
                'Price': info.get('currentPrice'),
                'P/E Ratio': info.get('trailingPE'),
                'Forward P/E': info.get('forwardPE'),
                'PEG Ratio': info.get('pegRatio'), # Great indicator combining P/E and Growth
                'Earnings Growth': info.get('earningsGrowth'), # YoY growth
                'Revenue Growth': info.get('revenueGrowth')
            })
            print(f"Fetched: {ticker}")
            
            # Be polite to the API to avoid rate limits
            time.sleep(0.2) 
            
        except Exception as e:
            print(f"Could not fetch {ticker}: {e}")

    # 4. Create DataFrame and save
    df = pd.DataFrame(stock_data)
    df.to_csv('sp500_pe_growth.csv', index=False)
    print("Done! Data saved to sp500_pe_growth.csv")
    return df

# Run the function
if __name__ == "__main__":
    get_sp500_data()
