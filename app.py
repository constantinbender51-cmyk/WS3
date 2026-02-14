import httpx
import time
from flask import Flask, jsonify

# Global cache
SCRAPED_DATA = {}

SUBS = ["CryptoCurrency", "Bitcoin", "ethereum", "WallStreetBets", "Solana"]

def fetch_all():
    print("Starting pre-fetch...")
    with httpx.Client(http2=True) as client:
        for sub in SUBS:
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit=5"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Accept": "application/json"
            }
            try:
                resp = client.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    posts = data.get('data', {}).get('children', [])
                    SCRAPED_DATA[sub] = [
                        {"title": p['data']['title'], "ups": p['data']['ups']} 
                        for p in posts
                    ]
                    print(f"Fetched /r/{sub}")
                else:
                    SCRAPED_DATA[sub] = f"Error: {resp.status_code}"
            except Exception as e:
                SCRAPED_DATA[sub] = f"Failed: {str(e)}"
            
            # Rate limit compliance
            time.sleep(6.5)
    print("Pre-fetch complete.")

# Initialize Flask
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify(SCRAPED_DATA)

if __name__ == '__main__':
    # 1. Fetch data first
    fetch_all()
    
    # 2. Start server only after data is populated
    app.run(host='0.0.0.0', port=8080, debug=False)
