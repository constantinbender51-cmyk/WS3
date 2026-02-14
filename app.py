import httpx
import time
from flask import Flask, jsonify

SCRAPED_DATA = {}
SUBS = ["CryptoCurrency", "Bitcoin", "ethereum", "WallStreetBets", "Solana"]

def fetch_all():
    print("🚀 Scraping all non-stickied posts from top 15 results...")
    
    with httpx.Client(http2=True) as client:
        for sub in SUBS:
            # Fetching 15 items
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit=15"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Accept": "application/json"
            }
            
            try:
                resp = client.get(url, headers=headers)
                if resp.status_code == 200:
                    raw_posts = resp.json().get('data', {}).get('children', [])
                    
                    # Filtering stickies but keeping ALL other results from the 15
                    clean_posts = [
                        {
                            "title": p['data'].get('title'),
                            "body": p['data'].get('selftext'),
                            "ups": p['data'].get('ups'),
                            "author": p['data'].get('author')
                        } 
                        for p in raw_posts if not p['data'].get('stickied')
                    ]
                    
                    SCRAPED_DATA[sub] = clean_posts
                    print(f"✅ /r/{sub}: Captured {len(clean_posts)} posts.")
                else:
                    print(f"❌ Error {resp.status_code} on /r/{sub}")
                    SCRAPED_DATA[sub] = f"Error: {resp.status_code}"
            except Exception as e:
                print(f"⚠️ Exception on /r/{sub}: {e}")
                SCRAPED_DATA[sub] = f"Failed: {str(e)}"
            
            # Your requested 6-second delay
            print("...waiting 6 seconds...")
            time.sleep(6)

    print("🏁 Fetch complete. Launching server on http://0.0.0.0:8080")

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "info": "Reddit Scrape - All non-stickied posts from top 15",
        "timestamp": time.ctime(),
        "results": SCRAPED_DATA
    })

if __name__ == '__main__':
    # Initial Fetch
    fetch_all()
    
    # Run server
    app.run(host='0.0.0.0', port=8080, debug=False)
