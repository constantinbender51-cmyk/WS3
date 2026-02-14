import httpx
import time
from flask import Flask, jsonify

# Storage for results
SCRAPED_DATA = {}

# Subreddits to crawl
SUBS = ["CryptoCurrency", "Bitcoin", "ethereum", "WallStreetBets", "Solana"]

def fetch_all():
    print("🚀 Initializing data retrieval...")
    
    with httpx.Client(http2=True) as client:
        for sub in SUBS:
            # We fetch 5 posts ('limit=5')
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
                        {
                            "title": p['data'].get('title'),
                            "body": p['data'].get('selftext'), # This is the actual post content
                            "ups": p['data'].get('ups')
                        } for p in posts
                    ]
                    print(f"✅ Success: /r/{sub}")
                else:
                    print(f"❌ Blocked: /r/{sub} ({resp.status_code})")
                    SCRAPED_DATA[sub] = f"Error: {resp.status_code}"
            except Exception as e:
                print(f"⚠️ Failed /r/{sub}: {e}")
                SCRAPED_DATA[sub] = f"Failed: {str(e)}"
            
            # Enforce 6.5s delay to stay under 10 requests/min
            time.sleep(6.5)

    print("🏁 Data populated. Starting server...")

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "timestamp": time.ctime(),
        "data": SCRAPED_DATA
    })

if __name__ == '__main__':
    # Step 1: Fetch
    fetch_all()
    
    # Step 2: Serve on 0.0.0.0:8080
    app.run(host='0.0.0.0', port=8080, debug=False)
