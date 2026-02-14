import httpx
import time
from flask import Flask, jsonify

SCRAPED_DATA = {}
SUBS = ["CryptoCurrency", "Bitcoin", "ethereum", "WallStreetBets", "Solana"]

def fetch_max_data():
    print("🚀 Extracting max data (100 posts per sub)...")
    
    with httpx.Client(http2=True) as client:
        for sub in SUBS:
            # limit=100 is the absolute maximum for a single listing call
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit=100"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Accept": "application/json"
            }
            
            try:
                resp = client.get(url, headers=headers)
                if resp.status_code == 200:
                    raw_items = resp.json().get('data', {}).get('children', [])
                    
                    # Capture everything available in this single call
                    posts = []
                    for p in raw_items:
                        d = p['data']
                        if d.get('stickied'): continue # Skip the noise
                        
                        posts.append({
                            "title": d.get('title'),
                            "author": d.get('author'),
                            "ups": d.get('ups'),
                            "upvote_ratio": d.get('upvote_ratio'),
                            "num_comments": d.get('num_comments'),
                            "created_utc": d.get('created_utc'),
                            "body": d.get('selftext'),
                            "permalink": f"https://reddit.com{d.get('permalink')}",
                            "is_video": d.get('is_video'),
                            "over_18": d.get('over_18')
                        })
                    
                    SCRAPED_DATA[sub] = posts
                    print(f"✅ /r/{sub}: Indexed {len(posts)} items.")
                else:
                    print(f"❌ Error {resp.status_code} on /r/{sub}")
            except Exception as e:
                print(f"⚠️ Exception: {e}")
            
            # The 6-second "Safe Zone" delay
            time.sleep(6)

    print("🏁 Buffer full. Server live at http://0.0.0.0:8080")

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "engine": "Max-Extraction-v1",
        "timestamp": time.time(),
        "total_subreddits": len(SCRAPED_DATA),
        "payload": SCRAPED_DATA
    })

if __name__ == '__main__':
    fetch_max_data()
    app.run(host='0.0.0.0', port=8080, debug=False)
