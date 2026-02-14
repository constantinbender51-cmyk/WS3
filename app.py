import httpx
import time
import json
import os
import threading
from flask import Flask, jsonify

# Config
DATA_DIR = "/app/data"
DATA_FILE = os.path.join(DATA_DIR, "reddit_store.json")
SUBS = ["CryptoCurrency", "Bitcoin", "ethereum", "WallStreetBets", "Solana"]

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__)

def load_db():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except: return {}
    return {}

def save_db(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def fetch_worker():
    """Background process: Fetches 100 posts + 5 comments per post."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }
    
    while True:
        db = load_db()
        
        with httpx.Client(http2=True, headers=headers, timeout=30) as client:
            for sub in SUBS:
                if sub not in db: db[sub] = []
                
                print(f"📡 Scanning /r/{sub} listing...")
                try:
                    # Request top 100 posts
                    list_url = f"https://www.reddit.com/r/{sub}/hot.json?limit=100"
                    r = client.get(list_url)
                    time.sleep(6) # Rate limit safety
                    
                    if r.status_code != 200: continue
                    
                    listing = r.json().get('data', {}).get('children', [])
                    
                    for entry in listing:
                        p_data = entry['data']
                        p_id = p_data['id']
                        
                        # Skip if already in database
                        if any(item['id'] == p_id for item in db[sub]):
                            continue
                            
                        # STEP: Fetch the ACTUAL post content + comments
                        print(f"   📥 Deep fetching: {p_id} from /r/{sub}")
                        post_url = f"https://www.reddit.com{p_data['permalink']}.json"
                        p_res = client.get(post_url)
                        
                        if p_res.status_code == 200:
                            raw_payload = p_res.json()
                            # Index 0 = Post Data, Index 1 = Comments
                            full_post = raw_payload[0]['data']['children'][0]['data']
                            raw_comments = raw_payload[1]['data']['children']
                            
                            # Grab 5 real comments
                            final_comments = []
                            for c in raw_comments[:10]: # Check first 10 for 5 valid ones
                                if c['kind'] == 't1' and len(final_comments) < 5:
                                    final_comments.append({
                                        "author": c['data'].get('author'),
                                        "body": c['data'].get('body'),
                                        "score": c['data'].get('ups')
                                    })
                            
                            db[sub].append({
                                "id": p_id,
                                "title": full_post.get('title'),
                                "content": full_post.get('selftext') or "[Link/Media Post]",
                                "url": full_post.get('url'),
                                "author": full_post.get('author'),
                                "score": full_post.get('ups'),
                                "comments": final_comments,
                                "scraped_at": time.time()
                            })
                            save_db(db) # Save progress after every successful fetch
                        
                        # The 6-second law
                        time.sleep(6)
                        
                except Exception as e:
                    print(f"🚨 Error scraping /r/{sub}: {e}")
                    
        print("✅ Cycle complete. Sleeping for 1 hour...")
        time.sleep(3600)

@app.route('/')
def get_data():
    return jsonify(load_db())

if __name__ == '__main__':
    # Start background scraper
    thread = threading.Thread(target=fetch_worker, daemon=True)
    thread.start()
    
    # Start web server immediately
    print("🌍 Web server live at http://0.0.0.0:8080. Scraping in background...")
    app.run(host='0.0.0.0', port=8080, debug=False)
