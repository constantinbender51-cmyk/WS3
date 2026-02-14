import httpx
import time
import json
import os
import threading
from flask import Flask, Response

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
        except:
            return {}
    return {}

def save_db(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def fetch_worker():
    """Background Deep Scraper"""
    headers = {"User-Agent": "Mozilla/5.0 Chrome/121.0.0.0", "Accept": "application/json"}
    while True:
        db = load_db()
        with httpx.Client(http2=True, headers=headers, timeout=30) as client:
            for sub in SUBS:
                if sub not in db: db[sub] = []
                try:
                    r = client.get(f"https://www.reddit.com/r/{sub}/hot.json?limit=100")
                    time.sleep(6.1)
                    if r.status_code != 200: continue
                    
                    listing = r.json().get('data', {}).get('children', [])
                    for entry in listing:
                        p_data = entry['data']
                        if any(item['id'] == p_data['id'] for item in db[sub]): continue
                        
                        # Deep fetch post content + 5 comments
                        p_res = client.get(f"https://www.reddit.com{p_data['permalink']}.json")
                        if p_res.status_code == 200:
                            raw = p_res.json()
                            post = raw[0]['data']['children'][0]['data']
                            coms = [{"u": c['data'].get('author'), "b": c['data'].get('body')} 
                                    for c in raw[1]['data']['children'][:5] if c['kind'] == 't1']
                            
                            db[sub].append({
                                "id": p_data['id'],
                                "title": post.get('title'),
                                "content": post.get('selftext') or "[Link]",
                                "comments": coms
                            })
                            save_db(db) # Save immediately
                        time.sleep(6.1)
                except Exception:
                    pass
        time.sleep(3600)

@app.route('/')
def index():
    # Force a fresh read from the disk
    db = load_db()
    
    # Calculate stats for the header
    p_count = sum(len(v) for v in db.values())
    c_count = sum(len(p.get('comments', [])) for v in db.values() for p in v)
    
    output = {
        "status": "RUNNING",
        "counts": {
            "total_posts": p_count,
            "total_comments": c_count
        },
        "database_location": DATA_FILE,
        "results": db
    }
    
    # Returning as a proper JSON response with indentation for readability
    return Response(
        json.dumps(output, indent=2),
        mimetype='application/json'
    )

if __name__ == '__main__':
    threading.Thread(target=fetch_worker, daemon=True).start()
    app.run(host='0.0.0.0', port=8080)
