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
    """Background process: 100 posts + 5 comments per post."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }
    
    while True:
        db = load_db()
        with httpx.Client(http2=True, headers=headers, timeout=30) as client:
            for sub in SUBS:
                if sub not in db: db[sub] = []
                
                try:
                    # Get top 100 listing
                    r = client.get(f"https://www.reddit.com/r/{sub}/hot.json?limit=100")
                    time.sleep(6) 
                    
                    if r.status_code != 200: continue
                    listing = r.json().get('data', {}).get('children', [])
                    
                    for entry in listing:
                        p_data = entry['data']
                        p_id = p_data['id']
                        
                        # Skip Duplicates
                        if any(item['id'] == p_id for item in db[sub]):
                            continue
                            
                        # Deep fetch for actual content + comments
                        post_url = f"https://www.reddit.com{p_data['permalink']}.json"
                        p_res = client.get(post_url)
                        
                        if p_res.status_code == 200:
                            raw_payload = p_res.json()
                            full_post = raw_payload[0]['data']['children'][0]['data']
                            raw_comments = raw_payload[1]['data']['children']
                            
                            comments = []
                            for c in raw_comments[:10]:
                                if c['kind'] == 't1' and len(comments) < 5:
                                    comments.append({
                                        "author": c['data'].get('author'),
                                        "body": c['data'].get('body')
                                    })
                            
                            db[sub].append({
                                "id": p_id,
                                "title": full_post.get('title'),
                                "content": full_post.get('selftext') or "[Link/Media]",
                                "comments": comments
                            })
                            save_db(db)
                        
                        time.sleep(6) # The mandatory 6s wait
                        
                except Exception as e:
                    print(f"Error: {e}")
        time.sleep(3600)

@app.route('/')
def get_data():
    db = load_db()
    
    # Calculate counts
    total_posts = sum(len(posts) for posts in db.values())
    total_comments = sum(
        len(post.get('comments', [])) 
        for posts in db.values() 
        for post in posts
    )
    
    return jsonify({
        "stats": {
            "total_posts_saved": total_posts,
            "total_comments_saved": total_comments,
            "subreddits_tracked": list(db.keys())
        },
        "data": db
    })

if __name__ == '__main__':
    thread = threading.Thread(target=fetch_worker, daemon=True)
    thread.start()
    app.run(host='0.0.0.0', port=8080)
