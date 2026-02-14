import httpx
import time
import json
import os
import threading
from flask import Flask, jsonify

# File configuration
DATA_DIR = "/app/data"
DATA_FILE = os.path.join(DATA_DIR, "reddit_store.json")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

app = Flask(__name__)
SUBS = ["CryptoCurrency", "Bitcoin", "ethereum", "WallStreetBets", "Solana"]

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except: return {}
    return {}

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def fetch_worker():
    """Background worker to fetch data without blocking the web server."""
    while True:
        stored_data = load_data()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }

        with httpx.Client(http2=True, headers=headers, timeout=30) as client:
            for sub in SUBS:
                if sub not in stored_data:
                    stored_data[sub] = []
                
                print(f"🔍 Background: Checking /r/{sub}...")
                try:
                    list_url = f"https://www.reddit.com/r/{sub}/hot.json?limit=100"
                    list_resp = client.get(list_url)
                    time.sleep(6) # Rate limit safety
                    
                    if list_resp.status_code != 200: continue
                    
                    items = [i for i in list_resp.json()['data']['children'] if not i['data']['stickied']]
                    
                    for item in items:
                        post_id = item['data']['id']
                        if any(p['id'] == post_id for p in stored_data[sub]):
                            continue
                        
                        # Deep fetch post + 5 comments
                        post_url = f"https://www.reddit.com{item['data']['permalink']}.json"
                        post_resp = client.get(post_url)
                        
                        if post_resp.status_code == 200:
                            raw_json = post_resp.json()
                            post_info = raw_json[0]['data']['children'][0]['data']
                            comment_data = raw_json[1]['data']['children']
                            
                            comments = []
                            for c in comment_data[:5]:
                                if c['kind'] == 't1':
                                    comments.append({
                                        "user": c['data'].get('author'),
                                        "text": c['data'].get('body'),
                                        "ups": c['data'].get('ups')
                                    })
                            
                            stored_data[sub].append({
                                "id": post_id,
                                "title": post_info.get('title'),
                                "body": post_info.get('selftext'),
                                "ups": post_info.get('ups'),
                                "comments": comments,
                                "timestamp": time.time()
                            })
                            save_data(stored_data) # Save immediately so web view updates
                            print(f"   ✨ Added: [{post_id}] to /r/{sub}")
                        
                        time.sleep(6) # Essential 6s delay
                except Exception as e:
                    print(f"⚠️ Worker Error: {e}")
        
        print("😴 Cycle complete. Sleeping 1 hour before next full scan...")
        time.sleep(3600)

@app.route('/')
def home():
    # Always reads the latest version of the file
    return jsonify(load_data())

if __name__ == '__main__':
    # Start the scraper in a separate thread
    scraper_thread = threading.Thread(target=fetch_worker, daemon=True)
    scraper_thread.start()
    
    # Start the Flask server immediately on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
