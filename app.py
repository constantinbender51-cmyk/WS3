from flask import Flask, jsonify
import httpx
import time

app = Flask(__name__)

# Target subreddits (5 max as requested)
SUBS = ["CryptoCurrency", "Bitcoin", "ethereum", "WallStreetBets", "Solana"]

def get_reddit_data():
    results = {}
    # HTTP/2 is necessary to bypass basic 2026 fingerprinting
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
                    results[sub] = [
                        {"title": p['data']['title'], "ups": p['data']['ups']} 
                        for p in posts
                    ]
                else:
                    results[sub] = f"Error: {resp.status_code}"
            except Exception as e:
                results[sub] = f"Failed: {str(e)}"
            
            # Rate limit safety: 10 requests per minute = 1 every 6 seconds
            time.sleep(6.5) 
            
    return results

@app.route('/')
def home():
    data = get_reddit_data()
    return jsonify(data)

if __name__ == '__main__':
    # Start server on port 8080
    app.run(port=8080, debug=False host='0.0.0.0')
