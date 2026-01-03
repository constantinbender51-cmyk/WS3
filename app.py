import http.server
import socketserver
import json
import os
import time
import random
import collections
import string
from urllib.parse import parse_qs

# --- CONFIGURATION ---
PORT = int(os.environ.get('PORT', 8080))
MIN_PRICE = 0
MAX_PRICE = 200000
NUM_BUCKETS = 20
BUCKET_SIZE = (MAX_PRICE - MIN_PRICE) / NUM_BUCKETS
BUCKET_CHARS = string.ascii_uppercase[:NUM_BUCKETS] # A-T
MAX_EDITS = 2
DATA_POINTS = 500
TRAIN_SPLIT = 0.5

# --- DATA GENERATION ---
def generate_mock_data(n=DATA_POINTS):
    prices = [100000]
    for _ in range(n - 1):
        # Weighted random walk to create semi-predictable patterns
        change = random.gauss(150, 4000)
        new_price = max(MIN_PRICE + 5000, min(MAX_PRICE - 5000, prices[-1] + change))
        prices.append(new_price)
    return prices

# --- PROBABILISTIC ENGINE ---
class MarketCorrector:
    def __init__(self, training_prices):
        self.loc_stream, self.mom_stream = self.discretize(training_prices)
        self.loc_counts = collections.defaultdict(collections.Counter)
        self.mom_counts = collections.defaultdict(collections.Counter)
        self.loc_trans = collections.defaultdict(collections.Counter)
        self.mom_trans = collections.defaultdict(collections.Counter)
        self.vocab_loc = set()
        self.vocab_mom = set()
        self.train()

    def get_idx(self, p):
        return max(0, min(NUM_BUCKETS - 1, int((p - MIN_PRICE) / BUCKET_SIZE)))

    def discretize(self, prices):
        locs, moms = [], []
        for i in range(1, len(prices)):
            p_idx = self.get_idx(prices[i-1])
            c_idx = self.get_idx(prices[i])
            locs.append(BUCKET_CHARS[c_idx])
            moms.append('U' if c_idx > p_idx else ('D' if c_idx < p_idx else 'F'))
        return locs, moms

    def train(self):
        print(f"Training Dictionary on {len(self.loc_stream)} steps...")
        time.sleep(0.1)
        for n in range(2, 7): # Support words of length 2, 3, 4, 5, 6
            for i in range(len(self.loc_stream) - n):
                l_w = tuple(self.loc_stream[i:i+n])
                m_w = tuple(self.mom_stream[i:i+n])
                n_l, n_m = self.loc_stream[i+n], self.mom_stream[i+n]
                self.loc_counts[l_w][n_l] += 1
                self.mom_counts[m_w][n_m] += 1
                self.loc_trans[l_w][n_l] += 1
                self.mom_trans[m_w][n_m] += 1
                self.vocab_loc.add(l_w)
                self.vocab_mom.add(m_w)
        self.total_l = sum(sum(c.values()) for c in self.loc_counts.values())
        self.total_m = sum(sum(c.values()) for c in self.mom_counts.values())

    def edits1(self, word, alphabet):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        replaces = [L + (c,) + R[1:] for L, R in splits if R for c in alphabet]
        return set(replaces) # Focus on replaces for fixed-length financial "typos"

    def candidates(self, word, vocab, alphabet):
        if word in vocab: return {word}
        curr = {word}
        for _ in range(MAX_EDITS):
            nxt = set()
            for w in curr: nxt.update(self.edits1(w, alphabet))
            known = {e for e in nxt if e in vocab}
            if known: return known
            curr = nxt
        return {word}

    def solve(self, in_loc, in_mom):
        l_cand = self.candidates(tuple(in_loc), self.vocab_loc, BUCKET_CHARS)
        m_cand = self.candidates(tuple(in_mom), self.vocab_mom, ['U', 'D', 'F'])
        last_idx = BUCKET_CHARS.index(in_loc[-1])
        results = []
        for move, shift in [('U', 1), ('D', -1), ('F', 0)]:
            t_idx = max(0, min(NUM_BUCKETS - 1, last_idx + shift))
            t_loc = BUCKET_CHARS[t_idx]
            p_abc = sum((sum(self.loc_counts[c].values())/self.total_l) * (self.loc_trans[c][t_loc]/sum(self.loc_trans[c].values())) 
                        for c in l_cand if sum(self.loc_trans[c].values()) > 0)
            p_udf = sum((sum(self.mom_counts[c].values())/self.total_m) * (self.mom_trans[c][move]/sum(self.mom_trans[c].values())) 
                        for c in m_cand if sum(self.mom_trans[c].values()) > 0)
            results.append({'target': t_loc, 'move': move, 'score': p_abc + p_udf})
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[0]

# --- SERVER ---
prices = generate_mock_data()
split_pt = int(len(prices) * TRAIN_SPLIT)
train_prices = prices[:split_pt]
test_prices = prices[split_pt:]
engine = MarketCorrector(train_prices)

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/data':
            # Run batch predictions for the plot
            test_locs, test_moms = engine.discretize(test_prices)
            payload = {'actual': test_locs, 'predictions': {}}
            for length in range(2, 7):
                preds = [None] * length
                for i in range(len(test_locs) - length):
                    l_in, m_in = test_locs[i:i+length], test_moms[i:i+length]
                    res = engine.solve(l_in, m_in)
                    preds.append(res['target'])
                payload['predictions'][length] = preds
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
            return
            
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"""
<!DOCTYPE html>
<html>
<head>
    <title>Market Spell Corrector Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: sans-serif; background: #121212; color: #e0e0e0; margin: 20px; }
        .container { max-width: 1100px; margin: auto; background: #1e1e1e; padding: 20px; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
        h1 { color: #00e676; border-bottom: 1px solid #333; padding-bottom: 10px; }
        canvas { background: #1a1a1a; border-radius: 4px; margin-top: 20px; }
        .controls { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
        .stat-card { background: #2a2a2a; padding: 15px; border-radius: 6px; flex: 1; min-width: 150px; border-left: 4px solid #00e676; }
        .legend { font-size: 0.8em; color: #888; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Market Spell Corrector: Multi-Length Analysis</h1>
        <div class="controls">
            <div class="stat-card"><b>Status:</b> Online</div>
            <div class="stat-card"><b>Train Set:</b> 50%</div>
            <div class="stat-card"><b>Alphabet:</b> A-T (Buckets)</div>
            <div class="stat-card"><b>Max Edits:</b> 2</div>
        </div>
        <canvas id="marketChart" width="800" height="400"></canvas>
        <div class="legend">
            * Chart displays Absolute Price Buckets (0-19) over time in the Unseen Test Set.
            Predictions for lengths 2, 3, 4, 5, 6 are overlaid to show how context length affects accuracy.
        </div>
    </div>
    <script>
        const bucketToVal = b => "ABCDEFGHIJKLMNOPQRST".indexOf(b);
        async function loadChart() {
            const resp = await fetch('/data');
            const data = await resp.json();
            const labels = data.actual.map((_, i) => i);
            const datasets = [{
                label: 'Actual Market',
                data: data.actual.map(bucketToVal),
                borderColor: '#ffffff',
                borderWidth: 3,
                pointRadius: 0,
                fill: false,
                tension: 0.1
            }];
            const colors = ['#ff5252', '#ffeb3b', '#2196f3', '#e91e63', '#00e676'];
            Object.keys(data.predictions).forEach((len, i) => {
                datasets.push({
                    label: `Length ${len} Prediction`,
                    data: data.predictions[len].map(b => b ? bucketToVal(b) : null),
                    borderColor: colors[i],
                    borderDash: [5, 5],
                    borderWidth: 1.5,
                    pointRadius: 1,
                    fill: false
                });
            });
            new Chart(document.getElementById('marketChart'), {
                type: 'line',
                data: { labels, datasets },
                options: {
                    responsive: true,
                    scales: {
                        y: { title: { display: true, text: 'Price Bucket Index (A-T)' }, min: 0, max: 20, grid: { color: '#333' } },
                        x: { title: { display: true, text: 'Time Step (Test Set)' }, grid: { display: false } }
                    },
                    plugins: { legend: { position: 'bottom' } }
                }
            });
        }
        loadChart();
    </script>
</body>
</html>
        """)

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Server live on port {PORT}. Analyzing sequences...")
        httpd.serve_forever()