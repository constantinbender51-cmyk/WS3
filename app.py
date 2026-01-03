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
DATA_POINTS = 800
TRAIN_SPLIT = 0.5
MAX_PATTERN_LENGTH = 6 # The "Dictionary" learns chunks from length 1 to 6

# --- DATA GENERATION ---
def generate_mock_data(n=DATA_POINTS):
    prices = [100000]
    for _ in range(n - 1):
        # We simulate market "regimes" so certain chunks repeat more than others
        regime = random.random()
        if regime > 0.8: # Trending
            change = random.gauss(500, 1000)
        elif regime < 0.2: # Crashing
            change = random.gauss(-500, 1000)
        else: # Sideways
            change = random.gauss(0, 2000)
        
        new_price = max(MIN_PRICE + 5000, min(MAX_PRICE - 5000, prices[-1] + change))
        prices.append(new_price)
    return prices

# --- THE VARIABLE-LENGTH NORVIG ENGINE ---
class MarketSpellCorrector:
    def __init__(self, training_prices):
        self.loc_stream, self.mom_stream = self.discretize(training_prices)
        
        # Unified Dictionary of "Valid" market patterns of various lengths
        self.loc_dict = collections.Counter()
        self.mom_dict = collections.Counter()
        
        # Totals per length for normalization
        self.loc_len_counts = collections.Counter()
        self.mom_len_counts = collections.Counter()
        
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
        print(f"Learning multi-length pattern probabilities (1 to {MAX_PATTERN_LENGTH})...")
        time.sleep(0.1)
        # We learn "words" of every length from 1 to MAX_PATTERN_LENGTH
        for length in range(1, MAX_PATTERN_LENGTH + 1):
            for i in range(len(self.loc_stream) - length + 1):
                l_word = tuple(self.loc_stream[i : i + length])
                m_word = tuple(self.mom_stream[i : i + length])
                self.loc_dict[l_word] += 1
                self.mom_dict[m_word] += 1
        
        # Populate length-specific totals for normalization
        # This allows us to say "This is a very common Length-5 pattern" 
        # vs "This is a common Length-1 pattern" without the Length-1 winning by raw volume.
        for word, count in self.loc_dict.items():
            self.loc_len_counts[len(word)] += count
            
        for word, count in self.mom_dict.items():
            self.mom_len_counts[len(word)] += count

    def P(self, word, dictionary, len_counts): 
        """
        Normalized probability of a pattern appearing relative to other patterns 
        OF THE SAME LENGTH. This removes the bias towards very short patterns.
        """
        if not word in dictionary: return 0
        n = len(word)
        if n not in len_counts or len_counts[n] == 0: return 0
        return dictionary[word] / len_counts[n]

    def edits1(self, word, alphabet):
        """Standard Norvig edits. These naturally change the length of the word."""
        splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [L + R[1:] for L, R in splits if R]
        transposes = [L + (R[1], R[0]) + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + (c,) + R[1:] for L, R in splits if R for c in alphabet]
        inserts    = [L + (c,) + R for L, R in splits for c in alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known(self, words, dictionary): 
        return set(w for w in words if w in dictionary)

    def get_all_candidates(self, word, dictionary, alphabet):
        """
        Collects all possible corrections within MAX_EDITS.
        """
        candidates = self.known([word], dictionary)
        
        # Get edits 1
        e1 = self.edits1(word, alphabet)
        candidates.update(self.known(e1, dictionary))
        
        # Get edits 2
        if MAX_EDITS >= 2:
            e2 = {e2 for e1_word in e1 for e2 in self.edits1(e1_word, alphabet)}
            candidates.update(self.known(e2, dictionary))
            
        return candidates if candidates else {word}

    def correct(self, in_loc, in_mom):
        """
        Corrects the input pattern to the highest probability pattern in history,
        using length-normalized probability.
        """
        # Find all historical chunks that are "near" the input
        c_locs = self.get_all_candidates(tuple(in_loc), self.loc_dict, BUCKET_CHARS)
        c_moms = self.get_all_candidates(tuple(in_mom), self.mom_dict, ['U', 'D', 'F'])
        
        # Select the best based on Length-Normalized Probability
        # We pass the length counts (self.loc_len_counts) instead of a global sum
        best_loc = max(c_locs, key=lambda w: self.P(w, self.loc_dict, self.loc_len_counts))
        best_mom = max(c_moms, key=lambda w: self.P(w, self.mom_dict, self.mom_len_counts))
        
        return best_loc, best_mom

# --- SERVER ---
prices = generate_mock_data()
split_pt = int(len(prices) * TRAIN_SPLIT)
train_prices = prices[:split_pt]
test_prices = prices[split_pt:]
engine = MarketSpellCorrector(train_prices)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Variable-Length Market Spell Corrector</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #020617; color: #f8fafc; margin: 20px; }
        .container { max-width: 1200px; margin: auto; background: #0f172a; padding: 30px; border-radius: 16px; border: 1px solid #1e293b; }
        h1 { color: #22d3ee; margin-top: 0; font-weight: 300; letter-spacing: -1px; }
        .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: #1e293b; padding: 20px; border-radius: 12px; border-left: 4px solid #22d3ee; }
        canvas { background: #020617; border-radius: 12px; padding: 15px; }
        .info { font-size: 0.9em; color: #94a3b8; margin-top: 20px; background: #1e293b; padding: 15px; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Market Sequence Spell Corrector</h1>
        <div class="stat-grid">
            <div class="stat-card"><b>Dictionary Depth:</b> Chunks 1-{{MAX_LEN}}</div>
            <div class="stat-card"><b>Logic:</b> Length-Normalized Prob.</div>
            <div class="stat-card"><b>Edit Distance:</b> Max {{MAX_EDITS}}</div>
        </div>
        <canvas id="marketChart" width="800" height="400"></canvas>
        <div class="info">
            <b>Theory:</b> The market is a continuous stream of tokens. We identify the most probable "words" (ABC/UDF chunks) 
            regardless of length. <br><br>
            <b>Normalization Update:</b> We now compare candidates against others of the <i>same length</i>. 
            This prevents short, high-frequency noise (Length 1) from always overpowering rarer, but highly distinct, long-form patterns (Length 6).
        </div>
    </div>
    <script>
        const bucketToVal = b => "ABCDEFGHIJKLMNOPQRST".indexOf(b);
        async function loadChart() {
            const resp = await fetch('/data');
            const data = await resp.json();
            const labels = data.actual.map((_, i) => i);
            
            new Chart(document.getElementById('marketChart'), {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Actual Market',
                            data: data.actual.map(bucketToVal),
                            borderColor: 'rgba(255, 255, 255, 0.8)',
                            borderWidth: 2,
                            pointRadius: 0,
                            fill: false
                        },
                        {
                            label: 'Corrected Context (Last State)',
                            data: data.prediction.map(b => b ? bucketToVal(b) : null),
                            borderColor: '#22d3ee',
                            backgroundColor: '#22d3ee',
                            borderWidth: 0,
                            pointRadius: 4,
                            showLine: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { min: 0, max: 20, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
                        x: { display: false }
                    },
                    plugins: { legend: { labels: { color: '#f8fafc' } } }
                }
            });
        }
        loadChart();
    </script>
</body>
</html>
"""

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/data':
            test_locs, test_moms = engine.discretize(test_prices)
            
            # We slide a window and "correct" it.
            # Unlike fixed-length prediction, we are validating if the model 
            # recognizes the current state as part of a highly probable known chunk.
            preds = [None] * MAX_PATTERN_LENGTH
            for i in range(MAX_PATTERN_LENGTH, len(test_locs)):
                # We take a context window of MAX_PATTERN_LENGTH
                context_l = test_locs[i - MAX_PATTERN_LENGTH : i]
                context_m = test_moms[i - MAX_PATTERN_LENGTH : i]
                
                # Correct this chunk (could be shortened or changed)
                corr_l, corr_m = engine.correct(context_l, context_m)
                
                # We map the "corrected" last token of the chunk as our expectation
                preds.append(corr_l[-1])
            
            payload = {'actual': test_locs, 'prediction': preds}
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
            return
            
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        content = HTML_TEMPLATE.replace("{{MAX_LEN}}", str(MAX_PATTERN_LENGTH)).replace("{{MAX_EDITS}}", str(MAX_EDITS))
        self.wfile.write(content.encode())

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Variable-Length Spell Corrector live on port {PORT}...")
        httpd.serve_forever()