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
DATA_POINTS = 600
TRAIN_SPLIT = 0.5
WORD_LENGTH = 4 # All "valid" market words are 4 tokens long

# --- DATA GENERATION ---
def generate_mock_data(n=DATA_POINTS):
    prices = [100000]
    for _ in range(n - 1):
        # Create patterns that repeat to give the "Dictionary" structure
        change = random.gauss(100, 3500)
        new_price = max(MIN_PRICE + 5000, min(MAX_PRICE - 5000, prices[-1] + change))
        prices.append(new_price)
    return prices

# --- THE NORVIG ENGINE ---
class MarketSpellCorrector:
    def __init__(self, training_prices):
        self.loc_stream, self.mom_stream = self.discretize(training_prices)
        
        # Dictionary of "Valid" 4-token words
        self.loc_dict = collections.Counter()
        self.mom_dict = collections.Counter()
        
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
        print(f"Building Dictionary of {WORD_LENGTH}-token words...")
        time.sleep(0.1)
        # We only store words of exactly WORD_LENGTH
        for i in range(len(self.loc_stream) - WORD_LENGTH + 1):
            l_word = tuple(self.loc_stream[i : i + WORD_LENGTH])
            m_word = tuple(self.mom_stream[i : i + WORD_LENGTH])
            self.loc_dict[l_word] += 1
            self.mom_dict[m_word] += 1

    def P(self, word, dictionary): 
        """Probability of a word based on dictionary frequency."""
        return dictionary[word] / sum(dictionary.values())

    def edits1(self, word, alphabet):
        """Standard Norvig edits: Deletes, Transposes, Replaces, Inserts."""
        splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [L + R[1:] for L, R in splits if R]
        transposes = [L + (R[1], R[0]) + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + (c,) + R[1:] for L, R in splits if R for c in alphabet]
        inserts    = [L + (c,) + R for L, R in splits for c in alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known(self, words, dictionary): 
        """Return the subset of words that actually exist in our market dictionary."""
        return set(w for w in words if w in dictionary)

    def candidates(self, word, dictionary, alphabet):
        """
        Norvig's logic: 
        1. If it's a known word, use it.
        2. Otherwise, look for known words at edit distance 1.
        3. Otherwise, look for known words at edit distance 2.
        4. Otherwise, return the word itself (even if unknown).
        """
        return (self.known([word], dictionary) or 
                self.known(self.edits1(word, alphabet), dictionary) or 
                self.known([e2 for e1 in self.edits1(word, alphabet) for e2 in self.edits1(e1, alphabet)], dictionary) or 
                [word])

    def correct(self, in_loc, in_mom):
        """
        Corrects/Completes the input pattern to the most likely 4-token word.
        """
        # Find candidate corrections for both streams
        c_locs = self.candidates(tuple(in_loc), self.loc_dict, BUCKET_CHARS)
        c_moms = self.candidates(tuple(in_mom), self.mom_dict, ['U', 'D', 'F'])
        
        # Among candidates, which one has the highest probability P(word)?
        # We sum ABC and UDF probabilities to find the most likely 'True' state
        
        # Scoring all possible 4-token combinations that could match these candidates
        best_word = None
        max_score = -1

        # To keep it efficient, we only check combinations of our best candidates
        for loc_w in c_locs:
            for mom_w in c_moms:
                # Basic Score: Freq(Loc_Word) + Freq(Mom_Word)
                score = self.loc_dict[loc_w] + self.mom_dict[mom_w]
                if score > max_score:
                    max_score = score
                    best_word = (loc_w, mom_w)
        
        return best_word # Returns (Location_Word, Momentum_Word)

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
    <title>Market Pattern Spell Corrector</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: sans-serif; background: #0f172a; color: #f8fafc; margin: 20px; }
        .container { max-width: 1100px; margin: auto; background: #1e293b; padding: 25px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        h1 { color: #38bdf8; border-bottom: 1px solid #334155; padding-bottom: 15px; margin-top: 0; }
        .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px; }
        .stat-card { background: #334155; padding: 15px; border-radius: 8px; border-top: 3px solid #38bdf8; }
        canvas { background: #0f172a; border-radius: 8px; padding: 10px; }
        .legend { font-size: 0.85em; color: #94a3b8; margin-top: 15px; line-height: 1.5; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Market Spell Corrector (Norvig Edition)</h1>
        <div class="stat-grid">
            <div class="stat-card"><b>Word Definition:</b> {{WORD_LENGTH}} Tokens</div>
            <div class="stat-card"><b>Input:</b> Sliding 3-step window</div>
            <div class="stat-card"><b>Correction:</b> Max Edit Distance 2</div>
            <div class="stat-card"><b>Alphabet:</b> A-T (Buckets) + UDF</div>
        </div>
        <canvas id="marketChart" width="800" height="400"></canvas>
        <div class="legend">
            <b>White Line:</b> Actual Market Buckets.<br>
            <b>Cyan Dotted:</b> The "Corrected" Completion. We feed the model 3 tokens (a "misspelled" 4-letter word); 
            it finds the most likely 4-letter completion from the historical dictionary.
        </div>
    </div>
    <script>
        const bucketToVal = b => "ABCDEFGHIJKLMNOPQRST".indexOf(b);
        async function loadChart() {
            const resp = await fetch('/data');
            const data = await resp.json();
            const labels = data.actual.map((_, i) => i);
            
            const datasets = [
                {
                    label: 'Actual Market',
                    data: data.actual.map(bucketToVal),
                    borderColor: '#ffffff',
                    borderWidth: 2.5,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.1
                },
                {
                    label: 'Spell-Corrected Completion',
                    data: data.prediction.map(b => b ? bucketToVal(b) : null),
                    borderColor: '#38bdf8',
                    borderDash: [4, 4],
                    borderWidth: 2,
                    pointRadius: 3,
                    pointBackgroundColor: '#38bdf8',
                    fill: false
                }
            ];

            new Chart(document.getElementById('marketChart'), {
                type: 'line',
                data: { labels, datasets },
                options: {
                    responsive: true,
                    scales: {
                        y: { 
                            title: { display: true, text: 'Price Bucket Index (A-T)', color: '#94a3b8' }, 
                            min: 0, max: 20, 
                            grid: { color: '#334155' },
                            ticks: { color: '#94a3b8' }
                        },
                        x: { 
                            title: { display: true, text: 'Time Step (Test Set)', color: '#94a3b8' }, 
                            grid: { display: false },
                            ticks: { color: '#94a3b8' }
                        }
                    },
                    plugins: { 
                        legend: { labels: { color: '#f8fafc' } } 
                    }
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
            # Run the Spell Corrector across the test set
            test_locs, test_moms = engine.discretize(test_prices)
            
            # We provide a 3-token input (a 'forgotten' letter typo)
            # The model completes it to the most likely 4-token word.
            preds = [None] * (WORD_LENGTH - 1)
            for i in range(len(test_locs) - (WORD_LENGTH - 1)):
                # This is our 'Misspelled' 3-letter word
                in_l = test_locs[i : i + (WORD_LENGTH - 1)]
                in_m = test_moms[i : i + (WORD_LENGTH - 1)]
                
                # Correct it to the best 4-letter word
                corrected_l, corrected_m = engine.correct(in_l, in_m)
                
                # The prediction is the 4th letter of the corrected word
                preds.append(corrected_l[-1])
            
            payload = {'actual': test_locs, 'prediction': preds}
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
            return
            
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        content = HTML_TEMPLATE.replace("{{WORD_LENGTH}}", str(WORD_LENGTH))
        self.wfile.write(content.encode())

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        time.sleep(0.1)
        print(f"Norvig Market Spell Corrector live on port {PORT}...")
        httpd.serve_forever()