import http.server
import socketserver
import re
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
WINDOW_SIZE = 6    # Max history depth
SIM_YEARS = 50
MAX_EDITS = 2      # Depth of Norvig edit search

# --- THE PROBABILISTIC ENGINE ---

class MarketCorrector:
    def __init__(self, prices):
        self.loc_stream, self.mom_stream = self.discretize(prices)
        
        # The 'Dictionary' (Counts of corrected historical sequences of varying lengths)
        self.loc_counts = collections.Counter()
        self.mom_counts = collections.Counter()
        
        # Transitions (What follows each sequence)
        self.loc_trans = collections.defaultdict(collections.Counter)
        self.mom_trans = collections.defaultdict(collections.Counter)
        
        self.train()

    def discretize(self, prices):
        locs, moms = [], []
        for i in range(1, len(prices)):
            p_idx = self.get_idx(prices[i-1])
            c_idx = self.get_idx(prices[i])
            locs.append(BUCKET_CHARS[c_idx])
            moms.append('U' if c_idx > p_idx else ('D' if c_idx < p_idx else 'F'))
        return locs, moms

    def get_idx(self, p):
        return max(0, min(NUM_BUCKETS - 1, int((p - MIN_PRICE) / BUCKET_SIZE)))

    def train(self):
        print(f"Training engine on sequences (1 to {WINDOW_SIZE} steps)...")
        time.sleep(0.1)
        
        # Train on multiple n-gram lengths to support variable input sizes
        for n in range(1, WINDOW_SIZE + 1):
            for i in range(len(self.loc_stream) - n):
                l_w = tuple(self.loc_stream[i:i+n])
                m_w = tuple(self.mom_stream[i:i+n])
                
                n_l = self.loc_stream[i+n]
                n_m = self.mom_stream[i+n]
                
                self.loc_counts[l_w] += 1
                self.mom_counts[m_w] += 1
                self.loc_trans[l_w][n_l] += 1
                self.mom_trans[m_w][n_m] += 1
            
        self.total_l = sum(self.loc_counts.values())
        self.total_m = sum(self.mom_counts.values())

    # --- Norvig Core ---

    def edits1(self, word, alphabet):
        splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [L + R[1:] for L, R in splits if R]
        transposes = [L + (R[1], R[0]) + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + (c,) + R[1:] for L, R in splits if R for c in alphabet]
        inserts    = [L + (c,) + R for L, R in splits for c in alphabet]
        return set(deletes + transposes + replaces + inserts)

    def candidates(self, word, counts, alphabet):
        """Finds 'Corrected' versions of the input pattern."""
        if word in counts:
            return set([word])
        
        current_edits = set([word])
        for _ in range(MAX_EDITS):
            next_generation = set()
            for w in current_edits:
                next_generation.update(self.edits1(w, alphabet))
            
            known_edits = set(e for e in next_generation if e in counts)
            if known_edits:
                return known_edits
            current_edits = next_generation
            
        return set([word])

    def solve(self, in_loc, in_mom):
        time.sleep(0.1)
        print(f"Correcting variable sequence: {''.join(in_loc)} | {''.join(in_mom)}")
        
        l_cand = self.candidates(tuple(in_loc), self.loc_counts, BUCKET_CHARS)
        m_cand = self.candidates(tuple(in_mom), self.mom_counts, ['U', 'D', 'F'])
        
        last_idx = BUCKET_CHARS.index(in_loc[-1])
        metrics = []

        for move, shift in [('U', 1), ('D', -1), ('F', 0)]:
            t_idx = last_idx + shift
            if 0 <= t_idx < NUM_BUCKETS:
                t_loc = BUCKET_CHARS[t_idx]
                
                # Probability = P(Pattern) * P(NextStep | Pattern)
                p_abc = sum((self.loc_counts[c]/self.total_l) * (self.loc_trans[c][t_loc]/sum(self.loc_trans[c].values())) 
                            for c in l_cand if sum(self.loc_trans[c].values()) > 0)
                
                p_udf = sum((self.mom_counts[c]/self.total_m) * (self.mom_trans[c][move]/sum(self.mom_trans[c].values())) 
                            for c in m_cand if sum(self.mom_trans[c].values()) > 0)
                
                metrics.append({
                    'target': t_loc, 'move': move,
                    'score': p_abc + p_udf,
                    'p_abc': p_abc, 'p_udf': p_udf
                })

        metrics.sort(key=lambda x: x['score'], reverse=True)
        return metrics

# --- DATA GENERATION ---
print(f"Simulating {SIM_YEARS} years of data...")
prices = [100000]
for _ in range(365 * SIM_YEARS):
    prices.append(max(MIN_PRICE, min(MAX_PRICE, prices[-1] + random.gauss(50, 2500))))
engine = MarketCorrector(prices)

# --- WEB SERVER ---

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(f"""
        <html><body style="font-family:sans-serif; max-width:600px; margin:40px auto; line-height:1.6">
            <h2>Market Spell Corrector</h2>
            <p>Input a sequence of any length (up to {WINDOW_SIZE}).</p>
            <form method="POST">
                Loc (A-T): <input name="l" value="KKK" style="text-transform:uppercase"><br>
                Mom (U/D/F): <input name="m" value="FF" style="text-transform:uppercase"><br><br>
                <button type="submit">Predict Completion</button>
            </form>
            <div style="font-size:0.8em; color:#888; margin-top:20px;">
                Alphabet: A-T (Price Buckets) | U=Up, D=Down, F=Flat
            </div>
        </body></html>
        """.encode())

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        data = parse_qs(self.rfile.read(length).decode())
        l_in = list(data.get('l', [''])[0].upper())
        m_in = list(data.get('m', [''])[0].upper())
        
        if not l_in or not m_in:
            self.send_error(400, "Empty Input")
            return

        results = engine.solve(l_in, m_in)
        
        rows = "".join([f"<tr><td>{r['target']}{r['move']}</td><td>{r['score']:.8f}</td><td>{r['p_abc']:.8f}</td><td>{r['p_udf']:.8f}</td></tr>" for r in results])
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(f"""
        <html><body style="font-family:sans-serif; max-width:600px; margin:40px auto;">
            <h2>Analysis Results</h2>
            <div style="background:#f4f4f4; padding:15px; border-radius:5px; border-left: 5px solid #007bff;">
                <b>Input Sequence:</b> {''.join(l_in)} | {''.join(m_in)}<br>
                <b>Most Likely Completion:</b> <span style="color:#007bff; font-weight:bold;">{results[0]['target']}</span>
            </div>
            <table border="1" style="width:100%; margin-top:20px; border-collapse:collapse" cellpadding="10">
                <thead><tr style="background:#eee"><th>Next Step</th><th>Total Score</th><th>P(ABC)</th><th>P(UDF)</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
            <br><a href="/" style="text-decoration:none; color:#007bff;">&larr; Try another pattern</a>
        </body></html>
        """.encode())

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Server started on port {PORT}")
        httpd.serve_forever()