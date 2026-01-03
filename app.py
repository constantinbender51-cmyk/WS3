import http.server
import socketserver
import re
import urllib.request
import os
import ssl
from collections import Counter
from urllib.parse import parse_qs

PORT = int(os.environ.get('PORT', 8080))
CORPUS_URL = "https://norvig.com/big.txt"
CORPUS_FILE = "big.txt"

class SpellingModel:
    def __init__(self):
        self.WORDS = Counter()
        self.N = 0
        self.download_corpus()
        self.train()

    def download_corpus(self):
        if not os.path.exists(CORPUS_FILE):
            print(f"Downloading {CORPUS_URL}...")
            # SSL context needed for some environments to trust norvig.com
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            try:
                with urllib.request.urlopen(CORPUS_URL, context=ctx) as response:
                    with open(CORPUS_FILE, 'wb') as f:
                        f.write(response.read())
            except Exception as e:
                print(f"Error downloading corpus: {e}")

    def train(self):
        if os.path.exists(CORPUS_FILE):
            with open(CORPUS_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().lower()
                self.WORDS = Counter(re.findall(r'\w+', text))
                self.N = sum(self.WORDS.values())
        print(f"Model trained. N={self.N}")

    def P(self, word): 
        return self.WORDS[word] / self.N

    def correction(self, word): 
        return max(self.candidates(word), key=self.P)

    def candidates(self, word): 
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words): 
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

model = SpellingModel()

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"""
            <html>
                <body>
                    <h1>Spelling Corrector</h1>
                    <form method="POST">
                        <textarea name="text" style="width:100%; height:100px"></textarea><br>
                        <button type="submit">Correct</button>
                    </form>
                </body>
            </html>
        """)

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        data = parse_qs(self.rfile.read(length).decode('utf-8'))
        text = data.get('text', [''])[0]
        
        # Tokenize preserving structure
        tokens = re.split(r'(\W+)', text)
        res_text = []
        metrics = []

        for token in tokens:
            if token.strip() and re.match(r'\w+', token):
                lower_token = token.lower()
                candidates = model.candidates(lower_token)
                best = max(candidates, key=model.P)
                
                # Case restoration
                final_word = best.title() if token.istitle() else (best.upper() if token.isupper() else best)
                res_text.append(final_word)

                if lower_token != best:
                    metrics.append(f"<b>{token} -> {best}</b><br>Candidates: {', '.join([f'{c} ({model.P(c):.2e})' for c in list(candidates)[:5]])}")
            else:
                res_text.append(token)

        output = "".join(res_text)
        metrics_html = "<hr>".join(metrics)

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        response = f"""
            <html>
                <body>
                    <h1>Result</h1>
                    <p style="white-space: pre-wrap; background: #eee; padding: 10px;">{output}</p>
                    <h3>Metrics</h3>
                    <div>{metrics_html}</div>
                    <a href="/">Back</a>
                </body>
            </html>
        """
        self.wfile.write(response.encode('utf-8'))

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("", PORT), Handler)
    print(f"Serving on port {PORT}")
    httpd.serve_forever()