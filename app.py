import http.server
import socketserver
import re
import urllib.request
import os
import sys
from collections import Counter
from urllib.parse import parse_qs

# --- Configuration ---
PORT = int(os.environ.get('PORT', 8080))
CORPUS_URL = "https://norvig.com/big.txt"
CORPUS_FILE = "big.txt"

# --- 1. The Probabilistic Model (Norvig's Algorithm) ---

class SpellingModel:
    def __init__(self):
        self.WORDS = Counter()
        self.N = 0
        self.ensure_corpus()
        self.train()

    def ensure_corpus(self):
        """Downloads the corpus if it doesn't exist locally."""
        if not os.path.exists(CORPUS_FILE):
            print(f"Downloading corpus from {CORPUS_URL}...")
            try:
                with urllib.request.urlopen(CORPUS_URL) as response:
                    data = response.read()
                    with open(CORPUS_FILE, 'wb') as f:
                        f.write(data)
                print("Download complete.")
            except Exception as e:
                print(f"Failed to download corpus: {e}")
                # Fallback to a tiny corpus for demo purposes if download fails
                with open(CORPUS_FILE, 'w') as f:
                    f.write("the of and to a in that is was he for it with as his on be at by i this had not are but from or have an they which one you were her all she there would their we him been has when who will more no if out so said what up its about into than them can only other new some could time these two may then do first any my now such like our over man me even most made after also did many before must through back years where much your way well down should because each just those people mr how too little state good very make world still own see men work long get here between both life being under never day same another know while last might great old year off come since go against came right used take three states himself few house use during without place american around however home small found mrs thought went say part once general high upon school every don does got united left number course war until always away something fact water less public put though meaning keep think week fathers reading fall saturn")

    def words(self, text): 
        return re.findall(r'\w+', text.lower())

    def train(self):
        """Trains the model on the corpus file."""
        print("Training model...")
        with open(CORPUS_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            self.WORDS = Counter(self.words(f.read()))
        self.N = sum(self.WORDS.values())
        print(f"Model trained on {self.N} words. Vocabulary size: {len(self.WORDS)}")

    def P(self, word): 
        """Probability of `word`."""
        return self.WORDS[word] / self.N

    def correction(self, word): 
        """Most probable spelling correction for word."""
        return max(self.candidates(word), key=self.P)

    def candidates(self, word): 
        """Generate possible spelling corrections for word."""
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words): 
        """The subset of `words` that appear in the dictionary of WORDS."""
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        """All edits that are one edit away from `word`."""
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        """All edits that are two edits away from `word`."""
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))
    
    def analyze(self, word):
        """Returns candidates and their probabilities for metrics."""
        word = word.lower()
        candidates = self.candidates(word)
        # Sort candidates by probability
        ranked = sorted([(c, self.P(c)) for c in candidates], key=lambda x: x[1], reverse=True)
        return ranked

# Initialize model globally
model = SpellingModel()

# --- 2. The Web Server ---

class SpellingHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Probabilistic Spelling Corrector</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f4f4f9; }
                .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                h1 { color: #333; }
                textarea { width: 100%; height: 100px; padding: 10px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 16px; }
                button { background: #007bff; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 4px; cursor: pointer; }
                button:hover { background: #0056b3; }
                .result { margin-top: 30px; border-top: 1px solid #eee; padding-top: 20px; }
                .word-analysis { margin-bottom: 20px; padding: 10px; background: #fafafa; border-left: 3px solid #007bff; }
                .metric { font-size: 0.85em; color: #666; font-family: monospace; }
                .corrected { color: #28a745; font-weight: bold; }
                .original { color: #dc3545; text-decoration: line-through; margin-right: 5px; }
                table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }
                th, td { text-align: left; padding: 8px; border-bottom: 1px solid #eee; }
                th { background-color: #f8f9fa; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Spelling Corrector</h1>
                <p>Based on Peter Norvig's probabilistic model. Enter text below:</p>
                <form method="POST">
                    <textarea name="text" placeholder="Type here (e.g., 'korrect this sentense')..."></textarea><br>
                    <button type="submit">Correct Text</button>
                </form>
            </div>
        </body>
        </html>
        """
        self.wfile.write(html.encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        params = parse_qs(post_data)
        input_text = params.get('text', [''])[0]
        
        # Process the text
        words = re.findall(r'\w+|[^\w\s]', input_text, re.UNICODE)
        full_analysis = []
        corrected_tokens = []
        
        for token in words:
            # Check if it's a word (not punctuation)
            if re.match(r'\w+', token):
                # Analyze
                candidates = model.analyze(token)
                best_word = candidates[0][0]
                
                # Preserve case roughly
                if token[0].isupper():
                    best_word = best_word.capitalize()
                
                corrected_tokens.append(best_word)
                
                # Only show analysis if correction happened or probability is interesting
                is_correction = best_word.lower() != token.lower()
                full_analysis.append({
                    'original': token,
                    'corrected': best_word,
                    'is_correction': is_correction,
                    'candidates': candidates[:5] # Top 5
                })
            else:
                corrected_tokens.append(token)
        
        corrected_text = " ".join(corrected_tokens)
        # Simple heuristic to fix spacing around punctuation (very basic)
        corrected_text = re.sub(r' \.', '.', corrected_text)
        corrected_text = re.sub(r' \,', ',', corrected_text)

        # Generate Results HTML
        analysis_html = ""
        for item in full_analysis:
            if item['is_correction']:
                rows = ""
                for cand, prob in item['candidates']:
                    # Scientific notation for very small probs
                    prob_str = f"{prob:.2e}" if prob < 0.001 else f"{prob:.4f}"
                    rows += f"<tr><td>{cand}</td><td>{prob_str}</td></tr>"
                
                analysis_html += f"""
                <div class="word-analysis">
                    <div>
                        <span class="original">{item['original']}</span>
                        <span>&rarr;</span>
                        <span class="corrected">{item['corrected']}</span>
                    </div>
                    <div class="metric">
                        <table>
                            <thead><tr><th>Candidate</th><th>Probability P(c)</th></tr></thead>
                            <tbody>{rows}</tbody>
                        </table>
                    </div>
                </div>
                """

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Results</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f4f4f9; }}
                .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; }}
                .result-box {{ background: #e9ecef; padding: 15px; border-radius: 4px; font-size: 1.1em; margin-bottom: 20px; }}
                .word-analysis {{ margin-bottom: 20px; padding: 15px; background: white; border: 1px solid #ddd; border-left: 4px solid #dc3545; border-radius: 4px; }}
                .metric {{ margin-top: 10px; }}
                .corrected {{ color: #28a745; font-weight: bold; font-size: 1.1em; }}
                .original {{ color: #dc3545; text-decoration: line-through; margin-right: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #eee; }}
                th {{ background-color: #f8f9fa; }}
                .back-link {{ display: inline-block; margin-top: 20px; color: #007bff; text-decoration: none; }}
                .back-link:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Correction Results</h1>
                
                <h3>Corrected Text:</h3>
                <div class="result-box">
                    {corrected_text}
                </div>

                <h3>Probabilistic Metrics (Corrections Only):</h3>
                {analysis_html if analysis_html else "<p>No spelling errors detected.</p>"}
                
                <a href="/" class="back-link">&larr; Try another</a>
            </div>
        </body>
        </html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

if __name__ == "__main__":
    # Allow address reuse prevents "Address already in use" errors on restart
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), SpellingHandler) as httpd:
        print(f"Server started at http://localhost:{PORT}")
        print(f"Using corpus: {CORPUS_FILE}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopping...")
            httpd.server_close()