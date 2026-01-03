import yfinance as yf
import time
import math
import numpy as np

# Override print for railway/console flushing
_builtin_print = print
def print(*args, **kwargs):
    time.sleep(0.1)
    _builtin_print(*args, **kwargs)

# ==============================================================================
# 1. DATA HANDLER (Yahoo Finance Version - No Key Needed)
# ==============================================================================

class MarketDataHandler:
    def __init__(self):
        # We need to remember how we converted $ to Categories
        # so we can convert the AI's answer back to $
        self.last_min = 0
        self.last_step = 0

    def fetch_candles(self, symbol="BTC-USD", days=60):
        """Fetches daily candles from Yahoo Finance."""
        print(f"Fetching {days} days of data for {symbol} via Yahoo Finance...")
        
        # yfinance handles the API work
        df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
        
        if df.empty:
            print("Error: No data found. Check symbol.")
            return []
            
        # Get the 'Close' column as a list
        # Handle cases where yfinance returns multi-index columns
        try:
            prices = df['Close'].values.flatten().tolist()
        except KeyError:
            # Fallback for different yfinance versions
            prices = df['Close'].tolist()
            
        # Filter out NaNs if any
        prices = [p for p in prices if not math.isnan(p)]
        return prices

    def discretize_sequence(self, prices, num_bins=20):
        """
        Converts raw dollar prices [150.5, 151.2...] into Categories [10, 11...]
        and Momentums [UP, UP...]
        """
        if not prices or len(prices) < 2: return []

        # 1. Determine Range (Dynamic Scaling)
        # We add a buffer so we don't constantly hit min/max edges
        min_p = min(prices) * 0.98
        max_p = max(prices) * 1.02
        price_range = max_p - min_p
        step = price_range / num_bins
        
        self.last_min = min_p
        self.last_step = step

        sequence = []
        
        # 2. Convert to (MOM, CAT)
        prev_cat = -1
        
        for i, p in enumerate(prices):
            # Map float to int bucket (1 to 20)
            cat = int((p - min_p) / step)
            cat = max(1, min(num_bins, cat)) # Clamp
            
            # Determine Momentum
            if i == 0:
                mom = 'FLAT' # No history
            else:
                if cat > prev_cat: mom = 'UP'
                elif cat < prev_cat: mom = 'DOWN'
                else: mom = 'FLAT' 
            
            prev_cat = cat
            sequence.append((mom, cat))
            
        return sequence

    def category_to_price(self, category):
        """Converts an AI Category back to a Dollar price."""
        return self.last_min + (category * self.last_step)

# ==============================================================================
# 2. THE AI ENGINE (Z-Score & Completion)
# ==============================================================================

class ZScoreEngine:
    def __init__(self):
        self.mom_map = {k: v for v, k in enumerate(['UP', 'DOWN', 'FLAT'])}
        # Trends are sticky
        self.mom_trans = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
        self.mom_start = np.array([0.4, 0.4, 0.2])
        # Prices move incrementally
        self.cat_trans = np.zeros((20, 20))
        for i in range(20):
            self.cat_trans[i] = [0.0001]*20 
            self.cat_trans[i][i] = 0.5      
            if i > 0: self.cat_trans[i][i-1] = 0.25 
            if i < 19: self.cat_trans[i][i+1] = 0.25 
            self.cat_trans[i] /= self.cat_trans[i].sum()
        self.cat_start = np.ones(20) / 20
        self.stats = {'mom': {}, 'cat': {}}

    def _get_log_prob(self, seq, trans, start):
        if not seq: return 0.0
        p = math.log(start[seq[0]] + 1e-9)
        for i in range(len(seq)-1):
            p += math.log(trans[seq[i]][seq[i+1]] + 1e-9)
        return p

    def calibrate(self):
        # Calibrate for length 2 to 20
        for length in range(2, 21): 
            m_logs, c_logs = [], []
            for _ in range(200):
                s_m = [np.random.choice(3, p=self.mom_start)]
                for _ in range(length-1): s_m.append(np.random.choice(3, p=self.mom_trans[s_m[-1]]))
                m_logs.append(self._get_log_prob(s_m, self.mom_trans, self.mom_start))
                s_c = [np.random.choice(20, p=self.cat_start)]
                for _ in range(length-1): s_c.append(np.random.choice(20, p=self.cat_trans[s_c[-1]]))
                c_logs.append(self._get_log_prob(s_c, self.cat_trans, self.cat_start))
            self.stats['mom'][length] = {'mu': np.mean(m_logs), 'var': np.var(m_logs)}
            self.stats['cat'][length] = {'mu': np.mean(c_logs), 'var': np.var(c_logs)}

    def get_z_score(self, sequence):
        length = len(sequence)
        if length not in self.stats['mom']: return 0.0
        m_idxs = [self.mom_map[m] for m, c in sequence]
        c_idxs = [c-1 for m, c in sequence]
        lp_m = self._get_log_prob(m_idxs, self.mom_trans, self.mom_start)
        lp_c = self._get_log_prob(c_idxs, self.cat_trans, self.cat_start)
        mu = self.stats['mom'][length]['mu'] + self.stats['cat'][length]['mu']
        std = math.sqrt(self.stats['mom'][length]['var'] + self.stats['cat'][length]['var'])
        return 0 if std == 0 else (lp_m + lp_c - mu) / std

class CompletionCorrector:
    def __init__(self, engine):
        self.engine = engine
        self.COST_INSERT = 1.6 # Cost to assume a prediction

    def repair_physics(self, sequence):
        if not sequence: return []
        repaired = [sequence[0]]
        for i in range(1, len(sequence)):
            prev_mom, prev_cat = repaired[-1]
            curr_mom, curr_cat = sequence[i]
            if curr_cat > prev_cat: mom = 'UP'
            elif curr_cat < prev_cat: mom = 'DOWN'
            else: mom = curr_mom
            repaired.append((mom, curr_cat))
        return repaired

    def solve(self, sequence):
        # Analyze last 8 candles
        analysis_window = sequence[-8:] 
        prefix = sequence[:-8]
        
        N = len(analysis_window)
        variants = [{'seq': analysis_window, 'op': 'HOLD', 'cost': 0}]
        
        # INSERT (Predict/Gap)
        for i in range(N + 1):
            ref_cat = analysis_window[min(i, N-1)][1]
            for cat in [ref_cat, ref_cat+1, ref_cat-1]:
                if 1 <= cat <= 20:
                    new_sub = analysis_window[:i] + [('FLAT', cat)] + analysis_window[i:]
                    label = "PREDICT" if i == N else "GAP_FILL"
                    variants.append({'seq': self.repair_physics(new_sub), 'op': label, 'cost': self.COST_INSERT})
        
        # Score
        results = []
        for v in variants:
            z = self.engine.get_z_score(v['seq'])
            results.append({**v, 'net': z - v['cost']})
            
        results.sort(key=lambda x: x['net'], reverse=True)
        winner = results[0]
        winner['full_seq'] = prefix + winner['seq']
        return winner

# ==============================================================================
# 3. MAIN RUNNER
# ==============================================================================

if __name__ == "__main__":
    # Settings
    SYMBOL = "BTC-USD" # Works for AAPL, TSLA, BTC-USD, EURUSD=X
    DAYS = 60
    
    market = MarketDataHandler()
    engine = ZScoreEngine()
    engine.calibrate()
    ai = CompletionCorrector(engine)

    # 1. Fetch
    raw_prices = market.fetch_candles(SYMBOL, days=DAYS)
    if not raw_prices:
        exit()
        
    # 2. Discretize
    seq = market.discretize_sequence(raw_prices, num_bins=20)
    
    # 3. Analyze
    print(f"\nAnalyzing {SYMBOL} (Price: ${raw_prices[-1]:.2f})...")
    
    result = ai.solve(seq)
    
    # 4. Result
    print("\n" + "="*40)
    print(f"AI DECISION: {result['op']}")
    print("="*40)
    
    if result['op'] == 'PREDICT':
        pred_cat = result['seq'][-1][1]
        pred_price = market.category_to_price(pred_cat)
        
        print(f"SIGNAL:     MOMENTUM TRADE DETECTED")
        print(f"Prediction: Sequence implies move to Cat {pred_cat}")
        print(f"Target:     ${pred_price:.2f}")
        print(f"Confidence: {result['net']:.2f}")
        
    elif result['op'] == 'GAP_FILL':
        print("SIGNAL:     HOLD (Noise)")
        print("Reason:     Internal gap fill detected. Market is choppy.")
        
    else:
        print("SIGNAL:     HOLD")
        print("Reason:     Current price action is statistically normal.")