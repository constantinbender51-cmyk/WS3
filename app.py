import requests
import time
import os
import math
import numpy as np

# ==============================================================================
# 1. API & DATA HANDLING
# ==============================================================================

class MarketDataHandler:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        
        # We need to remember how we converted $ to Categories
        # so we can convert the AI's answer back to $
        self.last_min = 0
        self.last_step = 0

    def fetch_candles(self, symbol="AAPL", days=60):
        """Fetches daily candles from Finnhub."""
        if not self.api_key:
            raise ValueError("Missing FINN_KEY environment variable.")

        end_t = int(time.time())
        start_t = end_t - (86400 * days) # 60 days ago
        
        url = f"{self.base_url}/stock/candle"
        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': start_t,
            'to': end_t,
            'token': self.api_key
        }
        
        print(f"Fetching {symbol} data from Finnhub...")
        r = requests.get(url, params=params)
        data = r.json()
        
        if data.get('s') != 'ok':
            print("API Error:", data)
            return []
            
        return data['c'] # Return list of Close prices

    def discretize_sequence(self, prices, num_bins=20):
        """
        Converts real prices [150.5, 151.2...] into Categories [10, 11...]
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
                else: mom = 'FLAT' # Or use price-level micro momentum
            
            prev_cat = cat
            sequence.append((mom, cat))
            
        return sequence

    def category_to_price(self, category):
        """Converts an AI Category back to a Dollar price."""
        return self.last_min + (category * self.last_step)

# ==============================================================================
# 2. THE AI ENGINE (Re-used)
# ==============================================================================

class ZScoreEngine:
    def __init__(self):
        self.mom_map = {k: v for v, k in enumerate(['UP', 'DOWN', 'FLAT'])}
        self.mom_trans = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
        self.mom_start = np.array([0.4, 0.4, 0.2])
        self.cat_trans = np.zeros((20, 20))
        for i in range(20):
            self.cat_trans[i] = [0.0001]*20 # Teleport prob
            self.cat_trans[i][i] = 0.5      # Stay
            if i > 0: self.cat_trans[i][i-1] = 0.25 # Down 1
            if i < 19: self.cat_trans[i][i+1] = 0.25 # Up 1
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
        # Silent calibration
        for length in range(2, 20): # Extended range for longer real sequences
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
        self.COST_SWAP = 2.0
        self.COST_INSERT = 1.6 # Aggressive insertion for prediction

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
        # Limit analysis to the last 8 candles to keep Z-Score relevant
        # (Looking at 60 days of Z-Score dilutes the signal)
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
        
        # Re-attach prefix for full context
        winner = results[0]
        winner['full_seq'] = prefix + winner['seq']
        return winner

# ==============================================================================
# 3. MAIN RUNNER
# ==============================================================================

if __name__ == "__main__":
    # 1. SETUP
    api_key = os.getenv("FINN_KEY")
    if not api_key:
        print("ERROR: Please set FINN_KEY in your .env file")
        exit()
        
    symbol = "AAPL" # Change to 'BINANCE:BTCUSDT' for crypto if Finnhub supports your plan
    
    market = MarketDataHandler(api_key)
    engine = ZScoreEngine()
    engine.calibrate()
    ai = CompletionCorrector(engine)

    # 2. FETCH REAL DATA
    raw_prices = market.fetch_candles(symbol, days=60)
    if not raw_prices:
        exit()
        
    # 3. DISCRETIZE
    # Convert $200 -> Category 15
    seq = market.discretize_sequence(raw_prices, num_bins=20)
    
    # 4. RUN AI ANALYSIS
    print(f"\nAnalyzing last 8 candles of {symbol}...")
    print(f"Current Price: ${raw_prices[-1]}")
    print(f"Current Cat:   {seq[-1][1]}")
    
    result = ai.solve(seq)
    
    # 5. DECODE RESULT
    print("\n" + "="*40)
    print(f"AI DECISION: {result['op']}")
    print("="*40)
    
    if result['op'] == 'PREDICT':
        # The last item in the sequence is the prediction
        pred_cat = result['seq'][-1][1]
        pred_price = market.category_to_price(pred_cat)
        
        print(f"SIGNAL:     BUY/SELL MOMENTUM")
        print(f"Prediction: Price moves to Category {pred_cat}")
        print(f"Target:     ${pred_price:.2f}")
        print(f"Confidence: {result['net']:.2f}")
        
    elif result['op'] == 'GAP_FILL':
        print("SIGNAL:     HOLD (Data Irregularity)")
        print("Reason:     The model found a gap in past data, implying volatility/noise.")
        
    else:
        print("SIGNAL:     HOLD (No clear edge)")
        print("Reason:     Current trend is statistically normal.")