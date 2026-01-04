import yfinance as yf
import time
import math
import numpy as np
import pandas as pd
from collections import Counter
import itertools

# ==============================================================================
# 1. DATA HANDLER
# ==============================================================================

class MarketDataHandler:
    def __init__(self):
        self.last_min = 0
        self.last_step = 0

    def fetch_candles(self, symbol="SPY", period="2y"):
        print(f"--- Fetching {period} data for {symbol} ---")
        try:
            df = yf.download(symbol, period=period, interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    prices = df.xs('Close', level=0, axis=1)[symbol].tolist()
                except KeyError:
                    prices = df['Close'].iloc[:, 0].tolist()
            elif 'Close' in df.columns:
                prices = df['Close'].tolist()
            else:
                prices = df.iloc[:, 0].tolist()
            
            prices = [p for p in prices if not math.isnan(p)]
            if not prices: raise ValueError("Empty data")
            
            # Keep raw prices for backtesting returns
            self.raw_prices = prices
            print(f"Loaded {len(prices)} candles.")
            return prices
        except Exception as e:
            print(f"Error fetching data: {e}")
            return []

    def discretize_sequence(self, prices, num_bins=40):
        if not prices or len(prices) < 2: return []
        min_p = min(prices) * 0.95
        max_p = max(prices) * 1.05
        step = (max_p - min_p) / num_bins
        
        self.last_min = min_p
        self.last_step = step

        sequence = []
        prev_cat = -1
        for i, p in enumerate(prices):
            cat = int((p - min_p) / step)
            cat = max(1, min(num_bins, cat))
            if i == 0: mom = 'FLAT'
            else:
                if cat > prev_cat: mom = 'UP'
                elif cat < prev_cat: mom = 'DOWN'
                else: mom = 'FLAT' 
            prev_cat = cat
            sequence.append((mom, cat))
        return sequence

# ==============================================================================
# 2. PROBABILISTIC ENGINE
# ==============================================================================

class ZScoreEngine:
    def __init__(self, num_categories=40):
        self.num_categories = num_categories
        self.mom_map = {k: v for v, k in enumerate(['UP', 'DOWN', 'FLAT'])}
        self.mom_trans = np.zeros((3, 3))
        self.mom_start = np.zeros(3)
        self.cat_trans = np.zeros((num_categories, num_categories))
        self.cat_start = np.zeros(num_categories)
        self.stats = {} 

    def train(self, sequence):
        print("Training Markov Probabilities...")
        mom_counts = np.ones((3, 3)) 
        cat_counts = np.ones((self.num_categories, self.num_categories)) 
        for i in range(len(sequence) - 1):
            curr_mom, curr_cat = sequence[i]
            next_mom, next_cat = sequence[i+1]
            m1, m2 = self.mom_map[curr_mom], self.mom_map[next_mom]
            c1, c2 = curr_cat - 1, next_cat - 1
            mom_counts[m1][m2] += 1
            cat_counts[c1][c2] += 1
            self.mom_start[m1] += 1
            self.cat_start[c1] += 1

        self.mom_trans = mom_counts / mom_counts.sum(axis=1, keepdims=True)
        self.cat_trans = cat_counts / cat_counts.sum(axis=1, keepdims=True)
        self.mom_start = self.mom_start / self.mom_start.sum()
        self.cat_start = self.cat_start / self.cat_start.sum()
        
    def _get_raw_log_prob(self, seq):
        if not seq: return -999.0
        m_idxs = [self.mom_map[m] for m, c in seq]
        c_idxs = [c-1 for m, c in seq]
        
        log_prob = 0.0
        log_prob += math.log(self.mom_start[m_idxs[0]] + 1e-9)
        log_prob += math.log(self.cat_start[c_idxs[0]] + 1e-9)
        for i in range(len(seq) - 1):
            log_prob += math.log(self.mom_trans[m_idxs[i]][m_idxs[i+1]] + 1e-9)
            log_prob += math.log(self.cat_trans[c_idxs[i]][c_idxs[i+1]] + 1e-9)
        return log_prob

    def calibrate(self, max_len=20, samples=2000):
        print(f"Calibrating baseline statistics (Max Len: {max_len})...")
        for length in range(1, max_len + 1):
            logs = []
            for _ in range(samples):
                m = np.random.choice(3, p=self.mom_start)
                c = np.random.choice(self.num_categories, p=self.cat_start)
                m_hist, c_hist = [m], [c]
                for _ in range(length - 1):
                    m = np.random.choice(3, p=self.mom_trans[m])
                    c = np.random.choice(self.num_categories, p=self.cat_trans[c])
                    m_hist.append(m)
                    c_hist.append(c)
                
                lp = math.log(self.mom_start[m_hist[0]] + 1e-9) + math.log(self.cat_start[c_hist[0]] + 1e-9)
                for i in range(length - 1):
                    lp += math.log(self.mom_trans[m_hist[i]][m_hist[i+1]] + 1e-9)
                    lp += math.log(self.cat_trans[c_hist[i]][c_hist[i+1]] + 1e-9)
                logs.append(lp)
            self.stats[length] = {'mu': np.mean(logs), 'std': np.std(logs)}

    def get_z_score(self, sequence):
        if not sequence: return 99.0
        L = len(sequence)
        if L not in self.stats: return 50.0 
        raw_lp = self._get_raw_log_prob(sequence)
        mu = self.stats[L]['mu']
        std = self.stats[L]['std']
        return 0 if std == 0 else -((raw_lp - mu) / std)

# ==============================================================================
# 3. EVOLUTIONARY SOLVER
# ==============================================================================

class EvolutionarySolver:
    def __init__(self, engine, num_bins):
        self.engine = engine
        self.num_bins = num_bins

    def _repair_physics(self, seq):
        repaired = []
        for i in range(len(seq)):
            cat = seq[i][1]
            if i == 0: mom = seq[i][0]
            else:
                prev_cat = repaired[-1][1]
                if cat > prev_cat: mom = 'UP'
                elif cat < prev_cat: mom = 'DOWN'
                else: mom = 'FLAT'
            repaired.append((mom, cat))
        return repaired

    def generate_mutations(self, seq):
        candidates = []
        L = len(seq)
        for i in range(L):
            curr_mom, curr_cat = seq[i]
            if curr_cat < self.num_bins: candidates.append(seq[:i] + [(curr_mom, curr_cat + 1)] + seq[i+1:])
            if curr_cat > 1: candidates.append(seq[:i] + [(curr_mom, curr_cat - 1)] + seq[i+1:])
        for i in range(L - 1):
            new_seq = seq[:]
            new_seq[i], new_seq[i+1] = new_seq[i+1], new_seq[i]
            candidates.append(new_seq)
        if L > 2:
            for i in range(L): candidates.append(seq[:i] + seq[i+1:])
        return candidates

    def solve(self, input_seq, horizon_steps=2):
        base_pool = [{'seq': input_seq, 'type': 'orig'}]
        L = len(input_seq)
        
        # Sub-sequences
        for length in range(3, L): 
            for i in range(L - length + 1):
                base_pool.append({'seq': input_seq[i : i+length], 'type': 'sub'})

        # Mutations
        muts = self.generate_mutations(input_seq)
        for m in muts:
            base_pool.append({'seq': m, 'type': 'mut'})

        # Extensions
        final_pool = []
        final_pool.extend(base_pool)
        
        for item in base_pool:
            current_tips = [item['seq']]
            for _ in range(horizon_steps):
                new_tips = []
                for tip in current_tips:
                    last_cat = tip[-1][1]
                    options = [last_cat]
                    if last_cat < self.num_bins: options.append(last_cat + 1)
                    if last_cat > 1: options.append(last_cat - 1)
                    for opt in options:
                        new_tip = tip + [('FLAT', opt)] 
                        new_tips.append(new_tip)
                        final_pool.append({'seq': new_tip, 'type': item['type'] + '+ext'})
                current_tips = new_tips

        # Evaluate
        scored = []
        unique_hashes = set()
        for cand in final_pool:
            fixed_seq = self._repair_physics(cand['seq'])
            s_hash = str(fixed_seq)
            if s_hash in unique_hashes: continue
            unique_hashes.add(s_hash)
            z = self.engine.get_z_score(fixed_seq)
            scored.append({'seq': fixed_seq, 'z': z, 'type': cand['type']})

        scored.sort(key=lambda x: x['z'])
        return scored[0]

# ==============================================================================
# 4. BACKTESTER & PORTFOLIO
# ==============================================================================

class Backtester:
    def __init__(self, initial_capital=10000.0):
        self.initial_capital = initial_capital
        self.equity = [initial_capital]
        self.returns = []
        self.positions = [] # +1 (Long), -1 (Short), 0 (Flat)
    
    def on_market_step(self, predicted_signal, actual_pct_change):
        """
        predicted_signal: 1 (Buy), -1 (Sell), 0 (Flat)
        actual_pct_change: The actual market move that happened AFTER the signal
        """
        # Calculate PnL for this step
        # If Signal=1 and Market=+2%, we make 2%. If Signal=-1 and Market=+2%, we lose 2%.
        step_return = predicted_signal * actual_pct_change
        
        # Update Equity
        curr_equity = self.equity[-1]
        new_equity = curr_equity * (1 + step_return)
        
        self.equity.append(new_equity)
        self.returns.append(step_return)
        self.positions.append(predicted_signal)
        
    def get_stats(self):
        if not self.returns: return {}
        
        equity_curve = np.array(self.equity)
        returns = np.array(self.returns)
        
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        # Annualized Statistics (Assuming Daily Data)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        # Sharpe Ratio (Mean / Std) * sqrt(252)
        # Using 0% Risk Free Rate for simplicity of raw alpha test
        if std_ret == 0: sharpe = 0
        else: sharpe = (mean_ret / std_ret) * math.sqrt(252)
        
        # Max Drawdown
        peak = equity_curve[0]
        max_dd = 0
        for val in equity_curve:
            if val > peak: peak = val
            dd = (peak - val) / peak
            if dd > max_dd: max_dd = dd
            
        return {
            "Total Return %": total_return * 100,
            "Sharpe Ratio": sharpe,
            "Max Drawdown %": max_dd * 100,
            "Win Rate %": (np.sum(returns > 0) / len(returns)) * 100 if len(returns) > 0 else 0,
            "Final Equity": equity_curve[-1]
        }

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    SYMBOL = "SPY"
    PERIOD = "2y" # Longer period for better Sharpe calc
    NUM_BINS = 50
    WINDOW_SIZE = 8
    HORIZON = 2 
    
    market = MarketDataHandler()
    raw_prices = market.fetch_candles(SYMBOL, PERIOD)
    if not raw_prices: exit()

    full_seq = market.discretize_sequence(raw_prices, num_bins=NUM_BINS)
    
    # 60% Train / 40% Backtest
    split_idx = int(len(full_seq) * 0.6)
    train_seq = full_seq[:split_idx]
    
    print(f"Training on first {split_idx} days, Backtesting on remaining {len(full_seq)-split_idx} days...")
    
    engine = ZScoreEngine(num_categories=NUM_BINS)
    engine.train(train_seq)
    engine.calibrate(max_len=WINDOW_SIZE + HORIZON + 5)
    
    solver = EvolutionarySolver(engine, NUM_BINS)
    backtester = Backtester()
    
    print("\nStarting Backtest Simulation...")
    print(f"{'IDX':<5} | {'SIGNAL':<8} | {'ACTUAL %':<10} | {'EQUITY ($)':<12}")
    print("-" * 50)
    
    # Run Backtest
    for i in range(split_idx, len(full_seq) - WINDOW_SIZE - 1):
        input_window = full_seq[i : i + WINDOW_SIZE]
        
        # Get Real Market Move for the NEXT day
        # Note: We make decision at 'i + WINDOW_SIZE', realize PnL at 'i + WINDOW_SIZE + 1'
        curr_price = raw_prices[i + WINDOW_SIZE]
        next_price = raw_prices[i + WINDOW_SIZE + 1]
        actual_pct_change = (next_price - curr_price) / curr_price
        
        # --- AI DECISION ---
        winner = solver.solve(input_window, horizon_steps=HORIZON)
        best_seq = winner['seq']
        
        # Signal Logic:
        # 1. If Extended (Length > Input) -> Forecast -> Trade
        # 2. If Shortened (Length < Input) -> Noise -> Flat
        # 3. If Same -> Neutral -> Flat
        
        signal = 0
        if len(best_seq) > len(input_window):
            # It's a prediction
            pred_cat = best_seq[len(input_window)][1] # First forecasted step
            curr_cat = input_window[-1][1]
            
            if pred_cat > curr_cat: signal = 1   # Long
            elif pred_cat < curr_cat: signal = -1 # Short
        
        # Record to Backtester
        backtester.on_market_step(signal, actual_pct_change)
        
        if i % 20 == 0: # Print every 20 steps to save space
            print(f"{i:<5} | {signal:<8} | {actual_pct_change*100:6.2f}% | {backtester.equity[-1]:,.2f}")

    # Final Stats
    stats = backtester.get_stats()
    
    print("\n" + "="*40)
    print(f"BACKTEST RESULTS ({SYMBOL})")
    print("="*40)
    print(f"Final Equity:   ${stats['Final Equity']:,.2f}")
    print(f"Total Return:   {stats['Total Return %']:.2f}%")
    print(f"Sharpe Ratio:   {stats['Sharpe Ratio']:.4f}")
    print(f"Max Drawdown:   {stats['Max Drawdown %']:.2f}%")
    print(f"Win Rate:       {stats['Win Rate %']:.2f}%")
    
    # Benchmarking (Buy & Hold)
    start_p = raw_prices[split_idx + WINDOW_SIZE]
    end_p = raw_prices[-1]
    bnh_ret = (end_p - start_p) / start_p * 100
    print("-" * 40)
    print(f"Buy & Hold Ret: {bnh_ret:.2f}%")