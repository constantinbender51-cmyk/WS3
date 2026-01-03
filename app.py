import numpy as np
import math
import itertools
import time

# Override print to ensure correct order of railway output
_builtin_print = print
def print(*args, **kwargs):
    time.sleep(0.1)
    _builtin_print(*args, **kwargs)

# ==============================================================================
# 1. CONFIGURATION: The "Cost of Truth"
# ==============================================================================
# The algorithm balances Probability (Z-Score) vs Complexity (Edit Cost).
# If a sequence is "UP UP UP", adding a 4th "UP" might increase the Z-Score 
# so much that it outweighs the cost of insertion.

COST_SWAP    = 2.0  # Cost to change a number (Fixing noise)
COST_DELETE  = 2.5  # Cost to remove a number (Filtering outliers)
COST_INSERT  = 1.8  # Cost to add a number (Gap filling OR Prediction)

NUM_CATEGORIES = 20
SEARCH_RADIUS  = 1  # How far to look for Swaps/Inserts (+/- 1 category)

# ==============================================================================
# 2. THE PROBABILISTIC ENGINE (Decoupled Z-Score)
# ==============================================================================
class ZScoreEngine:
    def __init__(self):
        self.mom_map = {k: v for v, k in enumerate(['UP', 'DOWN', 'FLAT'])}
        
        # --- PHYSICS SIMULATION ---
        # 1. Momentum: Strong Trend Bias (UP follows UP)
        self.mom_trans = np.array([
            [0.8, 0.1, 0.1], 
            [0.1, 0.8, 0.1], 
            [0.3, 0.3, 0.4]
        ])
        self.mom_start = np.array([0.4, 0.4, 0.2])
        
        # 2. Categories: Incremental movement
        self.cat_trans = np.zeros((NUM_CATEGORIES, NUM_CATEGORIES))
        for i in range(NUM_CATEGORIES):
            for j in range(NUM_CATEGORIES):
                dist = abs(i - j)
                if dist == 0: prob = 0.5
                elif dist == 1: prob = 0.25
                elif dist == 2: prob = 0.05
                else: prob = 0.0001
                self.cat_trans[i][j] = prob
            self.cat_trans[i] /= self.cat_trans[i].sum()
            
        self.cat_start = np.ones(NUM_CATEGORIES) / NUM_CATEGORIES
        self.stats = {'mom': {}, 'cat': {}}

    def _get_log_prob(self, seq, trans, start):
        if not seq: return 0.0
        p = math.log(start[seq[0]] + 1e-9)
        for i in range(len(seq)-1):
            p += math.log(trans[seq[i]][seq[i+1]] + 1e-9)
        return p

    def calibrate(self):
        # We calibrate a wide range of lengths to allow for 
        # deletions (shorter) and insertions/predictions (longer)
        print("Calibrating Probability Field (Lengths 2-8)...")
        for length in range(2, 9):
            m_logs, c_logs = [], []
            for _ in range(1000):
                # Mom
                s_m = [np.random.choice(3, p=self.mom_start)]
                for _ in range(length-1): s_m.append(np.random.choice(3, p=self.mom_trans[s_m[-1]]))
                m_logs.append(self._get_log_prob(s_m, self.mom_trans, self.mom_start))
                # Cat
                s_c = [np.random.choice(NUM_CATEGORIES, p=self.cat_start)]
                for _ in range(length-1): s_c.append(np.random.choice(NUM_CATEGORIES, p=self.cat_trans[s_c[-1]]))
                c_logs.append(self._get_log_prob(s_c, self.cat_trans, self.cat_start))
            
            self.stats['mom'][length] = {'mu': np.mean(m_logs), 'var': np.var(m_logs)}
            self.stats['cat'][length] = {'mu': np.mean(c_logs), 'var': np.var(c_logs)}

    def get_z_score(self, sequence):
        length = len(sequence)
        if length not in self.stats['mom']: return -50.0 # Impossible length
        
        m_idxs = [self.mom_map[m] for m, c in sequence]
        c_idxs = [c-1 for m, c in sequence]
        
        lp_m = self._get_log_prob(m_idxs, self.mom_trans, self.mom_start)
        lp_c = self._get_log_prob(c_idxs, self.cat_trans, self.cat_start)
        
        mu = self.stats['mom'][length]['mu'] + self.stats['cat'][length]['mu']
        std = math.sqrt(self.stats['mom'][length]['var'] + self.stats['cat'][length]['var'])
        
        return 0 if std == 0 else (lp_m + lp_c - mu) / std

# ==============================================================================
# 3. THE COMPLETION CORRECTOR
# ==============================================================================
class CompletionCorrector:
    def __init__(self, engine):
        self.engine = engine

    def repair_physics(self, sequence):
        """
        The Enforcer: Ensures Momentum matches the Categories.
        Essential for 'Prediction' because inserting a number at the end
        requires calculating the new momentum vector.
        """
        if not sequence: return []
        repaired = [sequence[0]]
        for i in range(1, len(sequence)):
            prev_mom, prev_cat = repaired[-1]
            curr_mom, curr_cat = sequence[i]
            
            if curr_cat > prev_cat: final_mom = 'UP'
            elif curr_cat < prev_cat: final_mom = 'DOWN'
            else: final_mom = curr_mom 
            
            repaired.append((final_mom, curr_cat))
        return repaired

    def generate_variants(self, sequence):
        """
        Generates all possible Single-Edit variants.
        Crucially, 'Insert' works from index 0 to len(sequence).
        """
        variants = []
        
        # 1. BASELINE (0 Edits)
        variants.append({
            'seq': self.repair_physics(sequence),
            'op': 'HOLD',
            'cost': 0
        })

        N = len(sequence)

        # 2. SWAPS (Fixing errors in place)
        for i in range(N):
            curr_mom, curr_cat = sequence[i]
            for delta in [-SEARCH_RADIUS, SEARCH_RADIUS]:
                new_cat = curr_cat + delta
                if 1 <= new_cat <= NUM_CATEGORIES:
                    # Construct
                    new_seq = list(sequence)
                    new_seq[i] = (curr_mom, new_cat)
                    variants.append({
                        'seq': self.repair_physics(new_seq),
                        'op': 'SWAP',
                        'cost': COST_SWAP
                    })

        # 3. DELETES (Removing outliers)
        if N > 2: # Don't delete if too short
            for i in range(N):
                new_seq = sequence[:i] + sequence[i+1:]
                variants.append({
                    'seq': self.repair_physics(new_seq),
                    'op': 'DELETE',
                    'cost': COST_DELETE
                })

        # 4. INSERTS (The Magic Step)
        # We loop from 0 to N. 
        # i=0 (Prepend), i=N//2 (Gap Fill), i=N (Append/Predict)
        for i in range(N + 1): 
            # Determine reference category for insertion
            if i == 0: ref_cat = sequence[0][1]
            elif i == N: ref_cat = sequence[-1][1]
            else: ref_cat = sequence[i][1] # Or average of i and i-1

            # Try inserting neighbors of the reference
            candidates = {ref_cat, ref_cat+1, ref_cat-1}
            
            for cat in candidates:
                if 1 <= cat <= NUM_CATEGORIES:
                    # Construct: Insert 'FLAT' (repair_physics will fix it)
                    new_seq = sequence[:i] + [('FLAT', cat)] + sequence[i:]
                    
                    # Logic to name the operation for the user
                    if i == N:
                        label = "INSERT (Predict)"
                    elif i == 0:
                        label = "INSERT (History)"
                    else:
                        label = "INSERT (Gap)"

                    variants.append({
                        'seq': self.repair_physics(new_seq),
                        'op': label,
                        'cost': COST_INSERT
                    })
        
        return variants

    def solve(self, sequence):
        variants = self.generate_variants(sequence)
        
        # Score all variants
        scored = []
        for v in variants:
            z = self.engine.get_z_score(v['seq'])
            net = z - v['cost']
            v['z_score'] = z
            v['net_score'] = net
            scored.append(v)
            
        # Sort by Net Score (High Z, Low Cost)
        scored.sort(key=lambda x: x['net_score'], reverse=True)
        return scored[0] # Return Winner

# ==============================================================================
# 4. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    engine = ZScoreEngine()
    engine.calibrate()
    corrector = CompletionCorrector(engine)

    def analyze_sequence(name, seq):
        print(f"\n--- {name} ---")
        print(f"INPUT: {seq}")
        
        winner = corrector.solve(seq)
        
        # INTERPRETATION LAYER
        # We check if the winner is longer than input and matches the prefix
        input_len = len(seq)
        winner_len = len(winner['seq'])
        
        print(f"WINNER: {winner['op']} (Net: {winner['net_score']:.2f})")
        print(f"OUTPUT: {winner['seq']}")
        
        if winner_len > input_len and winner['seq'][:input_len] == corrector.repair_physics(seq):
            # If we just added to the end...
            added = winner['seq'][input_len:]
            print(f">>> SIGNAL: PREDICTION DETECTED. The model completed the pattern with: {added}")
        elif winner_len > input_len:
            print(f">>> SIGNAL: GAP FILLED. The model inserted data into the past.")
        elif winner['op'] == 'HOLD':
            print(f">>> SIGNAL: NONE. Input data is solid.")
        else:
            print(f">>> SIGNAL: CORRECTION. The model altered the past data.")

    # CASE 1: Strong Trend (Implicit Prediction)
    # The model should see UP-UP-UP and decide that UP-UP-UP-UP is SO likely
    # that it's worth the insertion cost.
    analyze_sequence("Strong Trend", [('UP', 10), ('UP', 11), ('UP', 12)])

    # CASE 2: Broken History (Gap Fill)
    # The model should see 10 -> 12 and decide 10 -> 11 -> 12 is better.
    analyze_sequence("Data Gap", [('UP', 10), ('UP', 12)])

    # CASE 3: Weak/Noisy (Hold)
    # The model should see zig-zag and decide adding a prediction is risky.
    analyze_sequence("Choppy Noise", [('UP', 10), ('DOWN', 9), ('UP', 10)])