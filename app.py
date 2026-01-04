
import random
import time
from collections import defaultdict
from typing import List, Tuple

# ============================================================================
# PARAMETERS
# ============================================================================
N_PRICES = 20000
MIN_PRICE = 100
MAX_PRICE = 600
N_CATEGORIES = 100
TRAIN_SPLIT = 0.7
CATEGORY_STEP = 1
PRINT_DELAY = 0.1
STARTING_EQUITY = 10000
CATEGORY_TO_DOLLAR = 10  # Scaling factor for category movements to dollars
# ============================================================================


def custom_print(text: str, delay: float = PRINT_DELAY):
    """Print with delay"""
    print(text)
    time.sleep(delay)

def generate_prices(n: int, min_price: float, max_price: float) -> List[float]:
    """Generate mock prices"""
    return [random.uniform(min_price, max_price) for _ in range(n)]

def categorize_price(price: float, min_price: float, max_price: float, n_categories: int) -> int:
    """Categorize price into buckets"""
    step = (max_price - min_price) / n_categories
    if price < min_price:
        return 0
    elif price > max_price:
        return n_categories + int((price - max_price) / step)
    else:
        return int((price - min_price) / step)

def compute_category_sequence_probabilities(categories: List[int]) -> dict:
    """Compute probability of each 3-sequential category sequence"""
    sequence_counts = defaultdict(int)
    total_sequences = 0
    
    for i in range(len(categories) - 2):
        seq = tuple(categories[i:i+3])
        sequence_counts[seq] += 1
        total_sequences += 1
    
    probabilities = {seq: count / total_sequences for seq, count in sequence_counts.items()}
    return probabilities

def compute_directional_probabilities(categories: List[int]) -> dict:
    """Compute probability of directional sequences based on category differences"""
    directions = [categories[i+1] - categories[i] for i in range(len(categories) - 1)]
    
    dir_sequence_counts = defaultdict(int)
    total_sequences = 0
    
    for i in range(len(directions) - 2):
        seq = tuple(directions[i:i+3])
        dir_sequence_counts[seq] += 1
        total_sequences += 1
    
    probabilities = {seq: count / total_sequences for seq, count in dir_sequence_counts.items()}
    return probabilities

def generate_category_variants(last_two_cats: List[int], step: int) -> List[List[int]]:
    """Generate all category sequences 1 step away from last two categories"""
    c1, c2 = last_two_cats
    
    # Edit first two categories
    variants_step1 = [
        [c1 - step, c2],
        [c1 + step, c2],
        [c1, c2 - step],
        [c1, c2 + step]
    ]
    
    # Append third category with 1 step variations
    variants_step2 = []
    for var in variants_step1:
        variants_step2.extend([
            var + [c2 - step],
            var + [c2],
            var + [c2 + step]
        ])
    
    return variants_step2

def generate_directional_variants(cat_diffs: Tuple[int, int], step: int) -> List[List[int]]:
    """Generate directional variants based on category differences"""
    d1, d2 = cat_diffs
    
    # Edit directional values
    variants_step1 = [
        [d1 - step, d2],
        [d1 + step, d2],
        [d1, d2 - step],
        [d1, d2 + step]
    ]
    
    # Append third direction
    variants_step2 = []
    for var in variants_step1:
        variants_step2.extend([
            var + [d2 - step],
            var + [d2],
            var + [d2 + step]
        ])
    
    return variants_step2

def predict_next_category(
    last_two_cats: List[int],
    cat_probs: dict,
    dir_probs: dict,
    step: int
) -> Tuple[int, str]:
    """Predict next category and action"""
    
    # Generate category variants
    cat_variants = generate_category_variants(last_two_cats, step)
    
    # Convert to directional (category differences)
    directional = [last_two_cats[1] - last_two_cats[0]]
    
    # For directional variants, we need the difference pattern
    # We'll compute this for each category variant
    best_prob = -1
    best_prediction = None
    
    # Evaluate each variant
    for cat_var in cat_variants:
        # Get category sequence probability
        cat_seq = tuple(cat_var)
        cat_prob = cat_probs.get(cat_seq, 0)
        
        # Get directional sequence (differences between categories)
        dir_seq = tuple([cat_var[i+1] - cat_var[i] for i in range(len(cat_var) - 1)])
        dir_prob = dir_probs.get(dir_seq, 0)
        
        # Combined probability
        combined_prob = cat_prob + dir_prob
        
        if combined_prob > best_prob:
            best_prob = combined_prob
            best_prediction = cat_var[2]
    
    # Determine action
    current_cat = last_two_cats[1]
    if best_prediction is None:
        best_prediction = current_cat
    
    if best_prediction > current_cat:
        action = "buy"
    elif best_prediction < current_cat:
        action = "sell"
    else:
        action = "hold"
    
    return best_prediction, action

def evaluate_predictions(
    actual_cats: List[int],
    predictions: List[Tuple[int, str]],
    starting_equity: float,
    cat_to_dollar: float
) -> Tuple[float, float, float]:
    """Evaluate accuracy, equity, and Sharpe ratio"""
    correct = 0
    equity = starting_equity
    returns = []
    position = 0  # 0: no position, 1: long, -1: short
    entry_cat = 0
    
    for i in range(len(predictions)):
        pred_cat, action = predictions[i]
        actual = actual_cats[i]
        
        # Check direction accuracy
        if i > 0:
            actual_direction = actual - actual_cats[i-1]
            pred_direction = pred_cat - actual_cats[i-1]
            if (actual_direction > 0 and pred_direction > 0) or \
               (actual_direction < 0 and pred_direction < 0) or \
               (actual_direction == 0 and pred_direction == 0):
                correct += 1
        
        # Simulate trading (using category movements as proxy for price movements)
        if action == "buy" and position == 0:
            position = 1
            entry_cat = actual
        elif action == "sell" and position == 1:
            pnl = (actual - entry_cat) * cat_to_dollar
            equity += pnl
            returns.append(pnl / 1000 if pnl != 0 else 0)
            position = 0
        elif action == "sell" and position == 0:
            position = -1
            entry_cat = actual
        elif action == "buy" and position == -1:
            pnl = (entry_cat - actual) * cat_to_dollar
            equity += pnl
            returns.append(pnl / 1000 if pnl != 0 else 0)
            position = 0
    
    # Close any open position
    if position != 0 and len(actual_cats) > 0:
        final_cat = actual_cats[-1]
        if position == 1:
            pnl = (final_cat - entry_cat) * cat_to_dollar
            equity += pnl
            returns.append(pnl / 1000 if pnl != 0 else 0)
        else:
            pnl = (entry_cat - final_cat) * cat_to_dollar
            equity += pnl
            returns.append(pnl / 1000 if pnl != 0 else 0)
    
    accuracy = correct / max(len(predictions) - 1, 1)
    
    # Sharpe ratio
    if len(returns) > 0:
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_return = variance ** 0.5
        sharpe = avg_return / std_return if std_return > 0 else 0
    else:
        sharpe = 0
    
    return accuracy, equity, sharpe

def main():
    custom_print("=" * 60)
    custom_print("Category-Based Price Prediction Algorithm")
    custom_print("=" * 60)
    
    custom_print(f"\nGenerating {N_PRICES} mock prices...")
    prices = generate_prices(N_PRICES, MIN_PRICE, MAX_PRICE)
    
    custom_print("Converting prices to categories...")
    categories = [categorize_price(p, MIN_PRICE, MAX_PRICE, N_CATEGORIES) for p in prices]
    
    # Split data
    split_idx = int(N_PRICES * TRAIN_SPLIT)
    train_cats = categories[:split_idx]
    test_cats = categories[split_idx:]
    
    custom_print(f"Train set: {len(train_cats)} categories")
    custom_print(f"Test set: {len(test_cats)} categories")
    custom_print(f"Category range: {min(train_cats)} to {max(train_cats)}")
    
    # Compute probabilities on training set
    custom_print("\nComputing category sequence probabilities...")
    cat_probs = compute_category_sequence_probabilities(train_cats)
    custom_print(f"Found {len(cat_probs)} unique category sequences")
    
    custom_print("\nComputing directional sequence probabilities...")
    dir_probs = compute_directional_probabilities(train_cats)
    custom_print(f"Found {len(dir_probs)} unique directional sequences")
    
    # Make predictions
    custom_print("\nGenerating predictions...")
    predictions = []
    
    for i in range(2, len(test_cats)):
        last_two = test_cats[i-2:i]
        pred_cat, action = predict_next_category(
            last_two, cat_probs, dir_probs, CATEGORY_STEP
        )
        predictions.append((pred_cat, action))
        
        if i % 500 == 0:
            custom_print(f"  Processed {i}/{len(test_cats)} test samples...")
    
    custom_print(f"\nGenerated {len(predictions)} predictions")
    
    # Evaluate
    custom_print("\nEvaluating predictions...")
    actual_test = test_cats[2:]
    accuracy, equity, sharpe = evaluate_predictions(actual_test, predictions, STARTING_EQUITY, CATEGORY_TO_DOLLAR)
    
    custom_print("\n" + "=" * 60)
    custom_print("RESULTS")
    custom_print("=" * 60)
    custom_print(f"Direction Accuracy: {accuracy:.2%}")
    custom_print(f"Final Equity: ${equity:,.2f}")
    custom_print(f"Sharpe Ratio: {sharpe:.4f}")
    
    # Show sample predictions
    custom_print("\n" + "=" * 60)
    custom_print("Sample Predictions (first 10):")
    custom_print("=" * 60)
    for i in range(min(10, len(predictions))):
        actual = actual_test[i]
        pred_cat, action = predictions[i]
        prev_cat = test_cats[i+1]
        custom_print(f"Day {i+3}: Category {prev_cat} -> Predicted={pred_cat}, Actual={actual}, Action={action.upper()}")
    
    custom_print("\n" + "=" * 60)
    custom_print("Execution Complete")
    custom_print("=" * 60)

if __name__ == "__main__":
    main()