
import random
import time
from collections import defaultdict
from typing import List, Tuple

def custom_print(text: str, delay: float = 0.1):
    """Print with delay"""
    print(text)
    time.sleep(delay)

def generate_prices(n: int, min_price: float = 100, max_price: float = 600) -> List[float]:
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

def compute_sequence_probabilities(prices: List[float], min_price: float, max_price: float, n_categories: int) -> dict:
    """Compute probability of each 3-sequential price category sequence"""
    categorized = [categorize_price(p, min_price, max_price, n_categories) for p in prices]
    sequence_counts = defaultdict(int)
    total_sequences = 0
    
    for i in range(len(categorized) - 2):
        seq = tuple(categorized[i:i+3])
        sequence_counts[seq] += 1
        total_sequences += 1
    
    probabilities = {seq: count / total_sequences for seq, count in sequence_counts.items()}
    return probabilities

def compute_directional_probabilities(prices: List[float]) -> dict:
    """Compute probability of directional sequences"""
    directions = [prices[i+1] - prices[i] for i in range(len(prices) - 1)]
    
    dir_sequence_counts = defaultdict(int)
    total_sequences = 0
    
    for i in range(len(directions) - 2):
        seq = tuple(directions[i:i+3])
        dir_sequence_counts[seq] += 1
        total_sequences += 1
    
    probabilities = {seq: count / total_sequences for seq, count in dir_sequence_counts.items()}
    return probabilities

def generate_variants(last_two_prices: List[float], step: float = 1.0) -> List[List[float]]:
    """Generate all sequences 1 step away from last two prices"""
    p1, p2 = last_two_prices
    
    # Edit first price
    variants_step1 = [
        [p1 - step, p2],
        [p1 + step, p2],
        [p1, p2 - step],
        [p1, p2 + step]
    ]
    
    # Append third price with 1 step variations
    variants_step2 = []
    for var in variants_step1:
        variants_step2.extend([
            var + [p2 - step],
            var + [p2],
            var + [p2 + step]
        ])
    
    return variants_step2

def generate_directional_variants(price_diff: Tuple[float, float], step: float = 1.0) -> List[List[float]]:
    """Generate directional variants"""
    d1, d2 = price_diff
    
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

def predict_next_price(
    last_two_prices: List[float],
    price_probs: dict,
    dir_probs: dict,
    min_price: float,
    max_price: float,
    n_categories: int,
    step: float = 1.0
) -> Tuple[float, str]:
    """Predict next price and action"""
    
    # Generate price variants
    price_variants = generate_variants(last_two_prices, step)
    
    # Convert to directional
    p_prev = last_two_prices[0]
    directional = [last_two_prices[0] - p_prev, last_two_prices[1] - last_two_prices[0]]
    dir_variants = generate_directional_variants(tuple(directional), step)
    
    best_prob = -1
    best_prediction = None
    
    # Evaluate each variant
    for i, p_var in enumerate(price_variants):
        # Categorize price sequence
        cat_seq = tuple(categorize_price(p, min_price, max_price, n_categories) for p in p_var)
        price_prob = price_probs.get(cat_seq, 0)
        
        # Get directional sequence
        dir_seq = tuple(dir_variants[i])
        dir_prob = dir_probs.get(dir_seq, 0)
        
        # Combined probability
        combined_prob = price_prob + dir_prob
        
        if combined_prob > best_prob:
            best_prob = combined_prob
            best_prediction = p_var[2]
    
    # Determine action
    current_price = last_two_prices[1]
    if best_prediction is None:
        best_prediction = current_price
    
    if best_prediction > current_price:
        action = "buy"
    elif best_prediction < current_price:
        action = "sell"
    else:
        action = "hold"
    
    return best_prediction, action

def evaluate_predictions(actual_prices: List[float], predictions: List[Tuple[float, str]]) -> Tuple[float, float, float]:
    """Evaluate accuracy, equity, and Sharpe ratio"""
    correct = 0
    equity = 10000  # Starting equity
    returns = []
    position = 0  # 0: no position, 1: long, -1: short
    entry_price = 0
    
    for i in range(len(predictions)):
        pred_price, action = predictions[i]
        actual = actual_prices[i]
        
        # Check direction accuracy
        if i > 0:
            actual_direction = actual - actual_prices[i-1]
            pred_direction = pred_price - actual_prices[i-1]
            if (actual_direction > 0 and pred_direction > 0) or \
               (actual_direction < 0 and pred_direction < 0) or \
               (actual_direction == 0 and pred_direction == 0):
                correct += 1
        
        # Simulate trading
        if action == "buy" and position == 0:
            position = 1
            entry_price = actual
        elif action == "sell" and position == 1:
            pnl = actual - entry_price
            equity += pnl
            returns.append(pnl / entry_price)
            position = 0
        elif action == "sell" and position == 0:
            position = -1
            entry_price = actual
        elif action == "buy" and position == -1:
            pnl = entry_price - actual
            equity += pnl
            returns.append(pnl / entry_price)
            position = 0
    
    # Close any open position
    if position != 0 and len(actual_prices) > 0:
        final_price = actual_prices[-1]
        if position == 1:
            pnl = final_price - entry_price
            equity += pnl
            returns.append(pnl / entry_price)
        else:
            pnl = entry_price - final_price
            equity += pnl
            returns.append(pnl / entry_price)
    
    accuracy = correct / max(len(predictions) - 1, 1)
    
    # Sharpe ratio
    if len(returns) > 0:
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        sharpe = avg_return / std_return if std_return > 0 else 0
    else:
        sharpe = 0
    
    return accuracy, equity, sharpe

def main():
    custom_print("=" * 60)
    custom_print("Price Prediction Algorithm")
    custom_print("=" * 60)
    
    # Parameters
    n_prices = 20000
    min_price = 100
    max_price = 600
    n_categories = 100
    train_split = 0.7
    step = 5.0
    
    custom_print(f"\nGenerating {n_prices} mock prices...")
    prices = generate_prices(n_prices, min_price, max_price)
    
    # Split data
    split_idx = int(n_prices * train_split)
    train_prices = prices[:split_idx]
    test_prices = prices[split_idx:]
    
    custom_print(f"Train set: {len(train_prices)} prices")
    custom_print(f"Test set: {len(test_prices)} prices")
    
    # Compute probabilities on training set
    custom_print("\nComputing price sequence probabilities...")
    price_probs = compute_sequence_probabilities(train_prices, min_price, max_price, n_categories)
    custom_print(f"Found {len(price_probs)} unique price sequences")
    
    custom_print("\nComputing directional sequence probabilities...")
    dir_probs = compute_directional_probabilities(train_prices)
    custom_print(f"Found {len(dir_probs)} unique directional sequences")
    
    # Make predictions
    custom_print("\nGenerating predictions...")
    predictions = []
    
    for i in range(2, len(test_prices)):
        last_two = test_prices[i-2:i]
        pred_price, action = predict_next_price(
            last_two, price_probs, dir_probs, 
            min_price, max_price, n_categories, step
        )
        predictions.append((pred_price, action))
        
        if i % 500 == 0:
            custom_print(f"  Processed {i}/{len(test_prices)} test samples...")
    
    custom_print(f"\nGenerated {len(predictions)} predictions")
    
    # Evaluate
    custom_print("\nEvaluating predictions...")
    actual_test = test_prices[2:]
    accuracy, equity, sharpe = evaluate_predictions(actual_test, predictions)
    
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
        pred_price, action = predictions[i]
        custom_print(f"Day {i+3}: Predicted={pred_price:.2f}, Actual={actual:.2f}, Action={action.upper()}")
    
    custom_print("\n" + "=" * 60)
    custom_print("Execution Complete")
    custom_print("=" * 60)

if __name__ == "__main__":
    main()