# Optimal Trading Strategy Calculator

This project calculates optimal trading strategies with perfect foresight using dynamic programming. It accounts for transaction fees and slippage while ensuring non-overlapping trades.

## Features

- Dynamic programming approach for O(n) efficiency
- Handles 4+ million data points efficiently
- Accounts for transaction fees and slippage
- Ensures non-overlapping trades
- Tracks three position states: flat, long, short

## Usage

```python
from optimal_trading import OptimalTradingStrategy

# Initialize with transaction fee rate
strategy = OptimalTradingStrategy(fee_rate=0.002)

# Calculate optimal trades
result = strategy.calculate_optimal_trades(your_dataframe)
```

## Input Data Format

DataFrame must contain:
- timestamp
- open
- high  
- low
- close
- volume

## Output

Adds columns to input DataFrame:
- optimal_action: Trading action ('buy_long', 'sell_long', 'buy_short', 'sell_short', 'hold')
- optimal_capital: Capital value at each timestamp
- position_state: Current position (0=flat, 1=long, 2=short)

## Requirements

See requirements.txt for dependencies.