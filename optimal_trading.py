import pandas as pd
import numpy as np
from typing import Tuple, List

class OptimalTradingStrategy:
    """
    Calculate optimal trading strategy with perfect foresight using dynamic programming.
    Accounts for transaction fees and slippage while ensuring non-overlapping trades.
    """
    
    def __init__(self, fee_rate: float = 0.002):
        """
        Initialize with transaction fee rate.
        
        Args:
            fee_rate: Transaction fee as decimal (e.g., 0.002 for 0.2%)
        """
        self.fee_rate = fee_rate
    
    def calculate_optimal_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate optimal trading strategy for given OHLCV data.
        Accounts for 1-minute slippage by executing at next period's open price.
        
        Args:
            df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with added columns: 'optimal_action', 'optimal_capital', 'position_state'
        """
        n = len(df)
        
        # Initialize DP arrays
        capital_flat = np.zeros(n)  # Capital when flat
        capital_long = np.zeros(n)  # Capital when holding long
        capital_short = np.zeros(n)  # Capital when holding short
        
        # Track previous states for path reconstruction
        prev_flat = np.zeros(n, dtype=int)
        prev_long = np.zeros(n, dtype=int) 
        prev_short = np.zeros(n, dtype=int)
        
        # Initialize with starting capital (1 unit)
        capital_flat[0] = 1.0
        capital_long[0] = -np.inf  # Cannot start in long position
        capital_short[0] = -np.inf  # Cannot start in short position
        
        # Calculate price changes with 1-minute slippage
        # Entry: current close price + fee
        # Exit: next period's open price - fee (1-minute slippage)
        entry_prices = df['close'].values
        exit_prices = df['open'].shift(-1).fillna(df['close'].iloc[-1]).values
        
        for i in range(1, n):
            # From FLAT state:
            # Option 1: Stay flat
            flat_from_flat = capital_flat[i-1]
            # Option 2: Enter long from flat (execute at current close with fee)
            long_entry_cost = entry_prices[i] * (1 + self.fee_rate)
            long_from_flat = (capital_flat[i-1] / long_entry_cost) if long_entry_cost > 0 else -np.inf
            # Option 3: Enter short from flat (execute at current close with fee)
            short_from_flat = capital_flat[i-1] * (1 - self.fee_rate)
            
            # From LONG state:
            # Option 1: Stay in long
            long_from_long = capital_long[i-1]
            # Option 2: Exit long to flat (execute at next open with 1-minute slippage and fee)
            long_exit_value = capital_long[i-1] * exit_prices[i] * (1 - self.fee_rate)
            
            # From SHORT state:
            # Option 1: Stay in short
            short_from_short = capital_short[i-1]
            # Option 2: Exit short to flat (execute at next open with 1-minute slippage and fee)
            short_exit_value = capital_short[i-1] / exit_prices[i] * (1 - self.fee_rate)
            
            # Update current states
            # FLAT state: can come from previous flat or exiting positions
            flat_options = [
                (flat_from_flat, 0),  # Stay flat
                (long_exit_value, 1),  # Exit long
                (short_exit_value, 2)  # Exit short
            ]
            capital_flat[i], prev_flat[i] = max(flat_options, key=lambda x: x[0])
            
            # LONG state: can come from previous long or entering from flat
            long_options = [
                (long_from_long, 1),  # Stay long
                (long_from_flat, 0)   # Enter from flat
            ]
            capital_long[i], prev_long[i] = max(long_options, key=lambda x: x[0])
            
            # SHORT state: can come from previous short or entering from flat
            short_options = [
                (short_from_short, 2),  # Stay short
                (short_from_flat, 0)    # Enter from flat
            ]
            capital_short[i], prev_short[i] = max(short_options, key=lambda x: x[0])
        
        # Reconstruct optimal path
        optimal_actions = []
        optimal_capital = []
        position_states = []
        
        # Find best final state
        final_capitals = [capital_flat[-1], capital_long[-1], capital_short[-1]]
        final_state = np.argmax(final_capitals)
        
        # Backtrack to reconstruct path
        current_state = final_state
        for i in range(n-1, -1, -1):
            optimal_capital.append(max(capital_flat[i], capital_long[i], capital_short[i]))
            position_states.append(current_state)
            
            # Determine action based on state transition
            if i == 0:
                action = 'hold'
            else:
                if current_state == 0:  # Flat
                    prev_state = prev_flat[i]
                    if prev_state == 0:
                        action = 'hold'
                    elif prev_state == 1:
                        action = 'sell_long'
                    else:  # prev_state == 2
                        action = 'buy_short'
                elif current_state == 1:  # Long
                    prev_state = prev_long[i]
                    if prev_state == 1:
                        action = 'hold'
                    else:  # prev_state == 0
                        action = 'buy_long'
                else:  # current_state == 2 (Short)
                    prev_state = prev_short[i]
                    if prev_state == 2:
                        action = 'hold'
                    else:  # prev_state == 0
                        action = 'sell_short'
            
            optimal_actions.append(action)
            
            # Move to previous state
            if current_state == 0:
                current_state = prev_flat[i]
            elif current_state == 1:
                current_state = prev_long[i]
            else:  # current_state == 2
                current_state = prev_short[i]
        
        # Reverse the lists since we backtracked
        optimal_actions.reverse()
        optimal_capital.reverse()
        position_states.reverse()
        
        # Add results to dataframe
        result_df = df.copy()
        result_df['optimal_action'] = optimal_actions
        result_df['optimal_capital'] = optimal_capital
        result_df['position_state'] = position_states
        
        return result_df


def main():
    """Example usage"""
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    
    # Generate random walk prices
    returns = np.random.normal(0, 0.001, n_samples)
    prices = 100 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Calculate optimal strategy
    strategy = OptimalTradingStrategy(fee_rate=0.002)
    result = strategy.calculate_optimal_trades(df)
    
    print("Optimal trading strategy calculated:")
    print(f"Final capital: {result['optimal_capital'].iloc[-1]:.4f}")
    print(f"Total trades: {(result['optimal_action'] != 'hold').sum()}")
    
    # Save results
    result.to_csv('optimal_trading_results.csv', index=False)
    print("Results saved to optimal_trading_results.csv")


if __name__ == "__main__":
    main()