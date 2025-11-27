import gdown
import pandas as pd
import numpy as np
from flask import Flask, render_template_string
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Configuration
TRANSACTION_FEE = 0.002  # 0.2%
SLIPPAGE_MINUTES = 1
INITIAL_CAPITAL = 10000
POSITION_FLAT = 0
POSITION_LONG = 1
POSITION_SHORT = 2

app = Flask(__name__)

class OptimalTradingFinder:
    def __init__(self, df):
        self.df = df
        self.prices = df['close'].values
        self.n = len(self.prices)
        self.trades = []
        self.capital_history = []
        self.position_history = []
        
    def calculate_capital_change(self, entry_price, exit_price, position_type):
        """Calculate capital multiplier for a trade including fees"""
        if position_type == POSITION_LONG:
            # Long: profit when price goes up
            gross_return = exit_price / entry_price
        else:  # SHORT
            # Short: profit when price goes down
            # If price drops 10%, we gain 10%
            gross_return = 2 - (exit_price / entry_price)
        
        # Apply fees: entry fee and exit fee
        net_multiplier = gross_return * (1 - TRANSACTION_FEE) * (1 - TRANSACTION_FEE)
        return net_multiplier
    
    def find_optimal_trades(self):
        """Dynamic Programming to find optimal trade sequence"""
        print(f"Processing {self.n} data points...")
        print("Running DP optimization (this may take a few minutes)...")
        
        # DP state: dp[t][position] = (capital, previous_position, entry_index)
        # We track capital and backpointers
        dp = np.ones((self.n, 3)) * -np.inf  # Initialize with -inf
        backpointer = np.zeros((self.n, 3, 3), dtype=np.int32)  # [t][pos] = (prev_pos, prev_t, entry_t)
        
        # Initial state: start FLAT with initial capital
        dp[0][POSITION_FLAT] = INITIAL_CAPITAL
        
        # Forward pass
        for t in range(self.n - 1):
            if t % 100000 == 0:
                print(f"Progress: {t}/{self.n} ({100*t/self.n:.1f}%)")
            
            current_price = self.prices[t]
            next_price = self.prices[t + 1]  # Execution price (slippage)
            
            # From FLAT
            if dp[t][POSITION_FLAT] > 0:
                capital_flat = dp[t][POSITION_FLAT]
                
                # Stay FLAT
                if capital_flat > dp[t+1][POSITION_FLAT]:
                    dp[t+1][POSITION_FLAT] = capital_flat
                    backpointer[t+1][POSITION_FLAT] = [POSITION_FLAT, t, -1]
                
                # Enter LONG (pay fee, execute at next_price)
                capital_after_entry = capital_flat * (1 - TRANSACTION_FEE)
                if capital_after_entry > dp[t+1][POSITION_LONG]:
                    dp[t+1][POSITION_LONG] = capital_after_entry
                    backpointer[t+1][POSITION_LONG] = [POSITION_FLAT, t, t+1]
                
                # Enter SHORT (pay fee)
                capital_after_entry = capital_flat * (1 - TRANSACTION_FEE)
                if capital_after_entry > dp[t+1][POSITION_SHORT]:
                    dp[t+1][POSITION_SHORT] = capital_after_entry
                    backpointer[t+1][POSITION_SHORT] = [POSITION_FLAT, t, t+1]
            
            # From LONG
            if dp[t][POSITION_LONG] > 0:
                entry_idx = backpointer[t][POSITION_LONG][2]
                if entry_idx >= 0:
                    entry_price = self.prices[entry_idx]
                    capital_at_entry = dp[t][POSITION_LONG]
                    
                    # Hold LONG (capital changes with price)
                    price_ratio = next_price / entry_price
                    capital_held = capital_at_entry * price_ratio
                    if capital_held > dp[t+1][POSITION_LONG]:
                        dp[t+1][POSITION_LONG] = capital_held
                        backpointer[t+1][POSITION_LONG] = [POSITION_LONG, t, entry_idx]
                    
                    # Exit to FLAT (pay fee on exit)
                    capital_exit = capital_held * (1 - TRANSACTION_FEE)
                    if capital_exit > dp[t+1][POSITION_FLAT]:
                        dp[t+1][POSITION_FLAT] = capital_exit
                        backpointer[t+1][POSITION_FLAT] = [POSITION_LONG, t, -1]
                    
                    # Reverse to SHORT (exit long, enter short, pay 2 fees)
                    capital_reverse = capital_held * (1 - TRANSACTION_FEE) * (1 - TRANSACTION_FEE)
                    if capital_reverse > dp[t+1][POSITION_SHORT]:
                        dp[t+1][POSITION_SHORT] = capital_reverse
                        backpointer[t+1][POSITION_SHORT] = [POSITION_LONG, t, t+1]
            
            # From SHORT
            if dp[t][POSITION_SHORT] > 0:
                entry_idx = backpointer[t][POSITION_SHORT][2]
                if entry_idx >= 0:
                    entry_price = self.prices[entry_idx]
                    capital_at_entry = dp[t][POSITION_SHORT]
                    
                    # Hold SHORT (capital changes inversely with price)
                    price_ratio = next_price / entry_price
                    capital_held = capital_at_entry * (2 - price_ratio)
                    if capital_held > dp[t+1][POSITION_SHORT]:
                        dp[t+1][POSITION_SHORT] = capital_held
                        backpointer[t+1][POSITION_SHORT] = [POSITION_SHORT, t, entry_idx]
                    
                    # Exit to FLAT (pay fee on exit)
                    capital_exit = capital_held * (1 - TRANSACTION_FEE)
                    if capital_exit > dp[t+1][POSITION_FLAT]:
                        dp[t+1][POSITION_FLAT] = capital_exit
                        backpointer[t+1][POSITION_FLAT] = [POSITION_SHORT, t, -1]
                    
                    # Reverse to LONG (exit short, enter long, pay 2 fees)
                    capital_reverse = capital_held * (1 - TRANSACTION_FEE) * (1 - TRANSACTION_FEE)
                    if capital_reverse > dp[t+1][POSITION_LONG]:
                        dp[t+1][POSITION_LONG] = capital_reverse
                        backpointer[t+1][POSITION_LONG] = [POSITION_SHORT, t, t+1]
        
        # Find best final position
        final_capitals = dp[self.n-1]
        best_final_pos = np.argmax(final_capitals)
        final_capital = final_capitals[best_final_pos]
        
        print(f"\nOptimization complete!")
        print(f"Final capital: ${final_capital:,.2f}")
        print(f"Return: {(final_capital/INITIAL_CAPITAL - 1)*100:.2f}%")
        
        # Backtrack to reconstruct trades
        self._backtrack(dp, backpointer, best_final_pos)
        
        return final_capital
    
    def _backtrack(self, dp, backpointer, final_pos):
        """Reconstruct optimal trades from backpointers"""
        print("\nReconstructing optimal trade sequence...")
        
        trades = []
        capital_hist = [INITIAL_CAPITAL]
        position_hist = [POSITION_FLAT]
        
        # Backtrack from end to start
        t = self.n - 1
        current_pos = final_pos
        path = []
        
        while t > 0:
            prev_pos, prev_t, entry_t = backpointer[t][current_pos]
            path.append((t, current_pos, entry_t))
            t = prev_t
            current_pos = prev_pos
        
        path.reverse()
        
        # Forward pass to build trade list and history
        current_capital = INITIAL_CAPITAL
        current_position = POSITION_FLAT
        entry_index = -1
        entry_capital = 0
        
        for t, pos, entry_t in path:
            # Position change detected
            if pos != current_position:
                # Closing a position
                if current_position != POSITION_FLAT:
                    exit_price = self.prices[t]
                    trades.append({
                        'entry_time': self.df.index[entry_index],
                        'exit_time': self.df.index[t],
                        'entry_price': self.prices[entry_index],
                        'exit_price': exit_price,
                        'position': 'LONG' if current_position == POSITION_LONG else 'SHORT',
                        'entry_capital': entry_capital,
                        'exit_capital': dp[t][POSITION_FLAT] if pos == POSITION_FLAT else current_capital
                    })
                
                # Opening new position
                if pos != POSITION_FLAT:
                    entry_index = entry_t
                    entry_capital = dp[t][pos]
                
                current_position = pos
            
            capital_hist.append(dp[t][pos])
            position_hist.append(pos)
        
        self.trades = trades
        self.capital_history = capital_hist
        self.position_history = position_hist
        
        print(f"Total trades: {len(trades)}")
    
    def create_visualization(self):
        """Create interactive Plotly visualization"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price with Entry/Exit Points', 'Capital Curve'),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        # Subsample for visualization if too many points
        sample_rate = max(1, self.n // 10000)
        plot_indices = range(0, self.n, sample_rate)
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=self.df.index[plot_indices],
                y=self.prices[plot_indices],
                name='Price',
                line=dict(color='gray', width=1)
            ),
            row=1, col=1
        )
        
        # Entry points
        for trade in self.trades:
            color = 'green' if trade['position'] == 'LONG' else 'red'
            fig.add_trace(
                go.Scatter(
                    x=[trade['entry_time']],
                    y=[trade['entry_price']],
                    mode='markers',
                    marker=dict(color=color, size=8, symbol='triangle-up'),
                    name=f"{trade['position']} Entry",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Exit points
            fig.add_trace(
                go.Scatter(
                    x=[trade['exit_time']],
                    y=[trade['exit_price']],
                    mode='markers',
                    marker=dict(color=color, size=8, symbol='triangle-down'),
                    name=f"{trade['position']} Exit",
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Capital curve
        capital_sample = range(0, len(self.capital_history), max(1, len(self.capital_history) // 10000))
        fig.add_trace(
            go.Scatter(
                x=[self.df.index[i] if i < len(self.df.index) else self.df.index[-1] for i in capital_sample],
                y=[self.capital_history[i] for i in capital_sample],
                name='Capital',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Capital ($)", row=2, col=1)
        
        fig.update_layout(height=800, showlegend=True, title_text="Optimal Trading Strategy")
        
        return fig.to_html(full_html=False)

# Global variables
finder = None
chart_html = ""

def load_and_process_data():
    """Download and process the data"""
    global finder, chart_html
    
    print("Downloading data from Google Drive...")
    url = "https://drive.google.com/uc?id=1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o"
    output = "ohlcv_data.csv"
    gdown.download(url, output, quiet=False)
    
    print("Loading CSV...")
    df = pd.read_csv(output)
    
    # Assuming CSV has timestamp and OHLCV columns
    # Adjust column names based on actual CSV structure
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Initialize finder
    finder = OptimalTradingFinder(df)
    
    # Find optimal trades
    final_capital = finder.find_optimal_trades()
    
    # Create visualization
    print("Creating visualization...")
    chart_html = finder.create_visualization()
    
    print("Ready to serve web interface!")

@app.route('/')
def index():
    if finder is None:
        return "<h1>Processing data, please wait...</h1>"
    
    # Generate trade table HTML
    trades_html = "<table border='1' style='border-collapse: collapse; width: 100%;'>"
    trades_html += "<tr><th>Entry Time</th><th>Exit Time</th><th>Position</th><th>Entry Price</th><th>Exit Price</th><th>Entry Capital</th><th>Exit Capital</th><th>Return</th></tr>"
    
    for trade in finder.trades[:100]:  # Show first 100 trades
        ret = (trade['exit_capital'] / trade['entry_capital'] - 1) * 100
        trades_html += f"<tr><td>{trade['entry_time']}</td><td>{trade['exit_time']}</td><td>{trade['position']}</td>"
        trades_html += f"<td>${trade['entry_price']:.2f}</td><td>${trade['exit_price']:.2f}</td>"
        trades_html += f"<td>${trade['entry_capital']:.2f}</td><td>${trade['exit_capital']:.2f}</td>"
        trades_html += f"<td style='color: {'green' if ret > 0 else 'red'};'>{ret:.2f}%</td></tr>"
    
    trades_html += "</table>"
    
    final_capital = finder.capital_history[-1] if finder.capital_history else INITIAL_CAPITAL
    total_return = (final_capital / INITIAL_CAPITAL - 1) * 100
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Optimal Trading Strategy</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body style="font-family: Arial, sans-serif; margin: 20px;">
        <h1>Optimal Trading Strategy Results</h1>
        <div style="background: #f0f0f0; padding: 20px; margin-bottom: 20px; border-radius: 5px;">
            <h2>Summary Statistics</h2>
            <p><strong>Initial Capital:</strong> ${INITIAL_CAPITAL:,.2f}</p>
            <p><strong>Final Capital:</strong> ${final_capital:,.2f}</p>
            <p><strong>Total Return:</strong> <span style="color: green; font-size: 24px;">{total_return:.2f}%</span></p>
            <p><strong>Number of Trades:</strong> {len(finder.trades)}</p>
            <p><strong>Transaction Fee:</strong> {TRANSACTION_FEE*100}% per trade</p>
            <p><strong>Slippage:</strong> {SLIPPAGE_MINUTES} minute(s)</p>
        </div>
        
        {chart_html}
        
        <h2>Trade Details (First 100 Trades)</h2>
        {trades_html}
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    # Load and process data first
    load_and_process_data()
    
    # Start web server
    print("\nStarting web server on http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
