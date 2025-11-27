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
        
    def find_optimal_trades(self):
        """Dynamic Programming to find optimal trade sequence"""
        print(f"Processing {self.n} data points...")
        print("Running DP optimization (this may take a few minutes)...")
        
        # DP state: dp[t][position] = (capital, entry_price, entry_capital)
        # capital: current capital value
        # entry_price: price at which current position was entered (if holding)
        # entry_capital: capital when position was entered (for P&L calc)
        
        dp_capital = np.ones((self.n, 3)) * -np.inf
        dp_entry_price = np.zeros((self.n, 3))
        dp_entry_capital = np.zeros((self.n, 3))
        backpointer = np.zeros((self.n, 3, 2), dtype=np.int32)  # [t][pos] = (prev_pos, prev_t)
        
        # Initial state: start FLAT with initial capital
        dp_capital[0][POSITION_FLAT] = INITIAL_CAPITAL
        
        # Forward pass
        for t in range(self.n - 1):
            if t % 100000 == 0:
                print(f"Progress: {t}/{self.n} ({100*t/self.n:.1f}%)")
            
            next_price = self.prices[t + 1]  # Execution price due to slippage
            
            # From FLAT
            if dp_capital[t][POSITION_FLAT] > 0:
                capital_flat = dp_capital[t][POSITION_FLAT]
                
                # Stay FLAT
                if capital_flat > dp_capital[t+1][POSITION_FLAT]:
                    dp_capital[t+1][POSITION_FLAT] = capital_flat
                    backpointer[t+1][POSITION_FLAT] = [POSITION_FLAT, t]
                
                # Enter LONG (pay fee, execute at next_price)
                capital_after_fee = capital_flat * (1 - TRANSACTION_FEE)
                if capital_after_fee > dp_capital[t+1][POSITION_LONG]:
                    dp_capital[t+1][POSITION_LONG] = capital_after_fee
                    dp_entry_price[t+1][POSITION_LONG] = next_price
                    dp_entry_capital[t+1][POSITION_LONG] = capital_after_fee
                    backpointer[t+1][POSITION_LONG] = [POSITION_FLAT, t]
                
                # Enter SHORT (pay fee, execute at next_price)
                capital_after_fee = capital_flat * (1 - TRANSACTION_FEE)
                if capital_after_fee > dp_capital[t+1][POSITION_SHORT]:
                    dp_capital[t+1][POSITION_SHORT] = capital_after_fee
                    dp_entry_price[t+1][POSITION_SHORT] = next_price
                    dp_entry_capital[t+1][POSITION_SHORT] = capital_after_fee
                    backpointer[t+1][POSITION_SHORT] = [POSITION_FLAT, t]
            
            # From LONG
            if dp_capital[t][POSITION_LONG] > 0:
                entry_price = dp_entry_price[t][POSITION_LONG]
                entry_capital = dp_entry_capital[t][POSITION_LONG]
                
                if entry_price > 0:  # Valid entry
                    # Calculate current position value
                    # Long: if price goes up 10%, capital goes up 10%
                    price_change_pct = (next_price - entry_price) / entry_price
                    current_value = entry_capital * (1 + price_change_pct)
                    
                    # Hold LONG
                    if current_value > dp_capital[t+1][POSITION_LONG]:
                        dp_capital[t+1][POSITION_LONG] = current_value
                        dp_entry_price[t+1][POSITION_LONG] = entry_price
                        dp_entry_capital[t+1][POSITION_LONG] = entry_capital
                        backpointer[t+1][POSITION_LONG] = [POSITION_LONG, t]
                    
                    # Exit to FLAT (pay fee on current value)
                    capital_after_exit = current_value * (1 - TRANSACTION_FEE)
                    if capital_after_exit > dp_capital[t+1][POSITION_FLAT]:
                        dp_capital[t+1][POSITION_FLAT] = capital_after_exit
                        backpointer[t+1][POSITION_FLAT] = [POSITION_LONG, t]
                    
                    # Reverse to SHORT (exit + enter, pay 2 fees)
                    capital_after_reverse = current_value * (1 - TRANSACTION_FEE) * (1 - TRANSACTION_FEE)
                    if capital_after_reverse > dp_capital[t+1][POSITION_SHORT]:
                        dp_capital[t+1][POSITION_SHORT] = capital_after_reverse
                        dp_entry_price[t+1][POSITION_SHORT] = next_price
                        dp_entry_capital[t+1][POSITION_SHORT] = capital_after_reverse
                        backpointer[t+1][POSITION_SHORT] = [POSITION_LONG, t]
            
            # From SHORT
            if dp_capital[t][POSITION_SHORT] > 0:
                entry_price = dp_entry_price[t][POSITION_SHORT]
                entry_capital = dp_entry_capital[t][POSITION_SHORT]
                
                if entry_price > 0:  # Valid entry
                    # Calculate current position value
                    # Short: if price goes down 10%, capital goes up 10%
                    # Short: if price goes up 10%, capital goes down 10%
                    price_change_pct = (next_price - entry_price) / entry_price
                    current_value = entry_capital * (1 - price_change_pct)
                    
                    # Prevent negative capital from short positions
                    if current_value <= 0:
                        current_value = 0.01  # Margin call / liquidation
                    
                    # Hold SHORT
                    if current_value > dp_capital[t+1][POSITION_SHORT]:
                        dp_capital[t+1][POSITION_SHORT] = current_value
                        dp_entry_price[t+1][POSITION_SHORT] = entry_price
                        dp_entry_capital[t+1][POSITION_SHORT] = entry_capital
                        backpointer[t+1][POSITION_SHORT] = [POSITION_SHORT, t]
                    
                    # Exit to FLAT (pay fee on current value)
                    capital_after_exit = current_value * (1 - TRANSACTION_FEE)
                    if capital_after_exit > dp_capital[t+1][POSITION_FLAT]:
                        dp_capital[t+1][POSITION_FLAT] = capital_after_exit
                        backpointer[t+1][POSITION_FLAT] = [POSITION_SHORT, t]
                    
                    # Reverse to LONG (exit + enter, pay 2 fees)
                    capital_after_reverse = current_value * (1 - TRANSACTION_FEE) * (1 - TRANSACTION_FEE)
                    if capital_after_reverse > dp_capital[t+1][POSITION_LONG]:
                        dp_capital[t+1][POSITION_LONG] = capital_after_reverse
                        dp_entry_price[t+1][POSITION_LONG] = next_price
                        dp_entry_capital[t+1][POSITION_LONG] = capital_after_reverse
                        backpointer[t+1][POSITION_LONG] = [POSITION_SHORT, t]
        
        # Find best final position
        final_capitals = dp_capital[self.n-1]
        best_final_pos = np.argmax(final_capitals)
        final_capital = final_capitals[best_final_pos]
        
        print(f"\nOptimization complete!")
        print(f"Final capital: ${final_capital:,.2f}")
        print(f"Return: {(final_capital/INITIAL_CAPITAL - 1)*100:.2f}%")
        
        # Backtrack to reconstruct trades
        self._backtrack(dp_capital, dp_entry_price, backpointer, best_final_pos)
        
        return final_capital
    
    def _backtrack(self, dp_capital, dp_entry_price, backpointer, final_pos):
        """Reconstruct optimal trades from backpointers"""
        print("\nReconstructing optimal trade sequence...")
        
        trades = []
        capital_hist = []
        position_hist = []
        
        # Backtrack from end to start to build path
        t = self.n - 1
        current_pos = final_pos
        path = [(t, current_pos)]
        
        while t > 0:
            prev_pos, prev_t = backpointer[t][current_pos]
            path.append((prev_t, prev_pos))
            t = prev_t
            current_pos = prev_pos
        
        path.reverse()
        
        # Forward pass to build trade list and history
        current_position = POSITION_FLAT
        entry_index = -1
        entry_price = 0
        entry_capital = INITIAL_CAPITAL
        
        for i in range(len(path)):
            t, pos = path[i]
            capital = dp_capital[t][pos]
            
            capital_hist.append(capital)
            position_hist.append(pos)
            
            # Detect position changes
            if pos != current_position:
                # Record exit of previous position
                if current_position != POSITION_FLAT and entry_index >= 0:
                    # Find the actual exit point (when we transitioned to FLAT or reversed)
                    exit_price = self.prices[t]
                    exit_capital = capital if pos == POSITION_FLAT else dp_capital[t-1][current_position] if t > 0 else capital
                    
                    trades.append({
                        'entry_index': entry_index,
                        'exit_index': t,
                        'entry_time': self.df.index[entry_index],
                        'exit_time': self.df.index[t],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': 'LONG' if current_position == POSITION_LONG else 'SHORT',
                        'entry_capital': entry_capital,
                        'exit_capital': exit_capital
                    })
                
                # Record entry of new position
                if pos != POSITION_FLAT:
                    entry_index = t
                    entry_price = dp_entry_price[t][pos]
                    entry_capital = capital
                
                current_position = pos
        
        # Handle final open position if any
        if current_position != POSITION_FLAT and entry_index >= 0:
            t = self.n - 1
            exit_price = self.prices[t]
            exit_capital = dp_capital[t][current_position]
            
            trades.append({
                'entry_index': entry_index,
                'exit_index': t,
                'entry_time': self.df.index[entry_index],
                'exit_time': self.df.index[t],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': 'LONG' if current_position == POSITION_LONG else 'SHORT',
                'entry_capital': entry_capital,
                'exit_capital': exit_capital
            })
        
        self.trades = trades
        self.capital_history = capital_hist
        self.position_history = position_hist
        
        print(f"Total trades: {len(trades)}")
        
        # Debug: show first few trades
        for i, trade in enumerate(trades[:5]):
            ret = (trade['exit_capital'] / trade['entry_capital'] - 1) * 100
            print(f"Trade {i+1}: {trade['position']} @ ${trade['entry_price']:.2f} -> ${trade['exit_price']:.2f}, Return: {ret:.2f}%")
    
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
        
        # Entry and exit points
        long_entries_x, long_entries_y = [], []
        long_exits_x, long_exits_y = [], []
        short_entries_x, short_entries_y = [], []
        short_exits_x, short_exits_y = [], []
        
        for trade in self.trades:
            if trade['position'] == 'LONG':
                long_entries_x.append(trade['entry_time'])
                long_entries_y.append(trade['entry_price'])
                long_exits_x.append(trade['exit_time'])
                long_exits_y.append(trade['exit_price'])
            else:
                short_entries_x.append(trade['entry_time'])
                short_entries_y.append(trade['entry_price'])
                short_exits_x.append(trade['exit_time'])
                short_exits_y.append(trade['exit_price'])
        
        # Long entries
        if long_entries_x:
            fig.add_trace(
                go.Scatter(
                    x=long_entries_x, y=long_entries_y,
                    mode='markers',
                    marker=dict(color='green', size=10, symbol='triangle-up'),
                    name='Long Entry'
                ),
                row=1, col=1
            )
        
        # Long exits
        if long_exits_x:
            fig.add_trace(
                go.Scatter(
                    x=long_exits_x, y=long_exits_y,
                    mode='markers',
                    marker=dict(color='lightgreen', size=10, symbol='triangle-down'),
                    name='Long Exit'
                ),
                row=1, col=1
            )
        
        # Short entries
        if short_entries_x:
            fig.add_trace(
                go.Scatter(
                    x=short_entries_x, y=short_entries_y,
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='triangle-down'),
                    name='Short Entry'
                ),
                row=1, col=1
            )
        
        # Short exits
        if short_exits_x:
            fig.add_trace(
                go.Scatter(
                    x=short_exits_x, y=short_exits_y,
                    mode='markers',
                    marker=dict(color='pink', size=10, symbol='triangle-up'),
                    name='Short Exit'
                ),
                row=1, col=1
            )
        
        # Capital curve
        if len(self.capital_history) > 0:
            capital_sample = range(0, len(self.capital_history), max(1, len(self.capital_history) // 10000))
            capital_times = []
            for i in capital_sample:
                if i < len(self.df.index):
                    # Need to map capital_history index to dataframe index
                    # Capital history follows the path, not 1:1 with df
                    capital_times.append(self.df.index[min(i * (self.n // len(self.capital_history)), self.n-1)])
            
            fig.add_trace(
                go.Scatter(
                    x=capital_times,
                    y=[self.capital_history[i] for i in capital_sample],
                    name='Capital',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Capital ($)", row=2, col=1, type='log')
        
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
    
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")
    
    # Assuming CSV has timestamp and OHLCV columns
    # Adjust column names based on actual CSV structure
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    else:
        # Use first column as index if it looks like a timestamp
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
    
    print(f"Loaded {len(df)} rows")
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
    trades_html = "<table border='1' style='border-collapse: collapse; width: 100%; margin-top: 20px;'>"
    trades_html += "<tr style='background: #333; color: white;'><th>Entry Time</th><th>Exit Time</th><th>Position</th><th>Entry Price</th><th>Exit Price</th><th>Entry Capital</th><th>Exit Capital</th><th>Return %</th><th>Duration</th></tr>"
    
    for i, trade in enumerate(finder.trades[:100]):  # Show first 100 trades
        ret = (trade['exit_capital'] / trade['entry_capital'] - 1) * 100
        duration = trade['exit_index'] - trade['entry_index']
        bg = '#f9f9f9' if i % 2 == 0 else 'white'
        trades_html += f"<tr style='background: {bg};'><td>{trade['entry_time']}</td><td>{trade['exit_time']}</td><td>{trade['position']}</td>"
        trades_html += f"<td>${trade['entry_price']:.4f}</td><td>${trade['exit_price']:.4f}</td>"
        trades_html += f"<td>${trade['entry_capital']:.2f}</td><td>${trade['exit_capital']:.2f}</td>"
        trades_html += f"<td style='color: {'green' if ret > 0 else 'red'}; font-weight: bold;'>{ret:.2f}%</td>"
        trades_html += f"<td>{duration} min</td></tr>"
    
    trades_html += "</table>"
    
    final_capital = finder.capital_history[-1] if finder.capital_history else INITIAL_CAPITAL
    total_return = (final_capital / INITIAL_CAPITAL - 1) * 100
    
    # Calculate win rate
    winning_trades = sum(1 for t in finder.trades if t['exit_capital'] > t['entry_capital'])
    win_rate = (winning_trades / len(finder.trades) * 100) if finder.trades else 0
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Optimal Trading Strategy</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .summary {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric {{ display: inline-block; margin-right: 30px; }}
            .metric-label {{ color: #666; font-size: 14px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
            .positive {{ color: #00a000; }}
            .negative {{ color: #d00; }}
        </style>
    </head>
    <body>
        <h1>Optimal Trading Strategy Results</h1>
        <div class="summary">
            <h2>Summary Statistics</h2>
            <div class="metric">
                <div class="metric-label">Initial Capital</div>
                <div class="metric-value">${INITIAL_CAPITAL:,.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Final Capital</div>
                <div class="metric-value">${final_capital:,.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Return</div>
                <div class="metric-value positive">{total_return:.2f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Number of Trades</div>
                <div class="metric-value">{len(finder.trades)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{win_rate:.1f}%</div>
            </div>
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
                <p style="margin: 5px 0;"><strong>Transaction Fee:</strong> {TRANSACTION_FEE*100}% per trade leg (0.4% round-trip)</p>
                <p style="margin: 5px 0;"><strong>Slippage:</strong> {SLIPPAGE_MINUTES} minute execution delay</p>
            </div>
        </div>
        
        {chart_html}
        
        <h2 style="margin-top: 30px;">Trade Details (First 100)</h2>
        {trades_html}
        
        <div style="margin-top: 20px; padding: 15px; background: #ffffcc; border-radius: 5px;">
            <strong>Note:</strong> This shows the theoretically optimal trading sequence given perfect hindsight. 
            The algorithm found the sequence of long/short positions that maximizes final capital while accounting for 
            transaction fees and execution slippage.
        </div>
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
