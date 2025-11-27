import gdown
import pandas as pd
import numpy as np
from flask import Flask
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
TRANSACTION_FEE = 0.002  # 0.2% per trade leg
INITIAL_CAPITAL = 10000

app = Flask(__name__)

class OptimalTradingFinder:
    def __init__(self, df):
        self.df = df
        self.prices = df['close'].values
        self.n = len(self.prices)
        self.trades = []
        self.best_path = []
        self.best_capital = INITIAL_CAPITAL
        
    def find_optimal_trades_dp(self):
        """Simple DP: at each day, track best capital for FLAT/LONG/SHORT"""
        print(f"Processing {self.n} daily data points...")
        print("Running DP optimization...")
        
        # State: position (0=FLAT, 1=LONG, 2=SHORT)
        # dp[day][position] = (capital, entry_day, entry_price)
        INF = -1e18
        dp = np.full((self.n, 3, 3), INF, dtype=np.float64)
        # dp[day][position] = [capital, entry_day, entry_price]
        
        # Initial state: FLAT with initial capital
        dp[0][0] = [INITIAL_CAPITAL, -1, 0]
        
        for day in range(self.n - 1):
            if day % 100 == 0:
                print(f"Progress: {day}/{self.n}")
            
            price_today = self.prices[day]
            price_next = self.prices[day + 1]
            
            # FROM FLAT
            if dp[day][0][0] > 0:
                capital = dp[day][0][0]
                
                # Stay FLAT
                if capital > dp[day+1][0][0]:
                    dp[day+1][0] = [capital, -1, 0]
                
                # Enter LONG (pay entry fee)
                capital_after_fee = capital * (1 - TRANSACTION_FEE)
                if capital_after_fee > dp[day+1][1][0]:
                    dp[day+1][1] = [capital_after_fee, day+1, price_next]
                
                # Enter SHORT (pay entry fee)
                if capital_after_fee > dp[day+1][2][0]:
                    dp[day+1][2] = [capital_after_fee, day+1, price_next]
            
            # FROM LONG
            if dp[day][1][0] > 0:
                entry_capital = dp[day][1][0]
                entry_day = int(dp[day][1][1])
                entry_price = dp[day][1][2]
                
                if entry_price > 0:
                    # Calculate current value
                    price_change = (price_next - entry_price) / entry_price
                    current_value = entry_capital * (1 + price_change)
                    
                    # Hold LONG
                    if current_value > dp[day+1][1][0]:
                        dp[day+1][1] = [current_value, entry_day, entry_price]
                    
                    # Exit to FLAT (pay exit fee)
                    exit_capital = current_value * (1 - TRANSACTION_FEE)
                    if exit_capital > dp[day+1][0][0]:
                        dp[day+1][0] = [exit_capital, -1, 0]
                    
                    # Reverse to SHORT (pay exit + entry fees)
                    reverse_capital = current_value * (1 - TRANSACTION_FEE) * (1 - TRANSACTION_FEE)
                    if reverse_capital > dp[day+1][2][0]:
                        dp[day+1][2] = [reverse_capital, day+1, price_next]
            
            # FROM SHORT
            if dp[day][2][0] > 0:
                entry_capital = dp[day][2][0]
                entry_day = int(dp[day][2][1])
                entry_price = dp[day][2][2]
                
                if entry_price > 0:
                    # Calculate current value (inverse of price movement)
                    price_change = (price_next - entry_price) / entry_price
                    current_value = entry_capital * (1 - price_change)
                    
                    # Prevent going negative (margin call)
                    current_value = max(current_value, 0.01)
                    
                    # Hold SHORT
                    if current_value > dp[day+1][2][0]:
                        dp[day+1][2] = [current_value, entry_day, entry_price]
                    
                    # Exit to FLAT (pay exit fee)
                    exit_capital = current_value * (1 - TRANSACTION_FEE)
                    if exit_capital > dp[day+1][0][0]:
                        dp[day+1][0] = [exit_capital, -1, 0]
                    
                    # Reverse to LONG (pay exit + entry fees)
                    reverse_capital = current_value * (1 - TRANSACTION_FEE) * (1 - TRANSACTION_FEE)
                    if reverse_capital > dp[day+1][1][0]:
                        dp[day+1][1] = [reverse_capital, day+1, price_next]
        
        # Find best final position
        final_capitals = dp[self.n-1, :, 0]
        best_pos = np.argmax(final_capitals)
        final_capital = final_capitals[best_pos]
        
        print(f"\nOptimization complete!")
        print(f"Final capital: ${final_capital:,.2f}")
        print(f"Return: {(final_capital/INITIAL_CAPITAL - 1)*100:.2f}%")
        
        # Reconstruct trades
        self._reconstruct_trades_from_dp(dp, best_pos)
        
        return final_capital
    
    def _reconstruct_trades_from_dp(self, dp, final_pos):
        """Reconstruct the optimal path and extract trades"""
        print("\nReconstructing trades...")
        
        # Work backwards to find the path
        path = []  # (day, position, capital, entry_day, entry_price)
        day = self.n - 1
        pos = final_pos
        
        path.append((day, pos, dp[day][pos][0], int(dp[day][pos][1]), dp[day][pos][2]))
        
        # Backtrack by finding which previous state led to current state
        while day > 0:
            current_capital = dp[day][pos][0]
            current_entry_day = int(dp[day][pos][1])
            current_entry_price = dp[day][pos][2]
            
            found = False
            # Check previous day for matching state
            for prev_pos in range(3):
                if dp[day-1][prev_pos][0] <= 0:
                    continue
                
                # Simulate transition from prev_pos to pos
                prev_capital = dp[day-1][prev_pos][0]
                prev_entry_day = int(dp[day-1][prev_pos][1])
                prev_entry_price = dp[day-1][prev_pos][2]
                
                price_today = self.prices[day-1]
                price_next = self.prices[day]
                
                expected_capital = None
                expected_entry_day = current_entry_day
                expected_entry_price = current_entry_price
                
                # Check if this transition makes sense
                if prev_pos == pos and pos != 0:  # Holding position
                    if prev_entry_day == current_entry_day and abs(prev_entry_price - current_entry_price) < 1e-6:
                        price_change = (price_next - prev_entry_price) / prev_entry_price
                        if pos == 1:  # LONG
                            expected_capital = prev_capital * (1 + price_change)
                        else:  # SHORT
                            expected_capital = max(prev_capital * (1 - price_change), 0.01)
                        
                        if abs(expected_capital - current_capital) < 1e-6:
                            found = True
                            day = day - 1
                            pos = prev_pos
                            path.append((day, pos, prev_capital, prev_entry_day, prev_entry_price))
                            break
                
                elif prev_pos == 0 and pos == 0:  # Staying FLAT
                    if abs(prev_capital - current_capital) < 1e-6:
                        found = True
                        day = day - 1
                        pos = prev_pos
                        path.append((day, pos, prev_capital, -1, 0))
                        break
                
                elif prev_pos == 0 and pos != 0:  # Entering position from FLAT
                    expected_capital = prev_capital * (1 - TRANSACTION_FEE)
                    if abs(expected_capital - current_capital) < 1e-6 and current_entry_day == day:
                        found = True
                        day = day - 1
                        pos = prev_pos
                        path.append((day, pos, prev_capital, -1, 0))
                        break
                
                elif prev_pos != 0 and pos == 0:  # Exiting to FLAT
                    # Calculate what prev position value should be
                    implied_prev_value = current_capital / (1 - TRANSACTION_FEE)
                    
                    if prev_entry_price > 0:
                        price_change = (price_next - prev_entry_price) / prev_entry_price
                        if prev_pos == 1:  # Was LONG
                            expected_prev_capital = prev_capital * (1 + price_change)
                        else:  # Was SHORT
                            expected_prev_capital = max(prev_capital * (1 - price_change), 0.01)
                        
                        if abs(expected_prev_capital - implied_prev_value) < 1e-2:
                            found = True
                            day = day - 1
                            pos = prev_pos
                            path.append((day, pos, prev_capital, prev_entry_day, prev_entry_price))
                            break
                
                elif prev_pos != 0 and pos != 0 and prev_pos != pos:  # Reversal
                    implied_value = current_capital / ((1 - TRANSACTION_FEE) * (1 - TRANSACTION_FEE))
                    
                    if prev_entry_price > 0:
                        price_change = (price_next - prev_entry_price) / prev_entry_price
                        if prev_pos == 1:  # Was LONG
                            expected_prev_capital = prev_capital * (1 + price_change)
                        else:  # Was SHORT
                            expected_prev_capital = max(prev_capital * (1 - price_change), 0.01)
                        
                        if abs(expected_prev_capital - implied_value) < 1e-2 and current_entry_day == day:
                            found = True
                            day = day - 1
                            pos = prev_pos
                            path.append((day, pos, prev_capital, prev_entry_day, prev_entry_price))
                            break
            
            if not found:
                print(f"Warning: Could not backtrack from day {day+1}, breaking")
                break
        
        path.reverse()
        
        # Extract trades from path
        trades = []
        current_pos = 0  # Start FLAT
        entry_day = -1
        entry_price = 0
        entry_capital = INITIAL_CAPITAL
        
        for i in range(len(path)):
            day, pos, capital, e_day, e_price = path[i]
            
            if pos != current_pos:
                # Exit previous position
                if current_pos != 0 and entry_day >= 0:
                    exit_price = self.prices[day]
                    exit_capital = capital if pos == 0 else None  # Calculate properly
                    
                    # For exit capital, need to find the value just before exit
                    if i > 0:
                        prev_day, prev_pos, prev_capital, _, _ = path[i-1]
                        if prev_pos == current_pos:
                            # Calculate value at current day before exit
                            price_change = (self.prices[day] - entry_price) / entry_price
                            if current_pos == 1:  # LONG
                                value_before_exit = prev_capital * (1 + price_change)
                            else:  # SHORT
                                value_before_exit = max(prev_capital * (1 - price_change), 0.01)
                            exit_capital = value_before_exit * (1 - TRANSACTION_FEE)
                    
                    if exit_capital is None:
                        exit_capital = capital
                    
                    trades.append({
                        'entry_day': entry_day,
                        'exit_day': day,
                        'entry_time': self.df.index[entry_day],
                        'exit_time': self.df.index[day],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': 'LONG' if current_pos == 1 else 'SHORT',
                        'entry_capital': entry_capital,
                        'exit_capital': exit_capital
                    })
                
                # Enter new position
                if pos != 0:
                    entry_day = e_day
                    entry_price = e_price
                    entry_capital = capital
                
                current_pos = pos
        
        # Handle final position if still open
        if current_pos != 0 and entry_day >= 0:
            day = self.n - 1
            exit_price = self.prices[day]
            exit_capital = path[-1][2]
            
            trades.append({
                'entry_day': entry_day,
                'exit_day': day,
                'entry_time': self.df.index[entry_day],
                'exit_time': self.df.index[day],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': 'LONG' if current_pos == 1 else 'SHORT',
                'entry_capital': entry_capital,
                'exit_capital': exit_capital
            })
        
        self.trades = trades
        self.best_path = path
        
        print(f"Extracted {len(trades)} trades")
        for i, trade in enumerate(trades[:10]):
            ret = (trade['exit_capital'] / trade['entry_capital'] - 1) * 100
            print(f"  Trade {i+1}: {trade['position']} @ ${trade['entry_price']:.2f} -> ${trade['exit_price']:.2f}, Return: {ret:.2f}%")
    
    def create_visualization(self):
        """Create interactive Plotly visualization"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price with Entry/Exit Points', 'Capital Curve'),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.prices,
                name='Price',
                line=dict(color='lightgray', width=1)
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
        
        if long_entries_x:
            fig.add_trace(go.Scatter(x=long_entries_x, y=long_entries_y, mode='markers',
                marker=dict(color='green', size=12, symbol='triangle-up'), name='Long Entry'), row=1, col=1)
            fig.add_trace(go.Scatter(x=long_exits_x, y=long_exits_y, mode='markers',
                marker=dict(color='lightgreen', size=12, symbol='triangle-down'), name='Long Exit'), row=1, col=1)
        
        if short_entries_x:
            fig.add_trace(go.Scatter(x=short_entries_x, y=short_entries_y, mode='markers',
                marker=dict(color='red', size=12, symbol='triangle-down'), name='Short Entry'), row=1, col=1)
            fig.add_trace(go.Scatter(x=short_exits_x, y=short_exits_y, mode='markers',
                marker=dict(color='pink', size=12, symbol='triangle-up'), name='Short Exit'), row=1, col=1)
        
        # Capital curve from path
        capital_curve_x = []
        capital_curve_y = []
        for day, pos, capital, _, _ in self.best_path:
            capital_curve_x.append(self.df.index[day])
            capital_curve_y.append(capital)
        
        if capital_curve_x:
            fig.add_trace(go.Scatter(x=capital_curve_x, y=capital_curve_y, name='Capital',
                line=dict(color='blue', width=2)), row=2, col=1)
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Capital ($)", row=2, col=1, type='log')
        
        fig.update_layout(height=900, showlegend=True, title_text="Optimal Trading Strategy (Daily Data)")
        
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
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Parse timestamp
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
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
    
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Resample to daily
    print("Resampling to daily data...")
    df_daily = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    print(f"Daily data shape: {df_daily.shape}")
    print(f"Daily date range: {df_daily.index[0]} to {df_daily.index[-1]}")
    
    # Initialize finder with daily data
    finder = OptimalTradingFinder(df_daily)
    
    # Find optimal trades
    final_capital = finder.find_optimal_trades_dp()
    
    # Create visualization
    print("Creating visualization...")
    chart_html = finder.create_visualization()
    
    print("Ready to serve web interface!")

@app.route('/')
def index():
    if finder is None:
        return "<h1>Processing data, please wait...</h1>"
    
    # Generate trade table
    trades_html = "<table border='1' style='border-collapse: collapse; width: 100%; margin-top: 20px;'>"
    trades_html += "<tr style='background: #333; color: white;'><th>#</th><th>Entry Date</th><th>Exit Date</th><th>Position</th><th>Entry Price</th><th>Exit Price</th><th>Entry Capital</th><th>Exit Capital</th><th>Return %</th><th>Days Held</th></tr>"
    
    for i, trade in enumerate(finder.trades):
        ret = (trade['exit_capital'] / trade['entry_capital'] - 1) * 100
        duration = trade['exit_day'] - trade['entry_day']
        bg = '#f9f9f9' if i % 2 == 0 else 'white'
        trades_html += f"<tr style='background: {bg};'><td>{i+1}</td><td>{trade['entry_time'].date()}</td><td>{trade['exit_time'].date()}</td><td><strong>{trade['position']}</strong></td>"
        trades_html += f"<td>${trade['entry_price']:.4f}</td><td>${trade['exit_price']:.4f}</td>"
        trades_html += f"<td>${trade['entry_capital']:.2f}</td><td>${trade['exit_capital']:.2f}</td>"
        trades_html += f"<td style='color: {'green' if ret > 0 else 'red'}; font-weight: bold;'>{ret:.2f}%</td>"
        trades_html += f"<td>{duration} days</td></tr>"
    
    trades_html += "</table>"
    
    final_capital = finder.best_path[-1][2] if finder.best_path else INITIAL_CAPITAL
    total_return = (final_capital / INITIAL_CAPITAL - 1) * 100
    
    winning_trades = sum(1 for t in finder.trades if t['exit_capital'] > t['entry_capital'])
    win_rate = (winning_trades / len(finder.trades) * 100) if finder.trades else 0
    
    avg_return = sum((t['exit_capital']/t['entry_capital']-1)*100 for t in finder.trades) / len(finder.trades) if finder.trades else 0
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Optimal Trading Strategy - Daily</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .summary {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric {{ display: inline-block; margin: 10px 30px 10px 0; }}
            .metric-label {{ color: #666; font-size: 14px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
            .positive {{ color: #00a000; }}
            table {{ font-size: 14px; }}
            th {{ padding: 10px; }}
            td {{ padding: 8px; }}
        </style>
    </head>
    <body>
        <h1>Optimal Trading Strategy Results (Daily Data)</h1>
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
            <div class="metric">
                <div class="metric-label">Avg Return/Trade</div>
                <div class="metric-value">{avg_return:.2f}%</div>
            </div>
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
                <p style="margin: 5px 0;"><strong>Transaction Fee:</strong> {TRANSACTION_FEE*100}% per trade leg (0.4% round-trip)</p>
                <p style="margin: 5px 0;"><strong>Time Frame:</strong> Daily (no slippage)</p>
                <p style="margin: 5px 0;"><strong>Data Points:</strong> {finder.n} days</p>
            </div>
        </div>
        
        {chart_html}
        
        <h2 style="margin-top: 30px;">All Trades</h2>
        {trades_html}
        
        <div style="margin: 30px 0; padding: 15px; background: #ffffcc; border-radius: 5px;">
            <strong>Note:</strong> This algorithm uses dynamic programming to find the theoretically optimal 
            sequence of long/short positions that maximizes final capital with perfect hindsight. 
            Transaction fees (0.4% round-trip) are fully accounted for.
        </div>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    load_and_process_data()
    print("\nStarting web server on http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
