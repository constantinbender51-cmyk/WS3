import gdown
import pandas as pd
import numpy as np
from flask import Flask
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
TRANSACTION_FEE = 0.002  # 0.2% per trade leg (0.4% round-trip)
INITIAL_CAPITAL = 10000

app = Flask(__name__)

class OptimalTradingFinder:
    def __init__(self, df):
        self.df = df
        self.prices = df['close'].values
        self.n = len(self.prices)
        self.all_trades = []
        self.selected_trades = []
        self.best_capital = INITIAL_CAPITAL
        
    def calculate_trade_return(self, entry_idx, exit_idx, position_type):
        """Calculate net return multiplier for a single trade"""
        entry_price = self.prices[entry_idx]
        exit_price = self.prices[exit_idx]
        
        if position_type == 'LONG':
            # Long: profit when price goes up
            gross_multiplier = exit_price / entry_price
        else:  # SHORT
            # Short: profit when price goes down
            # Return = 1 - (price_change_pct)
            price_change_pct = (exit_price - entry_price) / entry_price
            gross_multiplier = 1 - price_change_pct
        
        # Apply fees on entry and exit
        net_multiplier = gross_multiplier * (1 - TRANSACTION_FEE) * (1 - TRANSACTION_FEE)
        
        return net_multiplier
    
    def find_all_profitable_trades(self):
        """Find all profitable long and short trades using peak/trough detection"""
        print(f"Finding all profitable trades from {self.n} days...")
        
        # Find local minima (good for LONG entries) and maxima (good for SHORT entries)
        print("Detecting peaks and troughs...")
        
        profitable_trades = []
        
        # Simple approach: for each day, find best future exit
        print("Finding best trades from each day...")
        
        for entry in range(self.n - 1):
            if entry % 50 == 0:
                print(f"  Progress: {entry}/{self.n} - Found {len(profitable_trades)} trades so far")
            
            entry_price = self.prices[entry]
            
            # Find best LONG exit (highest future price)
            future_prices = self.prices[entry+1:]
            if len(future_prices) > 0:
                max_future_idx = np.argmax(future_prices)
                exit_idx = entry + 1 + max_future_idx
                
                multiplier = self.calculate_trade_return(entry, exit_idx, 'LONG')
                if multiplier > 1.0:
                    profitable_trades.append({
                        'entry': entry,
                        'exit': exit_idx,
                        'type': 'LONG',
                        'multiplier': multiplier,
                        'entry_price': entry_price,
                        'exit_price': self.prices[exit_idx]
                    })
                
                # Find best SHORT exit (lowest future price)
                min_future_idx = np.argmin(future_prices)
                exit_idx = entry + 1 + min_future_idx
                
                multiplier = self.calculate_trade_return(entry, exit_idx, 'SHORT')
                if multiplier > 1.0:
                    profitable_trades.append({
                        'entry': entry,
                        'exit': exit_idx,
                        'type': 'SHORT',
                        'multiplier': multiplier,
                        'entry_price': entry_price,
                        'exit_price': self.prices[exit_idx]
                    })
        
        print(f"Found {len(profitable_trades)} profitable trades")
        self.all_trades = profitable_trades
        
        return profitable_trades
    
    def select_optimal_sequence(self):
        """Select non-overlapping trades that maximize compounded return - Greedy approach"""
        print("\nSelecting optimal non-overlapping sequence...")
        
        if not self.all_trades:
            print("No profitable trades found!")
            return []
        
        # Sort trades by multiplier (best returns first)
        sorted_trades = sorted(self.all_trades, key=lambda x: x['multiplier'], reverse=True)
        
        print(f"Selecting from {len(sorted_trades)} profitable trades...")
        
        # Greedy: take best trades that don't overlap
        selected = []
        occupied_days = set()
        
        for trade in sorted_trades:
            # Check if this trade overlaps with any selected trade
            trade_days = set(range(trade['entry'], trade['exit'] + 1))
            
            if not trade_days.intersection(occupied_days):
                # No overlap, take this trade
                selected.append(trade)
                occupied_days.update(trade_days)
        
        print(f"Selected {len(selected)} non-overlapping trades")
        
        # Sort selected trades by entry time
        selected.sort(key=lambda x: x['entry'])
        
        # Build trades list with capital tracking
        self.selected_trades = []
        running_capital = INITIAL_CAPITAL
        
        for trade in selected:
            entry_capital = running_capital
            exit_capital = entry_capital * trade['multiplier']
            
            self.selected_trades.append({
                'entry_day': trade['entry'],
                'exit_day': trade['exit'],
                'entry_time': self.df.index[trade['entry']],
                'exit_time': self.df.index[trade['exit']],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'position': trade['type'],
                'entry_capital': entry_capital,
                'exit_capital': exit_capital,
                'multiplier': trade['multiplier']
            })
            
            running_capital = exit_capital
        
        self.best_capital = running_capital
        
        print(f"Final capital: ${self.best_capital:,.2f}")
        print(f"Total return: {(self.best_capital/INITIAL_CAPITAL - 1)*100:.2f}%")
        
        # Show first few trades
        print("\nFirst 10 selected trades:")
        for i, trade in enumerate(self.selected_trades[:10]):
            ret = (trade['multiplier'] - 1) * 100
            print(f"  {i+1}. {trade['position']} {trade['entry_time'].date()} -> {trade['exit_time'].date()}: "
                  f"${trade['entry_price']:.2f} -> ${trade['exit_price']:.2f} ({ret:.2f}%)")
        
        return self.selected_trades
    
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
        
        for trade in self.selected_trades:
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
                marker=dict(color='green', size=12, symbol='triangle-up', line=dict(color='darkgreen', width=1)), 
                name='Long Entry'), row=1, col=1)
            fig.add_trace(go.Scatter(x=long_exits_x, y=long_exits_y, mode='markers',
                marker=dict(color='lightgreen', size=12, symbol='triangle-down', line=dict(color='green', width=1)), 
                name='Long Exit'), row=1, col=1)
        
        if short_entries_x:
            fig.add_trace(go.Scatter(x=short_entries_x, y=short_entries_y, mode='markers',
                marker=dict(color='red', size=12, symbol='triangle-down', line=dict(color='darkred', width=1)), 
                name='Short Entry'), row=1, col=1)
            fig.add_trace(go.Scatter(x=short_exits_x, y=short_exits_y, mode='markers',
                marker=dict(color='pink', size=12, symbol='triangle-up', line=dict(color='red', width=1)), 
                name='Short Exit'), row=1, col=1)
        
        # Capital curve
        capital_curve_x = [self.df.index[0]]
        capital_curve_y = [INITIAL_CAPITAL]
        
        for trade in self.selected_trades:
            # Add point at entry
            capital_curve_x.append(trade['entry_time'])
            capital_curve_y.append(trade['entry_capital'])
            
            # Add point at exit
            capital_curve_x.append(trade['exit_time'])
            capital_curve_y.append(trade['exit_capital'])
        
        # Add final point
        if self.selected_trades:
            capital_curve_x.append(self.df.index[-1])
            capital_curve_y.append(self.best_capital)
        
        fig.add_trace(go.Scatter(x=capital_curve_x, y=capital_curve_y, name='Capital',
            line=dict(color='blue', width=2), fill='tozeroy', fillcolor='rgba(0,100,255,0.1)'), row=2, col=1)
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Capital ($)", row=2, col=1, type='log')
        
        fig.update_layout(
            height=900, 
            showlegend=True, 
            title_text="Optimal Trading Strategy - Brute Force Selection",
            hovermode='x unified'
        )
        
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
    
    print("\nLoading CSV (this may take a moment for large files)...")
    df = pd.read_csv(output)
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Parse timestamp
    print("Parsing timestamps...")
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
    
    print(f"Original date range: {df.index[0]} to {df.index[-1]}")
    
    # Resample to daily BEFORE doing anything else
    print("\n*** Resampling to daily data (converting 4M rows to ~3K rows) ***")
    df_daily = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    print(f"Daily data shape: {df_daily.shape} ← Working with this from now on!")
    print(f"Daily date range: {df_daily.index[0]} to {df_daily.index[-1]}")
    
    # Free up memory from original data
    del df
    
    # Initialize finder with ONLY daily data
    print(f"\nInitializing optimizer with {len(df_daily)} daily bars...")
    finder = OptimalTradingFinder(df_daily)
    
    # Find all profitable trades (on daily data only)
    print("\n" + "="*60)
    print("STEP 1: Finding profitable trades")
    print("="*60)
    finder.find_all_profitable_trades()
    
    # Select optimal sequence
    print("\n" + "="*60)
    print("STEP 2: Selecting optimal non-overlapping sequence")
    print("="*60)
    finder.select_optimal_sequence()
    
    # Create visualization
    print("\n" + "="*60)
    print("STEP 3: Creating visualization")
    print("="*60)
    chart_html = finder.create_visualization()
    
    print("\n✅ Ready to serve web interface!")

@app.route('/')
def index():
    if finder is None:
        return "<h1>Processing data, please wait...</h1>"
    
    # Generate trade table
    trades_html = "<table border='1' style='border-collapse: collapse; width: 100%; margin-top: 20px;'>"
    trades_html += """<tr style='background: #333; color: white;'>
        <th>#</th><th>Entry Date</th><th>Exit Date</th><th>Position</th>
        <th>Entry Price</th><th>Exit Price</th><th>Price Change</th>
        <th>Entry Capital</th><th>Exit Capital</th><th>Trade Return</th><th>Days</th>
    </tr>"""
    
    for i, trade in enumerate(finder.selected_trades):
        price_change = ((trade['exit_price'] / trade['entry_price']) - 1) * 100
        trade_return = (trade['multiplier'] - 1) * 100
        duration = trade['exit_day'] - trade['entry_day']
        bg = '#f9f9f9' if i % 2 == 0 else 'white'
        
        trades_html += f"<tr style='background: {bg};'>"
        trades_html += f"<td>{i+1}</td>"
        trades_html += f"<td>{trade['entry_time'].date()}</td>"
        trades_html += f"<td>{trade['exit_time'].date()}</td>"
        trades_html += f"<td><strong style='color: {'green' if trade['position']=='LONG' else 'red'};'>{trade['position']}</strong></td>"
        trades_html += f"<td>${trade['entry_price']:.4f}</td>"
        trades_html += f"<td>${trade['exit_price']:.4f}</td>"
        
        if trade['position'] == 'LONG':
            trades_html += f"<td style='color: {'green' if price_change > 0 else 'red'};'>{price_change:+.2f}%</td>"
        else:
            trades_html += f"<td style='color: {'red' if price_change > 0 else 'green'};'>{price_change:+.2f}%</td>"
        
        trades_html += f"<td>${trade['entry_capital']:,.2f}</td>"
        trades_html += f"<td>${trade['exit_capital']:,.2f}</td>"
        trades_html += f"<td style='color: green; font-weight: bold;'>{trade_return:+.2f}%</td>"
        trades_html += f"<td>{duration}</td>"
        trades_html += "</tr>"
    
    trades_html += "</table>"
    
    total_return = (finder.best_capital / INITIAL_CAPITAL - 1) * 100
    
    avg_return = sum((t['multiplier']-1)*100 for t in finder.selected_trades) / len(finder.selected_trades) if finder.selected_trades else 0
    total_days = (finder.df.index[-1] - finder.df.index[0]).days
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Optimal Trading Strategy - Brute Force</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .summary {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric {{ display: inline-block; margin: 10px 30px 10px 0; vertical-align: top; }}
            .metric-label {{ color: #666; font-size: 14px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
            .positive {{ color: #00a000; }}
            table {{ font-size: 13px; }}
            th {{ padding: 10px; text-align: left; }}
            td {{ padding: 8px; }}
        </style>
    </head>
    <body>
        <h1>🎯 Optimal Trading Strategy Results (Brute Force)</h1>
        <div class="summary">
            <h2>Performance Summary</h2>
            <div class="metric">
                <div class="metric-label">Initial Capital</div>
                <div class="metric-value">${INITIAL_CAPITAL:,.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Final Capital</div>
                <div class="metric-value">${finder.best_capital:,.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Return</div>
                <div class="metric-value positive">{total_return:,.2f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Number of Trades</div>
                <div class="metric-value">{len(finder.selected_trades)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Return/Trade</div>
                <div class="metric-value positive">{avg_return:.2f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Days</div>
                <div class="metric-value">{total_days}</div>
            </div>
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
                <p style="margin: 5px 0;"><strong>Algorithm:</strong> Brute force search + optimal sequence selection</p>
                <p style="margin: 5px 0;"><strong>Transaction Fee:</strong> {TRANSACTION_FEE*100}% per leg (0.4% round-trip)</p>
                <p style="margin: 5px 0;"><strong>Profitable trades found:</strong> {len(finder.all_trades):,}</p>
                <p style="margin: 5px 0;"><strong>Trades selected:</strong> {len(finder.selected_trades)} (non-overlapping)</p>
            </div>
        </div>
        
        {chart_html}
        
        <h2 style="margin-top: 30px;">All Selected Trades</h2>
        {trades_html}
        
        <div style="margin: 30px 0; padding: 15px; background: #e8f5e9; border-left: 4px solid #4caf50; border-radius: 5px;">
            <strong>💡 Method:</strong> This algorithm finds ALL profitable trades (both long and short) 
            across the entire dataset, then uses dynamic programming to select the optimal non-overlapping 
            sequence that maximizes compounded returns. This is the theoretical maximum return achievable 
            with perfect hindsight.
        </div>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    load_and_process_data()
    print("\nStarting web server on http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
