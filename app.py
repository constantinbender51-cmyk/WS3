import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-GUI environments
from matplotlib.figure import Figure
import io
import base64
from flask import Flask, render_template_string
from datetime import datetime, timedelta
import time

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'
RSI_PERIOD = 14
LONG_ENTRY_LEVEL = 15
SHORT_ENTRY_LEVEL = 75
PORT = 8080

# --- Flask Setup ---
app = Flask(__name__)

# --- Data Fetching and Caching ---

def fetch_binance_data(symbol, timeframe, since_date_str):
    """Fetches historical OHLCV data from Binance, handling pagination."""
    print(f"Connecting to Binance and fetching {symbol} data from {since_date_str}...")
    binance = ccxt.binance({
        'enableRateLimit': True,
        'rateLimit': 500  # Adjust based on API limits
    })

    # Convert start date string to milliseconds timestamp
    since_ms = binance.parse8601(since_date_str)
    
    all_ohlcv = []
    limit = 1000 # Max limit per request for Binance 1d klines

    while True:
        try:
            # Fetch 1000 candles starting from 'since_ms'
            ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
            
            if not ohlcv:
                print("No more data found.")
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Set the 'since' to the timestamp of the last fetched candle to continue fetching
            since_ms = ohlcv[-1][0] + binance.parse_timeframe(timeframe) * 1000 
            
            print(f"Fetched {len(ohlcv)} candles. Last date: {binance.iso8601(ohlcv[-1][0])}")

            # Safety break for testing or very long history (Binance is rate limited)
            if len(ohlcv) < limit:
                break
            
            # Sleep to respect rate limits
            time.sleep(binance.rateLimit / 1000)

        except Exception as e:
            print(f"An error occurred while fetching data: {e}")
            break

    if not all_ohlcv:
        raise Exception("Failed to fetch any historical data.")
        
    # Convert list of lists to Pandas DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Total data points fetched: {len(df)}")
    return df

# --- Technical Analysis (RSI) ---

def calculate_rsi(data, window=14):
    """
    Calculates the Relative Strength Index (RSI) using the standard Wilders
    smoothing method (RMA/EWM).
    """
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Custom function for Wilders Smoothing (equivalent to RMA in trading platforms)
    def rma(series, periods):
        return series.ewm(alpha=1/periods, adjust=False).mean()

    avg_gain = rma(gain, window)
    avg_loss = rma(loss, window)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    data['RSI'] = rsi
    return data

# --- Backtesting Logic ---

def backtest_strategy(df):
    """
    Applies the RSI strategy and calculates compounded returns.
    Strategy: Long when RSI crosses above 15, Short when RSI crosses below 75.
    Daily compounding: 100% of capital is traded daily.
    """
    df = calculate_rsi(df, RSI_PERIOD)
    
    # 1. Generate Raw Signals
    # Long Signal (1): RSI crosses above 15
    df['Long_Signal'] = ((df['RSI'].shift(1) <= LONG_ENTRY_LEVEL) & (df['RSI'] > LONG_ENTRY_LEVEL)).astype(int)
    # Short Signal (-1): RSI crosses below 75
    df['Short_Signal'] = -((df['RSI'].shift(1) >= SHORT_ENTRY_LEVEL) & (df['RSI'] < SHORT_ENTRY_LEVEL)).astype(int)

    # Combine signals and fill gaps to hold position
    df['Signal'] = df['Long_Signal'] + df['Short_Signal']
    
    # Forward fill signals to maintain position until the next opposite signal
    # Start position is flat (0)
    df['Position'] = df['Signal'].replace(0, method='ffill')
    df['Position'] = df['Position'].fillna(0)
    
    # 2. Calculate Daily Returns
    # Daily asset return (close-to-close)
    df['Daily_Return'] = df['close'].pct_change()
    
    # Strategy Return: Position on Day T-1 * Asset Return on Day T
    # The 'Position' is the decision made at the end of the day, applied for the next day's movement.
    df['Strategy_Return'] = df['Position'].shift(1) * df['Daily_Return']
    
    # 3. Calculate Cumulative Equity
    # Strategy Return is compounded daily (1 + return). Start with $1 (or 100%).
    # Fill NaN from initial shift with 0 return (since no position was open)
    df['Strategy_Return'] = df['Strategy_Return'].fillna(0)
    df['Cumulative_Equity'] = (1 + df['Strategy_Return']).cumprod()
    
    # Calculate Buy & Hold baseline
    df['Buy_Hold_Equity'] = (1 + df['Daily_Return']).cumprod()
    
    return df.dropna(subset=['RSI'])

# --- Analysis & Plotting ---

def plot_to_base64(fig):
    """Converts a matplotlib figure to a base64 encoded string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode('ascii')
    return data

def create_equity_plot(df):
    """Generates the main equity curve plot."""
    fig = Figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    # Strategy Equity Curve
    ax.plot(df['Cumulative_Equity'], label='RSI Strategy Equity', color='blue')
    # Buy & Hold Benchmark
    ax.plot(df['Buy_Hold_Equity'], label='Buy & Hold Equity', color='grey', linestyle='--')
    
    # Calculate final performance metrics
    strategy_return = (df['Cumulative_Equity'].iloc[-1] - 1) * 100
    benchmark_return = (df['Buy_Hold_Equity'].iloc[-1] - 1) * 100
    
    ax.set_title(f"RSI Backtest Equity Curve for {SYMBOL} ({df.index.min().date()} to {df.index.max().date()})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Equity (starting at 1.0)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    
    # Add final returns text box
    ax.text(0.02, 0.98, 
            f'Strategy Return: {strategy_return:.2f}%\nBuy & Hold Return: {benchmark_return:.2f}%',
            transform=ax.transAxes, 
            fontsize=10, 
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.5))
            
    return plot_to_base64(fig)

def create_avg_returns_plot(df, period=14):
    """
    Calculates and plots the average return over the N days following a Long or Short signal.
    """
    long_signals = df[df['Long_Signal'] == 1].index
    short_signals = df[df['Short_Signal'] == -1].index
    
    daily_returns = df['Daily_Return']
    
    # Function to calculate rolling average returns for a given signal type
    def calculate_rolling_returns(signals, returns_series, days):
        returns_list = []
        for signal_date in signals:
            end_date = signal_date + timedelta(days=days)
            # Find the slice of returns for the next 'days' period
            future_returns = returns_series.loc[signal_date + timedelta(days=1):end_date]
            
            if len(future_returns) == days:
                # Calculate the cumulative return (1 + r1) * (1 + r2) ... - 1
                cumulative_return = (1 + future_returns).prod() - 1
                returns_list.append(cumulative_return)
        
        if not returns_list:
            return 0, 0
            
        return np.mean(returns_list) * 100, len(returns_list)

    long_avg_return, long_count = calculate_rolling_returns(long_signals, daily_returns, period)
    short_avg_return, short_count = calculate_rolling_returns(short_signals, daily_returns, period)

    # Plotting the results
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    data = {
        'Long Signal': long_avg_return,
        'Short Signal': short_avg_return
    }
    counts = {
        'Long Signal': long_count,
        'Short Signal': short_count
    }
    
    bars = ax.bar(data.keys(), data.values(), 
                  color=['green' if long_avg_return >= 0 else 'red', 
                         'red' if short_avg_return >= 0 else 'green']) # Note: Short profit is asset drop
                         
    ax.set_title(f"Avg. {period}-Day Return Following Signal Entry")
    ax.set_ylabel("Average Cumulative Return (%)")
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Add text labels on bars
    for i, bar in enumerate(bars):
        signal_type = list(data.keys())[i]
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., 
                height + (0.5 if height >= 0 else -1.5), 
                f'{height:.2f}%\n(N={counts[signal_type]})',
                ha='center', 
                va='bottom' if height >= 0 else 'top',
                fontsize=10)

    fig.tight_layout()
    return plot_to_base64(fig)


# --- Flask Routes and App Setup ---

@app.route('/')
def home():
    """Main route to run backtest, generate plots, and display HTML."""
    try:
        # Load or Fetch Data
        df = fetch_binance_data(SYMBOL, TIMEFRAME, START_DATE)
        
        # Run Backtest
        results_df = backtest_strategy(df)
        
        # Generate Plots
        equity_plot_base64 = create_equity_plot(results_df)
        avg_returns_plot_base64 = create_avg_returns_plot(results_df)
        
        # Prepare key metrics for display
        final_equity = results_df['Cumulative_Equity'].iloc[-1]
        benchmark_equity = results_df['Buy_Hold_Equity'].iloc[-1]
        
        summary = {
            'final_strategy_return': f"{(final_equity - 1) * 100:.2f}%",
            'final_benchmark_return': f"{(benchmark_equity - 1) * 100:.2f}%",
            'total_days': len(results_df),
            'start_date': results_df.index.min().strftime('%Y-%m-%d'),
            'end_date': results_df.index.max().strftime('%Y-%m-%d'),
        }

    except Exception as e:
        # Simple error handling for display
        error_message = f"Error during data fetching or backtesting: {e}"
        print(error_message)
        # Use a minimal template for error
        return render_template_string(ERROR_HTML, error=error_message), 500

    return render_template_string(HTML_TEMPLATE, 
                                  equity_plot=equity_plot_base64, 
                                  avg_returns_plot=avg_returns_plot_base64,
                                  summary=summary)

# --- HTML Template (Embedded in Python file) ---

# HTML template string using Tailwind CSS for clean, responsive styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Binance RSI Backtest Results</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #f7f9fb; }
        .card { background-color: white; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        h1 { border-bottom: 2px solid #3b82f6; padding-bottom: 0.5rem; }
    </style>
</head>
<body class="p-4 md:p-8">
    <div class="max-w-6xl mx-auto">
        <h1 class="text-3xl font-extrabold text-blue-600 mb-6">
            RSI (14) Backtesting Analysis: {{ summary.start_date }} to {{ summary.end_date }}
        </h1>

        <!-- Summary Metrics -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div class="card p-6 border-l-4 border-blue-500">
                <p class="text-sm text-gray-500">Total Days Analyzed</p>
                <p class="text-3xl font-bold text-gray-800">{{ summary.total_days }}</p>
            </div>
            <div class="card p-6 border-l-4 border-green-500">
                <p class="text-sm text-gray-500">Strategy Cumulative Return</p>
                <p class="text-3xl font-bold {{ 'text-green-600' if summary.final_strategy_return[0] != '-' else 'text-red-600' }}">{{ summary.final_strategy_return }}</p>
            </div>
            <div class="card p-6 border-l-4 border-gray-500">
                <p class="text-sm text-gray-500">Buy & Hold (Benchmark) Return</p>
                <p class="text-3xl font-bold {{ 'text-green-600' if summary.final_benchmark_return[0] != '-' else 'text-red-600' }}">{{ summary.final_benchmark_return }}</p>
            </div>
        </div>

        <!-- Plot 1: Equity Curve -->
        <div class="card p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-700 mb-4">
                Strategy Equity vs. Buy & Hold (Daily Compounding)
            </h2>
            <img src="data:image/png;base64,{{ equity_plot }}" alt="Equity Curve Plot" class="w-full h-auto rounded-lg"/>
            <p class="text-sm text-gray-600 mt-2">
                RSI Entry Rules: Long > {{ LONG_ENTRY_LEVEL }}, Short < {{ SHORT_ENTRY_LEVEL }}.
            </p>
        </div>

        <!-- Plot 2: Average Returns -->
        <div class="card p-6">
            <h2 class="text-xl font-semibold text-gray-700 mb-4">
                Average Future Cumulative Return ({{ RSI_PERIOD }} Days)
            </h2>
            <img src="data:image/png;base64,{{ avg_returns_plot }}" alt="Average Future Returns Plot" class="w-full h-auto rounded-lg"/>
            <p class="text-sm text-gray-600 mt-2">
                This plot shows the average cumulative return over the 14 days immediately following each signal. (N = number of trades).
            </p>
        </div>

    </div>
</body>
</html>
"""

# Error Template
ERROR_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>body { font-family: 'Inter', sans-serif; background-color: #fef2f2; }</style>
</head>
<body class="p-8">
    <div class="max-w-xl mx-auto bg-white p-6 rounded-lg shadow-lg border-l-4 border-red-500">
        <h1 class="text-2xl font-bold text-red-600 mb-4">Analysis Failed</h1>
        <p class="text-gray-700">The backtesting script encountered an error, likely due to a network issue or missing data.</p>
        <p class="mt-4 p-3 bg-gray-100 rounded text-sm font-mono text-red-700">Error: {{ error }}</p>
        <p class="mt-4 text-sm text-gray-500">Please check your internet connection and ensure all required Python libraries are installed.</p>
    </div>
</body>
</html>
"""

# --- Main Execution ---

if __name__ == '__main__':
    print(f"Starting web server on http://127.0.0.1:{PORT}")
    print("Fetching data (this may take a few minutes for 2018-present)...")
    
    # Run Flask application
    app.run(host='0.0.0.0', port=PORT)
