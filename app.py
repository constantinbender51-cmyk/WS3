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
from matplotlib.lines import Line2D

# --- Configuration ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'
RSI_PERIOD = 14
LONG_ENTRY_LEVEL = 15
SHORT_ENTRY_LEVEL = 75
PLOT_FUTURE_PERIOD = 14
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
            # Add one timeframe unit to the last timestamp to avoid fetching the same candle
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
    
    # 1. Generate Raw Signals (Crossover Logic)
    
    # Long Signal (1): RSI crosses UP above 15. Yesterday <= 15 AND Today > 15
    df['Long_Signal'] = ((df['RSI'].shift(1) <= LONG_ENTRY_LEVEL) & (df['RSI'] > LONG_ENTRY_LEVEL)).astype(int)
    
    # Short Signal (-1): RSI crosses DOWN below 75. Yesterday >= 75 AND Today < 75
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
    
    # Drop initial rows where RSI or shifted values are NaN
    return df.dropna(subset=['RSI']).copy()

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
    
    # Strategy Equity Curve (RED for a strategy aiming for UP movement/Long)
    ax.plot(df['Cumulative_Equity'], label='RSI Strategy Equity', color='red') 
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
            bbox=dict(boxstyle="round,pad=0.5", facecolor='mistyrose', alpha=0.5))
            
    return plot_to_base64(fig)

def create_rsi_plot(df):
    """Generates the RSI plot with entry/exit thresholds and signal markers."""
    fig = Figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    
    # Plot RSI Line
    ax.plot(df.index, df['RSI'], label=f'RSI ({RSI_PERIOD})', color='purple', linewidth=1.5)
    
    # Plot Thresholds (Red for Long Entry, Blue for Short Entry)
    ax.axhline(LONG_ENTRY_LEVEL, color='red', linestyle='--', linewidth=1.0, label=f'Long Entry ({LONG_ENTRY_LEVEL})')
    ax.axhline(SHORT_ENTRY_LEVEL, color='blue', linestyle='--', linewidth=1.0, label=f'Short Entry ({SHORT_ENTRY_LEVEL})')
    
    # Shade Overbought/Oversold regions
    ax.fill_between(df.index, SHORT_ENTRY_LEVEL, 100, color='blue', alpha=0.05) # Overbought (Short Entry Zone)
    ax.fill_between(df.index, 0, LONG_ENTRY_LEVEL, color='red', alpha=0.05)    # Oversold (Long Entry Zone)

    ax.set_title(f"Relative Strength Index (RSI) for {SYMBOL} with Entry Zones")
    ax.set_xlabel("Date")
    ax.set_ylabel("RSI Value")
    ax.set_ylim(0, 100) # Ensure y-axis is 0-100
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Highlight entry signals on the plot for visualization
    long_entries = df[df['Long_Signal'] == 1].index
    short_entries = df[df['Short_Signal'] == -1].index
    
    # Chinese convention: Red for Long (Buy), Blue for Short (Sell)
    ax.scatter(long_entries, df.loc[long_entries, 'RSI'], marker='^', color='red', s=50, label='Long Signal')
    ax.scatter(short_entries, df.loc[short_entries, 'RSI'], marker='v', color='blue', s=50, label='Short Signal')
    
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.05), ncol=4, borderaxespad=0.)

    fig.tight_layout()
    return plot_to_base64(fig)


def calculate_rolling_returns_series(signals, returns_series, days):
    """Calculates the average cumulative return for days 1 to 'days' following signals."""
    trade_returns = []
    
    # Iterate through all signal dates
    for signal_date in signals:
        # Define the 14-day window starting the day AFTER the signal
        start_date = signal_date + timedelta(days=1)
        end_date = signal_date + timedelta(days=days)
        
        # Get the returns slice (t+1 to t+14)
        future_returns = returns_series.loc[start_date:end_date]
        
        # Check if we have enough data points (14 days)
        if len(future_returns) == days:
            # Calculate cumulative return for each day in the window
            # (1 + r1), (1 + r1)(1 + r2), ..., (1 + r1)...(1 + r14)
            cum_returns = (1 + future_returns).cumprod() - 1
            trade_returns.append(cum_returns.values)
    
    if not trade_returns:
        # Return an array of zeros if no trades were executed (or not enough data)
        return np.zeros(days), 0
    
    # Convert list of arrays/series into a 2D numpy array
    returns_matrix = np.array(trade_returns)
    
    # Calculate the average return across all trades for each day (column)
    avg_cum_returns = np.mean(returns_matrix, axis=0) * 100 # Convert to percentage
    
    return avg_cum_returns, len(trade_returns)

def create_avg_returns_plot(df, period=PLOT_FUTURE_PERIOD):
    """
    Calculates and plots the average cumulative return line for days 1 to N 
    following a Long or Short signal, using Red for Long and Blue for Short.
    """
    # Long signals use Long_Signal == 1
    long_signals = df[df['Long_Signal'] == 1].index
    # Short signals use Short_Signal == -1
    short_signals = df[df['Short_Signal'] == -1].index 
    
    daily_returns = df['Daily_Return']
    
    long_avg_returns, long_count = calculate_rolling_returns_series(long_signals, daily_returns, period)
    short_avg_returns, short_count = calculate_rolling_returns_series(short_signals, daily_returns, period)

    # Plotting the results
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    x_days = np.arange(1, period + 1)
    
    # CHINESE CONVENTION: Long Plot in RED (Price movement after Long signal)
    ax.plot(x_days, long_avg_returns, 
            label=f'Long Signal Price Change (N={long_count})', 
            color='red', linewidth=2, marker='o', markersize=4)
    
    # CHINESE CONVENTION: Short Plot in BLUE (Price movement after Short signal)
    ax.plot(x_days, short_avg_returns, 
            label=f'Short Signal Price Change (N={short_count})', 
            color='blue', linewidth=2, marker='x', markersize=4)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    
    ax.set_title(f"Average Cumulative Price Change (Days 1 to {period} Post-Signal)")
    ax.set_xlabel("Days After Signal Entry")
    ax.set_ylabel("Average Cumulative Price Change (%)")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    fig.tight_layout()
    return plot_to_base64(fig)


def create_last_month_plot(df):
    """
    Generates a plot for the last month of data showing price and position shading.
    Uses Red for Long position and Blue for Short position backgrounds.
    """
    
    # Get the last 30 days of data
    last_month_df = df.iloc[-30:].copy()

    fig = Figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    # Calculate min/max range for fill_between
    y_min = last_month_df['low'].min() * 0.99
    y_max = last_month_df['high'].max() * 1.01

    # Create masks for Long, Short periods
    is_long = last_month_df['Position'] > 0
    is_short = last_month_df['Position'] < 0

    # 1. Draw shaded regions for Long positions (RED)
    ax.fill_between(last_month_df.index, y_min, y_max,
                    where=is_long, color='red', alpha=0.1, step='pre')

    # 2. Draw shaded regions for Short positions (BLUE)
    ax.fill_between(last_month_df.index, y_min, y_max,
                    where=is_short, color='blue', alpha=0.1, step='pre')

    # 3. Plot the Close price line
    ax.plot(last_month_df.index, last_month_df['close'], label='Close Price', color='gray', linewidth=2)
    
    # 4. Final plot settings
    ax.set_title(f"Position Visualization: Last 30 Days ({last_month_df.index.min().date()} to {last_month_df.index.max().date()})")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{SYMBOL} Close Price")
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Creating a custom legend for the line and fill areas using Chinese color convention
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, label='Close Price'),
        Line2D([0], [0], color='red', lw=10, alpha=0.1, label='Long Position'),
        Line2D([0], [0], color='blue', lw=10, alpha=0.1, label='Short Position'),
    ]
    ax.legend(handles=legend_elements, loc='best')

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
        rsi_plot_base64 = create_rsi_plot(results_df)  # NEW RSI Plot
        avg_returns_plot_base64 = create_avg_returns_plot(results_df)
        last_month_plot_base64 = create_last_month_plot(results_df) 
        
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
                                  rsi_plot=rsi_plot_base64,  # Pass new plot data
                                  avg_returns_plot=avg_returns_plot_base64,
                                  last_month_plot=last_month_plot_base64,
                                  summary=summary,
                                  RSI_PERIOD=RSI_PERIOD,
                                  LONG_ENTRY_LEVEL=LONG_ENTRY_LEVEL,
                                  SHORT_ENTRY_LEVEL=SHORT_ENTRY_LEVEL,
                                  PLOT_FUTURE_PERIOD=PLOT_FUTURE_PERIOD)

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
        h1 { border-bottom: 2px solid #ef4444; padding-bottom: 0.5rem; }
    </style>
</head>
<body class="p-4 md:p-8">
    <div class="max-w-6xl mx-auto">
        <h1 class="text-3xl font-extrabold text-red-600 mb-6">
            RSI ({{ RSI_PERIOD }}) Backtesting Analysis: {{ summary.start_date }} to {{ summary.end_date }}
        </h1>

        <!-- Summary Metrics -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div class="card p-6 border-l-4 border-red-500">
                <p class="text-sm text-gray-500">Total Days Analyzed</p>
                <p class="text-3xl font-bold text-gray-800">{{ summary.total_days }}</p>
            </div>
            <div class="card p-6 border-l-4 border-red-600">
                <p class="text-sm text-gray-500">Strategy Cumulative Return</p>
                <p class="text-3xl font-bold {{ 'text-red-600' if summary.final_strategy_return[0] != '-' else 'text-blue-600' }}">{{ summary.final_strategy_return }}</p>
            </div>
            <div class="card p-6 border-l-4 border-gray-500">
                <p class="text-sm text-gray-500">Buy & Hold (Benchmark) Return</p>
                <p class="text-3xl font-bold {{ 'text-red-600' if summary.final_benchmark_return[0] != '-' else 'text-blue-600' }}">{{ summary.final_benchmark_return }}</p>
            </div>
        </div>
        
        <!-- Plot 4: RSI Indicator Plot -->
        <div class="card p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-700 mb-4">
                Relative Strength Index (RSI) & Signal Zones
            </h2>
            <img src="data:image/png;base64,{{ rsi_plot }}" alt="RSI Indicator Plot" class="w-full h-auto rounded-lg"/>
            <p class="text-sm text-gray-600 mt-2">
                RSI thresholds: Long Entry at 15 (Red Zone), Short Entry at 75 (Blue Zone). Signals mark the crossover points.
            </p>
        </div>

        <!-- Plot 1: Equity Curve -->
        <div class="card p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-700 mb-4">
                Strategy Equity (Red=Long Focus) vs. Buy & Hold (Daily Compounding)
            </h2>
            <img src="data:image/png;base64,{{ equity_plot }}" alt="Equity Curve Plot" class="w-full h-auto rounded-lg"/>
            <p class="text-sm text-gray-600 mt-2">
                Strategy equity curve compared to the asset's Buy & Hold return.
            </p>
        </div>

        <!-- Plot 3: Last Month Position -->
        <div class="card p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-700 mb-4">
                Position Visualization: Last 30 Days (Red=Long, Blue=Short)
            </h2>
            <img src="data:image/png;base64,{{ last_month_plot }}" alt="Last Month Position Plot" class="w-full h-auto rounded-lg"/>
            <p class="text-sm text-gray-600 mt-2">
                Shows the price action and the strategy's position (background color) for the most recent 30 days.
            </p>
        </div>

        <!-- Plot 2: Average Returns -->
        <div class="card p-6">
            <h2 class="text-xl font-semibold text-gray-700 mb-4">
                Average Future Cumulative Price Change (Days 1 to {{ PLOT_FUTURE_PERIOD }})
            </h2>
            <img src="data:image/png;base64,{{ avg_returns_plot }}" alt="Average Future Returns Plot" class="w-full h-auto rounded-lg"/>
            <p class="text-sm text-gray-600 mt-2">
                This plot shows the average cumulative return of the underlying asset over the 14 days following each Long (Red line) or Short (Blue line) signal entry.
            </p>
        </div>

    </div>
</body>
</html>
"""

# Error Template (Unchanged)
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
