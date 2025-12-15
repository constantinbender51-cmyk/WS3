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
TIMEFRAME = '30m'
START_DATE = '2018-01-01 00:00:00'
RSI_PERIOD = 14
LONG_ENTRY_LEVEL = 30 # Long when RSI rises above 30
SHORT_ENTRY_LEVEL = 70 # Short when RSI drops below 70
PLOT_FUTURE_PERIOD = 14 # Used for both position duration and average return plot
PORT = 8080

# --- Global Caching ---
# These must be initialized to None or a default state
global_results_df = None
global_summary = None
global_equity_plot = None
global_rsi_plot = None
global_avg_returns_plot = None
global_last_month_plot = None
global_error_message = None

# --- Flask Setup ---
app = Flask(__name__)

# --- Data Fetching and Caching ---

def fetch_binance_data(symbol, timeframe, since_date_str):
    """Fetches historical OHLCV data from Binance, handling pagination."""
    print(f"Connecting to Binance and fetching {symbol} data from {since_date_str}...")
    # NOTE: Fetching 30m data since 2018-01-01 involves a very large number of data points.
    binance = ccxt.binance({
        'enableRateLimit': True,
        'rateLimit': 500  # Adjust based on API limits
    })

    since_ms = binance.parse8601(since_date_str)
    all_ohlcv = []
    limit = 1000 

    while True:
        try:
            ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Move to the start of the next candle
            since_ms = ohlcv[-1][0] + binance.parse_timeframe(timeframe) * 1000 
            
            # Print update on progress
            current_date = binance.iso8601(ohlcv[-1][0])
            print(f"Fetched {len(ohlcv)} candles. Latest date: {current_date}")

            if len(ohlcv) < limit:
                break
            
            time.sleep(binance.rateLimit / 1000)

        except Exception as e:
            print(f"An error occurred while fetching data: {e}")
            raise e

    if not all_ohlcv:
        raise Exception("Failed to fetch any historical data.")
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Total data points fetched: {len(df)}")
    return df

# --- Technical Analysis (RSI) ---

def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
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
    Applies the RSI strategy with decreasing weighted position size and calculates compounded returns.
    Position decreases linearly: 1 - (periods_since_entry / PLOT_FUTURE_PERIOD).
    """
    df = calculate_rsi(df, RSI_PERIOD)
    
    # 1. Generate Raw Signals (Crossover Logic)
    
    # Long Signal (1): RSI crosses UP above 30. Yesterday <= 30 AND Today > 30
    df['Long_Signal'] = ((df['RSI'].shift(1) <= LONG_ENTRY_LEVEL) & (df['RSI'] > LONG_ENTRY_LEVEL)).astype(int)
    
    # Short Signal (-1): RSI crosses DOWN below 70. Yesterday >= 70 AND Today < 70
    df['Short_Signal'] = -((df['RSI'].shift(1) >= SHORT_ENTRY_LEVEL) & (df['RSI'] < SHORT_ENTRY_LEVEL)).astype(int)

    df['Signal'] = df['Long_Signal'] + df['Short_Signal']
    
    # 2. Calculate Decreasing Weighted Position
    df['Position'] = 0.0
    periods_since_entry = 0
    current_signal = 0

    # Iterate through data to manage trade state and decaying position size
    for i in range(len(df)):
        new_signal = df.iloc[i]['Signal']
        
        # A new signal overrides any current trade
        if new_signal != 0:
            periods_since_entry = 1
            current_signal = new_signal
        # Continue trade, count periods
        elif current_signal != 0 and periods_since_entry < PLOT_FUTURE_PERIOD:
            periods_since_entry += 1
        # Trade duration expired, go flat
        elif current_signal != 0 and periods_since_entry >= PLOT_FUTURE_PERIOD:
            periods_since_entry = 0
            current_signal = 0
        
        # Calculate DECREASING weighted position size
        if current_signal != 0 and periods_since_entry > 0:
            # Weight decreases linearly: 1 - (periods_since_entry / PLOT_FUTURE_PERIOD)
            weight = 1.0 - (periods_since_entry / PLOT_FUTURE_PERIOD)
            
            # Apply the position size (weight) and direction (current_signal)
            df.iloc[i, df.columns.get_loc('Position')] = current_signal * weight
        else:
            # Flat (0) position
            df.iloc[i, df.columns.get_loc('Position')] = 0.0

    # 3. Calculate Cumulative Equity
    df['Daily_Return'] = df['close'].pct_change()
    
    # Strategy Return: Position on Period T-1 * Asset Return on Period T
    df['Strategy_Return'] = df['Position'].shift(1) * df['Daily_Return']
    
    df['Strategy_Return'] = df['Strategy_Return'].fillna(0)
    df['Cumulative_Equity'] = (1 + df['Strategy_Return']).cumprod()
    
    # Calculate Buy & Hold baseline
    df['Buy_Hold_Equity'] = (1 + df['Daily_Return']).cumprod()
    
    return df.dropna(subset=['RSI']).copy()

# --- Plotting Functions (Using global constants) ---

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
    ax.plot(df['Cumulative_Equity'], label='RSI Strategy Equity', color='red') 
    ax.plot(df['Buy_Hold_Equity'], label='Buy & Hold Equity', color='grey', linestyle='--')
    
    strategy_return = (df['Cumulative_Equity'].iloc[-1] - 1) * 100
    benchmark_return = (df['Buy_Hold_Equity'].iloc[-1] - 1) * 100
    
    ax.set_title(f"RSI Backtest Equity Curve for {SYMBOL} ({TIMEFRAME})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Equity (starting at 1.0)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    
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
    
    ax.plot(df.index, df['RSI'], label=f'RSI ({RSI_PERIOD})', color='purple', linewidth=1.5)
    
    ax.axhline(LONG_ENTRY_LEVEL, color='red', linestyle='--', linewidth=1.0, label=f'Long Entry ({LONG_ENTRY_LEVEL})')
    ax.axhline(SHORT_ENTRY_LEVEL, color='blue', linestyle='--', linewidth=1.0, label=f'Short Entry ({SHORT_ENTRY_LEVEL})')
    
    ax.fill_between(df.index, SHORT_ENTRY_LEVEL, 100, color='blue', alpha=0.05) 
    ax.fill_between(df.index, 0, LONG_ENTRY_LEVEL, color='red', alpha=0.05)    

    ax.set_title(f"Relative Strength Index (RSI) for {SYMBOL} ({TIMEFRAME}) with Entry Zones")
    ax.set_xlabel("Date")
    ax.set_ylabel("RSI Value")
    ax.set_ylim(0, 100) 
    ax.grid(True, linestyle=':', alpha=0.6)
    
    long_entries = df[df['Long_Signal'] == 1].index
    short_entries = df[df['Short_Signal'] == -1].index
    
    ax.scatter(long_entries, df.loc[long_entries, 'RSI'], marker='^', color='red', s=50, label='Long Signal')
    ax.scatter(short_entries, df.loc[short_entries, 'RSI'], marker='v', color='blue', s=50, label='Short Signal')
    
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.05), ncol=4, borderaxespad=0.)

    fig.tight_layout()
    return plot_to_base64(fig)


def calculate_rolling_returns_series(signals, returns_series, periods):
    """Calculates the average cumulative return for periods 1 to 'periods' following signals."""
    trade_returns = []
    
    num_periods = periods
    
    for signal_date in signals:
        future_returns = returns_series.loc[signal_date:].iloc[1:num_periods+1]
        
        if len(future_returns) == num_periods:
            cum_returns = (1 + future_returns).cumprod() - 1
            trade_returns.append(cum_returns.values)
    
    if not trade_returns:
        return np.zeros(num_periods), 0
    
    returns_matrix = np.array(trade_returns)
    avg_cum_returns = np.mean(returns_matrix, axis=0) * 100
    
    return avg_cum_returns, len(trade_returns)

def create_avg_returns_plot(df, period=PLOT_FUTURE_PERIOD):
    """
    Calculates and plots the average cumulative return line for periods 1 to N 
    following a Long or Short signal, using Red for Long and Blue for Short.
    """
    long_signals = df[df['Long_Signal'] == 1].index
    short_signals = df[df['Short_Signal'] == -1].index 
    daily_returns = df['Daily_Return']
    
    long_avg_returns, long_count = calculate_rolling_returns_series(long_signals, daily_returns, period)
    short_avg_returns, short_count = calculate_rolling_returns_series(short_signals, daily_returns, period)

    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    x_periods = np.arange(1, period + 1)
    
    # CHINESE CONVENTION: Long Plot in RED 
    ax.plot(x_periods, long_avg_returns, 
            label=f'Long Signal Price Change (N={long_count})', 
            color='red', linewidth=2, marker='o', markersize=4)
    
    # CHINESE CONVENTION: Short Plot in BLUE
    ax.plot(x_periods, short_avg_returns, 
            label=f'Short Signal Price Change (N={short_count})', 
            color='blue', linewidth=2, marker='x', markersize=4)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    
    ax.set_title(f"Avg Cumulative Price Change (Periods 1 to {period} Post-Signal)")
    ax.set_xlabel(f"Periods ({TIMEFRAME}) After Signal Entry")
    ax.set_ylabel("Average Cumulative Price Change (%)")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    fig.tight_layout()
    return plot_to_base64(fig)


def create_last_month_plot(df):
    """
    Generates a plot for the last month (30 calendar days) of data showing price and position shading.
    Uses Red for Long position and Blue for Short position backgrounds.
    """
    end_date = df.index.max()
    start_date = end_date - timedelta(days=30)
    last_month_df = df.loc[start_date:end_date].copy()

    fig = Figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    y_min = last_month_df['low'].min() * 0.99
    y_max = last_month_df['high'].max() * 1.01

    # Position is the *weighted* position, so we check if it was non-zero
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
    ax.set_title(f"Position Visualization: Last 30 Days ({TIMEFRAME} candles)")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{SYMBOL} Close Price")
    ax.grid(True, linestyle=':', alpha=0.7)
    
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, label='Close Price'),
        Line2D([0], [0], color='red', lw=10, alpha=0.1, label='Long Position'),
        Line2D([0], [0], color='blue', lw=10, alpha=0.1, label='Short Position'),
    ]
    ax.legend(handles=legend_elements, loc='best')

    fig.tight_layout()
    return plot_to_base64(fig)


def setup_backtest():
    """Initializes global variables by running the fetch and backtest logic."""
    global global_results_df, global_summary, global_equity_plot, global_rsi_plot, global_avg_returns_plot, global_last_month_plot, global_error_message

    try:
        # Load or Fetch Data
        df = fetch_binance_data(SYMBOL, TIMEFRAME, START_DATE)
        
        # Run Backtest
        global_results_df = backtest_strategy(df)
        
        # Generate Plots
        global_equity_plot = create_equity_plot(global_results_df)
        global_rsi_plot = create_rsi_plot(global_results_df) 
        global_avg_returns_plot = create_avg_returns_plot(global_results_df)
        global_last_month_plot = create_last_month_plot(global_results_df) 
        
        # Prepare key metrics for display
        final_equity = global_results_df['Cumulative_Equity'].iloc[-1]
        benchmark_equity = global_results_df['Buy_Hold_Equity'].iloc[-1]
        
        global_summary = {
            'final_strategy_return': f"{(final_equity - 1) * 100:.2f}%",
            'final_benchmark_return': f"{(benchmark_equity - 1) * 100:.2f}%",
            'total_days': len(global_results_df),
            'start_date': global_results_df.index.min().strftime('%Y-%m-%d %H:%M'),
            'end_date': global_results_df.index.max().strftime('%Y-%m-%d %H:%M'),
        }
        global_error_message = None # Clear error on success

    except Exception as e:
        # Catch and store the error message
        global_error_message = f"Error during initial setup (Data Fetching/Backtesting): {e}"
        print(global_error_message)


# --- Flask Routes and App Setup ---

@app.route('/')
def home():
    """Serves the pre-calculated results from global cache."""
    # ROBUSTNESS CHECK: Check if summary exists, not just if an error message was written.
    if global_summary is None:
        # If global_summary is None, we failed. Use stored error or a generic one.
        error = global_error_message if global_error_message else "Analysis failed during startup. Check terminal logs for API or calculation errors."
        return render_template_string(ERROR_HTML, error=error), 500

    return render_template_string(HTML_TEMPLATE, 
                                  equity_plot=global_equity_plot, 
                                  rsi_plot=global_rsi_plot, 
                                  avg_returns_plot=global_avg_returns_plot,
                                  last_month_plot=global_last_month_plot,
                                  summary=global_summary,
                                  TIMEFRAME=TIMEFRAME,
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
            RSI ({{ RSI_PERIOD }}) Backtesting Analysis: {{ TIMEFRAME }} Timeframe
        </h1>
        <p class="text-gray-600 mb-4">Data period: {{ summary.start_date }} to {{ summary.end_date }}</p>

        <!-- Summary Metrics -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div class="card p-6 border-l-4 border-red-500">
                <p class="text-sm text-gray-500">Total Periods Analyzed</p>
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
                RSI thresholds: Long Entry at {{ LONG_ENTRY_LEVEL }} (Red Zone), Short Entry at {{ SHORT_ENTRY_LEVEL }} (Blue Zone). Signals mark the crossover points.
            </p>
        </div>

        <!-- Plot 1: Equity Curve -->
        <div class="card p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-700 mb-4">
                Strategy Equity (Decreasing Weighted Position) vs. Buy & Hold (Compounded Per Period)
            </h2>
            <img src="data:image/png;base64,{{ equity_plot }}" alt="Equity Curve Plot" class="w-full h-auto rounded-lg"/>
            <p class="text-sm text-gray-600 mt-2">
                Strategy equity curve based on the weighted position size ({{ PLOT_FUTURE_PERIOD }}/{{ PLOT_FUTURE_PERIOD }} decreasing linearly to 0 over {{ PLOT_FUTURE_PERIOD }} periods after signal).
            </p>
        </div>

        <!-- Plot 3: Last Month Position -->
        <div class="card p-6 mb-8">
            <h2 class="text-xl font-semibold text-gray-700 mb-4">
                Position Visualization: Last 30 Calendar Days (Red=Long, Blue=Short)
            </h2>
            <img src="data:image/png;base64,{{ last_month_plot }}" alt="Last Month Position Plot" class="w-full h-auto rounded-lg"/>
            <p class="text-sm text-gray-600 mt-2">
                Shows the price action and the strategy's position (background color) for the most recent 30 calendar days ({{ TIMEFRAME }} candles).
            </p>
        </div>

        <!-- Plot 2: Average Returns -->
        <div class="card p-6">
            <h2 class="text-xl font-semibold text-gray-700 mb-4">
                Average Future Cumulative Price Change (Periods 1 to {{ PLOT_FUTURE_PERIOD }})
            </h2>
            <img src="data:image/png;base64,{{ avg_returns_plot }}" alt="Average Future Returns Plot" class="w-full h-auto rounded-lg"/>
            <p class="text-sm text-gray-600 mt-2">
                This plot shows the average cumulative return of the underlying asset over the {{ PLOT_FUTURE_PERIOD }} periods following each Long (Red line) or Short (Blue line) signal entry.
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
    print("--- Starting Backtest Setup ---")
    print("Fetching data and running backtest (this will take time for 30m since 2018)...")
    setup_backtest()
    
    if global_error_message is None:
        print(f"--- Setup Complete. Starting Web Server on http://127.0.0.1:{PORT} ---")
    else:
        print("--- Setup FAILED. Server will serve error message. ---")
    
    # Run Flask application
    app.run(host='0.0.0.0', port=PORT)
