import ccxt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from flask import Flask, send_file

# 1. Fetch Data
def fetch_btc_data():
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1d'
    since = exchange.parse8601('2018-01-01T00:00:00Z')
    
    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1
        if len(candles) < 1000:
            break
            
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    return df

# 2. Strategy Logic
def apply_strategy(df):
    # Anchor date: July 1, 2017
    anchor_date = pd.Timestamp('2017-07-01')
    
    df['returns'] = df['close'].pct_change()
    
    # 730 SMA
    df['sma730'] = df['close'].rolling(window=730).mean()
    
    # --- Curve Fitting Start ---
    # Log-transform SMA to handle exponential price action
    df['log_sma'] = np.log(df['sma730'])
    
    # Target: 730 SMA offset by -730 (Shift future values back to present)
    df['target_fit'] = df['log_sma'].shift(-730)
    
    # Numeric time axis for regression
    df['t'] = np.arange(len(df))
    
    # Clean data for fitting (remove NaNs from SMA window and Shift)
    fit_data = df.dropna(subset=['target_fit'])
    
    if not fit_data.empty:
        X = fit_data['t'].values
        Y = fit_data['target_fit'].values
        
        # Sine wave at a slope model: y = mt + c + A * sin(omega * t + phi)
        # Using fixed approx period of 1460 days (4 years) for stability, or optimizing it.
        # Optimizing period is preferred for exact fit.
        def sine_slope_model(t, slope, intercept, amp, phase, period):
            return slope * t + intercept + amp * np.sin(2 * np.pi * t / period + phase)
        
        # Initial guesses: Slope, Intercept, Amp, Phase, Period (approx 1460)
        p0 = [
            (Y[-1] - Y[0]) / (X[-1] - X[0]), # Slope
            Y[0],                            # Intercept
            0.5,                             # Amplitude
            0,                               # Phase
            1460                             # Period
        ]
        
        try:
            popt, _ = curve_fit(sine_slope_model, X, Y, p0=p0, maxfev=10000)
            
            # Apply fitted model to entire timeframe (including future projection)
            df['fitted_log'] = sine_slope_model(df['t'].values, *popt)
            df['sine_curve'] = np.exp(df['fitted_log'])
        except Exception as e:
            print(f"Fitting failed: {e}")
            df['sine_curve'] = np.nan
    else:
        df['sine_curve'] = np.nan
    # --- Curve Fitting End ---

    def get_signal(date):
        if date < anchor_date:
            return 0 
        
        days_since = (date - anchor_date).days
        years_passed = days_since / 365.25
        cycle_position = years_passed % 4
        
        # 0.0 - 1.0: Short (1st year)
        # 1.0 - 4.0: Long (Next 3 years)
        if 0 <= cycle_position < 1:
            return -1
        else:
            return 1

    df['signal'] = df.index.to_series().apply(get_signal)
    
    df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
    
    df['cum_bnh'] = (1 + df['returns']).cumprod()
    df['cum_strat'] = (1 + df['strategy_returns']).cumprod()
    
    return df

# 3. Server
app = Flask(__name__)

@app.route('/')
def serve_plot():
    df = fetch_btc_data()
    df = apply_strategy(df)
    
    # Create figure with secondary y-axis for price vs cumulative returns
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot Cumulative Returns (Left Axis)
    ax1.plot(df.index, df['cum_bnh'], label='Buy & Hold (Left)', alpha=0.5, color='gray')
    ax1.plot(df.index, df['cum_strat'], label='Strategy (Left)', color='blue')
    ax1.set_ylabel('Cumulative Return')
    ax1.set_yscale('log')
    
    # Plot Price and SMA (Right Axis) to scale correctly
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['sma730'], label='730 SMA (Right)', color='orange', linewidth=1.5)
    
    # Plot the Fitted Sine Curve
    if 'sine_curve' in df.columns:
        ax2.plot(df.index, df['sine_curve'], label='Fitted Sine (Offset -730)', color='magenta', linestyle='--', linewidth=1.5)

    ax2.set_ylabel('Price (USDT)')
    ax2.set_yscale('log')
    
    # Markers
    cycle_years = range(2017, df.index.year.max() + 1, 4)
    for y in cycle_years:
        d = pd.Timestamp(f'{y}-07-01')
        if d >= df.index.min() and d <= df.index.max():
            ax1.axvline(d, color='red', linestyle='--', alpha=0.3)

    plt.title('BTC 4-Year Cycle Strategy + 730 SMA + Sine Fit')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    print("Serving plot on port 8080...")
    app.run(host='0.0.0.0', port=8080)
