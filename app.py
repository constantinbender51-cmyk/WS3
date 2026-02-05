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
    anchor_date = pd.Timestamp('2017-07-01')
    
    df['returns'] = df['close'].pct_change()
    
    # 730 SMA
    df['sma730'] = df['close'].rolling(window=730).mean()
    
    # Offset SMA: Shift future values back to present (Peeking ahead for fitting/historical alignment)
    # Shift(-730) moves the value at index t+730 to index t
    df['sma730_shifted'] = df['sma730'].shift(-730)
    
    # --- Curve Fitting ---
    # Log-transform to handle exponential scale
    df['log_sma_shifted'] = np.log(df['sma730_shifted'])
    
    # Numeric time axis
    df['t'] = np.arange(len(df))
    
    # Fit only where we have valid shifted SMA data (stops 730 days before present)
    fit_data = df.dropna(subset=['log_sma_shifted'])
    
    if not fit_data.empty:
        X = fit_data['t'].values
        Y = fit_data['log_sma_shifted'].values
        
        # Model: Linear trend + Sine wave
        def sine_slope_model(t, slope, intercept, amp, phase, period):
            return slope * t + intercept + amp * np.sin(2 * np.pi * t / period + phase)
        
        # Initial guess
        p0 = [
            (Y[-1] - Y[0]) / (X[-1] - X[0]), # Slope
            Y[0],                            # Intercept
            0.5,                             # Amplitude
            0,                               # Phase
            1460                             # Period (~4 years)
        ]
        
        try:
            # Fit model
            popt, _ = curve_fit(sine_slope_model, X, Y, p0=p0, maxfev=10000)
            
            # Generate sine curve for WHOLE timeframe (including recent days where SMA is NaN)
            df['fitted_log'] = sine_slope_model(df['t'].values, *popt)
            df['sine_curve'] = np.exp(df['fitted_log'])
        except Exception:
            df['sine_curve'] = np.nan
    else:
        df['sine_curve'] = np.nan

    # Strategy Signal (Fixed Time-based)
    def get_signal(date):
        if date < anchor_date:
            return 0 
        days_since = (date - anchor_date).days
        years_passed = days_since / 365.25
        cycle_position = years_passed % 4
        # 0.0 - 1.0: Short (1st year of cycle)
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
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Left Axis: Returns
    ax1.plot(df.index, df['cum_bnh'], label='Buy & Hold', alpha=0.3, color='gray')
    ax1.plot(df.index, df['cum_strat'], label='Strategy', color='blue', alpha=0.6)
    ax1.set_ylabel('Cumulative Return')
    ax1.set_yscale('log')
    
    # Right Axis: Price & Models
    ax2 = ax1.twinx()
    
    # Plot the SHIFTED SMA (stops early)
    ax2.plot(df.index, df['sma730_shifted'], label='730 SMA (Offset -730)', color='orange', linewidth=2)
    
    # Plot the FITTED SINE (extends to present)
    if 'sine_curve' in df.columns:
        ax2.plot(df.index, df['sine_curve'], label='Fitted Sine Slope', color='magenta', linestyle='--', linewidth=1.5)
        
    ax2.set_ylabel('Price / Model (USDT)')
    ax2.set_yscale('log')
    
    # Cycle Markers
    cycle_years = range(2017, df.index.year.max() + 1, 4)
    for y in cycle_years:
        d = pd.Timestamp(f'{y}-07-01')
        if d >= df.index.min() and d <= df.index.max():
            ax1.axvline(d, color='red', linestyle=':', alpha=0.5)

    plt.title('BTC: 4-Year Cycle + Sine Fit to Offset SMA')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.grid(True, which="both", ls="-", alpha=0.1)
    
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    print("Serving plot on port 8080...")
    app.run(host='0.0.0.0', port=8080)
