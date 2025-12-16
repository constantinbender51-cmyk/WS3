import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import io
import base64
from dash import Dash, dcc, html, dash_table
import datetime
import time

# --- Configuration ---
# To ensure fair comparison, we try to fetch a consistent time window.
# 30m data from 2018 is too heavy for a quick script, so we start from 2021.
START_YEAR = 2021 
SYMBOL = 'BTC/USDT'
PORT = 8080

TIMEFRAMES = ['30m', '1h', '4h', '1d', '1w']
DECAY_PERIODS = [1, 2, 4, 8, 16, 32, 64]

# -----------------------------------------------------------------------------
# 1. Data Fetching
# -----------------------------------------------------------------------------
def fetch_data_for_timeframe(symbol, timeframe, start_year):
    print(f"Fetching {timeframe} data for {symbol} starting {start_year}...")
    exchange = ccxt.binance({'enableRateLimit': True})
    
    start_date = datetime.datetime(start_year, 1, 1)
    since = int(start_date.timestamp() * 1000)
    
    all_ohlcv = []
    limit = 1000
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            
            # Progress update for large fetches
            if len(all_ohlcv) % 10000 == 0:
                print(f"  {timeframe}: {len(all_ohlcv)} candles...")

            # Stop if we reach current time
            if last_timestamp > (time.time() * 1000) - 60000:
                break
            
            time.sleep(0.05) 
        except Exception as e:
            print(f"  Error fetching {timeframe}: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    
    return df

def fetch_all_timeframes():
    data_store = {}
    for tf in TIMEFRAMES:
        df = fetch_data_for_timeframe(SYMBOL, tf, START_YEAR)
        data_store[tf] = df
    return data_store

# -----------------------------------------------------------------------------
# 2. Strategy Logic
# -----------------------------------------------------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_decay_weight(elapsed_steps, max_steps):
    if max_steps == 0: return 0
    days_elapsed = elapsed_steps # We treat 'steps' as the unit here
    weight = 1 - (days_elapsed / max_steps)**2
    return np.maximum(weight, 0)

def run_strategy(df, duration_periods):
    """
    Runs the Mutually Exclusive strategy with SYMMETRIC duration.
    duration_periods: The decay length in candles (1, 2, 4, 8...).
    """
    # Create copy to avoid modifying the cached dataframe
    df = df.copy()
    
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    # Crossover Signals
    long_signals = (df['rsi'] < 30) & (df['rsi'].shift(1) >= 30)
    short_signals = (df['rsi'] > 70) & (df['rsi'].shift(1) <= 70)
    
    n = len(df)
    position_weights = np.zeros(n)
    
    long_timer = 0
    short_timer = 0
    
    # Fast loop using numpy arrays for read-access where possible might be faster,
    # but the state-machine nature requires iteration.
    # We optimize by extracting numpy arrays first.
    long_sig_arr = long_signals.values
    short_sig_arr = short_signals.values
    
    for i in range(1, n):
        if long_sig_arr[i]:
            long_timer = 1
            short_timer = 0
        elif short_sig_arr[i]:
            short_timer = 1
            long_timer = 0
        else:
            if long_timer > 0: long_timer += 1
            if short_timer > 0: short_timer += 1
            
        long_weight = 0
        if long_timer > 0 and long_timer <= duration_periods:
            # 1 - (elapsed/duration)^2
            long_weight = 1 - (long_timer / duration_periods)**2
            if long_weight < 0: long_weight = 0

        short_weight = 0
        if short_timer > 0 and short_timer <= duration_periods:
            short_weight = 1 - (short_timer / duration_periods)**2
            if short_weight < 0: short_weight = 0
            
        position_weights[i] = long_weight - short_weight

    df['position'] = position_weights
    
    # Calculate Returns
    df['returns'] = df['close'].pct_change().fillna(0)
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['strategy_returns'] = df['strategy_returns'].fillna(0)
    
    return df

def calculate_sharpe(df, timeframe):
    """Calculates Annualized Sharpe Ratio."""
    if df.empty: return -999
    
    returns = df['strategy_returns']
    if returns.std() == 0: return -999
    
    # Annualization Factors
    factors = {
        '30m': 365 * 48,
        '1h':  365 * 24,
        '4h':  365 * 6,
        '1d':  365,
        '1w':  52
    }
    
    N = factors.get(timeframe, 365)
    sharpe = np.sqrt(N) * (returns.mean() / returns.std())
    return sharpe

# -----------------------------------------------------------------------------
# 3. Grid Search
# -----------------------------------------------------------------------------
def run_grid_search(data_store):
    results = []
    
    print("\n--- Starting Grid Search ---")
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"Decay Periods: {DECAY_PERIODS}")
    
    for tf in TIMEFRAMES:
        df_base = data_store.get(tf)
        if df_base is None or df_base.empty:
            print(f"Skipping {tf} (No Data)")
            continue
            
        for periods in DECAY_PERIODS:
            # Run Backtest
            df_res = run_strategy(df_base, periods)
            
            # Calculate Metrics
            sharpe = calculate_sharpe(df_res, tf)
            total_return = (1 + df_res['strategy_returns']).cumprod().iloc[-1] - 1
            
            results.append({
                'Timeframe': tf,
                'Decay (Candles)': periods,
                'Sharpe Ratio': round(sharpe, 4),
                'Total Return': round(total_return * 100, 2), # %
                'df': df_res # Store df for plotting the winner
            })
            
    # Sort by Sharpe Ratio (Descending)
    sorted_results = sorted(results, key=lambda x: x['Sharpe Ratio'], reverse=True)
    return sorted_results

# -----------------------------------------------------------------------------
# 4. Plotting
# -----------------------------------------------------------------------------
def plot_winner(result_dict):
    df = result_dict['df']
    tf = result_dict['Timeframe']
    decay = result_dict['Decay (Candles)']
    sharpe = result_dict['Sharpe Ratio']
    
    df['equity'] = (1 + df['strategy_returns']).cumprod() * 100
    
    plt.style.use('ggplot') 
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 12), sharex=True, 
                             gridspec_kw={'height_ratios': [2, 1, 1], 'hspace': 0.1})
    
    # Row 1: Price & Equity
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='BTC Price', color='#333333', linewidth=1)
    ax1.set_ylabel('Price', color='#333333')
    
    ax1b = ax1.twinx()
    ax1b.plot(df.index, df['equity'], label='Equity', color='#0077B6', linewidth=2)
    ax1b.set_ylabel('Equity (Start=100)', color='#0077B6')
    
    # Background Shading
    trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
    is_long = df['position'] > 0.05
    is_short = df['position'] < -0.05
    ax1.fill_between(df.index, 0, 1, where=is_long, transform=trans, color='green', alpha=0.1, linewidth=0)
    ax1.fill_between(df.index, 0, 1, where=is_short, transform=trans, color='red', alpha=0.1, linewidth=0)
    
    ax1.set_title(f"WINNER: {tf} | Decay: {decay} Candles | Sharpe: {sharpe}", fontsize=16)
    ax1.legend(loc='upper left')
    ax1b.legend(loc='upper right')

    # Row 2: RSI
    ax2 = axes[1]
    ax2.plot(df.index, df['rsi'], color='#724C9F', linewidth=1)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)

    # Row 3: Position
    ax3 = axes[2]
    ax3.fill_between(df.index, df['position'], 0, where=(df['position']>=0), color='green', alpha=0.6, step='mid')
    ax3.fill_between(df.index, df['position'], 0, where=(df['position']<0), color='red', alpha=0.6, step='mid')
    ax3.set_ylabel('Position')
    ax3.set_xlabel('Date')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) 
    return f'data:image/png;base64,{img_base64}'

# -----------------------------------------------------------------------------
# 5. Execution
# -----------------------------------------------------------------------------
print("Step 1: Fetching Data...")
data_store = fetch_all_timeframes()

print("Step 2: Running Grid Search...")
sorted_results = run_grid_search(data_store)

top_5 = sorted_results[:5]
winner = top_5[0]

print("\n--- TOP 5 RESULTS ---")
for i, res in enumerate(top_5):
    print(f"{i+1}. {res['Timeframe']} | Decay: {res['Decay (Candles)']} | Sharpe: {res['Sharpe Ratio']} | Ret: {res['Total Return']}%")

print("\nStep 3: Generating Plot for Winner...")
winner_plot_base64 = plot_winner(winner)

# -----------------------------------------------------------------------------
# 6. Web Server
# -----------------------------------------------------------------------------
app = Dash(__name__)
server = app.server 

# Prepare table data
table_data = []
for i, res in enumerate(top_5):
    table_data.append({
        'Rank': i+1,
        'Timeframe': res['Timeframe'],
        'Decay (Candles)': res['Decay (Candles)'],
        'Sharpe Ratio': res['Sharpe Ratio'],
        'Total Return (%)': res['Total Return']
    })

app.layout = html.Div(style={'fontFamily': 'sans-serif', 'padding': '20px', 'maxWidth': '1000px', 'margin': '0 auto'}, children=[
    html.H1("RSI Strategy Optimization", style={'textAlign': 'center'}),
    html.P(f"Backtest Range: {START_YEAR} - Present", style={'textAlign': 'center', 'color': '#666'}),
    
    html.Div([
        html.H3("Top 5 Configurations (Sharpe Ratio)"),
        dash_table.DataTable(
            data=table_data,
            columns=[{'name': i, 'id': i} for i in ['Rank', 'Timeframe', 'Decay (Candles)', 'Sharpe Ratio', 'Total Return (%)']],
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f2f2f2'},
            style_data_conditional=[
                {'if': {'row_index': 0}, 'backgroundColor': '#d1e7dd', 'fontWeight': 'bold'}
            ]
        )
    ], style={'marginBottom': '40px'}),
    
    html.Div([
        html.H3(f"Best Performer: {winner['Timeframe']} (Decay {winner['Decay (Candles)']})"),
        html.Img(src=winner_plot_base64, style={'width': '100%', 'borderRadius': '8px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.1)'})
    ])
])

if __name__ == '__main__':
    print("Starting Server...")
    app.run_server(debug=True, port=PORT)
