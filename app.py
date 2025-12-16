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
START_YEAR = 2021
SYMBOL = 'BTC/USDT'
PORT = 8080
FIXED_DECAY = 16 

# -----------------------------------------------------------------------------
# 1. Efficient Data Fetching & Resampling
# -----------------------------------------------------------------------------
def fetch_base_data(symbol, start_year):
    """
    Fetches ONLY the base 30m data. 
    We will calculate all other timeframes from this to save API calls.
    """
    timeframe = '30m'
    print(f"Fetching base {timeframe} data for {symbol} starting {start_year}...")
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
            
            # Progress indicator
            if len(all_ohlcv) % 10000 == 0:
                print(f"  Fetched {len(all_ohlcv)} base candles...")

            if last_timestamp > (time.time() * 1000) - 60000:
                break
            
            time.sleep(0.05) 
        except Exception as e:
            print(f"  Error fetching base data: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    
    print(f"Base data fetch complete. {len(df)} rows.")
    return df

def resample_data(df_30m):
    """
    Generates higher timeframe data from 30m source.
    """
    print("Resampling data to higher timeframes...")
    data_store = {'30m': df_30m.copy()}
    
    # Aggregation rules for OHLCV
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # 1 Hour
    data_store['1h'] = df_30m.resample('1h').agg(agg_dict).dropna()
    
    # 4 Hour
    data_store['4h'] = df_30m.resample('4h').agg(agg_dict).dropna()
    
    # 1 Day (Daily)
    data_store['1d'] = df_30m.resample('1D').agg(agg_dict).dropna()
    
    # 1 Week (Weekly) - 'W-MON' ensures weeks start on Monday (standard for crypto)
    data_store['1w'] = df_30m.resample('W-MON').agg(agg_dict).dropna()
    
    return data_store

# -----------------------------------------------------------------------------
# 2. Strategy Logic (RSI 50 Cross Momentum)
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

def run_strategy(df, duration_periods):
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    # Momentum 50 Cross Signals
    long_signals = (df['rsi'] > 50) & (df['rsi'].shift(1) <= 50)
    short_signals = (df['rsi'] < 50) & (df['rsi'].shift(1) >= 50)
    
    n = len(df)
    position_weights = np.zeros(n)
    
    long_timer = 0
    short_timer = 0
    
    long_sig_arr = long_signals.values
    short_sig_arr = short_signals.values
    
    for i in range(1, n):
        # Mutual Exclusion State Machine
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
            long_weight = 1 - (long_timer / duration_periods)**2
            if long_weight < 0: long_weight = 0

        short_weight = 0
        if short_timer > 0 and short_timer <= duration_periods:
            short_weight = 1 - (short_timer / duration_periods)**2
            if short_weight < 0: short_weight = 0
            
        position_weights[i] = long_weight - short_weight

    df['position'] = position_weights
    
    # Returns
    df['returns'] = df['close'].pct_change().fillna(0)
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['strategy_returns'] = df['strategy_returns'].fillna(0)
    
    return df

def calculate_sharpe(returns, timeframe):
    if returns.std() == 0: return -999
    # Approximate annualized factors
    factors = {'30m': 365*48, '1h': 365*24, '4h': 365*6, '1d': 365, '1w': 52}
    N = factors.get(timeframe, 365)
    return np.sqrt(N) * (returns.mean() / returns.std())

def calculate_metrics(df, timeframe):
    if df.empty: return {}
    
    strat_ret = df['strategy_returns']
    bh_ret = df['returns']
    
    sharpe_strat = calculate_sharpe(strat_ret, timeframe)
    sharpe_bh = calculate_sharpe(bh_ret, timeframe)
    
    total_ret_strat = (1 + strat_ret).cumprod().iloc[-1] - 1
    total_ret_bh = (1 + bh_ret).cumprod().iloc[-1] - 1
    
    return {
        'sharpe': round(sharpe_strat, 4),
        'bh_sharpe': round(sharpe_bh, 4),
        'total_ret': round(total_ret_strat * 100, 2),
        'bh_total_ret': round(total_ret_bh * 100, 2)
    }

# -----------------------------------------------------------------------------
# 3. Targeted Analysis (Fixed 16-Period Decay)
# -----------------------------------------------------------------------------
def run_targeted_analysis(data_store):
    results = []
    print(f"\n--- Running Analysis (Decay: {FIXED_DECAY}) ---")
    
    for tf, df_base in data_store.items():
        if df_base is None or df_base.empty: continue
            
        df_res = run_strategy(df_base, FIXED_DECAY)
        metrics = calculate_metrics(df_res, tf)
        
        results.append({
            'Timeframe': tf,
            'Decay': FIXED_DECAY,
            'Sharpe Ratio': metrics['sharpe'],
            'B&H Sharpe': metrics['bh_sharpe'],
            'Total Return (%)': metrics['total_ret'],
            'B&H Return (%)': metrics['bh_total_ret'],
            'df': df_res
        })
            
    return sorted(results, key=lambda x: x['Sharpe Ratio'], reverse=True)

# -----------------------------------------------------------------------------
# 4. Plotting
# -----------------------------------------------------------------------------
def plot_winner(result_dict):
    df = result_dict['df']
    tf = result_dict['Timeframe']
    sharpe = result_dict['Sharpe Ratio']
    
    df['equity'] = (1 + df['strategy_returns']).cumprod() * 100
    df['bh_equity'] = (1 + df['returns']).cumprod() * 100
    
    plt.style.use('ggplot') 
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 12), sharex=True, 
                             gridspec_kw={'height_ratios': [2, 1, 1], 'hspace': 0.1})
    
    # Row 1: Equity
    ax1 = axes[0]
    ax1.plot(df.index, df['bh_equity'], label='Buy & Hold', color='gray', alpha=0.5, linestyle='--')
    ax1.plot(df.index, df['equity'], label=f'RSI 50 Mom ({FIXED_DECAY} decay)', color='#0077B6', linewidth=2)
    ax1.set_ylabel('Equity')
    ax1.set_title(f"Performance: {tf} | Sharpe: {sharpe}", fontsize=16)
    ax1.legend()

    # Row 2: RSI
    ax2 = axes[1]
    ax2.plot(df.index, df['rsi'], color='#724C9F', linewidth=1)
    ax2.axhline(50, color='black', linestyle='--', linewidth=1.5)
    ax2.fill_between(df.index, 50, 100, where=(df['rsi']>50), color='green', alpha=0.05)
    ax2.fill_between(df.index, 0, 50, where=(df['rsi']<50), color='red', alpha=0.05)
    ax2.set_ylabel('RSI')

    # Row 3: Position
    ax3 = axes[2]
    ax3.fill_between(df.index, df['position'], 0, where=(df['position']>=0), color='green', alpha=0.6, step='mid', label='Long')
    ax3.fill_between(df.index, df['position'], 0, where=(df['position']<0), color='red', alpha=0.6, step='mid', label='Short')
    ax3.set_ylabel('Weight')
    ax3.set_xlabel('Date')
    ax3.legend()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) 
    return f'data:image/png;base64,{img_base64}'

# -----------------------------------------------------------------------------
# 5. Execution
# -----------------------------------------------------------------------------
# Step 1: Fetch 30m ONLY
df_30m = fetch_base_data(SYMBOL, START_YEAR)

# Step 2: Resample in memory
data_store = resample_data(df_30m)

# Step 3: Run Strategy
sorted_results = run_targeted_analysis(data_store)
winner = sorted_results[0]

# Step 4: Plot
print("\nGenerating Plot...")
winner_plot_base64 = plot_winner(winner)

# -----------------------------------------------------------------------------
# 6. Web Server
# -----------------------------------------------------------------------------
app = Dash(__name__)
server = app.server 

table_data = []
for i, res in enumerate(sorted_results):
    table_data.append({
        'Rank': i+1,
        'Timeframe': res['Timeframe'],
        'Strat Sharpe': res['Sharpe Ratio'],
        'B&H Sharpe': res['B&H Sharpe'],
        'Strat Return': f"{res['Total Return (%)']}%",
        'B&H Return': f"{res['B&H Return (%)']}%"
    })

app.layout = html.Div(style={'fontFamily': 'sans-serif', 'padding': '20px'}, children=[
    html.H1(f"RSI 50 Momentum: Optimized Fetching", style={'textAlign': 'center'}),
    html.P("Fetching 30m base data -> Resampling to 1h, 4h, 1d, 1w.", style={'textAlign': 'center'}),
    
    html.Div([
        dash_table.DataTable(
            data=table_data,
            columns=[{'name': k, 'id': k} for k in table_data[0].keys()],
            style_cell={'textAlign': 'center'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f2f2f2'}
        )
    ], style={'marginBottom': '40px'}),
    html.Div([
        html.Img(src=winner_plot_base64, style={'width': '100%'})
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True, port=PORT)
