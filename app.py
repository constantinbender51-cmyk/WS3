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
START_YEAR = 2017
SYMBOL = 'BTC/USDT'
PORT = 8080

# Grid Search Parameters
TIMEFRAMES = ['30m', '1h', '4h', '1d', '1w']

# Fine-grained SMA: 10, 20, 30 ... up to 400
SMA_PERIODS = list(range(10, 401, 10)) 

# Added 128 to decay periods
DECAY_PERIODS = [1, 2, 4, 8, 16, 32, 64, 128]

# -----------------------------------------------------------------------------
# 1. Efficient Data Fetching & Resampling
# -----------------------------------------------------------------------------
def fetch_base_data(symbol, start_year):
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
            
            if len(all_ohlcv) % 20000 == 0:
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
    print("Resampling data to higher timeframes...")
    data_store = {'30m': df_30m.copy()}
    
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    data_store['1h'] = df_30m.resample('1h').agg(agg_dict).dropna()
    data_store['4h'] = df_30m.resample('4h').agg(agg_dict).dropna()
    data_store['1d'] = df_30m.resample('1D').agg(agg_dict).dropna()
    data_store['1w'] = df_30m.resample('W-MON').agg(agg_dict).dropna()
    
    return data_store

# -----------------------------------------------------------------------------
# 2. Strategy Logic (SMA Crossover + Variable Decay)
# -----------------------------------------------------------------------------
def run_strategy(df, sma_period, decay_period):
    if len(df) < sma_period:
        return pd.DataFrame()
        
    df = df.copy()
    
    # Calculate SMA
    df['sma'] = df['close'].rolling(window=sma_period).mean()
    
    # --- STRATEGY: SMA CROSSOVER ---
    # Long: Close crosses ABOVE SMA
    long_signals = (df['close'] > df['sma']) & (df['close'].shift(1) <= df['sma'].shift(1))
    
    # Short: Close crosses BELOW SMA
    short_signals = (df['close'] < df['sma']) & (df['close'].shift(1) >= df['sma'].shift(1))
    
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
        if long_timer > 0 and long_timer <= decay_period:
            long_weight = 1 - (long_timer / decay_period)**2
            if long_weight < 0: long_weight = 0

        short_weight = 0
        if short_timer > 0 and short_timer <= decay_period:
            short_weight = 1 - (short_timer / decay_period)**2
            if short_weight < 0: short_weight = 0
            
        position_weights[i] = long_weight - short_weight

    df['position'] = position_weights
    
    # Returns
    df['returns'] = df['close'].pct_change().fillna(0)
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['strategy_returns'] = df['strategy_returns'].fillna(0)
    
    return df

def calculate_sharpe(returns, timeframe):
    if len(returns) < 2 or returns.std() == 0: return -999
    factors = {'30m': 365*48, '1h': 365*24, '4h': 365*6, '1d': 365, '1w': 52}
    N = factors.get(timeframe, 365)
    return np.sqrt(N) * (returns.mean() / returns.std())

def calculate_metrics(df, timeframe):
    if df.empty: return {}
    
    valid_df = df.dropna(subset=['sma', 'strategy_returns'])
    if valid_df.empty: return {}
    
    strat_ret = valid_df['strategy_returns']
    bh_ret = valid_df['returns']
    
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
# 3. Full Grid Search (SMA + Decay)
# -----------------------------------------------------------------------------
def run_full_grid_search(data_store):
    results = []
    total_combinations = len(TIMEFRAMES) * len(SMA_PERIODS) * len(DECAY_PERIODS)
    print(f"\n--- Running Full Grid Search ({total_combinations} combinations) ---")
    
    count = 0
    start_time = time.time()
    
    for tf, df_base in data_store.items():
        if df_base is None or df_base.empty: continue
            
        for sma in SMA_PERIODS:
            for decay in DECAY_PERIODS:
                # Run Strategy
                df_res = run_strategy(df_base, sma, decay)
                
                if df_res.empty: continue

                metrics = calculate_metrics(df_res, tf)
                if not metrics: continue
                
                results.append({
                    'Timeframe': tf,
                    'SMA Period': sma,
                    'Decay Period': decay,
                    'Sharpe Ratio': metrics['sharpe'],
                    'B&H Sharpe': metrics['bh_sharpe'],
                    'Total Return (%)': metrics['total_ret'],
                    'B&H Return (%)': metrics['bh_total_ret'],
                    'df': df_res
                })
                
                count += 1
                if count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    print(f"  Processed {count}/{total_combinations} ({rate:.1f} iter/s)...")
            
    return sorted(results, key=lambda x: x['Sharpe Ratio'], reverse=True)

# -----------------------------------------------------------------------------
# 4. Plotting
# -----------------------------------------------------------------------------
def plot_winner(result_dict):
    df = result_dict['df']
    tf = result_dict['Timeframe']
    sharpe = result_dict['Sharpe Ratio']
    sma = result_dict['SMA Period']
    decay = result_dict['Decay Period']
    
    df['equity'] = (1 + df['strategy_returns']).cumprod() * 100
    df['bh_equity'] = (1 + df['returns']).cumprod() * 100
    
    plt.style.use('ggplot') 
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 12), sharex=True, 
                             gridspec_kw={'height_ratios': [2, 1, 1], 'hspace': 0.1})
    
    # Row 1: Equity
    ax1 = axes[0]
    ax1.plot(df.index, df['bh_equity'], label='Buy & Hold', color='gray', alpha=0.5, linestyle='--')
    ax1.plot(df.index, df['equity'], label=f'SMA {sma} (Decay {decay})', color='#0077B6', linewidth=2)
    ax1.set_ylabel('Equity')
    ax1.set_title(f"Best Config: {tf} | SMA {sma} | Decay {decay} | Sharpe: {sharpe}", fontsize=16)
    ax1.legend()

    # Row 2: Price & SMA
    ax2 = axes[1]
    ax2.plot(df.index, df['close'], label='Price', color='black', alpha=0.6, linewidth=1)
    ax2.plot(df.index, df['sma'], label=f'SMA {sma}', color='orange', linewidth=2)
    # Shade zones based on signal state
    ax2.fill_between(df.index, df['close'].max(), df['close'].min(), 
                     where=(df['close'] > df['sma']), color='green', alpha=0.05)
    ax2.fill_between(df.index, df['close'].max(), df['close'].min(), 
                     where=(df['close'] < df['sma']), color='red', alpha=0.05)
    ax2.set_ylabel('Price')
    ax2.legend(loc='upper left')

    # Row 3: Position Weight
    ax3 = axes[2]
    ax3.fill_between(df.index, df['position'], 0, where=(df['position']>=0), color='green', alpha=0.6, step='mid', label='Long')
    ax3.fill_between(df.index, df['position'], 0, where=(df['position']<0), color='red', alpha=0.6, step='mid', label='Short')
    ax3.set_ylabel('Decaying Weight')
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
print("Step 1: Fetching Data...")
df_30m = fetch_base_data(SYMBOL, START_YEAR)
data_store = resample_data(df_30m)

print("Step 2: Running Full Grid Search...")
sorted_results = run_full_grid_search(data_store)

if sorted_results:
    winner = sorted_results[0]
    print(f"\nWinner found: {winner['Timeframe']} SMA {winner['SMA Period']} Decay {winner['Decay Period']} (Sharpe: {winner['Sharpe Ratio']})")
    print("\nStep 3: Generating Plot...")
    winner_plot_base64 = plot_winner(winner)
else:
    print("No valid results found.")
    winner_plot_base64 = ""

# -----------------------------------------------------------------------------
# 6. Web Server
# -----------------------------------------------------------------------------
app = Dash(__name__)
server = app.server 

table_data = []
# Show top 15 results
for i, res in enumerate(sorted_results[:15]):
    table_data.append({
        'Rank': i+1,
        'Timeframe': res['Timeframe'],
        'SMA': res['SMA Period'],
        'Decay': res['Decay Period'],
        'Sharpe': res['Sharpe Ratio'],
        'Return': f"{res['Total Return (%)']}%",
        'B&H Sharpe': res['B&H Sharpe']
    })

app.layout = html.Div(style={'fontFamily': 'sans-serif', 'padding': '20px'}, children=[
    html.H1(f"SMA Crossover Fine-Grained Search", style={'textAlign': 'center'}),
    html.P(f"Signal: Cross SMA(X). Weight Decays over Y periods. Range: {START_YEAR}-Present", style={'textAlign': 'center'}),
    
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
