import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, Response
import io
import time
from datetime import timedelta

app = Flask(__name__)

# --- Configuration ---
SYMBOL = 'ETH/USDT'
TIMEFRAME = '1d'
START_DATE_STR = '2018-01-01 00:00:00'
VOLATILITY_THRESHOLD = 0.10
DEFAULT_DURATION_DAYS = 365
TRAILING_TP_PCT = 0.01

# Global cache
_cache = {'data': None, 'timestamp': 0}

def fetch_data():
    if _cache['data'] is not None and time.time() - _cache['timestamp'] < 3600:
        return _cache['data']

    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601(START_DATE_STR)
    all_candles = []
    
    while True:
        try:
            candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since)
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            if candles[-1][0] >= exchange.milliseconds() - 24*60*60*1000: 
                break
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    _cache['data'] = df
    _cache['timestamp'] = time.time()
    return df

def backtest_strategies(df):
    df['range_pct'] = (df['high'] / df['low']) - 1
    
    active_positions = [] 
    completed_trades = []
    triangles = []
    
    # 1. Identify Non-Overlapping Triangles
    next_available_date = pd.Timestamp.min
    
    for date, row in df.iterrows():
        # Check if we are inside a blocking triangle window
        if date < next_available_date:
            continue
            
        if row['range_pct'] > VOLATILITY_THRESHOLD:
            y_start = row['high']
            y_end = row['low']
            direction = 1 if row['close'] > row['open'] else -1
            slope = (y_end - y_start) / DEFAULT_DURATION_DAYS
            
            triangle = {
                'start_date': date,
                'end_date': date + timedelta(days=DEFAULT_DURATION_DAYS),
                'm': slope,
                'c': y_start,
                'direction': direction
            }
            triangles.append(triangle)
            
            # Block new triangles until this one finishes
            next_available_date = triangle['end_date']

    # 2. Iterate through history to execute trades
    if not triangles:
        return [], []
        
    start_sim = triangles[0]['start_date']
    sim_data = df.loc[start_sim:]
    
    for current_date, row in sim_data.iterrows():
        day_high = row['high']
        day_low = row['low']
        
        # A. Check Triangle Entries
        for tri in triangles:
            if tri['start_date'] < current_date <= tri['end_date']:
                days_passed = (current_date - tri['start_date']).days
                hypo_price = tri['m'] * days_passed + tri['c']
                
                # Check fill: Price crossed hypotenuse
                if day_low <= hypo_price <= day_high:
                    active_positions.append({
                        'entry_price': hypo_price,
                        'direction': tri['direction'],
                        'extreme_price': hypo_price, 
                        'entry_date': current_date
                    })

        # B. Check Trailing TP
        remaining_positions = []
        for pos in active_positions:
            triggered = False
            
            if pos['direction'] == 1: # Long
                if day_high > pos['extreme_price']:
                    pos['extreme_price'] = day_high
                
                stop_price = pos['extreme_price'] * (1 - TRAILING_TP_PCT)
                if day_low <= stop_price:
                    completed_trades.append({'date': current_date, 'price': stop_price, 'type': 'exit_long'})
                    triggered = True
            else: # Short
                if day_low < pos['extreme_price']:
                    pos['extreme_price'] = day_low
                
                stop_price = pos['extreme_price'] * (1 + TRAILING_TP_PCT)
                if day_high >= stop_price:
                    completed_trades.append({'date': current_date, 'price': stop_price, 'type': 'exit_short'})
                    triggered = True
            
            if not triggered:
                remaining_positions.append(pos)
        
        active_positions = remaining_positions

    return triangles, completed_trades

def analyze_and_plot():
    df = fetch_data()
    triangles, trades = backtest_strategies(df)
    
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, df['close'], label='Close', color='gray', alpha=0.3, linewidth=1)

    for tri in triangles:
        plt.plot([tri['start_date'], tri['end_date']], 
                 [tri['c'], tri['m'] * DEFAULT_DURATION_DAYS + tri['c']], 
                 'r--', linewidth=0.8, alpha=0.7)

    if trades:
        exit_df = pd.DataFrame(trades)
        long_exits = exit_df[exit_df['type'] == 'exit_long']
        short_exits = exit_df[exit_df['type'] == 'exit_short']
        
        if not long_exits.empty:
            plt.scatter(long_exits['date'], long_exits['price'], c='green', marker='^', s=30, label='Long Exit')
        if not short_exits.empty:
            plt.scatter(short_exits['date'], short_exits['price'], c='red', marker='v', s=30, label='Short Exit')

    plt.title(f"ETH/USDT 1D: >10% Volatility Triangles (Non-Overlapping) & 1% Trailing TP")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    buf.seek(0)
    return buf

@app.route('/')
def serve_plot():
    try:
        img_buf = analyze_and_plot()
        return Response(img_buf, mimetype='image/png')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Server Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
