import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for web server plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import io
import base64
import http.server
import socketserver
import urllib.parse
import socket
import time
import random

# --- Configuration ---
SYMBOL = 'BTC/USDT'  # Default starting symbol
TIMEFRAME = '1h'
DAYS_BACK = 30           
STARTING_BALANCE = 10000
PORT = 8000

# Available Cryptocurrencies for Dropdown
CRYPTO_CHOICES = {
    'BTC/USDT': '#1 Bitcoin (BTC)',
    'ETH/USDT': '#2 Ethereum (ETH)',
    'XRP/USDT': '#5 XRP (XRP)',
    'SOL/USDT': '#6 Solana (SOL)',
    'DOGE/USDT': '#9 Dogecoin (DOGE)',
    'ADA/USDT': '#10 Cardano (ADA)',
    'BCH/USDT': '#11 Bitcoin Cash (BCH)',
    'LINK/USDT': '#13 Chainlink (LINK)',
    'XLM/USDT': '#15 Stellar (XLM)',
    'SUI/USDT': '#19 Sui (SUI)',
    'AVAX/USDT': '#22 Avalanche (AVAX)',
    'LTC/USDT': '#23 Litecoin (LTC)',
    'HBAR/USDT': '#24 Hedera (HBAR)',
    'SHIB/USDT': '#25 Shiba Inu (SHIB)',
    'TON/USDT': '#28 Toncoin (TON)',
    'PEPE/USDT': 'Pepe (PEPE)'
}

# Global variable to hold market data so we only fetch it once per symbol
GLOBAL_DF = None

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def fetch_binance_data_accurate(symbol, days):
    print(f"\n==========================================")
    print(f"Fetching 1h & 1m data for {symbol}...")
    print(f"==========================================")
    exchange = ccxt.binance({'enableRateLimit': True})
    since_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    
    # 1. Fetch 1h Data
    ohlcv_1h = exchange.fetch_ohlcv(symbol, '1h', since=since_ms, limit=1000)
    df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms')
    
    # Pre-calculate Wick and Body metrics
    df_1h['body'] = abs(df_1h['close'] - df_1h['open'])
    df_1h['top_wick'] = df_1h['high'] - df_1h[['open', 'close']].max(axis=1)
    df_1h['bot_wick'] = df_1h[['open', 'close']].min(axis=1) - df_1h['low']
    df_1h['max_wick'] = df_1h[['top_wick', 'bot_wick']].max(axis=1)
    
    df_1h['wick_pct'] = df_1h['max_wick'] / df_1h['open']
    df_1h['wick_body_ratio'] = np.where(df_1h['body'] == 0, np.inf, df_1h['max_wick'] / df_1h['body'])

    # 2. Fetch 1m Data
    print(f"Fetching ~43,200 intraday 1m candles for precise Trailing Stops (Takes ~3-5 seconds)...")
    all_1m = []
    current_since = since_ms
    while True:
        ohlcv_1m = exchange.fetch_ohlcv(symbol, '1m', since=current_since, limit=1000)
        if not ohlcv_1m: break
        all_1m.extend(ohlcv_1m)
        current_since = ohlcv_1m[-1][0] + 60000 
        if len(ohlcv_1m) < 1000: break
            
    df_1m = pd.DataFrame(all_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms')
    df_1m['hour_ts'] = df_1m['timestamp'].dt.floor('h')
    
    # 3. Map 1m arrays to their corresponding 1h candle
    grouped = df_1m.groupby('hour_ts')
    m1_ts, m1_opens, m1_highs, m1_lows, m1_closes = [], [], [], [], []
    
    for ts in df_1h['timestamp']:
        if ts in grouped.groups:
            grp = grouped.get_group(ts)
            m1_ts.append(grp['timestamp'].values)
            m1_opens.append(grp['open'].values)
            m1_highs.append(grp['high'].values)
            m1_lows.append(grp['low'].values)
            m1_closes.append(grp['close'].values)
        else:
            m1_ts.append(np.array([])); m1_opens.append(np.array([])); m1_highs.append(np.array([])); m1_lows.append(np.array([])); m1_closes.append(np.array([]))
            
    df_1h['m1_ts'] = m1_ts
    df_1h['m1_opens'] = m1_opens
    df_1h['m1_highs'] = m1_highs
    df_1h['m1_lows'] = m1_lows
    df_1h['m1_closes'] = m1_closes
    print("✅ Data Fetch Complete!\n")
    return df_1h

# ==========================================
# 1. CORE ENGINE (With Deep Dive Capture)
# ==========================================
def run_backtest(df, sl_pct, tsl_pct, f_pct, g_pct, u_pct):
    balance = STARTING_BALANCE
    sl_history, tsl_history = [False], [False]
    position_history, exit_price_history = [0], [0.0]
    state_history, equity_history = [2], [balance]
    
    state = 2
    low_profit_count = 0
    all_trades = []  
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # State Transitions
        if prev['wick_pct'] > f_pct or prev['wick_body_ratio'] > g_pct:
            state = 1
            low_profit_count = 0 
        elif low_profit_count >= 3:
            state = 2
            
        if state == 2:
            sl_history.append(False); tsl_history.append(False); position_history.append(0)
            exit_price_history.append(current['close']); state_history.append(2); equity_history.append(balance)
            continue
            
        # --- State 1 Trading ---
        position = 1 if prev['close'] >= prev['open'] else -1
        entry_price = current['open']
        
        m1_h, m1_l = current['m1_highs'], current['m1_lows']
        
        sl_hit, tsl_hit = False, False
        exit_price = current['close']
        pnl_pct = 0.0
        
        # Deep Dive Tracking Variables
        rolling_stops = []
        exit_idx = len(m1_h) - 1
        exit_reason = "End of Hour"
        initial_sl = entry_price
        act_price = entry_price
        
        if len(m1_h) == 0:
            pnl_pct = (current['close'] - entry_price) / entry_price if position == 1 else (entry_price - current['close']) / entry_price
        else:
            if position == 1:
                initial_sl = entry_price * (1 - sl_pct)
                act_price = entry_price * (1 + tsl_pct)
                highest = entry_price
                active_tsl = False
                tsl_price = 0.0
                
                for j, (mh, ml) in enumerate(zip(m1_h, m1_l)):
                    curr_stop = tsl_price if active_tsl else initial_sl
                    rolling_stops.append(curr_stop)
                    
                    if ml <= curr_stop:
                        exit_price = curr_stop
                        sl_hit = not active_tsl
                        tsl_hit = active_tsl
                        exit_idx = j
                        exit_reason = "Trailing SL Hit" if active_tsl else "Initial SL Hit"
                        break
                        
                    if mh > highest:
                        highest = mh
                        if highest >= act_price:
                            active_tsl = True
                            if highest * (1 - tsl_pct) > tsl_price:
                                tsl_price = highest * (1 - tsl_pct)
                                
                pnl_pct = (exit_price - entry_price) / entry_price
                
            elif position == -1:
                initial_sl = entry_price * (1 + sl_pct)
                act_price = entry_price * (1 - tsl_pct)
                lowest = entry_price
                active_tsl = False
                tsl_price = float('inf')
                
                for j, (mh, ml) in enumerate(zip(m1_h, m1_l)):
                    curr_stop = tsl_price if active_tsl else initial_sl
                    rolling_stops.append(curr_stop)
                    
                    if mh >= curr_stop:
                        exit_price = curr_stop
                        sl_hit = not active_tsl
                        tsl_hit = active_tsl
                        exit_idx = j
                        exit_reason = "Trailing SL Hit" if active_tsl else "Initial SL Hit"
                        break
                        
                    if ml < lowest:
                        lowest = ml
                        if lowest <= act_price:
                            active_tsl = True
                            if lowest * (1 + tsl_pct) < tsl_price:
                                tsl_price = lowest * (1 + tsl_pct)
                                
                pnl_pct = (entry_price - exit_price) / entry_price
                    
        # State 2 Check Triggers
        if pnl_pct < u_pct: low_profit_count += 1
        else: low_profit_count = 0
                
        balance *= (1 + pnl_pct) 
            
        sl_history.append(sl_hit); tsl_history.append(tsl_hit); position_history.append(position)
        exit_price_history.append(exit_price); state_history.append(1); equity_history.append(balance)
        
        # --- Capture Trade Details for deep dive ---
        if position != 0:
            rolling_stops_padded = rolling_stops + [rolling_stops[-1]] * (len(m1_h) - len(rolling_stops)) if rolling_stops else []
            all_trades.append({
                'trigger_ts': prev['timestamp'], 'trade_ts': current['timestamp'],
                'trigger_w_pct': prev['wick_pct'], 'trigger_wb_ratio': prev['wick_body_ratio'],
                'f': f_pct, 'g': g_pct, 'prev_color': 'Green' if prev['close'] >= prev['open'] else 'Red',
                'direction': 'LONG' if position == 1 else 'SHORT',
                'entry_price': entry_price, 'initial_sl': initial_sl, 'act_price': act_price,
                'exit_price': exit_price, 'exit_reason': exit_reason, 'exit_idx': exit_idx,
                'pnl_pct': pnl_pct,
                'prev_m1_ts': prev['m1_ts'], 'prev_m1_o': prev['m1_opens'], 'prev_m1_h': prev['m1_highs'], 'prev_m1_l': prev['m1_lows'], 'prev_m1_c': prev['m1_closes'],
                'm1_ts': current['m1_ts'], 'm1_o': current['m1_opens'], 'm1_h': current['m1_highs'], 'm1_l': current['m1_lows'], 'm1_c': current['m1_closes'],
                'rolling_stops': rolling_stops_padded
            })
                
    return balance, sl_history, tsl_history, position_history, exit_price_history, state_history, equity_history, all_trades

# ==========================================
# 2. FAST GA ENGINE 
# ==========================================
def fast_ga_backtest(opens, closes, wick_pcts, wick_body_ratios, m1_highs_list, m1_lows_list, sl_pct, tsl_pct, f_pct, g_pct, u_pct):
    balance = 10000.0
    equity = np.empty(len(opens))
    equity[0] = balance
    state, low_profit_count = 2, 0
    for i in range(1, len(opens)):
        if wick_pcts[i-1] > f_pct or wick_body_ratios[i-1] > g_pct: state, low_profit_count = 1, 0
        elif low_profit_count >= 3: state = 2
        if state == 2:
            equity[i] = balance
            continue
        position = 1 if closes[i-1] >= opens[i-1] else -1
        c_open, c_close, m1_h, m1_l = opens[i], closes[i], m1_highs_list[i], m1_lows_list[i]
        exit_price = c_close
        if len(m1_h) > 0:
            if position == 1:
                initial_sl, act_price, highest, active_tsl, tsl_price = c_open*(1-sl_pct), c_open*(1+tsl_pct), c_open, False, 0.0
                for mh, ml in zip(m1_h, m1_l):
                    curr_stop = tsl_price if active_tsl else initial_sl
                    if ml <= curr_stop: exit_price = curr_stop; break
                    if mh > highest:
                        highest = mh
                        if highest >= act_price:
                            active_tsl = True
                            if highest*(1-tsl_pct) > tsl_price: tsl_price = highest*(1-tsl_pct)
            else:
                initial_sl, act_price, lowest, active_tsl, tsl_price = c_open*(1+sl_pct), c_open*(1-tsl_pct), c_open, False, float('inf')
                for mh, ml in zip(m1_h, m1_l):
                    curr_stop = tsl_price if active_tsl else initial_sl
                    if mh >= curr_stop: exit_price = curr_stop; break
                    if ml < lowest:
                        lowest = ml
                        if lowest <= act_price:
                            active_tsl = True
                            if lowest*(1+tsl_pct) < tsl_price: tsl_price = lowest*(1+tsl_pct)
        pnl = (exit_price - c_open)/c_open if position == 1 else (c_open - exit_price)/c_open
        if pnl < u_pct: low_profit_count += 1
        else: low_profit_count = 0
        balance *= (1 + pnl); equity[i] = balance
    return equity

def run_ga_optimization(df):
    print(f"\n🧬 Starting Genetic Algorithm for {SYMBOL}...")
    start_time = time.time()
    opens, closes = df['open'].values, df['close'].values
    wick_pcts, wick_body_ratios = df['wick_pct'].values, df['wick_body_ratio'].values
    m1_highs_list, m1_lows_list = df['m1_highs'].values, df['m1_lows'].values

    bounds = np.array([[0.1, 4.0], [50.0, 400.0], [-1.0, 2.0], [0.5, 6.0], [0.5, 6.0]])
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (50, 5))

    def evaluate(ind):
        equity = fast_ga_backtest(opens, closes, wick_pcts, wick_body_ratios, m1_highs_list, m1_lows_list, ind[3]/100.0, ind[4]/100.0, ind[0]/100.0, ind[1]/100.0, ind[2]/100.0)
        returns = np.diff(equity) / equity[:-1]
        std = np.std(returns)
        if std == 0: return -999.0
        return (np.mean(returns) / std) * np.sqrt(365*24)

    for gen in range(10):
        fitness = np.array([evaluate(ind) for ind in pop])
        next_pop = [pop[np.argmax(fitness)], pop[np.argmax(fitness)].copy()]
        while len(next_pop) < 50:
            i1, i2 = np.random.choice(50, 2, replace=False)
            p1 = pop[i1] if fitness[i1] > fitness[i2] else pop[i2]
            i1, i2 = np.random.choice(50, 2, replace=False)
            p2 = pop[i1] if fitness[i1] > fitness[i2] else pop[i2]
            child = np.concatenate([p1[:2], p2[2:]]) if np.random.rand() < 0.6 else p1.copy()
            for i in range(5):
                if np.random.rand() < 0.25:
                    child[i] += np.random.normal(0, (bounds[i, 1] - bounds[i, 0]) * 0.1)
                    child[i] = np.clip(child[i], bounds[i, 0], bounds[i, 1])
            next_pop.append(child)
        pop = np.array(next_pop)
        
    fitness = np.array([evaluate(ind) for ind in pop])
    best = pop[np.argmax(fitness)]
    print(f"✅ Optimization Finished in {time.time()-start_time:.1f}s!")
    return { 'f': round(best[0], 2), 'g': round(best[1], 2), 'u': round(best[2], 2), 'sl': round(best[3], 2), 'tsl': round(best[4], 2) }

# ==========================================
# 3. WEB VISUALIZATION & REPORTING
# ==========================================
def generate_html_report(df, params, final_balance, roi, sharpe, selected_trade):
    print(f"Generating charts for web display...")
    
    # Generate Crypto Dropdown HTML
    options_html = ""
    for val, label in CRYPTO_CHOICES.items():
        selected = "selected" if val == SYMBOL else ""
        options_html += f'<option value="{val}" {selected}>{label}</option>\n'

    # === CHART 1: 48H MACRO VIEW ===
    plt.figure(figsize=(16, 10))
    df_2d = df.tail(48).copy()
    
    ax1 = plt.subplot(2, 1, 1)
    up, down = df_2d[df_2d['close'] >= df_2d['open']], df_2d[df_2d['close'] < df_2d['open']]
    ax1.bar(up['timestamp'], up['close'] - up['open'], bottom=up['open'], color='green', width=0.03, zorder=3)
    ax1.vlines(up['timestamp'], up['low'], up['high'], color='green', linewidth=1, zorder=3)
    ax1.bar(down['timestamp'], down['open'] - down['close'], bottom=down['close'], color='red', width=0.03, zorder=3)
    ax1.vlines(down['timestamp'], down['low'], down['high'], color='red', linewidth=1, zorder=3)

    y_max, y_min = df_2d['high'].max() * 1.01, df_2d['low'].min() * 0.99
    ax1.fill_between(df_2d['timestamp'], y_max, y_min, where=(df_2d['position'] == 1), color='green', alpha=0.15, label='State 1: Long', zorder=1)
    ax1.fill_between(df_2d['timestamp'], y_max, y_min, where=(df_2d['position'] == -1), color='red', alpha=0.15, label='State 1: Short', zorder=1)
    ax1.fill_between(df_2d['timestamp'], y_max, y_min, where=(df_2d['state'] == 2), color='gray', alpha=0.10, label='State 2: Flat', zorder=1)

    sl_data, tsl_data = df_2d[df_2d['sl_hit'] == True], df_2d[df_2d['tsl_hit'] == True]
    if not sl_data.empty: ax1.scatter(sl_data['timestamp'], sl_data['exit_price'], marker='X', color='black', s=150, label='Initial SL', zorder=5)
    if not tsl_data.empty: ax1.scatter(tsl_data['timestamp'], tsl_data['exit_price'], marker='o', color='darkorange', s=120, label='Trailing SL', zorder=5)

    ax1.set_ylim(y_min, y_max)
    ax1.set_xlim(df_2d['timestamp'].min() - pd.Timedelta(hours=1), df_2d['timestamp'].max() + pd.Timedelta(hours=1))
    ax1.set_title(f"LAST 48 HOURS PRICE ({SYMBOL})")
    ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3, zorder=0)
    
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(df_2d['timestamp'], df_2d['equity'], color='#2980b9', linewidth=2.5, label='Account Balance (USDT)')
    ax2.fill_between(df_2d['timestamp'], df_2d['equity'], df_2d['equity'].min() * 0.999, color='#2980b9', alpha=0.1)
    ax2.set_title("LAST 48 HOURS RETURNS")
    ax2.set_ylabel('Balance (USDT)'); ax2.legend(loc='upper left'); ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    
    plt.xticks(rotation=45); plt.tight_layout()
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', dpi=100); buf1.seek(0); plt.close()
    img_48h = base64.b64encode(buf1.read()).decode('utf-8')
    
    # === CHART 2: RANDOM TRADE DEEP DIVE (1m Intrabar) ===
    img_trade = ""
    trade_text = f"<p style='color: #7f8c8d;'>No trades occurred with these parameters for {SYMBOL}.</p>"
    
    if selected_trade and len(selected_trade['m1_ts']) > 0:
        plt.figure(figsize=(16, 6))
        
        all_ts = np.concatenate([selected_trade['prev_m1_ts'], selected_trade['m1_ts'][:selected_trade['exit_idx']+1]])
        all_o = np.concatenate([selected_trade['prev_m1_o'], selected_trade['m1_o'][:selected_trade['exit_idx']+1]])
        all_h = np.concatenate([selected_trade['prev_m1_h'], selected_trade['m1_h'][:selected_trade['exit_idx']+1]])
        all_l = np.concatenate([selected_trade['prev_m1_l'], selected_trade['m1_l'][:selected_trade['exit_idx']+1]])
        all_c = np.concatenate([selected_trade['prev_m1_c'], selected_trade['m1_c'][:selected_trade['exit_idx']+1]])
        
        up = all_c >= all_o
        down = all_c < all_o
        width = 0.0004 
        
        plt.bar(all_ts[up], all_c[up] - all_o[up], bottom=all_o[up], color='green', width=width, zorder=3)
        plt.vlines(all_ts[up], all_l[up], all_h[up], color='green', linewidth=1, zorder=3)
        plt.bar(all_ts[down], all_o[down] - all_c[down], bottom=all_c[down], color='red', width=width, zorder=3)
        plt.vlines(all_ts[down], all_l[down], all_h[down], color='red', linewidth=1, zorder=3)
        
        plt.axvspan(selected_trade['prev_m1_ts'][0], selected_trade['trade_ts'], color='gray', alpha=0.15, label='Trigger Hour (Evaluating)')
        trade_color = 'green' if selected_trade['direction'] == 'LONG' else 'red'
        plt.axvspan(selected_trade['trade_ts'], all_ts[-1], color=trade_color, alpha=0.15, label=f"Trade Hour ({selected_trade['direction']})")
        
        plt.hlines(selected_trade['entry_price'], selected_trade['trade_ts'], all_ts[-1], color='black', linestyle='-', linewidth=2, label='Entry Price')
        plt.hlines(selected_trade['act_price'], selected_trade['trade_ts'], all_ts[-1], color='blue', linestyle='--', linewidth=1.5, label='TSL Activation Level')
        
        tsl_times = selected_trade['m1_ts'][:selected_trade['exit_idx']+1]
        tsl_vals = selected_trade['rolling_stops'][:selected_trade['exit_idx']+1]
        plt.step(tsl_times, tsl_vals, where='post', color='darkorange', linewidth=2, label='Dynamic Stop Loss')
        
        exit_marker = 'X' if selected_trade['exit_reason'] == 'Initial SL Hit' else ('o' if 'Trailing' in selected_trade['exit_reason'] else 's')
        exit_color = 'black' if exit_marker == 'X' else 'darkorange'
        plt.scatter(all_ts[-1], selected_trade['exit_price'], color=exit_color, marker=exit_marker, s=200, zorder=5, label=selected_trade['exit_reason'])
        
        plt.title(f"🔍 RANDOM TRADE MICROSCOPIC VIEW (1-Minute Resolution)")
        plt.legend(loc='best'); plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.tight_layout()
        
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', dpi=100); buf2.seek(0); plt.close()
        img_trade = base64.b64encode(buf2.read()).decode('utf-8')
        
        trade_text = f"""
        <div style="text-align: left; background: #eafaf1; padding: 20px; border-left: 5px solid #27ae60; border-radius: 4px; font-size: 16px; line-height: 1.6;">
            <strong>1. THE TRIGGER:</strong> During the hour starting at <code>{selected_trade['trigger_ts']}</code>, the candle closed with a Wick-to-Price ratio of <b>{selected_trade['trigger_w_pct']*100:.2f}%</b> and a Wick-to-Body ratio of <b>{selected_trade['trigger_wb_ratio']*100:.0f}%</b>. Because this exceeded your required parameters (f={selected_trade['f']*100}% or g={selected_trade['g']*100}%), the bot transitioned into <b>State 1</b>.<br><br>
            
            <strong>2. THE DECISION:</strong> The trigger candle closed <b>{selected_trade['prev_color']}</b>. Therefore, at exactly <code>{selected_trade['trade_ts']}</code>, the bot entered a <b>{selected_trade['direction']}</b> trade with 100% of the account at an entry price of <b>${selected_trade['entry_price']:.8f}</b>.<br><br>
            
            <strong>3. THE EXECUTION:</strong> The Initial Stop Loss was placed at <b>${selected_trade['initial_sl']:.8f}</b>, and the bot watched the 1-minute chart waiting for the price to reach <b>${selected_trade['act_price']:.8f}</b> to activate the trailing stop.<br><br>
            
            <strong>4. THE EXIT:</strong> After <b>{selected_trade['exit_idx']} minutes</b> into the hour, the trade was closed at <b>${selected_trade['exit_price']:.8f}</b>. <br>
            <b>Reason:</b> {selected_trade['exit_reason']}. <br>
            <b>Trade PnL:</b> <span style="color:{'green' if selected_trade['pnl_pct'] > 0 else 'red'}; font-weight:bold;">{selected_trade['pnl_pct']*100:.2f}%</span>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Algorithmic Strategy Backtest</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f4f9; color: #333; text-align: center; padding: 20px; }}
            .form-container {{ background: #2c3e50; color: white; border-radius: 8px; padding: 20px; margin: 0 auto 20px; width: 85%; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .form-container input, .form-container select {{ margin: 0 5px; padding: 5px; text-align: center; border-radius: 4px; border: none; font-size: 15px;}}
            .form-container input {{ width: 60px; }}
            .form-container select {{ width: 220px; font-weight: bold; cursor: pointer; }}
            .btn-run {{ padding: 10px 10px; background-color: #27ae60; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 16px; margin: 15px 5px; width: 180px;}}
            .btn-run:hover {{ background-color: #219150; }}
            .btn-shuffle {{ padding: 10px 10px; background-color: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 16px; margin: 15px 5px; width: 180px;}}
            .btn-shuffle:hover {{ background-color: #2980b9; }}
            .btn-opt {{ padding: 10px 10px; background-color: #f39c12; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 16px; margin: 15px 5px; width: 180px;}}
            .btn-opt:hover {{ background-color: #e67e22; }}
            .stats-container {{ background: #fff; border-radius: 8px; padding: 20px; margin: 0 auto 20px; width: 85%; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: flex; justify-content: space-around; }}
            .stat-box {{ margin: 0 10px; }}
            h1, h2 {{ color: #2c3e50; margin-bottom: 5px;}}
            p.subtitle {{ color: #7f8c8d; margin-top: 0; margin-bottom: 20px; }}
            .value {{ font-size: 22px; font-weight: bold; color: #2980b9; margin-top: 5px; }}
            .chart-container {{ margin: 0 auto 40px; width: 85%; }}
            .chart-container img {{ width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}}
            .input-group {{ display: inline-block; margin: 5px 15px; text-align: right;}}
        </style>
    </head>
    <body>
        <h1>Advanced Hourly Reversal Engine</h1>
        <p class="subtitle">1m Intrabar Accuracy | Genetic Optimizer | Trade Deep Dive</p>
        
        <div class="form-container">
            <form method="POST">
                <!-- Cryptocurrency Dropdown -->
                <div style="margin-bottom: 20px; border-bottom: 1px solid #4a6278; padding-bottom: 15px;">
                    <label style="font-size: 18px; font-weight: bold; margin-right: 10px;">Select Asset:</label>
                    <select name="symbol">
                        {options_html}
                    </select>
                </div>

                <div class="input-group"><label>Wick Size > (f): <input type="number" step="0.1" name="f" value="{params['f']}"> %</label></div>
                <div class="input-group"><label>Wick/Body > (g): <input type="number" step="1" name="g" value="{params['g']}"> %</label></div>
                <div class="input-group"><label>Profit Threshold (u): <input type="number" step="0.1" name="u" value="{params['u']}"> %</label></div>
                <br>
                <div class="input-group"><label>Initial SL: <input type="number" step="0.1" name="sl" value="{params['sl']}"> %</label></div>
                <div class="input-group"><label>Trailing Act/Dist (c): <input type="number" step="0.1" name="tsl" value="{params['tsl']}"> %</label></div>
                <br>
                <button type="submit" name="action" value="run" class="btn-run">▶️ Run Backtest</button>
                <button type="submit" name="action" value="shuffle" class="btn-shuffle">🎲 Shuffle Trade</button>
                <button type="submit" name="action" value="optimize" class="btn-opt">⚡ Optimize GA</button>
            </form>
        </div>

        <div class="stats-container">
            <div class="stat-box"><div>Starting Balance</div><div class="value">${STARTING_BALANCE:,.2f}</div></div>
            <div class="stat-box"><div>Final Balance (30D)</div><div class="value">${final_balance:,.2f}</div></div>
            <div class="stat-box"><div>30-Day Net ROI</div><div class="value" style="color:{'#27ae60' if roi >= 0 else '#c0392b'};">{roi:.2f}%</div></div>
            <div class="stat-box"><div>30-Day Sharpe Ratio</div><div class="value">{sharpe:.2f}</div></div>
        </div>
        
        <div class="chart-container">
            <img src="data:image/png;base64,{img_48h}" alt="48 Hour Chart">
        </div>
        
        <hr style="width: 85%; border: 1px solid #ccc; margin: 40px auto;">
        
        <h2>🔍 Random Trade Deep Dive ({SYMBOL})</h2>
        <div style="width: 85%; margin: 0 auto 20px;">
            {trade_text}
        </div>
        {f'<div class="chart-container"><img src="data:image/png;base64,{img_trade}" alt="Random Trade Micro Chart"></div>' if img_trade else ''}
        
    </body>
    </html>
    """
    return html

def execute_run(params):
    df_run = GLOBAL_DF.copy()
    
    sl, tsl, f, g, u = params['sl']/100.0, params['tsl']/100.0, params['f']/100.0, params['g']/100.0, params['u']/100.0
    final_balance, sls, tsls, positions, exit_prices, states, equity, all_trades = run_backtest(df_run, sl, tsl, f, g, u)
    
    df_run['sl_hit'], df_run['tsl_hit'], df_run['position'] = sls, tsls, positions
    df_run['exit_price'], df_run['state'], df_run['equity'] = exit_prices, states, equity
    
    roi = ((final_balance - STARTING_BALANCE) / STARTING_BALANCE) * 100
    returns = np.diff(equity) / equity[:-1]
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(365*24)) if np.std(returns) > 0 else 0
    
    selected_trade = random.choice(all_trades) if all_trades else None
    
    return generate_html_report(df_run, params, final_balance, roi, sharpe, selected_trade)

class BacktestServer(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        default_params = {'f': 1.0, 'g': 150.0, 'u': 0.5, 'sl': 2.0, 'tsl': 2.0}
        self.send_response(200); self.send_header('Content-type', 'text/html'); self.end_headers()
        self.wfile.write(execute_run(default_params).encode('utf-8'))

    def do_POST(self):
        global SYMBOL, GLOBAL_DF
        
        length = int(self.headers['Content-Length'])
        parsed = urllib.parse.parse_qs(self.rfile.read(length).decode('utf-8'))
        action = parsed.get('action', ['run'])[0]
        
        # Check if user selected a new cryptocurrency
        req_symbol = parsed.get('symbol', [SYMBOL])[0]
        if req_symbol != SYMBOL or GLOBAL_DF is None:
            SYMBOL = req_symbol
            GLOBAL_DF = fetch_binance_data_accurate(SYMBOL, DAYS_BACK)
        
        if action == 'optimize':
            params = run_ga_optimization(GLOBAL_DF)
        else:
            params = {
                'f': float(parsed.get('f', ['1.0'])[0]), 'g': float(parsed.get('g', ['150.0'])[0]),
                'u': float(parsed.get('u', ['0.5'])[0]), 'sl': float(parsed.get('sl', ['2.0'])[0]),
                'tsl': float(parsed.get('tsl', ['2.0'])[0])
            }
        
        self.send_response(200); self.send_header('Content-type', 'text/html'); self.end_headers()
        self.wfile.write(execute_run(params).encode('utf-8'))

if __name__ == "__main__":
    # Fetch default asset (BTC) on script startup
    GLOBAL_DF = fetch_binance_data_accurate(SYMBOL, DAYS_BACK)
    
    handler = BacktestServer
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"🌐 Web server running! Open browser to: http://{get_local_ip()}:{PORT}  (or localhost:{PORT})")
        print("Press Ctrl+C to stop.")
        try: httpd.serve_forever()
        except KeyboardInterrupt: print("\nShutting down.")