import requests
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from collections import deque
import statistics
import numpy as np
import datetime

# --- Configuration ---
SYMBOL = "PF_XBTUSD"
URL = "https://futures.kraken.com/derivatives/api/v3/orderbook"
PORT = 8080
UPDATE_INTERVAL_MS = 5000  # 5 seconds
MAX_HISTORY = 720  # 60 minutes
ORDER_USD_VALUE = 1000 
STRATEGY_INTERVAL_HOURS = 1  # <--- CHANGED TO 1 HOUR

# --- Paper Trading Engine (Hedge Mode) ---
class PaperTrader:
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.positions = {
            'long': {'size': 0.0, 'entry': 0.0},
            'short': {'size': 0.0, 'entry': 0.0}
        }
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.active_orders = [] 
        self.trade_log = deque(maxlen=50)
        self.last_entry_time = None

        self.FEE_RATE = 0.002      
        self.SLIPPAGE_RATE = 0.001 
        self.TOTAL_COST_RATE = self.FEE_RATE + self.SLIPPAGE_RATE 

    def check_entry_signal(self, current_price):
        now = datetime.datetime.now()
        # Check 1-hour Timer
        if self.last_entry_time is None or (now - self.last_entry_time).total_seconds() >= STRATEGY_INTERVAL_HOURS * 3600:
            self._open_positions(current_price)
            self.last_entry_time = now
            return True
        return False

    def _open_positions(self, mid):
        base_qty = ORDER_USD_VALUE / mid 
        
        # 1. Market Entries
        self._execute_trade('long', base_qty, mid, "Entry Long")
        self._execute_trade('short', -base_qty, mid, "Entry Short")

        # 2. Hard Take Profit (2% - Single Order)
        # Long TP (Sell Limit @ +2%)
        self.active_orders.append({'scope': 'long', 'side': 'sell', 'type': 'limit', 'price': mid * 1.02, 'size': base_qty})
        
        # Short TP (Buy Limit @ -2%)
        self.active_orders.append({'scope': 'short', 'side': 'buy', 'type': 'limit', 'price': mid * 0.98, 'size': base_qty})

        # 3. Hard Stop Loss (6.5% - Single Order)
        # Long Stop (Sell Stop @ -6.5%)
        self.active_orders.append({'scope': 'long', 'side': 'sell', 'type': 'stop', 'price': mid * 0.935, 'size': base_qty})
        
        # Short Stop (Buy Stop @ +6.5%)
        self.active_orders.append({'scope': 'short', 'side': 'buy', 'type': 'stop', 'price': mid * 1.065, 'size': base_qty})

    def close_all(self, bid, ask, reason):
        """Immediate Market Close for all positions"""
        # Close Long (Sell at Bid)
        if self.positions['long']['size'] > 1e-9:
            self._execute_trade('long', -self.positions['long']['size'], bid, reason)
        
        # Close Short (Buy at Ask)
        if abs(self.positions['short']['size']) > 1e-9:
            self._execute_trade('short', abs(self.positions['short']['size']), ask, reason)
            
        self.active_orders = []

    def process_tick(self, bid, ask, avg_60m):
        # 1. Check Signal Stop (Ratio > 0.1 ABS)
        if abs(avg_60m) > 0.1:
            if abs(self.positions['long']['size']) > 1e-9 or abs(self.positions['short']['size']) > 1e-9:
                self.close_all(bid, ask, f"Ratio Stop ({avg_60m:.2f})")
            return

        # 2. Process Pending Orders
        filled_indices = []
        for i, order in enumerate(self.active_orders):
            executed = False
            scope = order['scope']
            curr_pos_size = self.positions[scope]['size']
            
            # Orphan check
            if abs(curr_pos_size) < 1e-9:
                filled_indices.append(i)
                continue

            # Limit Buy (Short TP)
            if order['type'] == 'limit' and order['side'] == 'buy':
                if ask <= order['price']:
                    qty = min(order['size'], abs(curr_pos_size))
                    if qty > 1e-9:
                        self._execute_trade(scope, qty, order['price'], "Short TP 2%")
                        executed = True
            
            # Limit Sell (Long TP)
            elif order['type'] == 'limit' and order['side'] == 'sell':
                if bid >= order['price']:
                    qty = min(order['size'], abs(curr_pos_size))
                    if qty > 1e-9:
                        self._execute_trade(scope, -qty, order['price'], "Long TP 2%")
                        executed = True
            
            # Stop Buy (Short Hard Stop)
            elif order['type'] == 'stop' and order['side'] == 'buy':
                if ask >= order['price']:
                    qty = abs(curr_pos_size)
                    if qty > 1e-9:
                        self._execute_trade(scope, qty, order['price'], "Short Stop 6.5%")
                        executed = True 
            
            # Stop Sell (Long Hard Stop)
            elif order['type'] == 'stop' and order['side'] == 'sell':
                if bid <= order['price']:
                    qty = abs(curr_pos_size)
                    if qty > 1e-9:
                        self._execute_trade(scope, -qty, order['price'], "Long Stop 6.5%")
                        executed = True

            if executed:
                filled_indices.append(i)
        
        for i in sorted(filled_indices, reverse=True):
            del self.active_orders[i]

    def _execute_trade(self, scope, size, price, reason):
        trade_value = abs(size * price) 
        cost = trade_value * self.TOTAL_COST_RATE 
        self.fees_paid += cost
        self.balance -= cost 

        pos_data = self.positions[scope]
        old_size = pos_data['size']
        old_entry = pos_data['entry']

        # Increase Position
        if (scope == 'long' and size > 0) or (scope == 'short' and size < 0):
            new_size = old_size + size
            if abs(new_size) > 1e-9:
                total_cost = (old_size * old_entry) + (size * price)
                pos_data['entry'] = total_cost / new_size
            pos_data['size'] = new_size
            
        # Decrease Position (Realize PnL)
        else:
            closing_qty = min(abs(size), abs(old_size))
            pnl = (price - old_entry) * closing_qty
            if scope == 'short': pnl = -pnl 
            
            self.realized_pnl += pnl
            self.balance += pnl
            pos_data['size'] += size 
            
            if abs(pos_data['size']) < 1e-9:
                pos_data['size'] = 0.0
                pos_data['entry'] = 0.0

        self.trade_log.append(f"[{scope.upper()}] {reason} | {size:+.4f} @ {price:.2f}")

    def get_stats(self, current_price):
        unrealized_long = 0.0
        unrealized_short = 0.0
        
        if abs(self.positions['long']['size']) > 1e-9:
            unrealized_long = (current_price - self.positions['long']['entry']) * self.positions['long']['size']
            
        if abs(self.positions['short']['size']) > 1e-9:
            unrealized_short = (current_price - self.positions['short']['entry']) * self.positions['short']['size']

        total_unrealized = unrealized_long + unrealized_short
        return {
            'balance': self.balance,
            'long_pos': self.positions['long']['size'],
            'short_pos': self.positions['short']['size'],
            'realized': self.realized_pnl,
            'unrealized': total_unrealized,
            'fees': self.fees_paid,
            'total_equity': self.balance + total_unrealized
        }

# --- Global State ---
ratio_history = deque(maxlen=MAX_HISTORY)
trader = PaperTrader()

app = Dash(__name__)

def fetch_order_book():
    try:
        response = requests.get(URL, params={'symbol': SYMBOL}, timeout=3)
        response.raise_for_status()
        data = response.json()
        if data.get('result') == 'success':
            return data.get('orderBook', {})
        return None
    except:
        return None

def process_data(order_book):
    bids = pd.DataFrame(order_book.get('bids', []), columns=['price', 'size']).astype(float)
    bids = bids.sort_values(by='price', ascending=False)
    bids['cumulative'] = bids['size'].cumsum()

    asks = pd.DataFrame(order_book.get('asks', []), columns=['price', 'size']).astype(float)
    asks = asks.sort_values(by='price', ascending=True)
    asks['cumulative'] = asks['size'].cumsum()
    return bids, asks

def build_figure(bids, asks, title, log_scale=False, active_orders=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bids['price'], y=bids['cumulative'], fill='tozeroy', name='Bids', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=asks['price'], y=asks['cumulative'], fill='tozeroy', name='Asks', line=dict(color='red')))

    if active_orders:
        y_max = max(bids['cumulative'].max(), asks['cumulative'].max())
        y_level = y_max * 0.5
        
        for o in active_orders:
            color = 'cyan' if o['scope'] == 'long' else 'orange'
            symbol = 'triangle-up' if o['side'] == 'buy' else 'triangle-down'
            if o['type'] == 'stop':
                symbol = 'x'
                color = 'magenta'
            
            fig.add_trace(go.Scatter(
                x=[o['price']], y=[y_level], mode='markers', 
                marker=dict(symbol=symbol, size=10, color=color), 
                name=f"{o['scope']} {o['type']}", showlegend=False
            ))

    layout_args = dict(title=title, xaxis_title="Price", yaxis_title="Vol", template="plotly_dark", height=400, margin=dict(l=40, r=40, t=40, b=40))
    if log_scale: layout_args['xaxis_type'] = "log"
    fig.update_layout(**layout_args)
    return fig

app.layout = html.Div([
    html.H2(f"Kraken: {SYMBOL} + Hedge Bot (1H | TP: 2% | SL: 6.5%)", style={'textAlign': 'center', 'color': '#eee', 'fontFamily': 'sans-serif'}),
    
    html.Div([
        html.Div([
            html.H4("Equity", style={'margin': '0', 'color': '#aaa'}),
            html.H2(id='equity-display', style={'margin': '5px', 'color': '#fff'})
        ], style={'flex': 1, 'textAlign': 'center', 'borderRight': '1px solid #444'}),
        html.Div([
            html.H4("Unrealized PnL", style={'margin': '0', 'color': '#aaa'}),
            html.H2(id='pnl-display', style={'margin': '5px'})
        ], style={'flex': 1, 'textAlign': 'center', 'borderRight': '1px solid #444'}),
        html.Div([
            html.H4("Fees Paid", style={'margin': '0', 'color': '#aaa'}),
            html.H2(id='fees-display', style={'margin': '5px', 'color': '#FF851B'})
        ], style={'flex': 1, 'textAlign': 'center', 'borderRight': '1px solid #444'}),
        html.Div([
            html.H4("L/S Exposure", style={'margin': '0', 'color': '#aaa'}),
            html.H3(id='pos-display', style={'margin': '5px'})
        ], style={'flex': 1, 'textAlign': 'center'}),
        html.Div([
            html.H4("Last Entry", style={'margin': '0', 'color': '#aaa'}),
            html.H3(id='status-display', style={'margin': '5px'})
        ], style={'flex': 1, 'textAlign': 'center', 'borderLeft': '1px solid #444'}),
    ], style={'display': 'flex', 'backgroundColor': '#222', 'marginBottom': '20px', 'padding': '10px', 'borderRadius': '8px'}),

    html.Div([
        html.Div([html.H5("Current Ratio"), html.H3(id='ratio-val')], style={'display': 'inline-block', 'width': '45%', 'textAlign': 'center', 'color': '#ccc'}),
        html.Div([html.H5("60m Avg Ratio"), html.H3(id='avg-val')], style={'display': 'inline-block', 'width': '45%', 'textAlign': 'center', 'color': '#ccc'})
    ], style={'textAlign': 'center'}),

    dcc.Graph(id='focused-chart'),
    dcc.Graph(id='ten-percent-chart'),
    dcc.Graph(id='full-chart'),
    dcc.Interval(id='timer', interval=UPDATE_INTERVAL_MS, n_intervals=0)
], style={'backgroundColor': '#111', 'padding': '20px', 'minHeight': '100vh', 'fontFamily': 'sans-serif'})

@app.callback(
    [Output('focused-chart', 'figure'), Output('ten-percent-chart', 'figure'), Output('full-chart', 'figure'),
     Output('ratio-val', 'children'), Output('avg-val', 'children'),
     Output('equity-display', 'children'), Output('pnl-display', 'children'),
     Output('pnl-display', 'style'), Output('fees-display', 'children'),
     Output('pos-display', 'children'), Output('pos-display', 'style'),
     Output('status-display', 'children'), Output('status-display', 'style')],
    [Input('timer', 'n_intervals')]
)
def update(n):
    ob = fetch_order_book()
    if not ob: return go.Figure(), go.Figure(), go.Figure(), "-", "-", "-", "-", {}, "-", "-", {}, "OFFLINE", {'color': 'red'}

    bids, asks = process_data(ob)
    best_bid = bids['price'].iloc[0]
    best_ask = asks['price'].iloc[0]
    mid = (best_bid + best_ask) / 2

    # 1. Calc Ratio
    b_sub = bids[bids['price'] >= mid * 0.98]
    a_sub = asks[asks['price'] <= mid * 1.02]
    b_10 = bids[bids['price'] >= mid * 0.90]
    a_10 = asks[asks['price'] <= mid * 1.10]
    
    vb, va = b_sub['size'].sum(), a_sub['size'].sum()
    ratio = 0 if va == 0 else 1 - (vb / va)
    
    ratio_history.append(ratio)
    avg_60m = statistics.mean(ratio_history) if ratio_history else 0

    # 2. Strategy Logic
    trader.check_entry_signal(mid)
    trader.process_tick(best_bid, best_ask, avg_60m)
    
    stats = trader.get_stats(mid)

    pnl_col = {'color': '#2ECC40'} if stats['unrealized'] >= 0 else {'color': '#FF4136'}
    pos_txt = f"L: {stats['long_pos']:.3f} | S: {stats['short_pos']:.3f}"
    pos_col = {'color': '#fff'}
    
    last_time_str = "Waiting..."
    if trader.last_entry_time:
         last_time_str = trader.last_entry_time.strftime("%H:%M:%S")
    status_col = {'color': '#2ECC40'}

    fig1 = build_figure(b_sub, a_sub, f"Active Depth ±2% ({mid:.1f})", False, trader.active_orders)
    fig2 = build_figure(b_10, a_10, f"Depth ±10% ({mid:.1f})", False, trader.active_orders)
    fig3 = build_figure(bids, asks, "Full Book", True, trader.active_orders)

    return (fig1, fig2, fig3, f"{ratio:.4f}", f"{avg_60m:.4f}", 
            f"${stats['total_equity']:,.2f}", f"${stats['unrealized']:,.2f}", pnl_col,
            f"${stats['fees']:,.2f}", pos_txt, pos_col, last_time_str, status_col)

if __name__ == '__main__':
    app.run(debug=False, port=PORT, host='0.0.0.0')
