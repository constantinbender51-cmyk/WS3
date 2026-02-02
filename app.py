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
ORDER_USD_VALUE = 1000 # $1000 per side (Long and Short)
STRATEGY_INTERVAL_HOURS = 4

# --- Paper Trading Engine (Hedge Mode) ---
class PaperTrader:
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        
        # Hedge Mode State
        self.positions = {
            'long': {'size': 0.0, 'entry': 0.0},
            'short': {'size': 0.0, 'entry': 0.0}
        }
        
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.active_orders = [] 
        self.trade_log = deque(maxlen=50)
        self.last_entry_time = None

        # Cost Model: 0.2% Fee + 0.1% Slippage = 0.3% per side
        self.FEE_RATE = 0.002      
        self.SLIPPAGE_RATE = 0.001 
        self.TOTAL_COST_RATE = self.FEE_RATE + self.SLIPPAGE_RATE 

    def check_entry_signal(self, current_price):
        now = datetime.datetime.now()
        
        # Check 4-hour Timer
        if self.last_entry_time is None or (now - self.last_entry_time).total_seconds() >= STRATEGY_INTERVAL_HOURS * 3600:
            self._open_positions(current_price)
            self.last_entry_time = now
            return True
        return False

    def _open_positions(self, mid):
        # Calculate Base Size
        base_qty = ORDER_USD_VALUE / mid 
        
        # 1. Execute Market Entry (Hedge: Long + Short)
        self._execute_trade('long', base_qty, mid, "Entry Long")
        self._execute_trade('short', -base_qty, mid, "Entry Short")

        # 2. Generate Arrays
        # TP: 0.12% to 4% (6 orders)
        tp_pcts = np.linspace(0.0012, 0.04, 6)
        tp_qty = base_qty / 6
        
        # SL: 0.12% to 2% (3 orders)
        sl_pcts = np.linspace(0.0012, 0.02, 3)
        sl_qty = base_qty / 3

        # 3. Place Long Exits (TP = Sell Limit, SL = Sell Stop)
        for pct in tp_pcts:
            price = mid * (1 + pct)
            self.active_orders.append({'scope': 'long', 'side': 'sell', 'type': 'limit', 'price': price, 'size': tp_qty})
            
        for pct in sl_pcts:
            price = mid * (1 - pct)
            self.active_orders.append({'scope': 'long', 'side': 'sell', 'type': 'stop', 'price': price, 'size': sl_qty})

        # 4. Place Short Exits (TP = Buy Limit, SL = Buy Stop)
        for pct in tp_pcts:
            price = mid * (1 - pct)
            self.active_orders.append({'scope': 'short', 'side': 'buy', 'type': 'limit', 'price': price, 'size': tp_qty})
            
        for pct in sl_pcts:
            price = mid * (1 + pct)
            self.active_orders.append({'scope': 'short', 'side': 'buy', 'type': 'stop', 'price': price, 'size': sl_qty})

    def process_tick(self, bid, ask):
        filled_indices = []

        for i, order in enumerate(self.active_orders):
            executed = False
            scope = order['scope']
            # Safety check: if position closed, cancel remaining orders for that scope (optional, but good practice)
            # Keeping it simple: Allow orders to execute if position exists
            
            curr_pos_size = self.positions[scope]['size']
            if abs(curr_pos_size) < 1e-9:
                filled_indices.append(i) # Cancel orphaned orders
                continue

            # Limit Buy (Short TP)
            if order['type'] == 'limit' and order['side'] == 'buy':
                if ask <= order['price']:
                    qty = min(order['size'], abs(curr_pos_size)) # Cap at remaining pos
                    if qty > 1e-9:
                        self._execute_trade(scope, qty, order['price'], "Short TP")
                        executed = True
            
            # Limit Sell (Long TP)
            elif order['type'] == 'limit' and order['side'] == 'sell':
                if bid >= order['price']:
                    qty = min(order['size'], abs(curr_pos_size))
                    if qty > 1e-9:
                        self._execute_trade(scope, -qty, order['price'], "Long TP")
                        executed = True
            
            # Stop Buy (Short SL)
            elif order['type'] == 'stop' and order['side'] == 'buy':
                if ask >= order['price']:
                    qty = min(order['size'], abs(curr_pos_size))
                    if qty > 1e-9:
                        self._execute_trade(scope, qty, order['price'], "Short SL")
                        executed = True 
            
            # Stop Sell (Long SL)
            elif order['type'] == 'stop' and order['side'] == 'sell':
                if bid <= order['price']:
                    qty = min(order['size'], abs(curr_pos_size))
                    if qty > 1e-9:
                        self._execute_trade(scope, -qty, order['price'], "Long SL")
                        executed = True

            if executed:
                filled_indices.append(i)
        
        for i in sorted(filled_indices, reverse=True):
            del self.active_orders[i]

    def _execute_trade(self, scope, size, price, reason):
        # 1. Transaction Cost
        trade_value = abs(size * price) 
        cost = trade_value * self.TOTAL_COST_RATE 
        self.fees_paid += cost
        self.balance -= cost 

        pos_data = self.positions[scope]
        old_size = pos_data['size']
        old_entry = pos_data['entry']

        # 2. Logic for Hedge Buckets
        # If increasing position (Long buy or Short sell)
        if (scope == 'long' and size > 0) or (scope == 'short' and size < 0):
            new_size = old_size + size
            # Update weighted average entry
            if abs(new_size) > 1e-9:
                total_cost = (old_size * old_entry) + (size * price)
                pos_data['entry'] = total_cost / new_size
            pos_data['size'] = new_size
            
        # If decreasing position (Long sell or Short buy) -> Realize PnL
        else:
            closing_qty = min(abs(size), abs(old_size))
            pnl = (price - old_entry) * closing_qty
            
            # Inverse PnL for shorts
            if scope == 'short': 
                pnl = -pnl # (Entry - Exit) * Qty, but here formulation is (Exit - Entry) * Qty
                # Short: Entry 50k, Exit 40k. Price - Entry = -10k. Qty (pos) is negative? 
                # Let's standardize: 
                # Long: (Exit - Entry) * Qty
                # Short: (Entry - Exit) * Qty
            
            self.realized_pnl += pnl
            self.balance += pnl
            
            pos_data['size'] += size # size is negative for sell, positive for buy
            if abs(pos_data['size']) < 1e-9:
                pos_data['size'] = 0.0
                pos_data['entry'] = 0.0

        self.trade_log.append(f"[{scope.upper()}] {reason} | {size:+.4f} @ {price:.2f} | PnL: {self.realized_pnl:.2f}")

    def get_stats(self, current_price):
        # Calculate Unrealized PnL for both buckets
        unrealized_long = 0.0
        unrealized_short = 0.0
        
        if abs(self.positions['long']['size']) > 1e-9:
            unrealized_long = (current_price - self.positions['long']['entry']) * self.positions['long']['size']
            
        if abs(self.positions['short']['size']) > 1e-9:
            # Short PnL: (Entry - Current) * Abs(Size) 
            # OR: (Current - Entry) * NegativeSize
            unrealized_short = (current_price - self.positions['short']['entry']) * self.positions['short']['size']

        total_unrealized = unrealized_long + unrealized_short
        net_position = self.positions['long']['size'] + self.positions['short']['size'] # Should be near 0 if perfectly hedged initially

        return {
            'balance': self.balance,
            'position': net_position, # Net exposure
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
        
        # Color code by Scope
        for o in active_orders:
            color = 'cyan' if o['scope'] == 'long' else 'orange'
            symbol = 'triangle-up' if o['side'] == 'buy' else 'triangle-down'
            if o['type'] == 'stop':
                symbol = 'x'
                color = 'magenta'
            
            fig.add_trace(go.Scatter(
                x=[o['price']], 
                y=[y_level], 
                mode='markers', 
                marker=dict(symbol=symbol, size=10, color=color), 
                name=f"{o['scope']} {o['type']}",
                showlegend=False
            ))

    layout_args = dict(title=title, xaxis_title="Price", yaxis_title="Vol", template="plotly_dark", height=400, margin=dict(l=40, r=40, t=40, b=40))
    if log_scale: layout_args['xaxis_type'] = "log"
    fig.update_layout(**layout_args)
    return fig

app.layout = html.Div([
    html.H2(f"Kraken: {SYMBOL} + Hedge Bot (4H)", style={'textAlign': 'center', 'color': '#eee', 'fontFamily': 'sans-serif'}),
    
    # --- Paper Trading Account Panel ---
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

    # --- Metrics ---
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

    # 1. Calc Ratio (Kept for visualization, disconnected from logic)
    b_sub = bids[bids['price'] >= mid * 0.98]
    a_sub = asks[asks['price'] <= mid * 1.02]
    
    # 10% Depth
    b_10 = bids[bids['price'] >= mid * 0.90]
    a_10 = asks[asks['price'] <= mid * 1.10]
    
    vb, va = b_sub['size'].sum(), a_sub['size'].sum()
    ratio = 0 if va == 0 else 1 - (vb / va)
    
    ratio_history.append(ratio)
    avg_60m = statistics.mean(ratio_history) if ratio_history else 0

    # 2. Strategy Logic (4H Interval)
    entered = trader.check_entry_signal(mid)
    
    # Processing
    trader.process_tick(best_bid, best_ask)
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
