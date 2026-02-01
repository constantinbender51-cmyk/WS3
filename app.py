import requests
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from collections import deque
import statistics
import numpy as np

# --- Configuration ---
SYMBOL = "PF_XBTUSD"
URL = "https://futures.kraken.com/derivatives/api/v3/orderbook"
PORT = 8080
UPDATE_INTERVAL_MS = 5000  # 5 seconds
MAX_HISTORY = 720  # 60 minutes
POSITION_SIZE_USD = 10000  # Target position size in USD

# --- Paper Trading Engine ---
class PaperTrader:
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.position = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.trade_log = deque(maxlen=50)
        self.reference_price = 0.0

        # Cost Model: 0.05% Taker Fee (Market Order) + 0.01% Slippage
        self.FEE_RATE = 0.0005      
        self.SLIPPAGE_RATE = 0.0001 
        self.TOTAL_COST_RATE = self.FEE_RATE + self.SLIPPAGE_RATE 

    def rebalance_position(self, direction, current_price):
        """
        Ensures position matches the target direction.
        Only trades if the current position is neutral or opposing the signal.
        """
        self.reference_price = current_price
        
        # Determine Target Quantity (BTC)
        # If price is 50k, target is 0.2 BTC
        target_qty = POSITION_SIZE_USD / current_price
        
        if direction == 'short':
            target_qty = -target_qty
        
        trade_needed = 0.0
        
        # Logic: Only trade if we are not already in the correct direction
        # This prevents fee churn from micro-adjustments
        if direction == 'long':
            if self.position <= 0: # We are Short or Neutral, need to go Long
                trade_needed = target_qty - self.position
        elif direction == 'short':
            if self.position >= 0: # We are Long or Neutral, need to go Short
                trade_needed = target_qty - self.position

        if abs(trade_needed) > 1e-9:
            self._execute_trade(trade_needed, current_price, f"Flip to {direction.upper()}")

    def _execute_trade(self, size, price, reason):
        # 1. Calculate Transaction Cost
        trade_value = abs(size * price) 
        cost = trade_value * self.TOTAL_COST_RATE 
        self.fees_paid += cost
        self.balance -= cost 

        # 2. PnL Calculation
        # If reducing position (closing) or flipping
        if (self.position > 1e-9 and size < 0) or (self.position < -1e-9 and size > 0):
            closing_qty = min(abs(size), abs(self.position))
            pnl = (price - self.avg_entry_price) * closing_qty
            if self.position < 0: pnl = -pnl # Short PnL logic
            
            self.realized_pnl += pnl
            self.balance += pnl
            
        # 3. Update Position
        new_pos = self.position + size
        
        if abs(new_pos) < 1e-9: 
            self.position = 0.0
            self.avg_entry_price = 0.0
        elif (self.position >= 0 and size > 0) or (self.position <= 0 and size < 0):
            # Increasing position (averaging in) - though with this logic we usually jump straight to target
            total_cost = (self.position * self.avg_entry_price) + (size * price)
            self.avg_entry_price = total_cost / new_pos
            self.position = new_pos
        else:
            # Crossing 0 (Flip)
            self.position = new_pos
            self.avg_entry_price = price # Entry price resets at the flip

        self.trade_log.append(f"{reason} | {size:+.4f} @ {price:.2f} | Fee: ${cost:.2f}")

    def get_stats(self):
        unrealized = 0.0
        if self.position != 0:
            unrealized = (self.reference_price - self.avg_entry_price) * self.position
        
        return {
            'balance': self.balance,
            'position': self.position,
            'realized': self.realized_pnl,
            'unrealized': unrealized,
            'fees': self.fees_paid,
            'total_equity': self.balance + unrealized
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

def build_figure(bids, asks, title, mid):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=bids['price'], y=bids['cumulative'], fill='tozeroy', name='Bids', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=asks['price'], y=asks['cumulative'], fill='tozeroy', name='Asks', line=dict(color='red')))

    # Add vertical line for mid price
    fig.add_vline(x=mid, line_width=1, line_dash="dash", line_color="white")

    fig.update_layout(
        title=title, 
        xaxis_title="Price", 
        yaxis_title="Vol", 
        template="plotly_dark", 
        height=500, 
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

app.layout = html.Div([
    html.H2(f"Kraken: {SYMBOL} | Directional Strategy", style={'textAlign': 'center', 'color': '#eee', 'fontFamily': 'sans-serif'}),
    
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
            html.H4("Position", style={'margin': '0', 'color': '#aaa'}),
            html.H2(id='pos-display', style={'margin': '5px'})
        ], style={'flex': 1, 'textAlign': 'center'}),
        html.Div([
            html.H4("Signal", style={'margin': '0', 'color': '#aaa'}),
            html.H2(id='status-display', style={'margin': '5px'})
        ], style={'flex': 1, 'textAlign': 'center', 'borderLeft': '1px solid #444'}),
    ], style={'display': 'flex', 'backgroundColor': '#222', 'marginBottom': '20px', 'padding': '10px', 'borderRadius': '8px'}),

    # --- Metrics ---
    html.Div([
        html.Div([html.H5("Current 10% Ratio"), html.H3(id='ratio-val')], style={'display': 'inline-block', 'width': '45%', 'textAlign': 'center', 'color': '#ccc'}),
        html.Div([html.H5("60m Avg Ratio"), html.H3(id='avg-val')], style={'display': 'inline-block', 'width': '45%', 'textAlign': 'center', 'color': '#ccc'})
    ], style={'textAlign': 'center'}),

    dcc.Graph(id='ten-percent-chart'),
    dcc.Interval(id='timer', interval=UPDATE_INTERVAL_MS, n_intervals=0)
], style={'backgroundColor': '#111', 'padding': '20px', 'minHeight': '100vh', 'fontFamily': 'sans-serif'})

@app.callback(
    [Output('ten-percent-chart', 'figure'),
     Output('ratio-val', 'children'), Output('avg-val', 'children'),
     Output('equity-display', 'children'), Output('pnl-display', 'children'),
     Output('pnl-display', 'style'), Output('fees-display', 'children'),
     Output('pos-display', 'children'), Output('pos-display', 'style'),
     Output('status-display', 'children'), Output('status-display', 'style')],
    [Input('timer', 'n_intervals')]
)
def update(n):
    ob = fetch_order_book()
    if not ob: return go.Figure(), "-", "-", "-", "-", {}, "-", "-", {}, "OFFLINE", {'color': 'red'}

    bids, asks = process_data(ob)
    best_bid = bids['price'].iloc[0]
    best_ask = asks['price'].iloc[0]
    mid = (best_bid + best_ask) / 2

    # 1. Calc 10% Ratio
    b_10 = bids[bids['price'] >= mid * 0.90]
    a_10 = asks[asks['price'] <= mid * 1.10]
    
    vb, va = b_10['size'].sum(), a_10['size'].sum()
    ratio = 0 if va == 0 else 1 - (vb / va)
    
    ratio_history.append(ratio)
    avg_60m = statistics.mean(ratio_history) if ratio_history else 0

    # 2. Strategy Logic: Simple Directional Flip
    direction = 'neutral'
    status_txt = "NEUTRAL"
    status_col = {'color': '#999'}

    if avg_60m < 0:
        direction = 'short'
        status_txt = "SHORT"
        status_col = {'color': '#FF4136'}
    elif avg_60m > 0:
        direction = 'long'
        status_txt = "LONG"
        status_col = {'color': '#2ECC40'}

    # 3. Execute Flip
    if direction != 'neutral':
        trader.rebalance_position(direction, mid)

    stats = trader.get_stats()

    pnl_col = {'color': '#2ECC40'} if stats['unrealized'] >= 0 else {'color': '#FF4136'}
    pos_col = {'color': '#2ECC40'} if stats['position'] > 0 else ({'color': '#FF4136'} if stats['position'] < 0 else {'color': '#ccc'})
    
    fig = build_figure(b_10, a_10, f"Depth ±10% ({mid:.1f})", mid)

    return (fig, f"{ratio:.4f}", f"{avg_60m:.4f}", 
            f"${stats['total_equity']:,.2f}", f"${stats['unrealized']:,.2f}", pnl_col,
            f"${stats['fees']:,.2f}", f"{stats['position']:.4f}", pos_col, status_txt, status_col)

if __name__ == '__main__':
    app.run(debug=False, port=PORT, host='0.0.0.0')
