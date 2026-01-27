import http.server
import socketserver
import json
import threading
import time
import uuid
import requests
import pandas as pd
import io
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from enum import Enum

# ==========================================
# 1. BLOCKING DATA DOWNLOAD (Global Scope)
# ==========================================
DATA_URL = "https://ohlcendpoint.up.railway.app/data/btc1m.csv"

print(f"[*] Initializing module. Starting blocking download from {DATA_URL}...")

try:
    # Blocking request immediately on import
    response = requests.get(DATA_URL)
    response.raise_for_status()
    
    # Parse CSV data into DataFrame
    # Assuming standard OHLC columns; handling potentially dirty data implicitly by pandas
    GLOBAL_OHLC_DATA = pd.read_csv(io.StringIO(response.text))
    print(f"[*] Data download complete. Loaded {len(GLOBAL_OHLC_DATA)} rows.")

except Exception as e:
    # Critical failure in initialization
    print(f"[!] CRITICAL: Failed to download or parse data: {e}")
    GLOBAL_OHLC_DATA = pd.DataFrame()

# ==========================================
# 2. TRADING & ORDER LOGIC
# ==========================================

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"

@dataclass
class Order:
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: float
    status: OrderStatus = OrderStatus.PENDING
    parent_id: Optional[str] = None  # Trace back to the filled order that triggered this (for TP/SL)

@dataclass
class Position:
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    
    @property
    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.side == OrderSide.BUY:
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price

class TradingEngine:
    def __init__(self, data: pd.DataFrame, start_balance: float = 10000.0):
        self.data = data
        self.balance = start_balance
        self.equity = start_balance
        self.orders: List[Order] = []
        self.positions: List[Position] = []
        self.realized_pnl = 0.0
        self.current_price = 0.0
        self.current_index = 0
        
        # Configuration
        self.trade_qty = 0.1  # Fixed quantity for simplicity
        self.limit_offset = 0.005  # 0.5%
        self.sl_offset = 0.005     # 0.5%
        self.tp_offset = 0.01      # 1.0%

    def get_state(self):
        """Returns the current state for the API."""
        return {
            "current_price": self.current_price,
            "balance": self.balance,
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "open_positions": len(self.positions),
            "pending_orders": [asdict(o) for o in self.orders if o.status == OrderStatus.PENDING],
            "active_triggers": {
                "stop_losses": [o.price for o in self.orders if o.order_type == OrderType.STOP_LOSS and o.status == OrderStatus.PENDING],
                "take_profits": [o.price for o in self.orders if o.order_type == OrderType.TAKE_PROFIT and o.status == OrderStatus.PENDING]
            }
        }

    def place_order(self, side: OrderSide, order_type: OrderType, price: float, qty: float, parent_id: str = None):
        order = Order(
            id=str(uuid.uuid4())[:8],
            symbol="BTC",
            side=side,
            order_type=order_type,
            price=price,
            quantity=qty,
            parent_id=parent_id
        )
        self.orders.append(order)
        # print(f"Placed {order_type.name} {side.name} @ {price:.2f}")

    def on_fill(self, filled_order: Order):
        """Handle logic when an order is filled (Entry, TP, or SL)."""
        filled_order.status = OrderStatus.FILLED
        
        # Calculate PnL if closing a position
        if filled_order.order_type in [OrderType.TAKE_PROFIT, OrderType.STOP_LOSS]:
            # Close the position logic (FIFO or matching)
            # For this complex simulation, we simply remove the matching position
            match_side = OrderSide.SELL if filled_order.side == OrderSide.BUY else OrderSide.BUY
            
            # Find a matching position to close
            for pos in self.positions:
                if pos.side == match_side:
                    pnl = (filled_order.price - pos.entry_price) * pos.quantity if pos.side == OrderSide.BUY else (pos.entry_price - filled_order.price) * pos.quantity
                    self.balance += pnl
                    self.realized_pnl += pnl
                    self.positions.remove(pos)
                    print(f"[-] Position Closed via {filled_order.order_type.name}. PnL: {pnl:.2f}")
                    break
                    
        elif filled_order.order_type == OrderType.LIMIT:
            # Entry logic: Create Position
            new_pos = Position(symbol="BTC", side=filled_order.side, entry_price=filled_order.price, quantity=filled_order.quantity)
            self.positions.append(new_pos)
            print(f"[+] Position Opened: {filled_order.side.name} @ {filled_order.price:.2f}")

            # 3. When an order is executed place a 0.5% stop loss and a 1% take profit
            if filled_order.side == OrderSide.BUY:
                sl_price = filled_order.price * (1 - self.sl_offset)
                tp_price = filled_order.price * (1 + self.tp_offset)
                sl_side, tp_side = OrderSide.SELL, OrderSide.SELL
            else:
                sl_price = filled_order.price * (1 + self.sl_offset)
                tp_price = filled_order.price * (1 - self.tp_offset)
                sl_side, tp_side = OrderSide.BUY, OrderSide.BUY

            self.place_order(sl_side, OrderType.STOP_LOSS, sl_price, filled_order.quantity, parent_id=filled_order.id)
            self.place_order(tp_side, OrderType.TAKE_PROFIT, tp_price, filled_order.quantity, parent_id=filled_order.id)

    def process_market_tick(self, high: float, low: float, close: float):
        """Simulate order matching against High/Low of the minute candle."""
        self.current_price = close
        
        # Check trigger conditions for all pending orders
        # We must iterate a copy because we might modify the list
        for order in list(self.orders):
            if order.status != OrderStatus.PENDING:
                continue

            executed = False
            
            # BUY Logic
            if order.side == OrderSide.BUY:
                # Limit Buy: Low <= Price
                if order.order_type == OrderType.LIMIT and low <= order.price:
                    executed = True
                # SL Buy (Short cover): High >= Price
                elif order.order_type == OrderType.STOP_LOSS and high >= order.price:
                    executed = True
                # TP Buy (Short cover): Low <= Price
                elif order.order_type == OrderType.TAKE_PROFIT and low <= order.price:
                    executed = True
            
            # SELL Logic
            elif order.side == OrderSide.SELL:
                # Limit Sell: High >= Price
                if order.order_type == OrderType.LIMIT and high >= order.price:
                    executed = True
                # SL Sell (Long exit): Low <= Price
                elif order.order_type == OrderType.STOP_LOSS and low <= order.price:
                    executed = True
                # TP Sell (Long exit): High >= Price
                elif order.order_type == OrderType.TAKE_PROFIT and high >= order.price:
                    executed = True

            if executed:
                self.on_fill(order)

    def run_strategy_step(self, close_price: float):
        """
        2. Every minute place a buy lmt order 0.5% lower and sell order 0.5% higher
        """
        buy_limit_price = close_price * (1 - self.limit_offset)
        sell_limit_price = close_price * (1 + self.limit_offset)

        self.place_order(OrderSide.BUY, OrderType.LIMIT, buy_limit_price, self.trade_qty)
        self.place_order(OrderSide.SELL, OrderType.LIMIT, sell_limit_price, self.trade_qty)

    def update_equity(self):
        """Mark to market."""
        unrealized = 0.0
        for pos in self.positions:
            if pos.side == OrderSide.BUY:
                unrealized += (self.current_price - pos.entry_price) * pos.quantity
            else:
                unrealized += (pos.entry_price - self.current_price) * pos.quantity
        self.equity = self.balance + unrealized

# Global Engine Instance
ENGINE = TradingEngine(GLOBAL_OHLC_DATA)

# ==========================================
# 3. SERVER IMPLEMENTATION (Port 8080)
# ==========================================

class TradingInfoHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        # Serve JSON state
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        state = ENGINE.get_state()
        
        # Custom JSON encoder for Enums
        class EnumEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Enum):
                    return obj.name
                return super().default(obj)

        self.wfile.write(json.dumps(state, cls=EnumEncoder, indent=2).encode('utf-8'))

    def log_message(self, format, *args):
        return # Silence console noise

def start_server():
    port = 8080
    handler = TradingInfoHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"[*] Serving PnL and Triggers on port {port}")
        httpd.serve_forever()

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================

def simulation_loop():
    """
    Iterates through the downloaded data to simulate the 'Every Minute' logic.
    Includes a sleep to allow the server to be queried in real-time.
    """
    if GLOBAL_OHLC_DATA.empty:
        print("[!] No data to process.")
        return

    print(f"[*] Starting simulation loop over {len(GLOBAL_OHLC_DATA)} candles...")
    
    # Identify column names (handling case sensitivity or variations standard in crypto csvs)
    # Falling back to index-based if specific names aren't found, assuming: time, open, high, low, close...
    cols = GLOBAL_OHLC_DATA.columns.str.lower()
    
    # Helper to safe-get column data
    def get_col(name, idx):
        if name in cols:
            return GLOBAL_OHLC_DATA.iloc[idx][name]
        # Fallback map for common OHLC indices
        map_idx = {'open': 1, 'high': 2, 'low': 3, 'close': 4}
        return GLOBAL_OHLC_DATA.iloc[idx, map_idx.get(name, 4)]

    for i in range(len(GLOBAL_OHLC_DATA)):
        try:
            # Extract candle data
            row = GLOBAL_OHLC_DATA.iloc[i]
            # Assumes columns exist, or uses pandas smart indexing. 
            # We standardize to 'close', 'high', 'low' if possible, otherwise rely on iloc/standard names
            # Adjust these keys based on the actual CSV format from the endpoint
            c = row.get('close') or row.iloc[4] 
            h = row.get('high') or row.iloc[2]
            l = row.get('low') or row.iloc[3]
            
            # 1. Process Order Execution (Mocking the 'during the minute' movement)
            ENGINE.process_market_tick(float(h), float(l), float(c))
            
            # 2. Update Equity
            ENGINE.update_equity()
            
            # 3. Strategy Logic (Every minute place orders)
            ENGINE.run_strategy_step(float(c))
            
            # Simulation delay to allow time for checking port 8080
            # Running at 10x speed (0.1s per minute) to make it observable but not instant
            time.sleep(0.5) 
            
            if i % 10 == 0:
                print(f"Step {i}/{len(GLOBAL_OHLC_DATA)} | Price: {c} | Equity: {ENGINE.equity:.2f}")

        except Exception as e:
            print(f"Error in simulation step {i}: {e}")
            break

if __name__ == "__main__":
    # Start Web Server in separate thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Give server a moment to bind
    time.sleep(1)

    # Start Simulation
    simulation_loop()
