import sys
import time
import json
import threading
import logging
from typing import Dict, List, Optional

# Third-party libraries
import websocket # pip install websocket-client
from binance import ThreadedWebsocketManager, ThreadedDepthCacheManager # pip install python-binance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class KrakenAutoClient:
    """
    Connects to Kraken WebSockets automatically on instantiation.
    Maintains complexity: Raw WebSocket implementation without wrapper libraries.
    """
    WS_URL = 'wss://ws.kraken.com/'

    def __init__(self, pairs: List[str]):
        self.logger = logging.getLogger("KrakenClient")
        self.pairs = pairs
        self.ws: Optional[websocket.WebSocketApp] = None
        self.wst: Optional[threading.Thread] = None
        
        # Automatic startup mechanism
        self._initiate_connection()

    def _initiate_connection(self):
        """Internal method to start the WS connection immediately."""
        self.logger.info("Initializing Kraken WebSocket connection...")
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            self.WS_URL,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        # Daemon thread to ensure it doesn't block program exit if needed
        self.wst = threading.Thread(target=self.ws.run_forever)
        self.wst.daemon = True
        self.wst.start()
        self.logger.info("Kraken background thread started.")

    def _on_open(self, ws):
        self.logger.info("Kraken Connection Opened. Auto-subscribing...")
        
        # Subscribe to Ticker (Tick Data)
        ticker_payload = {
            "event": "subscribe",
            "pair": self.pairs,
            "subscription": {"name": "ticker"}
        }
        ws.send(json.dumps(ticker_payload))
        self.logger.info(f"Subscribed to Ticker: {self.pairs}")

        # Subscribe to Order Book (Depth) - maintaining full precision
        book_payload = {
            "event": "subscribe",
            "pair": self.pairs,
            "subscription": {"name": "book", "depth": 100}
        }
        ws.send(json.dumps(book_payload))
        self.logger.info(f"Subscribed to OrderBook (100 depth): {self.pairs}")

    def _on_message(self, ws, message):
        """
        Complex handling of raw JSON messages.
        Differentiates between heartbeats, system status, and data channels.
        """
        try:
            data = json.loads(message)
            
            # Filter heartbeats
            if isinstance(data, dict) and data.get("event") == "heartbeat":
                return 

            # Handle System Events
            if isinstance(data, dict) and "event" in data:
                self.logger.debug(f"System Event: {data}")
                return

            # Handle Channel Data (List format in Kraken V1)
            # Format: [channelID, {data}, channelName, pair]
            if isinstance(data, list):
                channel_name = data[-2]
                pair = data[-1]
                
                if channel_name == "book-100":
                    self._process_orderbook(data, pair)
                elif channel_name == "ticker":
                    self._process_tick(data, pair)

        except Exception as e:
            self.logger.error(f"Message processing error: {e}")

    def _process_orderbook(self, data, pair):
        """
        Kraken sends snapshots (as/bs) initially, then updates (a/b).
        Complexity: Real implementation requires merging these updates into a local map.
        """
        payload = data[1]
        if "as" in payload or "bs" in payload:
            self.logger.info(f"[Kraken][{pair}] OrderBook SNAPSHOT received.")
            # In a full impl, you would store this snapshot
        elif "a" in payload or "b" in payload:
            # Differential update
            # self.logger.info(f"[Kraken][{pair}] OrderBook UPDATE received.")
            pass

    def _process_tick(self, data, pair):
        """
        Process tick data (c: close price/lot volume).
        """
        # Ticker format: [channelID, {c: [price, vol], ...}, 'ticker', pair]
        payload = data[1]
        if 'c' in payload:
            price = payload['c'][0]
            vol = payload['c'][1]
            self.logger.info(f"[Kraken][{pair}] TICK: {price} (Vol: {vol})")

    def _on_error(self, ws, error):
        self.logger.error(f"Kraken Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        self.logger.warning("Kraken Connection Closed.")


class BinanceAutoClient:
    """
    Uses python-binance Managers. 
    Complexity: Uses ThreadedDepthCacheManager for accurate local order book replication
    and ThreadedWebsocketManager for raw tick streams.
    """
    def __init__(self, symbol: str):
        self.logger = logging.getLogger("BinanceClient")
        self.symbol = symbol.upper()
        
        # Initialize Managers
        self.twm = ThreadedWebsocketManager()
        self.dcm = ThreadedDepthCacheManager()
        
        # Automatic startup
        self._initiate_streams()

    def _initiate_streams(self):
        self.logger.info("Initializing Binance Stream Managers...")
        
        # Start the internal event loops
        self.twm.start()
        self.dcm.start()

        # 1. Start Trade Stream (Tick Data)
        self.logger.info(f"Subscribing to Binance Trade Stream: {self.symbol}")
        self.twm.start_trade_socket(
            callback=self._handle_trade_message,
            symbol=self.symbol
        )

        # 2. Start Depth Cache (Order Book)
        # This handles the complexity of fetching a REST snapshot and applying WS updates
        self.logger.info(f"Subscribing to Binance Depth Cache: {self.symbol}")
        self.dcm.start_depth_cache(
            callback=self._handle_depth_message,
            symbol=self.symbol,
            refresh_interval=None # Set to None to maintain connection indefinitely
        )

    def _handle_trade_message(self, msg):
        """
        Handles raw trade events.
        msg['e'] = event type, msg['p'] = price, msg['q'] = quantity
        """
        if msg['e'] == 'error':
            self.logger.error(f"Binance Trade Error: {msg}")
        else:
            price = msg['p']
            qty = msg['q']
            # self.logger.info(f"[Binance][{self.symbol}] TRADE: {price} (Qty: {qty})")

    def _handle_depth_message(self, depth_cache):
        """
        Receives a DepthCache object which is automatically kept in sync.
        This hides the complexity of U/u ids but ensures data integrity.
        """
        if depth_cache is not None:
            best_bid = depth_cache.get_bids()[0]
            best_ask = depth_cache.get_asks()[0]
            self.logger.info(f"[Binance][{self.symbol}] BOOK: Bid {best_bid[0]} | Ask {best_ask[0]}")
        else:
            self.logger.warning("Binance Depth Cache update failed")

    def stop(self):
        self.twm.stop()
        self.dcm.stop()

# --- Execution Block ---
# Automatically runs on script execution or import
try:
    # 1. Initialize Kraken (XBT/USD)
    kraken_pairs = ["XBT/USD"]
    kraken_client = KrakenAutoClient(kraken_pairs)

    # 2. Initialize Binance (BTCUSDT)
    binance_symbol = "BTCUSDT"
    binance_client = BinanceAutoClient(binance_symbol)

    print("----------------------------------------------------------------")
    print(">> DATA STREAMS INITIATED <<")
    print("Both exchanges are now streaming data in background threads.")
    print("Press Ctrl+C to stop.")
    print("----------------------------------------------------------------")

    # Keep main thread alive to allow background threads to work
    if __name__ == "__main__":
        while True:
            time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping clients...")
    binance_client.stop()
    # Kraken thread is daemon, will die with main
    print("Shutdown complete.")
