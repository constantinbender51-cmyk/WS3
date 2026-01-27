import sys
import time
import json
import threading
import asyncio
import logging
from typing import List, Optional

# Third-party libraries
import websocket  # pip install websocket-client
from binance import AsyncClient, BinanceSocketManager
from binance.depth import AsyncDepthCacheManager

# Configure logging
# Reduced logging level for 'binance' to prevent I/O blocking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("binance").setLevel(logging.WARNING) 
logging.getLogger("websockets").setLevel(logging.WARNING)

class KrakenAutoClient:
    """
    Connects to Kraken WebSockets automatically on instantiation.
    Uses raw WebSocket implementation.
    """
    WS_URL = 'wss://ws.kraken.com/'

    def __init__(self, pairs: List[str]):
        self.logger = logging.getLogger("KrakenClient")
        self.pairs = pairs
        self.ws: Optional[websocket.WebSocketApp] = None
        self.wst: Optional[threading.Thread] = None
        
        self._initiate_connection()

    def _initiate_connection(self):
        self.logger.info("Initializing Kraken WebSocket connection...")
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            self.WS_URL,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        self.wst = threading.Thread(target=self.ws.run_forever)
        self.wst.daemon = True
        self.wst.start()
        self.logger.info("Kraken background thread started.")

    def _on_open(self, ws):
        self.logger.info("Kraken Connection Opened. Auto-subscribing...")
        
        # Subscribe to Ticker
        ticker_payload = {
            "event": "subscribe",
            "pair": self.pairs,
            "subscription": {"name": "ticker"}
        }
        ws.send(json.dumps(ticker_payload))

        # Subscribe to Order Book
        book_payload = {
            "event": "subscribe",
            "pair": self.pairs,
            "subscription": {"name": "book", "depth": 100}
        }
        ws.send(json.dumps(book_payload))
        self.logger.info(f"Subscribed to Kraken streams for {self.pairs}")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            
            # Filter heartbeats and status events
            if isinstance(data, dict): 
                return 

            # Handle Channel Data [channelID, data, channelName, pair]
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
        payload = data[1]
        if "as" in payload or "bs" in payload:
            self.logger.info(f"[Kraken][{pair}] SNAPSHOT received.")
        # Differential updates (a/b) are silenced for log clarity but processed here

    def _process_tick(self, data, pair):
        payload = data[1]
        if 'c' in payload:
            price = payload['c'][0]
            # self.logger.info(f"[Kraken][{pair}] TICK: {price}")

    def _on_error(self, ws, error):
        self.logger.error(f"Kraken Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        self.logger.warning("Kraken Connection Closed.")


class BinanceAutoClient:
    """
    Revised Client: Uses pure asyncio inside a dedicated thread.
    Fixes 'Event loop already running' on Python 3.13 by isolating the loop.
    Fixes 'QueueOverflow' by removing the synchronous queue bridge.
    """
    def __init__(self, symbol: str):
        self.logger = logging.getLogger("BinanceClient")
        self.symbol = symbol.upper()
        self.loop = asyncio.new_event_loop()
        self.t = threading.Thread(target=self._start_async_loop, args=(self.loop,))
        self.t.daemon = True
        self.t.start()

    def _start_async_loop(self, loop):
        """
        Runs a dedicated asyncio loop for Binance in a background thread.
        """
        asyncio.set_event_loop(loop)
        self.logger.info("Binance Async Loop Started.")
        loop.run_until_complete(self._main_stream_logic())

    async def _main_stream_logic(self):
        """
        Manages the AsyncClient and streams.
        """
        self.client = await AsyncClient.create()
        self.bm = BinanceSocketManager(self.client)
        
        # 1. Start Depth Cache (Order Book)
        # AsyncDepthCacheManager handles the REST snapshot + WS delta sync automatically
        self.dcm = AsyncDepthCacheManager(self.client, symbol=self.symbol, refresh_interval=None)
        
        # 2. Start Trade Socket
        ts = self.bm.trade_socket(self.symbol)

        self.logger.info(f"Subscribing to Binance streams: {self.symbol}")

        # Run both consumers concurrently
        await asyncio.gather(
            self._watch_depth_cache(),
            self._watch_trades(ts)
        )

    async def _watch_depth_cache(self):
        """
        Consumes updates from the Depth Cache Manager.
        """
        async with self.dcm as dcm_socket:
            while True:
                depth_cache = await dcm_socket.recv()
                if depth_cache:
                    # Accessing the bid/ask directly from the cache object
                    # This object is automatically mutated/updated by the library
                    bids = depth_cache.get_bids()
                    asks = depth_cache.get_asks()
                    if bids and asks:
                        # Log only occasionally or on significant change to prevent spam
                        # self.logger.info(f"[Binance][{self.symbol}] BOOK: Bid {bids[0][0]} | Ask {asks[0][0]}")
                        pass

    async def _watch_trades(self, trade_socket):
        """
        Consumes raw trade messages directly.
        """
        async with trade_socket as tscm:
            while True:
                msg = await tscm.recv()
                if msg.get('e') == 'error':
                    self.logger.error(f"Binance Socket Error: {msg}")
                else:
                    price = msg['p']
                    # self.logger.info(f"[Binance][{self.symbol}] TRADE: {price}")

    def stop(self):
        # Graceful shutdown logic would go here
        pass

# --- Execution Block ---
try:
    print("----------------------------------------------------------------")
    print(">> INITIALIZING HIGH-FREQUENCY DATA STREAMS <<")
    print("----------------------------------------------------------------")

    # 1. Initialize Kraken (XBT/USD)
    kraken_client = KrakenAutoClient(["XBT/USD"])

    # 2. Initialize Binance (BTCUSDT)
    # Using the new Async-in-Thread architecture
    binance_client = BinanceAutoClient("BTCUSDT")

    print(">> STREAMS RUNNING. PRESS CTRL+C TO STOP <<")
    
    # Keep main thread alive
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping clients...")
    sys.exit(0)
