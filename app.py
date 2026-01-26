import requests
import threading
import sys
import time

# Force unbuffered output so logs appear immediately in Railway/Docker
sys.stdout.reconfigure(line_buffering=True)

# Configuration for data sources
DATA_SOURCES = {
    "BTC": "https://ohlcendpoint.up.railway.app/data/btc?limit=4000000",
    "ETH": "https://ohlcendpoint.up.railway.app/data/eth?limit=4000000"
}

def _execute_blocking_download(symbol: str, url: str) -> None:
    """
    Internal function to handle the synchronous download and validation 
    of OHLC data.
    """
    print(f"[{symbol}] Automatic download task initiated...", flush=True)
    try:
        # High timeout allowed for large datasets
        response = requests.get(url, stream=False, timeout=300)
        response.raise_for_status()
        
        payload = response.json()
        
        # Validate structure based on provided schema
        if "data" in payload and isinstance(payload["data"], list):
            record_count = len(payload["data"])
            print(f"Success: {symbol} - Retrieved {record_count} records.", flush=True)
        else:
            print(f"Failure: {symbol} - Invalid JSON structure.", flush=True)
            
    except requests.exceptions.RequestException as e:
        print(f"Failure: {symbol} - Network/HTTP Error: {e}", flush=True)
    except ValueError as e:
        print(f"Failure: {symbol} - JSON Decode Error: {e}", flush=True)
    except Exception as e:
        print(f"Failure: {symbol} - Unexpected Error: {e}", flush=True)

# Automatic Initialization
def _initialize_startup_tasks():
    threads = []
    print("System startup: Initiating background data ingestion...", flush=True)
    
    for symbol, endpoint in DATA_SOURCES.items():
        # Threads are non-daemon to ensure they complete before script exit
        downloader_thread = threading.Thread(
            target=_execute_blocking_download, 
            args=(symbol, endpoint),
            name=f"Downloader-{symbol}"
        )
        downloader_thread.start()
        threads.append(downloader_thread)

    # Explicitly join threads to keep the main process alive in the container
    # until all downloads finalize.
    for t in threads:
        t.join()

    print("All startup tasks completed.", flush=True)

# Trigger execution automatically at module level
if __name__ == "__main__":
    _initialize_startup_tasks()
