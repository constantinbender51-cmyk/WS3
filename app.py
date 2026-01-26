import requests
import threading
import sys

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
    try:
        # High timeout allowed for large datasets
        response = requests.get(url, stream=False, timeout=300)
        response.raise_for_status()
        
        payload = response.json()
        
        # Validate structure based on provided schema
        if "data" in payload and isinstance(payload["data"], list):
            record_count = len(payload["data"])
            print(f"Success: {symbol} - Retrieved {record_count} records.")
        else:
            print(f"Failure: {symbol} - Invalid JSON structure.")
            
    except requests.exceptions.RequestException as e:
        print(f"Failure: {symbol} - Network/HTTP Error: {e}")
    except ValueError as e:
        print(f"Failure: {symbol} - JSON Decode Error: {e}")
    except Exception as e:
        print(f"Failure: {symbol} - Unexpected Error: {e}")

# Automatic Initialization
# Iterates through configured sources and initiates a thread for each
# to ensure the download begins immediately upon module import or startup.
def _initialize_startup_tasks():
    threads = []
    for symbol, endpoint in DATA_SOURCES.items():
        # Daemon threads used so the script can exit if the main program finishes, 
        # though for data integrity joining them is often preferred in a persistent app.
        downloader_thread = threading.Thread(
            target=_execute_blocking_download, 
            args=(symbol, endpoint),
            name=f"Downloader-{symbol}"
        )
        downloader_thread.start()
        threads.append(downloader_thread)

# Trigger execution automatically at module level
_initialize_startup_tasks()
