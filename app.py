import requests
import threading
import sys
import json
import io
from tqdm import tqdm  # pip install tqdm

# Configuration for data sources
DATA_SOURCES = {
    "BTC": "https://ohlcendpoint.up.railway.app/data/btc?limit=4000000",
    "ETH": "https://ohlcendpoint.up.railway.app/data/eth?limit=4000000"
}

def _execute_streaming_download(symbol: str, url: str, position: int) -> None:
    """
    Handles streaming download with chunked iteration to support progress visualization.
    Uses a BytesIO buffer to reconstruct the payload in memory before JSON parsing.
    
    Args:
        symbol: The asset symbol (e.g., BTC).
        url: The endpoint URL.
        position: The line offset for the progress bar (for multi-threaded display).
    """
    buffer = io.BytesIO()
    
    try:
        # stream=True is required to prevent immediate content download
        with requests.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            
            # Extract header content length for progress calculation
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192  # 8KB chunks

            # Initialize tqdm with specific position to handle multi-threading
            with tqdm(
                total=total_size, 
                unit='B', 
                unit_scale=True, 
                desc=symbol, 
                position=position, 
                leave=True,
                mininterval=1.0 # Prevent log flooding in container environments
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        buffer.write(chunk)
                        pbar.update(len(chunk))
        
        # Reset buffer cursor to beginning before reading
        buffer.seek(0)
        
        # Decode bytes to string, then load JSON
        # This maintains the complexity of manual buffer management
        payload = json.loads(buffer.read().decode('utf-8'))

        # Validate structure
        if "data" in payload and isinstance(payload["data"], list):
            record_count = len(payload["data"])
            # Using tqdm.write to avoid interfering with progress bars
            tqdm.write(f"Success: {symbol} - Retrieved {record_count} records.")
        else:
            tqdm.write(f"Failure: {symbol} - Invalid JSON structure.")

    except requests.exceptions.RequestException as e:
        tqdm.write(f"Failure: {symbol} - Network Error: {e}")
    except json.JSONDecodeError as e:
        tqdm.write(f"Failure: {symbol} - JSON Decode Error: {e}")
    except Exception as e:
        tqdm.write(f"Failure: {symbol} - Unexpected Error: {e}")
    finally:
        buffer.close()

def _initialize_startup_tasks():
    threads = []
    print("System startup: Initiating background data ingestion with progress tracking...", flush=True)
    
    # Enumerate sources to assign specific progress bar positions (0, 1, etc.)
    for index, (symbol, endpoint) in enumerate(DATA_SOURCES.items()):
        downloader_thread = threading.Thread(
            target=_execute_streaming_download, 
            args=(symbol, endpoint, index),
            name=f"Downloader-{symbol}"
        )
        downloader_thread.start()
        threads.append(downloader_thread)

    # Block main thread until downloads complete
    for t in threads:
        t.join()

    print("All startup tasks completed.", flush=True)

if __name__ == "__main__":
    _initialize_startup_tasks()
