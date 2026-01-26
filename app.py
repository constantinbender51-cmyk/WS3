import os
import sys
import requests
import pandas as pd

# --- Configuration ---
BASE_URL = "https://ohlcendpoint.up.railway.app"
DATA_DIR = "./downloaded_ohlc_data"
# List of tickers based on the previous context provided in the prompt
SYMBOLS = [
    "BTC", "ETH", "XRP", "SOL", "DOGE",
    "ADA", "BCH", "LINK", "XLM", "SUI",
    "AVAX", "LTC", "HBAR", "SHIB", "TON",
]

# --- Force Unbuffered Logging ---
sys.stdout.reconfigure(line_buffering=True)

def initiate_download_sequence():
    """
    Iterates through the symbol list and downloads the corresponding CSV 
    files from the endpoint. 
    """
    print("--- INITIALIZING AUTO-DOWNLOAD SEQUENCE ---")
    
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR)
            print(f"Created data directory: {os.path.abspath(DATA_DIR)}")
        except OSError as e:
            print(f"CRITICAL: Failed to create directory {DATA_DIR}. Error: {e}")
            return

    for ticker in SYMBOLS:
        # Construct the URL based on the API structure provided in the context
        url = f"{BASE_URL}/download/{ticker}"
        # Standardizing filename convention to match the source script's logic (Symbol_USDT)
        filename = os.path.join(DATA_DIR, f"{ticker}_USDT.csv")
        
        print(f"[{ticker}] Initiating download from {url}...")
        
        try:
            # Using stream=True to handle potentially large files without loading into memory at once
            with requests.get(url, stream=True) as r:
                if r.status_code == 200:
                    with open(filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk: 
                                f.write(chunk)
                    
                    # Verification step: ensure the file is valid CSV readable by pandas
                    # This maintains the complexity of data integrity checks.
                    try:
                        df = pd.read_csv(filename)
                        row_count = len(df)
                        print(f"[{ticker}] SUCCESS: Downloaded and verified {row_count} rows.")
                    except pd.errors.EmptyDataError:
                        print(f"[{ticker}] WARNING: File downloaded but contains no data.")
                    except Exception as pd_e:
                        print(f"[{ticker}] WARNING: File downloaded but failed CSV parsing: {pd_e}")
                        
                else:
                    print(f"[{ticker}] FAILURE: Server returned status code {r.status_code}")
        
        except requests.exceptions.RequestException as e:
            print(f"[{ticker}] ERROR: Network or connection failure - {e}")
        except Exception as e:
            print(f"[{ticker}] ERROR: Unexpected error - {e}")

    print("--- AUTO-DOWNLOAD SEQUENCE COMPLETED ---")

# --- Execution Entry Point ---
# The task is triggered immediately on import or execution.
initiate_download_sequence()
