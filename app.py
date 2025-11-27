import pandas as pd
import numpy as np
from optimal_trading import OptimalTradingStrategy
import os
import gdown

def download_data_at_startup():
    """Download data automatically at script startup"""
    try:
        data_url = 'https://drive.google.com/file/d/1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o/view?usp=drivesdk'
        print("DEBUG: Starting automatic data download at startup...")
        print(f"DEBUG: Fetching data from URL: {data_url}")
        
        # Handle Google Drive URLs with gdown
        file_id = data_url.split('/d/')[1].split('/')[0]
        download_url = f'https://drive.google.com/uc?id={file_id}'
        print(f"DEBUG: Downloading from Google Drive. File ID: {file_id}")
        
        # Download file temporarily
        output_path = 'temp_data.csv'
        gdown.download(download_url, output_path, quiet=False)
        print(f"DEBUG: File downloaded to {output_path}")
        
        df = pd.read_csv(output_path)
        print(f"DEBUG: CSV loaded. Shape: {df.shape}")
        
        # Clean up temporary file
        if os.path.exists(output_path):
            os.remove(output_path)
            print("DEBUG: Temporary file cleaned up")
        
        # Validate required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"ERROR: Missing columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        print(f"DEBUG: All required columns present: {required_columns}")
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            print("DEBUG: Converting timestamp to datetime")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"DEBUG: Timestamp dtype: {df['timestamp'].dtype}")
        
        print("DEBUG: Data downloaded and stored successfully at startup")
        return df
        
    except Exception as e:
        print(f"ERROR: Failed to download data at startup: {str(e)}")
        import traceback
        print(f"ERROR: Traceback: {traceback.format_exc()}")
        raise e

if __name__ == '__main__':
    # Download data and run analysis
    print("DEBUG: Starting application - downloading data at startup...")
    downloaded_data = download_data_at_startup()
    print("DEBUG: Data download completed at startup")
    
    print("DEBUG: Starting automatic analysis at startup...")
    try:
        strategy = OptimalTradingStrategy(fee_rate=0.002)
        analysis_result = strategy.calculate_optimal_trades(downloaded_data)
        print("DEBUG: Automatic analysis completed at startup")
        
        # Display results
        final_capital = float(analysis_result['optimal_capital'].iloc[-1])
        total_trades = int((analysis_result['optimal_action'] != 'hold').sum())
        long_trades = int((analysis_result['optimal_action'] == 'buy_long').sum())
        short_trades = int((analysis_result['optimal_action'] == 'sell_short').sum())
        
        print("\n=== Analysis Results ===")
        print(f"Final Capital: {final_capital:.4f}")
        print(f"Total Trades: {total_trades}")
        print(f"Long Trades: {long_trades}")
        print(f"Short Trades: {short_trades}")
        print("=== End of Results ===")
        
    except Exception as e:
        print(f"ERROR: Failed to run automatic analysis at startup: {str(e)}")
        import traceback
        print(f"ERROR: Traceback: {traceback.format_exc()}")
        raise e