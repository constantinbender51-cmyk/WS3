import pandas as pd
import numpy as np
from optimal_trading import OptimalTradingStrategy
import os
import gdown
from flask import Flask
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

# Create Flask app instance
app = Flask(__name__)

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

        # Define route for displaying results
        @app.route('/')
        def display_results():
            # Create plot with background colors for entries and exits
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # Add background colors based on position state
            position_state = analysis_result['position_state'].values
            for i in range(len(position_state)):
                if position_state[i] == 1:  # Long position
                    ax1.axvspan(analysis_result['timestamp'].iloc[i], analysis_result['timestamp'].iloc[i], 
                                ymin=0, ymax=1, color='green', alpha=0.3)
                elif position_state[i] == 2:  # Short position
                    ax1.axvspan(analysis_result['timestamp'].iloc[i], analysis_result['timestamp'].iloc[i], 
                                ymin=0, ymax=1, color='red', alpha=0.3)
                else:  # Flat position
                    ax1.axvspan(analysis_result['timestamp'].iloc[i], analysis_result['timestamp'].iloc[i], 
                                ymin=0, ymax=1, color='gray', alpha=0.1)
            
            # Plot price data
            ax1.plot(analysis_result['timestamp'], analysis_result['close'], label='Close Price', color='black', linewidth=1)
            
            # Mark entry and exit points
            long_entries = analysis_result[analysis_result['optimal_action'] == 'buy_long']
            long_exits = analysis_result[analysis_result['optimal_action'] == 'sell_long']
            short_entries = analysis_result[analysis_result['optimal_action'] == 'sell_short']
            short_exits = analysis_result[analysis_result['optimal_action'] == 'buy_short']
            
            ax1.scatter(long_entries['timestamp'], long_entries['close'], color='green', marker='^', s=50, label='Long Entry', zorder=5)
            ax1.scatter(long_exits['timestamp'], long_exits['close'], color='red', marker='v', s=50, label='Long Exit', zorder=5)
            ax1.scatter(short_entries['timestamp'], short_entries['close'], color='blue', marker='v', s=50, label='Short Entry', zorder=5)
            ax1.scatter(short_exits['timestamp'], short_exits['close'], color='orange', marker='^', s=50, label='Short Exit', zorder=5)
            
            ax1.set_xlabel('Timestamp')
            ax1.set_ylabel('Price', color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.legend(loc='upper left')
            ax1.grid(True)
            
            # Plot capital on secondary axis
            ax2 = ax1.twinx()
            ax2.plot(analysis_result['timestamp'], analysis_result['optimal_capital'], label='Capital', color='purple', linewidth=2)
            ax2.set_ylabel('Capital', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
            ax2.legend(loc='upper right')
            
            plt.title('Price Entries, Exits, and Capital Over Time')
            fig.tight_layout()
            
            # Save plot to a bytes buffer and encode as base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            plt.close(fig)
            
            return f"""
            <h1>Optimal Trading Strategy Results</h1>
            <p>Final Capital: {final_capital:.4f}</p>
            <p>Total Trades: {total_trades}</p>
            <p>Long Trades: {long_trades}</p>
            <p>Short Trades: {short_trades}</p>
            <h2>Plot: Price Entries, Exits, and Capital</h2>
            <img src="data:image/png;base64,{plot_data}" alt="Trading Plot">
            """

        print("Starting web server on port 8080 at 0.0.0.0...")
        app.run(host='0.0.0.0', port=8080)
# Run the app when executed directly
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

        print("Starting web server on port 8080 at 0.0.0.0...")
        app.run(host='0.0.0.0', port=8080)
        
    except Exception as e:
        print(f"ERROR: Failed to run automatic analysis at startup: {str(e)}")
        import traceback
        print(f"ERROR: Traceback: {traceback.format_exc()}")
        raise e
        
    except Exception as e:
        print(f"ERROR: Failed to run automatic analysis at startup: {str(e)}")
        import traceback
        print(f"ERROR: Traceback: {traceback.format_exc()}")
        raise e