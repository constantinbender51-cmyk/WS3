from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
from datetime import datetime
from optimal_trading import OptimalTradingStrategy
import os
import gdown

app = Flask(__name__)

# Global variable to store downloaded data
downloaded_data = None

def download_data_at_startup():
    """Download data automatically at script startup"""
    global downloaded_data
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
        
        downloaded_data = df
        print("DEBUG: Data downloaded and stored successfully at startup")
        
    except Exception as e:
        print(f"ERROR: Failed to download data at startup: {str(e)}")
        import traceback
        print(f"ERROR: Traceback: {traceback.format_exc()}")
        raise e

# Download data when the script starts
download_data_at_startup()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get parameters from request
        data = request.get_json()
        fee_rate = float(data.get('fee_rate', 0.002))
        print(f"DEBUG: Starting analysis with fee_rate={fee_rate}")
        
        # Use pre-downloaded data
        if downloaded_data is None:
            raise ValueError("Data not available. Please restart the application.")
        df = downloaded_data
        print(f"DEBUG: Using pre-downloaded data. Shape: {df.shape}, Columns: {list(df.columns)}")
        print(f"DEBUG: First few rows:\n{df.head()}")
        
        # Calculate optimal strategy
        print("DEBUG: Starting optimal strategy calculation...")
        strategy = OptimalTradingStrategy(fee_rate=fee_rate)
        result = strategy.calculate_optimal_trades(df)
        print(f"DEBUG: Strategy calculation completed. Result shape: {result.shape}")
        print(f"DEBUG: Result columns: {list(result.columns)}")
        print(f"DEBUG: Optimal actions distribution: {result['optimal_action'].value_counts().to_dict()}")
        
        # Prepare data for visualization
        print("DEBUG: Preparing chart data...")
        chart_data = prepare_chart_data(result)
        print(f"DEBUG: Chart data prepared. Timestamps: {len(chart_data['timestamps'])}, Prices: {len(chart_data['prices'])}, Trade markers: {len(chart_data['trade_markers'])}")
        
        return jsonify({
            'success': True,
            'chart_data': chart_data,
            'summary': {
                'final_capital': float(result['optimal_capital'].iloc[-1]),
                'total_trades': int((result['optimal_action'] != 'hold').sum()),
                'long_trades': int((result['optimal_action'] == 'buy_long').sum()),
                'short_trades': int((result['optimal_action'] == 'sell_short').sum())
            }
        })
        
    except Exception as e:
        print(f"ERROR: Exception in analyze route: {str(e)}")
        import traceback
        print(f"ERROR: Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        })









def prepare_chart_data(result_df):
    """Prepare data for chart visualization"""
    print(f"DEBUG: Preparing chart data from result_df with shape: {result_df.shape}")
    
    # Convert to list format for JSON serialization
    timestamps = result_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
    prices = result_df['close'].tolist()
    capital = result_df['optimal_capital'].tolist()
    actions = result_df['optimal_action'].tolist()
    
    print(f"DEBUG: Converted {len(timestamps)} timestamps, {len(prices)} prices, {len(capital)} capital values")
    
    # Prepare trade markers
    trade_markers = []
    for i, action in enumerate(actions):
        if action != 'hold':
            trade_markers.append({
                'x': timestamps[i],
                'y': prices[i],
                'action': action,
                'color': 'green' if 'buy' in action else 'red'
            })
    
    print(f"DEBUG: Created {len(trade_markers)} trade markers")
    
    chart_data = {
        'timestamps': timestamps,
        'prices': prices,
        'capital': capital,
        'trade_markers': trade_markers,
        'actions': actions
    }
    
    print(f"DEBUG: Chart data keys: {list(chart_data.keys())}")
    return chart_data

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
