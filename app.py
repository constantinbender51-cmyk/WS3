from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
from datetime import datetime
from optimal_trading import OptimalTradingStrategy

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get parameters from request
        data = request.get_json()
        fee_rate = float(data.get('fee_rate', 0.002))
        
        # Fetch data from URL or use default Google Drive URL
        data_url = data.get('data_url', 'https://drive.google.com/file/d/1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o/view?usp=drivesdk')
        df = fetch_data_from_url(data_url)
        
        # Calculate optimal strategy
        strategy = OptimalTradingStrategy(fee_rate=fee_rate)
        result = strategy.calculate_optimal_trades(df)
        
        # Prepare data for visualization
        chart_data = prepare_chart_data(result)
        
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
        return jsonify({
            'success': False,
            'error': str(e)
        })





def fetch_data_from_url(url):
    """Fetch OHLCV data from a URL (CSV format expected)"""
    try:
        # Handle Google Drive URLs with gdown
        if 'drive.google.com' in url:
            import gdown
            # Extract file ID from Google Drive URL
            file_id = url.split('/d/')[1].split('/')[0]
            download_url = f'https://drive.google.com/uc?id={file_id}'
            
            # Download file temporarily
            output_path = 'temp_data.csv'
            gdown.download(download_url, output_path, quiet=False)
            
            df = pd.read_csv(output_path)
            
            # Clean up temporary file
            import os
            os.remove(output_path)
        else:
            # Regular URL
            df = pd.read_csv(url)
        
        # Validate required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except Exception as e:
        raise ValueError(f"Failed to fetch data from URL: {str(e)}")

def prepare_chart_data(result_df):
    """Prepare data for chart visualization"""
    # Convert to list format for JSON serialization
    timestamps = result_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
    prices = result_df['close'].tolist()
    capital = result_df['optimal_capital'].tolist()
    actions = result_df['optimal_action'].tolist()
    
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
    
    return {
        'timestamps': timestamps,
        'prices': prices,
        'capital': capital,
        'trade_markers': trade_markers,
        'actions': actions
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)