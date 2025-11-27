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
        
        # Generate sample data or use uploaded data
        if data.get('use_sample_data', True):
            df = generate_sample_data()
        else:
            # In a real implementation, handle file upload
            df = generate_sample_data()
        
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

def generate_sample_data(n_samples=1000):
    """Generate sample OHLCV data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    
    # Generate random walk prices with some trend
    returns = np.random.normal(0.0001, 0.002, n_samples)
    prices = 100 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_samples))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    return df

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
    app.run(debug=True, host='0.0.0.0', port=5000)