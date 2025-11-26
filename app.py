import os
import pandas as pd
import matplotlib.pyplot as plt
import gdown
from flask import Flask, render_template_string
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

# Download the CSV file
file_id = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
url = f'https://drive.google.com/uc?id={file_id}'
output = '1m.csv'
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Load the data
df = pd.read_csv(output)

# Ensure the timestamp column is in datetime format and set as index
# Assuming the CSV has columns like 'timestamp', 'open', 'high', 'low', 'close', 'volume'
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Filter data to start from 2022
df = df[df.index >= '2022-01-01']

# Resample to different timeframes
timeframes = {
    '15min': '15T',
    '30min': '30T',
    '1h': '1H',
    '4h': '4H',
    '1d': '1D',
    '1w': '1W'
}

resampled_data = {}
for tf_name, tf in timeframes.items():
    resampled = df.resample(tf).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    resampled_data[tf_name] = resampled

# Prepare data for linear regression models
data_1h = resampled_data['1h'].copy()
data_1d = resampled_data['1d'].copy()
data_1w = resampled_data['1w'].copy()

# Calculate features and targets for 1-day prediction
# Features: 24 individual hourly price changes (percent) - using data up to t-1
for i in range(1, 25):
    data_1h[f'price_change_{i}h_pct'] = data_1h['close'].shift(i).pct_change(periods=1) * 100
# Target: price change over next 1 day (percent) - calculated from hourly data
data_1h['target_1d_pct'] = (data_1h['close'].shift(-24) / data_1h['close'] - 1) * 100

# Calculate features and targets for 1-week prediction
# Features: 7 individual daily price changes (percent) - using data up to t-1
for i in range(1, 8):
    data_1d[f'price_change_{i}d_pct'] = data_1d['close'].shift(i).pct_change(periods=1) * 100
# Target: price change over next 1 week (percent) - calculated from daily data
data_1d['target_1w_pct'] = (data_1d['close'].shift(-7) / data_1d['close'] - 1) * 100

# Drop rows with NaN values (due to pct_change calculations)
feature_cols_1d = [f'price_change_{i}h_pct' for i in range(1, 25)]
data_1h_clean = data_1h.dropna(subset=feature_cols_1d + ['target_1d_pct'])
feature_cols_1w = [f'price_change_{i}d_pct' for i in range(1, 8)]
data_1d_clean = data_1d.dropna(subset=feature_cols_1w + ['target_1w_pct'])

# Prepare features and targets for modeling
X_1d = data_1h_clean[feature_cols_1d]
y_1d = data_1h_clean['target_1d_pct']
X_1w = data_1d_clean[feature_cols_1w]
y_1w = data_1d_clean['target_1w_pct']

# Split data into train and test sets (using 80% train, 20% test)
X_1d_train, X_1d_test, y_1d_train, y_1d_test = train_test_split(X_1d, y_1d, test_size=0.2, random_state=42)
X_1w_train, X_1w_test, y_1w_train, y_1w_test = train_test_split(X_1w, y_1w, test_size=0.2, random_state=42)

# Train linear regression models
model_1d = LinearRegression()
model_1d.fit(X_1d_train, y_1d_train)

model_1w = LinearRegression()
model_1w.fit(X_1w_train, y_1w_train)

# Train neural network models
nn_model_1d = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
nn_model_1d.fit(X_1d_train, y_1d_train)

nn_model_1w = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
nn_model_1w.fit(X_1w_train, y_1w_train)

# Calculate predictions and errors for 1-day linear model
y_1d_pred = model_1d.predict(X_1d_test)
mae_1d = np.mean(np.abs(y_1d_pred - y_1d_test))
rmse_1d = np.sqrt(np.mean((y_1d_pred - y_1d_test) ** 2))
prediction_distances_1d = np.abs(y_1d_pred - y_1d_test)

# Calculate predictions and errors for 1-week linear model
y_1w_pred = model_1w.predict(X_1w_test)
mae_1w = np.mean(np.abs(y_1w_pred - y_1w_test))
rmse_1w = np.sqrt(np.mean((y_1w_pred - y_1w_test) ** 2))
prediction_distances_1w = np.abs(y_1w_pred - y_1w_test)

# Calculate predictions and errors for 1-day neural network model
y_1d_nn_pred = nn_model_1d.predict(X_1d_test)
mae_1d_nn = np.mean(np.abs(y_1d_nn_pred - y_1d_test))
rmse_1d_nn = np.sqrt(np.mean((y_1d_nn_pred - y_1d_test) ** 2))
prediction_distances_1d_nn = np.abs(y_1d_nn_pred - y_1d_test)

# Calculate predictions and errors for 1-week neural network model
y_1w_nn_pred = nn_model_1w.predict(X_1w_test)
mae_1w_nn = np.mean(np.abs(y_1w_nn_pred - y_1w_test))
rmse_1w_nn = np.sqrt(np.mean((y_1w_nn_pred - y_1w_test) ** 2))
prediction_distances_1w_nn = np.abs(y_1w_nn_pred - y_1w_test)

# Trading simulation with 1000 euro initial investment
def simulate_trading(data, predictions, actual_returns, timeframe_name, initial_capital=1000):
    """
    Simulate trading based on prediction direction:
    - Long if prediction > 0
    - Short if prediction < 0
    - No position if prediction == 0
    """
    capital = [initial_capital]
    positions = []
    trades = []
    
    for i in range(len(predictions)):
        current_capital = capital[-1]
        
        # Determine position based on prediction direction
        if predictions[i] > 0:
            position = 'long'
            # Long position: profit = actual_return
            profit_pct = actual_returns.iloc[i]
        elif predictions[i] < 0:
            position = 'short'
            # Short position: profit = -actual_return
            profit_pct = -actual_returns.iloc[i]
        else:
            position = 'none'
            profit_pct = 0
        
        # Calculate new capital (assuming full investment each period)
        new_capital = current_capital * (1 + profit_pct / 100)
        capital.append(new_capital)
        positions.append(position)
        
        trades.append({
            'index': i,
            'prediction': predictions[i],
            'actual_return': actual_returns.iloc[i],
            'position': position,
            'capital_before': current_capital,
            'capital_after': new_capital,
            'profit_pct': profit_pct
        })
    
    # Create trading results DataFrame
    trades_df = pd.DataFrame(trades)
    trades_df['cumulative_return_pct'] = (trades_df['capital_after'] / initial_capital - 1) * 100
    
    # Calculate performance metrics
    total_return_pct = (capital[-1] / initial_capital - 1) * 100
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['profit_pct'] > 0])
    losing_trades = len([t for t in trades if t['profit_pct'] < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'capital_series': capital,
        'trades_df': trades_df,
        'total_return_pct': total_return_pct,
        'final_capital': capital[-1],
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'timeframe': timeframe_name
    }

# Run trading simulations for all models
print("Running trading simulations...")

# 1-day linear model simulation
trading_1d_linear = simulate_trading(
    data_1h_clean.loc[X_1d_test.index],
    y_1d_pred,
    y_1d_test,
    '1-day Linear'
)

# 1-week linear model simulation
trading_1w_linear = simulate_trading(
    data_1d_clean.loc[X_1w_test.index],
    y_1w_pred,
    y_1w_test,
    '1-week Linear'
)

# 1-day neural network simulation
trading_1d_nn = simulate_trading(
    data_1h_clean.loc[X_1d_test.index],
    y_1d_nn_pred,
    y_1d_test,
    '1-day Neural Network'
)

# 1-week neural network simulation
trading_1w_nn = simulate_trading(
    data_1d_clean.loc[X_1w_test.index],
    y_1w_nn_pred,
    y_1w_test,
    '1-week Neural Network'
)

# Function to generate trading performance plot
def generate_trading_plot(trading_results, title):
    plt.figure(figsize=(12, 8))
    
    # Plot capital development
    plt.subplot(2, 1, 1)
    plt.plot(trading_results['capital_series'], linewidth=2)
    plt.title(f'{title} - Capital Development')
    plt.xlabel('Trade Number')
    plt.ylabel('Capital (EUR)')
    plt.grid(True)
    plt.axhline(y=1000, color='r', linestyle='--', alpha=0.7, label='Initial Capital')
    plt.legend()
    
    # Plot cumulative returns
    plt.subplot(2, 1, 2)
    plt.plot(trading_results['trades_df']['cumulative_return_pct'], linewidth=2)
    plt.title(f'{title} - Cumulative Return (%)')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative Return (%)')
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Break-even')
    plt.legend()
    
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

# Generate trading performance plots
trading_plots = {}
trading_plots['1d_linear'] = generate_trading_plot(trading_1d_linear, '1-Day Linear Model')
trading_plots['1w_linear'] = generate_trading_plot(trading_1w_linear, '1-Week Linear Model')
trading_plots['1d_nn'] = generate_trading_plot(trading_1d_nn, '1-Day Neural Network')
trading_plots['1w_nn'] = generate_trading_plot(trading_1w_nn, '1-Week Neural Network')

# Function to generate plot as base64 image
def generate_plot(data, title, model=None, feature_name=None, nn_model=None):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['close'], label='Close Price')
    
    # Add prediction lines if models and feature are provided
    if model is not None and feature_name is not None:
        # Calculate feature values for all data points using data up to t-1
        if feature_name == '1d':
            feature_values = pd.DataFrame({f'price_change_{i}h_pct': data['close'].shift(i).pct_change(periods=1) * 100 for i in range(1, 25)})
        elif feature_name == '1w':
            feature_values = pd.DataFrame({f'price_change_{i}d_pct': data['close'].shift(i).pct_change(periods=1) * 100 for i in range(1, 8)})
        else:
            feature_values = pd.DataFrame()
        
        # Predict percentage changes for all data points with non-NaN features
        valid_indices = feature_values.dropna().index
        if len(valid_indices) > 0:
            # Linear regression predictions
            predicted_pct_changes = model.predict(feature_values.loc[valid_indices])
            # Calculate predicted prices for the next period (shift indices forward)
            if feature_name == '1d':
                predicted_indices = valid_indices + pd.Timedelta(hours=24)
            elif feature_name == '1w':
                predicted_indices = valid_indices + pd.Timedelta(days=7)
            else:
                predicted_indices = valid_indices
            predicted_prices = data.loc[valid_indices, 'close'] * (1 + predicted_pct_changes / 100)
            plt.plot(predicted_indices, predicted_prices, color='red', linestyle='--', label='Linear Regression Prediction')
            
            # Neural network predictions if model provided
            if nn_model is not None:
                nn_predicted_pct_changes = nn_model.predict(feature_values.loc[valid_indices])
                nn_predicted_prices = data.loc[valid_indices, 'close'] * (1 + nn_predicted_pct_changes / 100)
                plt.plot(predicted_indices, nn_predicted_prices, color='green', linestyle='--', label='Neural Network Prediction')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def index():
    plots = {}
    for tf_name, data in resampled_data.items():
        # For 1d and 1w timeframes, include predictions
        if tf_name == '1d':
            plots[tf_name] = generate_plot(data, f'BTC OHLCV - {tf_name} with Predictions', model=model_1d, feature_name='1d', nn_model=nn_model_1d)
        elif tf_name == '1w':
            plots[tf_name] = generate_plot(data, f'BTC OHLCV - {tf_name} with Predictions', model=model_1w, feature_name='1w', nn_model=nn_model_1w)
        else:
            plots[tf_name] = generate_plot(data, f'BTC OHLCV - {tf_name}')
    
    # Prepare model info for display
    model_info = {
        '1d': {
            'coefficients': model_1d.coef_.tolist(),
            'intercept': model_1d.intercept_,
            'features': ['1-hour price change (%)', '2-hour price change (%)', '3-hour price change (%)', '4-hour price change (%)', '5-hour price change (%)', '6-hour price change (%)', '7-hour price change (%)', '8-hour price change (%)', '9-hour price change (%)', '10-hour price change (%)', '11-hour price change (%)', '12-hour price change (%)', '13-hour price change (%)', '14-hour price change (%)', '15-hour price change (%)', '16-hour price change (%)', '17-hour price change (%)', '18-hour price change (%)', '19-hour price change (%)', '20-hour price change (%)', '21-hour price change (%)', '22-hour price change (%)', '23-hour price change (%)', '24-hour price change (%)'],
            'target': '1-day price change (%)',
            'mae': mae_1d,
            'rmse': rmse_1d,
            'test_set_size': len(y_1d_test),
            'prediction_distances': prediction_distances_1d,
            'trading_results': trading_1d_linear
        },
        '1w': {
            'coefficients': model_1w.coef_.tolist(),
            'intercept': model_1w.intercept_,
            'features': ['1-day price change (%)', '2-day price change (%)', '3-day price change (%)', '4-day price change (%)', '5-day price change (%)', '6-day price change (%)', '7-day price change (%)'],
            'target': '1-week price change (%)',
            'mae': mae_1w,
            'rmse': rmse_1w,
            'test_set_size': len(y_1w_test),
            'prediction_distances': prediction_distances_1w,
            'trading_results': trading_1w_linear
        },
        '1d_nn': {
            'features': ['1-hour price change (%)', '2-hour price change (%)', '3-hour price change (%)', '4-hour price change (%)', '5-hour price change (%)', '6-hour price change (%)', '7-hour price change (%)', '8-hour price change (%)', '9-hour price change (%)', '10-hour price change (%)', '11-hour price change (%)', '12-hour price change (%)', '13-hour price change (%)', '14-hour price change (%)', '15-hour price change (%)', '16-hour price change (%)', '17-hour price change (%)', '18-hour price change (%)', '19-hour price change (%)', '20-hour price change (%)', '21-hour price change (%)', '22-hour price change (%)', '23-hour price change (%)', '24-hour price change (%)'],
            'target': '1-day price change (%)',
            'mae': mae_1d_nn,
            'rmse': rmse_1d_nn,
            'test_set_size': len(y_1d_test),
            'prediction_distances': prediction_distances_1d_nn,
            'trading_results': trading_1d_nn
        },
        '1w_nn': {
            'features': ['1-day price change (%)', '2-day price change (%)', '3-day price change (%)', '4-day price change (%)', '5-day price change (%)', '6-day price change (%)', '7-day price change (%)'],
            'target': '1-week price change (%)',
            'mae': mae_1w_nn,
            'rmse': rmse_1w_nn,
            'test_set_size': len(y_1w_test),
            'prediction_distances': prediction_distances_1w_nn,
            'trading_results': trading_1w_nn
        }
    }
    
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>BTC OHLCV Timeframes and Linear Regression Models</title>
    </head>
    <body>
        <h1>BTC OHLCV Data - Various Timeframes</h1>
        {% for tf, img in plots.items() %}
            <h2>{{ tf }}</h2>
            <img src="data:image/png;base64,{{ img }}" alt="{{ tf }} Plot">
        {% endfor %}
        
        <h1>Trading Simulation Results (1000 EUR Initial Investment)</h1>
        {% for model_name, info in model_info.items() %}
            {% if info.trading_results %}
                <h2>{{ info.trading_results.timeframe }} Model Trading Performance</h2>
                <p><strong>Initial Capital:</strong> 1,000 EUR</p>
                <p><strong>Final Capital:</strong> {{ "%.2f"|format(info.trading_results.final_capital) }} EUR</p>
                <p><strong>Total Return:</strong> {{ "%.2f"|format(info.trading_results.total_return_pct) }}%</p>
                <p><strong>Total Trades:</strong> {{ info.trading_results.total_trades }}</p>
                <p><strong>Winning Trades:</strong> {{ info.trading_results.winning_trades }}</p>
                <p><strong>Losing Trades:</strong> {{ info.trading_results.losing_trades }}</p>
                <p><strong>Win Rate:</strong> {{ "%.1f"|format(info.trading_results.win_rate) }}%</p>
                
                {% if model_name == '1d' %}
                    <img src="data:image/png;base64,{{ trading_plots['1d_linear'] }}" alt="1-Day Linear Trading Plot">
                {% elif model_name == '1w' %}
                    <img src="data:image/png;base64,{{ trading_plots['1w_linear'] }}" alt="1-Week Linear Trading Plot">
                {% elif model_name == '1d_nn' %}
                    <img src="data:image/png;base64,{{ trading_plots['1d_nn'] }}" alt="1-Day Neural Network Trading Plot">
                {% elif model_name == '1w_nn' %}
                    <img src="data:image/png;base64,{{ trading_plots['1w_nn'] }}" alt="1-Week Neural Network Trading Plot">
                {% endif %}
                <hr>
            {% endif %}
        {% endfor %}
        
        <h1>Linear Regression Models for Price Prediction</h1>
        {% for model_name, info in model_info.items() if 'nn' not in model_name %}
            <h2>Linear Regression Model for {{ model_name }} Prediction</h2>
            <p>Features: {{ info.features | join(', ') }}</p>
            <p>Target: {{ info.target }}</p>
            <p>Coefficients: {% for coef in info.coefficients %}{{ "%.6f"|format(coef) }}{% if not loop.last %}, {% endif %}{% endfor %}</p>
            <p>Intercept: {{ "%.6f"|format(info.intercept) }}</p>
            <p>Equation: Target = {% for coef in info.coefficients %}{{ "%.6f"|format(coef) }} * Feature_{{ loop.index }}{% if not loop.last %} + {% endif %}{% endfor %} + {{ "%.6f"|format(info.intercept) }}</p>
            <p>Mean Absolute Error (MAE): {{ "%.4f"|format(info.mae) }}%</p>
            <p>Root Mean Squared Error (RMSE): {{ "%.4f"|format(info.rmse) }}%</p>
            <p>Test Set Size: {{ info.test_set_size }} samples</p>
        {% endfor %}
        
        <h1>Neural Network Models for Price Prediction</h1>
        {% for model_name, info in model_info.items() if 'nn' in model_name %}
            <h2>Neural Network Model for {{ model_name[:-3] }} Prediction</h2>
            <p>Features: {{ info.features | join(', ') }}</p>
            <p>Target: {{ info.target }}</p>
            <p>Mean Absolute Error (MAE): {{ "%.4f"|format(info.mae) }}%</p>
            <p>Root Mean Squared Error (RMSE): {{ "%.4f"|format(info.rmse) }}%</p>
            <p>Test Set Size: {{ info.test_set_size }} samples</p>
        {% endfor %}
        
        <h1>Prediction Distance Analysis</h1>
        <h2>1-Day Linear Model Prediction Distances</h2>
        <p>Average distance between predicted and actual: {{ "%.4f"|format(model_info['1d'].mae) }}%</p>
        <p>Maximum distance: {{ "%.4f"|format(model_info['1d'].prediction_distances.max()) }}%</p>
        <p>Minimum distance: {{ "%.4f"|format(model_info['1d'].prediction_distances.min()) }}%</p>
        
        <h2>1-Week Linear Model Prediction Distances</h2>
        <p>Average distance between predicted and actual: {{ "%.4f"|format(model_info['1w'].mae) }}%</p>
        <p>Maximum distance: {{ "%.4f"|format(model_info['1w'].prediction_distances.max()) }}%</p>
        <p>Minimum distance: {{ "%.4f"|format(model_info['1w'].prediction_distances.min()) }}%</p>
        
        <h2>1-Day Neural Network Model Prediction Distances</h2>
        <p>Average distance between predicted and actual: {{ "%.4f"|format(model_info['1d_nn'].mae) }}%</p>
        <p>Maximum distance: {{ "%.4f"|format(model_info['1d_nn'].prediction_distances.max()) }}%</p>
        <p>Minimum distance: {{ "%.4f"|format(model_info['1d_nn'].prediction_distances.min()) }}%</p>
        
        <h2>1-Week Neural Network Model Prediction Distances</h2>
        <p>Average distance between predicted and actual: {{ "%.4f"|format(model_info['1w_nn'].mae) }}%</p>
        <p>Maximum distance: {{ "%.4f"|format(model_info['1w_nn'].prediction_distances.max()) }}%</p>
        <p>Minimum distance: {{ "%.4f"|format(model_info['1w_nn'].prediction_distances.min()) }}%</p>
    </body>
    </html>
    '''
    return render_template_string(html_template, plots=plots, model_info=model_info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)