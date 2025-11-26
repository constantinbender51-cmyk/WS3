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
            'prediction_distances': prediction_distances_1d
        },
        '1w': {
            'coefficients': model_1w.coef_.tolist(),
            'intercept': model_1w.intercept_,
            'features': ['1-day price change (%)', '2-day price change (%)', '3-day price change (%)', '4-day price change (%)', '5-day price change (%)', '6-day price change (%)', '7-day price change (%)'],
            'target': '1-week price change (%)',
            'mae': mae_1w,
            'rmse': rmse_1w,
            'test_set_size': len(y_1w_test),
            'prediction_distances': prediction_distances_1w
        },
        '1d_nn': {
            'features': ['1-hour price change (%)', '2-hour price change (%)', '3-hour price change (%)', '4-hour price change (%)', '5-hour price change (%)', '6-hour price change (%)', '7-hour price change (%)', '8-hour price change (%)', '9-hour price change (%)', '10-hour price change (%)', '11-hour price change (%)', '12-hour price change (%)', '13-hour price change (%)', '14-hour price change (%)', '15-hour price change (%)', '16-hour price change (%)', '17-hour price change (%)', '18-hour price change (%)', '19-hour price change (%)', '20-hour price change (%)', '21-hour price change (%)', '22-hour price change (%)', '23-hour price change (%)', '24-hour price change (%)'],
            'target': '1-day price change (%)',
            'mae': mae_1d_nn,
            'rmse': rmse_1d_nn,
            'test_set_size': len(y_1d_test),
            'prediction_distances': prediction_distances_1d_nn
        },
        '1w_nn': {
            'features': ['1-day price change (%)', '2-day price change (%)', '3-day price change (%)', '4-day price change (%)', '5-day price change (%)', '6-day price change (%)', '7-day price change (%)'],
            'target': '1-week price change (%)',
            'mae': mae_1w_nn,
            'rmse': rmse_1w_nn,
            'test_set_size': len(y_1w_test),
            'prediction_distances': prediction_distances_1w_nn
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