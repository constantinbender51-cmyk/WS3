import os
import pandas as pd
import matplotlib.pyplot as plt
import gdown
from flask import Flask, render_template_string
import io
import base64
from sklearn.linear_model import LinearRegression
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
data_1d = resampled_data['1d'].copy()
data_1w = resampled_data['1w'].copy()

# Calculate features and targets for 1-day prediction
# Feature: price change over last 24 hours (percent) - using data up to t-1
data_1d['price_change_24h_pct'] = data_1d['close'].shift(1).pct_change(periods=1) * 100
# Target: price change over next 1 day (percent)
data_1d['target_1d_pct'] = data_1d['close'].pct_change(periods=-1) * 100

# Calculate features and targets for 1-week prediction
# Feature: price change over last 7 days (percent) - using data up to t-1
data_1w['price_change_7d_pct'] = data_1w['close'].shift(1).pct_change(periods=1) * 100
# Target: price change over next 1 week (percent)
data_1w['target_1w_pct'] = data_1w['close'].pct_change(periods=-1) * 100

# Drop rows with NaN values (due to pct_change calculations)
data_1d_clean = data_1d.dropna(subset=['price_change_24h_pct', 'target_1d_pct'])
data_1w_clean = data_1w.dropna(subset=['price_change_7d_pct', 'target_1w_pct'])

# Prepare features and targets for modeling
X_1d = data_1d_clean[['price_change_24h_pct']]
y_1d = data_1d_clean['target_1d_pct']
X_1w = data_1w_clean[['price_change_7d_pct']]
y_1w = data_1w_clean['target_1w_pct']

# Split data into train and test sets (using 80% train, 20% test)
X_1d_train, X_1d_test, y_1d_train, y_1d_test = train_test_split(X_1d, y_1d, test_size=0.2, random_state=42)
X_1w_train, X_1w_test, y_1w_train, y_1w_test = train_test_split(X_1w, y_1w, test_size=0.2, random_state=42)

# Train linear regression models
model_1d = LinearRegression()
model_1d.fit(X_1d_train, y_1d_train)

model_1w = LinearRegression()
model_1w.fit(X_1w_train, y_1w_train)

# Function to generate plot as base64 image
def generate_plot(data, title, model=None, feature_name=None):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['close'], label='Close Price')
    
    # Add prediction line if model and feature are provided
    if model is not None and feature_name is not None:
        # Calculate feature values for all data points using data up to t-1
        if feature_name == 'price_change_24h_pct':
            feature_values = data['close'].shift(1).pct_change(periods=1) * 100
        elif feature_name == 'price_change_7d_pct':
            feature_values = data['close'].shift(1).pct_change(periods=1) * 100
        else:
            feature_values = pd.Series([0] * len(data), index=data.index)
        
        # Predict percentage changes for all data points with non-NaN features
        valid_indices = feature_values.dropna().index
        if len(valid_indices) > 0:
            predicted_pct_changes = model.predict(feature_values.loc[valid_indices].values.reshape(-1, 1))
            # Calculate predicted prices
            predicted_prices = data.loc[valid_indices, 'close'] * (1 + predicted_pct_changes / 100)
            # Plot the predicted prices as a line
            plt.plot(valid_indices, predicted_prices, color='red', linestyle='--', label='Predicted Price')
    
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
            plots[tf_name] = generate_plot(data, f'BTC OHLCV - {tf_name} with Prediction', model=model_1d, feature_name='price_change_24h_pct')
        elif tf_name == '1w':
            plots[tf_name] = generate_plot(data, f'BTC OHLCV - {tf_name} with Prediction', model=model_1w, feature_name='price_change_7d_pct')
        else:
            plots[tf_name] = generate_plot(data, f'BTC OHLCV - {tf_name}')
    
    # Prepare model info for display
    model_info = {
        '1d': {
            'coefficient': model_1d.coef_[0],
            'intercept': model_1d.intercept_,
            'feature': '24-hour price change (%)',
            'target': '1-day price change (%)',
            'mae': mae_1d,
            'rmse': rmse_1d,
            'test_set_size': len(y_1d_test)
        },
        '1w': {
            'coefficient': model_1w.coef_[0],
            'intercept': model_1w.intercept_,
            'feature': '7-day price change (%)',
            'target': '1-week price change (%)',
            'mae': mae_1w,
            'rmse': rmse_1w,
            'test_set_size': len(y_1w_test)
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
        {% for model_name, info in model_info.items() %}
            <h2>Model for {{ model_name }} Prediction</h2>
            <p>Feature: {{ info.feature }}</p>
            <p>Target: {{ info.target }}</p>
            <p>Coefficient: {{ "%.6f"|format(info.coefficient) }}</p>
            <p>Intercept: {{ "%.6f"|format(info.intercept) }}</p>
            <p>Equation: Target = {{ "%.6f"|format(info.coefficient) }} * Feature + {{ "%.6f"|format(info.intercept) }}</p>
            <p>Mean Absolute Error (MAE): {{ "%.4f"|format(info.mae) }}%</p>
            <p>Root Mean Squared Error (RMSE): {{ "%.4f"|format(info.rmse) }}%</p>
            <p>Test Set Size: {{ info.test_set_size }} samples</p>
        {% endfor %}
        
        <h1>Prediction Distance Analysis</h1>
        <h2>1-Day Model Prediction Distances</h2>
        <p>Average distance between predicted and actual: {{ "%.4f"|format(model_info['1d'].mae) }}%</p>
        <p>Maximum distance: {{ "%.4f"|format(prediction_distances_1d.max()) }}%</p>
        <p>Minimum distance: {{ "%.4f"|format(prediction_distances_1d.min()) }}%</p>
        
        <h2>1-Week Model Prediction Distances</h2>
        <p>Average distance between predicted and actual: {{ "%.4f"|format(model_info['1w'].mae) }}%</p>
        <p>Maximum distance: {{ "%.4f"|format(prediction_distances_1w.max()) }}%</p>
        <p>Minimum distance: {{ "%.4f"|format(prediction_distances_1w.min()) }}%</p>
    </body>
    </html>
    '''
    return render_template_string(html_template, plots=plots, model_info=model_info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)