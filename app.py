import gdown
import pandas as pd
import numpy as np
from flask import Flask, render_template_string
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import os

# Download the file from Google Drive
file_id = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'data.csv'
gdown.download(url, output, quiet=False)

# Load the data
data = pd.read_csv(output)

# Assume the data has columns: timestamp, open, high, low, close, volume
# Convert timestamp to datetime and set as index
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Resample to daily OHLCV
daily_data = data.resample('D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# Calculate 365-day SMA
daily_data['sma_365'] = daily_data['close'].rolling(window=365).mean()

# Define target: next day movement (1 for up, 0 for flat, -1 for down)
daily_data['next_close'] = daily_data['close'].shift(-1)
daily_data['movement'] = np.where(daily_data['next_close'] > daily_data['close'], 1, 
                                  np.where(daily_data['next_close'] < daily_data['close'], -1, 0))

# Drop rows with NaN values (due to SMA and shift)
daily_data = daily_data.dropna()

# Prepare features and target
features = daily_data[['sma_365']].values
target = daily_data['movement'].values

# Normalize features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Reshape for LSTM (samples, time steps, features)
# Using a single time step for simplicity
features_reshaped = features_scaled.reshape((features_scaled.shape[0], 1, features_scaled.shape[1]))

# Split data into train and test (using 80-20 split)
train_size = int(len(features_reshaped) * 0.8)
X_train, X_test = features_reshaped[:train_size], features_reshaped[train_size:]
y_train, y_test = target[:train_size], target[train_size:]

# Build LSTM model
model = Sequential([
    Input(shape=(1, 1)),
    LSTM(50, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: up, flat, down
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train + 1, epochs=100, batch_size=32, validation_data=(X_test, y_test + 1), verbose=1)  # Adjust labels to 0,1,2

# Predict on the entire dataset
predictions = model.predict(features_reshaped)
predicted_classes = np.argmax(predictions, axis=1) - 1  # Convert back to -1,0,1

# Calculate strategy returns
initial_capital = 10000  # Starting capital
capital = [initial_capital] * len(daily_data)  # Initialize capital list with same length
position = 0  # 0 for no position, 1 for long
for i in range(1, len(daily_data)):
    if predicted_classes[i-1] == 1:  # Predicted up
        if position == 0:
            # Buy at open
            shares = capital[i-1] / daily_data['open'].iloc[i]
            position = shares
            capital[i] = capital[i-1]  # Capital unchanged when buying
        else:
            capital[i] = capital[i-1]  # Hold position
    elif predicted_classes[i-1] == -1:  # Predicted down
        if position > 0:
            # Sell at open
            capital[i] = position * daily_data['open'].iloc[i]
            position = 0
        else:
            capital[i] = capital[i-1]  # No position
    else:  # Predicted flat
        if position > 0:
            # Sell at open
            capital[i] = position * daily_data['open'].iloc[i]
            position = 0
        else:
            capital[i] = capital[i-1]  # No position

# If position is held at the end, sell at last close
if position > 0:
    capital[-1] = position * daily_data['close'].iloc[-1]

# Prepare data for the web server
daily_data['prediction'] = predicted_classes
daily_data['actual'] = daily_data['movement']
daily_data['capital'] = capital

# Start Flask app
app = Flask(__name__)

@app.route('/')
def index():
    # Create HTML content to display BTC price with prediction-based background colors
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BTC Price with Prediction Background</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>BTC Price with Prediction-Based Background Colors</h1>
        <div id="price-plot"></div>
        <div id="capital-plot"></div>
        <script>
            var priceData = [
                {
                    x: {{ dates | safe }},
                    y: {{ prices | safe }},
                    type: 'scatter',
                    mode: 'lines',
                    name: 'BTC Close Price',
                    line: {color: 'black', width: 2}
                },
                {
                    x: {{ dates | safe }},
                    y: {{ sma_365 | safe }},
                    type: 'scatter',
                    mode: 'lines',
                    name: 'SMA 365',
                    line: {color: 'blue', width: 1}
                }
            ];
            
            // Add background shapes based on predictions
            var shapes = [];
            var predictions = {{ predictions | safe }};
            var dates = {{ dates | safe }};
            
            for (var i = 0; i < predictions.length; i++) {
                var color;
                if (predictions[i] === 1) {
                    color = 'rgba(0, 255, 0, 0.2)';  // Green for up prediction
                } else if (predictions[i] === -1) {
                    color = 'rgba(255, 0, 0, 0.2)';  // Red for down prediction
                } else {
                    color = 'rgba(255, 255, 0, 0.2)';  // Yellow for flat prediction
                }
                
                shapes.push({
                    type: 'rect',
                    xref: 'x',
                    yref: 'paper',
                    x0: dates[i],
                    x1: i < predictions.length - 1 ? dates[i + 1] : dates[i],
                    y0: 0,
                    y1: 1,
                    fillcolor: color,
                    line: {width: 0},
                    layer: 'below'
                });
            }
            
            var priceLayout = {
                title: 'BTC Close Price with Prediction Background Colors',
                xaxis: {title: 'Date'},
                yaxis: {title: 'Price ($)'},
                shapes: shapes,
                showlegend: true
            };
            Plotly.newPlot('price-plot', priceData, priceLayout);

            var capitalData = [
                {
                    x: {{ dates | safe }},
                    y: {{ capital | safe }},
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Capital',
                    line: {color: 'green'}
                }
            ];
            var capitalLayout = {
                title: 'Capital Development Over Time',
                xaxis: {title: 'Date'},
                yaxis: {title: 'Capital ($)'}
            };
            Plotly.newPlot('capital-plot', capitalData, capitalLayout);
        </script>
    </body>
    </html>
    """
    
    dates = daily_data.index.strftime('%Y-%m-%d').tolist()
    predictions = daily_data['prediction'].tolist()
    prices = daily_data['close'].tolist()
    sma_365 = daily_data['sma_365'].tolist()
    capital_vals = daily_data['capital'].tolist()
    
    return render_template_string(html_content, dates=dates, predictions=predictions, prices=prices, sma_365=sma_365, capital=capital_vals)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)