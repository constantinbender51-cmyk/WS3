import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from flask import Flask, send_file
import io
import os

def fetch_binance_data(symbol='BTCUSDT', start_date='2018-01-01', interval='1d'):
    """Fetch OHLCV data from Binance API."""
    base_url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': int(pd.Timestamp(start_date).timestamp() * 1000),
        'limit': 1000
    }
    all_data = []
    while True:
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data: {response.status_code}")
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        params['startTime'] = data[-1][0] + 1
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df[['open', 'high', 'low', 'close', 'volume']]

def calculate_atr(df, period=14):
    """Calculate Average True Range (ATR)."""
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def prepare_data(atr_series, lookback_days=14, forecast_days=14):
    """Prepare data for LSTM model."""
    data = atr_series.dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(lookback_days, len(data_scaled) - forecast_days + 1):
        X.append(data_scaled[i-lookback_days:i, 0])
        y.append(data_scaled[i+forecast_days-1, 0])  # Predict ATR at the end of forecast period
    
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, scaler

def build_lstm_model(input_shape):
    """Build and compile LSTM model."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X_train, y_train, epochs=50, batch_size=32):
    """Train the LSTM model."""
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)
    return model

def predict_future(model, last_sequence, scaler):
    """Predict future ATR values."""
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    prediction_scaled = model.predict(last_sequence_scaled.reshape(1, last_sequence_scaled.shape[0], 1))
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0, 0]

def create_plot(df, atr_actual, atr_predicted):
    """Create a plot of actual vs predicted ATR."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, atr_actual, label='Actual ATR', color='blue')
    plt.plot(df.index[-len(atr_predicted):], atr_predicted, label='Predicted ATR', color='red', linestyle='--')
    plt.title('Actual vs Predicted ATR')
    plt.xlabel('Date')
    plt.ylabel('ATR')
    plt.legend()
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# Main execution
if __name__ == '__main__':
    # Fetch and prepare data
    df = fetch_binance_data()
    atr = calculate_atr(df)
    
    # Prepare data for model
    X, y, scaler = prepare_data(atr)
    
    # Split data (simple split for demonstration)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Predict on test set
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate accuracy metrics for evaluation
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    print(f"Test RMSE: {rmse}")
    print(f"Test MAE: {mae}")
    print(f"Test R²: {r2}")
    
    # Prepare data for plotting (align dates)
    # Adjust indices to account for lookback_days and forecast_days in prepare_data
    lookback_days = 14
    forecast_days = 14
    start_idx = lookback_days + forecast_days - 1 + split  # Start index for test set in original data
    plot_dates = df.index[start_idx:start_idx + len(y_actual)]
    atr_actual_plot = y_actual
    atr_predicted_plot = y_pred
    
    # Set up Flask app
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        buf = create_plot(pd.DataFrame(index=plot_dates), atr_actual_plot, atr_predicted_plot)
        return send_file(buf, mimetype='image/png')
    
    # Run the server
    app.run(host='0.0.0.0', port=8080, debug=False)