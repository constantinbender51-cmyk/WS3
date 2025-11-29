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

def prepare_data(df, lookback_days=14, forecast_days=14):
    """Prepare data for LSTM model using OHLCV features."""
    # Use OHLCV columns as features
    data = df[['open', 'high', 'low', 'close', 'volume']].dropna().values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(lookback_days, len(data_scaled) - forecast_days + 1):
        X.append(data_scaled[i-lookback_days:i, :])  # Use all OHLCV features
        y.append(data_scaled[i+forecast_days-1, 3])  # Predict close price at the end of forecast period
    
    X = np.array(X)
    y = np.array(y)
    
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

def train_model(X_train, y_train, epochs=100, batch_size=32):
    """Train the LSTM model."""
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)
    return model, history

def predict_future(model, last_sequence, scaler):
    """Predict future close price values."""
    last_sequence_scaled = scaler.transform(last_sequence)
    prediction_scaled = model.predict(last_sequence_scaled.reshape(1, last_sequence_scaled.shape[0], last_sequence_scaled.shape[1]))
    # Inverse transform for close price (index 3 in scaled data)
    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[0, 3] = prediction_scaled[0, 0]
    prediction = scaler.inverse_transform(dummy)[0, 3]
    return prediction

def create_combined_plot(train_dates, train_actual, train_predicted, test_dates, test_actual, test_predicted, history, last_month_dates, last_month_actual, last_month_predicted):
    """Create a combined plot with close price predictions and loss over epochs."""
    plt.figure(figsize=(14, 16))
    
    # Subplot 1: Close price training phase
    plt.subplot(4, 1, 1)
    plt.plot(train_dates, train_actual, label='Actual Close Price', color='blue')
    plt.plot(train_dates, train_predicted, label='Predicted Close Price', color='red', linestyle='--')
    plt.title('Training Phase: Actual vs Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Close price testing phase
    plt.subplot(4, 1, 2)
    plt.plot(test_dates, test_actual, label='Actual Close Price', color='blue')
    plt.plot(test_dates, test_predicted, label='Predicted Close Price', color='red', linestyle='--')
    plt.title('Testing Phase: Actual vs Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    
    # Subplot 3: Last month of close price data
    plt.subplot(4, 1, 3)
    plt.plot(last_month_dates, last_month_actual, label='Actual Close Price', color='blue', marker='o', markersize=3)
    plt.plot(last_month_dates, last_month_predicted, label='Predicted Close Price', color='red', linestyle='--', marker='s', markersize=3)
    plt.title('Last Month: Actual vs Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # Subplot 4: Loss over epochs
    plt.subplot(4, 1, 4)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Training Loss vs Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def create_loss_plot(history):
    """Create a plot of training loss vs validation loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Training Loss vs Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
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
    
    # Prepare data for model using OHLCV features
    X, y, scaler = prepare_data(df, forecast_days=1)
    
    # Split data (simple split for demonstration)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model
    model, history = train_model(X_train, y_train)
    
    # Predict on training set
    y_train_pred_scaled = model.predict(X_train)
    y_train_pred = scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    
    # Predict on test set
    y_test_pred_scaled = model.predict(X_test)
    y_test_pred = scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate accuracy metrics for evaluation
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    train_r2 = r2_score(y_train_actual, y_train_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    test_mae = mean_absolute_error(y_test_actual, y_test_pred)
    test_r2 = r2_score(y_test_actual, y_test_pred)
    print(f"Training RMSE: {train_rmse}")
    print(f"Training MAE: {train_mae}")
    print(f"Training R²: {train_r2}")
    print(f"Test RMSE: {test_rmse}")
    print(f"Test MAE: {test_mae}")
    print(f"Test R²: {test_r2}")
    
    # Prepare data for plotting (align dates)
    # Adjust indices to account for lookback_days and forecast_days in prepare_data
    lookback_days = 14
    forecast_days = 1
    train_start_idx = lookback_days + forecast_days - 1  # Start index for training set in original data
    train_end_idx = train_start_idx + len(y_train_actual)
    test_start_idx = lookback_days + forecast_days - 1 + split  # Start index for test set in original data
    test_end_idx = test_start_idx + len(y_test_actual)
    train_dates = df.index[train_start_idx:train_end_idx]
    test_dates = df.index[test_start_idx:test_end_idx]
    
    # Prepare last month data for plotting
    last_month_days = 30
    last_month_start_idx = max(0, len(df) - last_month_days)
    last_month_dates = df.index[last_month_start_idx:]
    last_month_df = df.iloc[last_month_start_idx:]
    
    # Prepare sequences for last month predictions
    last_month_actual = []
    last_month_predicted = []
    for i in range(len(last_month_df) - lookback_days):
        sequence = last_month_df.iloc[i:i+lookback_days][['open', 'high', 'low', 'close', 'volume']].values
        actual_value = last_month_df.iloc[i+lookback_days]['close']
        prediction = predict_future(model, sequence, scaler)
        
        last_month_actual.append(actual_value)
        last_month_predicted.append(prediction)
    
    # Adjust last month dates to match prediction length
    last_month_plot_dates = last_month_dates[lookback_days:lookback_days + len(last_month_actual)]
    
    # Set up Flask app
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        buf = create_combined_plot(train_dates, y_train_actual, y_train_pred, test_dates, y_test_actual, y_test_pred, history, last_month_plot_dates, last_month_actual, last_month_predicted)
        return send_file(buf, mimetype='image/png')
    @app.route('/loss')
    def loss_plot():
        buf = create_loss_plot(history)
        return send_file(buf, mimetype='image/png')
    
    # Run the server
    app.run(host='0.0.0.0', port=8080, debug=False)