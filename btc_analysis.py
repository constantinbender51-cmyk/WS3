import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
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

def prepare_data(df, lookback_days=30, forecast_days=8):
    """Prepare data for LSTM model using log returns and volume as features to predict ATR."""
    # Calculate ATR and add to dataframe
    df['atr'] = calculate_atr(df)
    
    # Use high-low range and its moving averages as features
    df['high_low_feature'] = df['high'] - df['low']
    df['sma_7_high_low'] = df['high_low_feature'].rolling(window=7).mean()
    df['sma_29_high_low'] = df['high_low_feature'].rolling(window=29).mean()
    df['sma_60_high_low'] = df['high_low_feature'].rolling(window=60).mean()
    
    feature_columns = ['high_low_feature', 'sma_7_high_low', 'sma_29_high_low', 'sma_60_high_low']
    target_column = 'atr'
    
    # Remove rows with NaN values (from ATR calculation and log derivatives)
    df_clean = df[feature_columns + [target_column]].dropna()
    
    # Scale features and target separately
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    features_scaled = feature_scaler.fit_transform(df_clean[feature_columns])
    target_scaled = target_scaler.fit_transform(df_clean[[target_column]])
    
    X, y = [], []
    for i in range(lookback_days, len(features_scaled) - forecast_days + 1):
        X.append(features_scaled[i-lookback_days:i, :])  # Use high_low_feature only
        y.append(target_scaled[i+forecast_days-1])  # Predict ATR at the end of forecast period
    
    X = np.array(X)
    y = np.array(y).reshape(-1)
    
    return X, y, feature_scaler, target_scaler

def build_lstm_model(input_shape, units=200):
    """Build and compile LSTM model."""
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units, return_sequences=True))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X_train, y_train, epochs=100, batch_size=32, units=200):
    """Train the LSTM model."""
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), units=units)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1, callbacks=[early_stopping])
    return model, history

def predict_future(model, last_sequence, feature_scaler, target_scaler):
    """Predict future ATR values."""
    last_sequence_scaled = feature_scaler.transform(last_sequence)
    prediction_scaled = model.predict(last_sequence_scaled.reshape(1, last_sequence_scaled.shape[0], last_sequence_scaled.shape[1]))
    # Inverse transform for ATR
    prediction = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]
    return prediction

def create_combined_plot(train_dates, train_actual, train_predicted, test_dates, test_actual, test_predicted, history, train_baseline=None, test_baseline=None):
    """Create a combined plot with ATR predictions and loss over epochs."""
    plt.figure(figsize=(14, 12))
    
    # Subplot 1: ATR training phase
    plt.subplot(3, 1, 1)
    plt.plot(train_dates, train_actual, label='Actual ATR', color='blue')
    plt.plot(train_dates, train_predicted, label='Predicted ATR', color='red', linestyle='--')
    if train_baseline is not None:
        plt.plot(train_dates, train_baseline, label='Baseline (Shifted ATR)', color='green', linestyle=':')
    plt.title('Training Phase: Actual vs Predicted ATR')
    plt.xlabel('Date')
    plt.ylabel('ATR')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: ATR testing phase
    plt.subplot(3, 1, 2)
    plt.plot(test_dates, test_actual, label='Actual ATR', color='blue')
    plt.plot(test_dates, test_predicted, label='Predicted ATR', color='red', linestyle='--')
    if test_baseline is not None:
        plt.plot(test_dates, test_baseline, label='Baseline (Shifted ATR)', color='green', linestyle=':')
    plt.title('Testing Phase: Actual vs Predicted ATR')
    plt.xlabel('Date')
    plt.ylabel('ATR')
    plt.legend()
    plt.grid(True)
    
    # Subplot 3: Loss over epochs
    plt.subplot(3, 1, 3)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Training Loss vs Validation Loss Over Epochs (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.yscale('log')
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
    plt.title('Training Loss vs Validation Loss Over Epochs (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.yscale('log')
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
    
    # Prepare data for model using OHLCV features to predict ATR
    X, y, feature_scaler, target_scaler = prepare_data(df, lookback_days=30, forecast_days=8)
    
    # Split data (simple split for demonstration)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model with 16 units
    units = 16
    print(f"Training model with {units} units...")
    model, history = train_model(X_train, y_train, units=units)
    
    # Predict on training set
    y_train_pred_scaled = model.predict(X_train)
    y_train_pred = target_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_train_actual = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    
    # Predict on test set
    y_test_pred_scaled = model.predict(X_test)
    y_test_pred = target_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate accuracy metrics for evaluation
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    train_r2 = r2_score(y_train_actual, y_train_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    test_mae = mean_absolute_error(y_test_actual, y_test_pred)
    test_r2 = r2_score(y_test_actual, y_test_pred)
    
    # Calculate baseline metrics using shifted ATR (shift by 8 days)
    # Align shifted ATR with predictions (shift ATR series by forecast_days)
    forecast_days = 8
    atr_shifted = df['atr'].shift(-forecast_days).dropna()
    # Get indices for training and testing sets
    train_shifted = atr_shifted.iloc[train_start_idx:train_end_idx]
    test_shifted = atr_shifted.iloc[test_start_idx:test_end_idx]
    # Calculate baseline metrics
    train_baseline_rmse = np.sqrt(mean_squared_error(y_train_actual, train_shifted))
    train_baseline_mae = mean_absolute_error(y_train_actual, train_shifted)
    train_baseline_r2 = r2_score(y_train_actual, train_shifted)
    test_baseline_rmse = np.sqrt(mean_squared_error(y_test_actual, test_shifted))
    test_baseline_mae = mean_absolute_error(y_test_actual, test_shifted)
    test_baseline_r2 = r2_score(y_test_actual, test_shifted)
    
    print(f"\nModel Metrics (with {units} units, predicting ATR 8 days ahead):")
    print(f"Training RMSE: {train_rmse}")
    print(f"Training MAE: {train_mae}")
    print(f"Training R²: {train_r2}")
    print(f"Test RMSE: {test_rmse}")
    print(f"Test MAE: {test_mae}")
    print(f"Test R²: {test_r2}")
    print(f"\nBaseline Metrics (shifted ATR by 8 days):")
    print(f"Training RMSE: {train_baseline_rmse}")
    print(f"Training MAE: {train_baseline_mae}")
    print(f"Training R²: {train_baseline_r2}")
    print(f"Test RMSE: {test_baseline_rmse}")
    print(f"Test MAE: {test_baseline_mae}")
    print(f"Test R²: {test_baseline_r2}")
    
    # Prepare data for plotting (align dates)
    # Adjust indices to account for lookback_days and forecast_days in prepare_data
    lookback_days = 30
    forecast_days = 8
    train_start_idx = lookback_days + forecast_days - 1  # Start index for training set in original data
    train_end_idx = train_start_idx + len(y_train_actual)
    test_start_idx = lookback_days + forecast_days - 1 + split  # Start index for test set in original data
    test_end_idx = test_start_idx + len(y_test_actual)
    train_dates = df.index[train_start_idx:train_end_idx]
    test_dates = df.index[test_start_idx:test_end_idx]
    
    # Set up Flask app
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        buf = create_combined_plot(train_dates, y_train_actual, y_train_pred, test_dates, y_test_actual, y_test_pred, history, train_baseline=train_shifted, test_baseline=test_shifted)
        return send_file(buf, mimetype='image/png')
    
    @app.route('/loss')
    def loss_plot():
        buf = create_loss_plot(history)
        return send_file(buf, mimetype='image/png')
    

    
    # Run the server
    app.run(host='0.0.0.0', port=8080, debug=False)