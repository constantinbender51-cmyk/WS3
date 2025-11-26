import pandas as pd
import numpy as np
import gdown
from flask import Flask, render_template_string
import matplotlib.pyplot as plt
import io
import base64
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# Download the CSV file from Google Drive
file_id = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'ohlcv_data.csv'
gdown.download(url, output, quiet=False)

# Load the data
df = pd.read_csv(output)

# Ensure the timestamp column is parsed correctly (adjust column name if needed)
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
else:
    # If no timestamp, assume first column is datetime; adjust as per actual data
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df.set_index(df.columns[0], inplace=True)

# Filter data for time period 2021 to September 2025
start_date = '2021-01-01'
end_date = '2025-09-30'
df = df.loc[start_date:end_date]

# Resample to different timeframes
ohlcv_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
df_1h = df.resample('1H').apply(ohlcv_dict).dropna()
df_4h = df.resample('4H').apply(ohlcv_dict).dropna()
df_1d = df.resample('1D').apply(ohlcv_dict).dropna()
df_1w = df.resample('1W').apply(ohlcv_dict).dropna()

# Use daily data for model training and prediction
df_daily = df_1d.copy()

# Calculate SMAs for 7, 29, 365 days
df_daily['sma_7'] = df_daily['close'].rolling(window=7).mean()
df_daily['sma_29'] = df_daily['close'].rolling(window=29).mean()
df_daily['sma_365'] = df_daily['close'].rolling(window=365).mean()

# Calculate Bollinger Bands for 20-day period (standard)
window = 20
df_daily['bb_middle'] = df_daily['close'].rolling(window=window).mean()
df_daily['bb_std'] = df_daily['close'].rolling(window=window).std()
df_daily['bb_upper'] = df_daily['bb_middle'] + 2 * df_daily['bb_std']
df_daily['bb_lower'] = df_daily['bb_middle'] - 2 * df_daily['bb_std']

# Calculate MACD line and signal line for weekly data (using weekly close)
# First, ensure we have weekly data; if not, approximate from daily
df_weekly_for_macd = df_1w[['close']].copy() if not df_1w.empty else df_daily.resample('1W').last()[['close']]
ema_12 = df_weekly_for_macd['close'].ewm(span=12).mean()
ema_26 = df_weekly_for_macd['close'].ewm(span=26).mean()
macd_line = ema_12 - ema_26
signal_line = macd_line.ewm(span=9).mean()
# Align MACD data to daily index by forward filling (since MACD is weekly)
df_daily['macd_line'] = macd_line.reindex(df_daily.index, method='ffill')
df_daily['signal_line'] = signal_line.reindex(df_daily.index, method='ffill')
df_daily['macd_signal_diff'] = df_daily['macd_line'] - df_daily['signal_line']

# Fetch hashrate from blockchain.com API (example endpoint; adjust if needed)
try:
    hashrate_response = requests.get('https://api.blockchain.info/charts/hash-rate?timespan=all&sampled=true&format=json')
    hashrate_data = hashrate_response.json()
    hashrate_df = pd.DataFrame(hashrate_data['values'])
    hashrate_df['x'] = pd.to_datetime(hashrate_df['x'], unit='s')
    hashrate_df.set_index('x', inplace=True)
    hashrate_df.rename(columns={'y': 'hashrate'}, inplace=True)
    # Filter for time period 2021 to September 2025
    hashrate_df = hashrate_df.loc[start_date:end_date]
    # Resample to daily and merge
    hashrate_daily = hashrate_df.resample('1D').mean()
    df_daily = df_daily.merge(hashrate_daily, left_index=True, right_index=True, how='left')
    df_daily['hashrate'].fillna(method='ffill', inplace=True)
except Exception as e:
    print(f"Error fetching hashrate: {e}")
    df_daily['hashrate'] = np.nan

# Fetch active addresses from blockchain.com API (example endpoint; adjust if needed)
try:
    active_addr_response = requests.get('https://api.blockchain.info/charts/n-unique-addresses?timespan=all&sampled=true&format=json')
    active_addr_data = active_addr_response.json()
    active_addr_df = pd.DataFrame(active_addr_data['values'])
    active_addr_df['x'] = pd.to_datetime(active_addr_df['x'], unit='s')
    active_addr_df.set_index('x', inplace=True)
    active_addr_df.rename(columns={'y': 'active_addresses'}, inplace=True)
    # Filter for time period 2021 to September 2025
    active_addr_df = active_addr_df.loc[start_date:end_date]
    # Resample to daily and merge
    active_addr_daily = active_addr_df.resample('1D').mean()
    df_daily = df_daily.merge(active_addr_daily, left_index=True, right_index=True, how='left')
    df_daily['active_addresses'].fillna(method='ffill', inplace=True)
except Exception as e:
    print(f"Error fetching active addresses: {e}")
    df_daily['active_addresses'] = np.nan

# Prepare features and target
features = ['open', 'high', 'low', 'close', 'volume', 'sma_7', 'sma_29', 'sma_365', 'bb_upper', 'bb_lower', 'macd_signal_diff', 'hashrate', 'active_addresses']
df_daily_clean = df_daily[features].dropna()

# Define target: next day price change direction with meaningful threshold
# Use percentage change to avoid simple up/down bias
df_daily_clean['next_close'] = df_daily_clean['close'].shift(-1)
df_daily_clean['price_change_pct'] = (df_daily_clean['next_close'] - df_daily_clean['close']) / df_daily_clean['close']

# Use a threshold to define meaningful up/down movements (e.g., >0.5% change)
threshold = 0.005  # 0.5%
df_daily_clean['target'] = np.where(
    df_daily_clean['price_change_pct'] > threshold, 1,  # Significant up
    np.where(df_daily_clean['price_change_pct'] < -threshold, 0, 2)  # Significant down or neutral
)

# Remove neutral cases for binary classification
df_daily_clean = df_daily_clean[df_daily_clean['target'] != 2]
df_daily_clean = df_daily_clean.dropna()

# Check class distribution
class_counts = df_daily_clean['target'].value_counts()
print(f"Class distribution - Up: {class_counts.get(1, 0)}, Down: {class_counts.get(0, 0)}")
print(f"Class imbalance ratio: {class_counts.get(1, 1) / class_counts.get(0, 1):.2f}")

# Skip Boruta feature selection and use all features
print("Skipping Boruta feature selection - using all available features")
selected_features = features.copy()
print(f"Using all features: {selected_features}")

# Use all features for model training
df_daily_clean_selected = df_daily_clean[selected_features + ['target']]

# Normalize features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_daily_clean_selected[selected_features])

# Prepare sequences for LSTM (using 60 days of history)
sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(scaled_features)):
    X.append(scaled_features[i-sequence_length:i])
    y.append(df_daily_clean_selected['target'].iloc[i])
X, y = np.array(X), np.array(y)

# Split data: 80% train, 20% test
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Build LSTM model with class weighting
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights to handle imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(100, return_sequences=True),
    Dropout(0.3),
    LSTM(100, return_sequences=False),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with class weights
model.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_test, y_test), verbose=1, class_weight=class_weight_dict)

# Make predictions and analyze prediction distribution
pred_probs = model.predict(X_test).flatten()
predictions = (pred_probs > 0.5).astype(int)

# Analyze prediction distribution
print(f"Prediction distribution - Up: {np.sum(predictions == 1)}, Down: {np.sum(predictions == 0)}")
print(f"Prediction probabilities - Mean: {np.mean(pred_probs):.3f}, Std: {np.std(pred_probs):.3f}")

# Calculate accuracy and additional metrics
from sklearn.metrics import classification_report, confusion_matrix

accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=['Down', 'Up']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Prepare data for plotting
test_dates = df_daily_clean_selected.index[split_idx + sequence_length:]
actual_changes = df_daily_clean['close'].pct_change().shift(-1).loc[test_dates]
predicted_directions = predictions

# Calculate capital development (start with 100 units)
capital = 100
capital_history = [capital]
positions = []  # 1 for long, -1 for short, 0 for flat

# Use actual price changes for capital calculation
for i in range(len(predicted_directions)):
    if i < len(actual_changes):
        change = actual_changes.iloc[i] if i < len(actual_changes) else 0
        if predicted_directions[i] == 1:  # Predict up, go long
            capital *= (1 + change)
            positions.append(1)
        elif predicted_directions[i] == 0:  # Predict down, go short
            capital *= (1 - change)
            positions.append(-1)
        else:
            positions.append(0)  # Flat, no change
        capital_history.append(capital)

print(f"Final capital: {capital:.2f}")
print(f"Positions taken - Long: {positions.count(1)}, Short: {positions.count(-1)}, Flat: {positions.count(0)}")

# Prepare data for full time period visualization
train_dates = df_daily_clean_selected.index[sequence_length:split_idx + sequence_length]
full_dates = df_daily_clean_selected.index[sequence_length:]
full_close_prices = df_daily_clean['close'].loc[full_dates]
train_close_prices = df_daily_clean['close'].loc[train_dates]

# Calculate squared returns for volatility prediction
df_daily_clean['squared_returns'] = df_daily_clean['close'].pct_change() ** 2
df_daily_clean['volatility_target'] = df_daily_clean['squared_returns'].shift(-1)  # Predict next day volatility

# Prepare data for volatility LSTM model
df_vol_clean = df_daily_clean[selected_features + ['volatility_target']].dropna()
scaled_features_vol = scaler.transform(df_vol_clean[selected_features])  # Use same scaler

# Prepare sequences for volatility LSTM
X_vol, y_vol = [], []
for i in range(sequence_length, len(scaled_features_vol)):
    X_vol.append(scaled_features_vol[i-sequence_length:i])
    y_vol.append(df_vol_clean['volatility_target'].iloc[i])
X_vol, y_vol = np.array(X_vol), np.array(y_vol)

# Split data for volatility model
split_idx_vol = int(0.8 * len(X_vol))
X_train_vol, X_test_vol = X_vol[:split_idx_vol], X_vol[split_idx_vol:]
y_train_vol, y_test_vol = y_vol[:split_idx_vol], y_vol[split_idx_vol:]

# Build and train volatility LSTM model
model_vol = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train_vol.shape[1], X_train_vol.shape[2])),
    Dropout(0.3),
    LSTM(100, return_sequences=True),
    Dropout(0.3),
    LSTM(100, return_sequences=False),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1, activation='linear')  # Linear for regression
])
model_vol.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model_vol.fit(X_train_vol, y_train_vol, batch_size=32, epochs=30, validation_data=(X_test_vol, y_test_vol), verbose=1)

# Make volatility predictions
vol_preds = model_vol.predict(X_test_vol).flatten()
vol_test_dates = df_vol_clean.index[split_idx_vol + sequence_length:]
actual_vol = df_vol_clean['volatility_target'].loc[vol_test_dates]

# Start Flask web server
app = Flask(__name__)

@app.route('/')
def index():
    # Plot 1: Full Time Period Price Data
    plt.figure(figsize=(14, 8))
    plt.plot(full_dates, full_close_prices, label='Close Price', color='blue', alpha=0.7)
    plt.axvline(x=train_dates[-1], color='red', linestyle='--', label='Train/Test Split')
    plt.fill_betweenx(y=[full_close_prices.min(), full_close_prices.max()], 
                     x1=full_dates[0], x2=train_dates[-1], alpha=0.2, color='green', label='Training Period')
    plt.fill_betweenx(y=[full_close_prices.min(), full_close_prices.max()], 
                     x1=train_dates[-1], x2=full_dates[-1], alpha=0.2, color='orange', label='Test Period')
    plt.title('Full Time Period: Close Prices with Training/Test Split')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img1 = io.BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)
    plot_url1 = base64.b64encode(img1.getvalue()).decode()
    plt.close()
    
    # Plot 2: Predictions vs Actual Price Changes
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, actual_changes.values, label='Actual Price Changes', alpha=0.7)
    plt.scatter(test_dates, [0.01 if p == 1 else -0.01 for p in predicted_directions], 
                color='red', label='Predicted Direction (Up/Down)', marker='o')
    plt.title('Predictions vs Actual Price Changes (Test Period Only)')
    plt.xlabel('Date')
    plt.ylabel('Price Change')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img2 = io.BytesIO()
    plt.savefig(img2, format='png')
    img2.seek(0)
    plot_url2 = base64.b64encode(img2.getvalue()).decode()
    plt.close()
    
    # Plot 3: Capital Development
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates[:len(capital_history)-1], capital_history[:-1], label='Capital', color='green')
    plt.title('Capital Development Based on Predictions')
    plt.xlabel('Date')
    plt.ylabel('Capital')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img3 = io.BytesIO()
    plt.savefig(img3, format='png')
    img3.seek(0)
    plot_url3 = base64.b64encode(img3.getvalue()).decode()
    plt.close()

    # Plot 4: Volatility Predictions
    plt.figure(figsize=(12, 6))
    plt.plot(vol_test_dates, actual_vol.values, label='Actual Volatility (Squared Returns)', alpha=0.7)
    plt.plot(vol_test_dates, vol_preds, label='Predicted Volatility', alpha=0.7)
    plt.title('Volatility Predictions vs Actual (Test Period Only)')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Squared Returns)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img4 = io.BytesIO()
    plt.savefig(img4, format='png')
    img4.seek(0)
    plot_url4 = base64.b64encode(img4.getvalue()).decode()
    plt.close()
    
    # HTML template to display plots
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>LSTM Prediction Results</title>
    </head>
    <body>
        <h1>LSTM Model for Next-Day Price Direction Prediction</h1>
        <p>Model Accuracy: {accuracy:.2f}</p>
        <p>Selected Features: {', '.join(selected_features)}</p>
        <p>Training Period: {train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')}</p>
        <p>Test Period: {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}</p>
        <h2>Full Time Period: Close Prices with Training/Test Split</h2>
        <img src="data:image/png;base64,{plot_url1}" alt="Full Time Period">
        <h2>Predictions vs Actual Price Changes (Test Period Only)</h2>
        <img src="data:image/png;base64,{plot_url2}" alt="Predictions vs Actual">
        <h2>Capital Development</h2>
        <img src="data:image/png;base64,{plot_url3}" alt="Capital Development">
        <h2>Volatility Predictions</h2>
        <img src="data:image/png;base64,{plot_url4}" alt="Volatility Predictions">
    </body>
    </html>
    '''
    return render_template_string(html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)