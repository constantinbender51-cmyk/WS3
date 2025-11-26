import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from flask import Flask, render_template_string
import io
import base64

app = Flask(__name__)

# Generate mock price data
def generate_mock_data():
    start_date = datetime(2022, 1, 1)
    end_date = start_date + timedelta(days=730)  # 2 years
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    prices = [1000.0]  # Start at 1000
    for i in range(1, len(dates)):
        date = dates[i]
        prev_price = prices[-1]
        
        # Base daily pattern
        if date.weekday() == 0 or date.weekday() == 1:  # Monday, Tuesday: up
            base_return = 0.002  # 0.2% up
        elif date.weekday() == 3 or date.weekday() == 4:  # Thursday, Friday: down
            base_return = -0.002  # 0.2% down
        else:  # Wednesday, Saturday, Sunday: flat
            base_return = 0.0
        
        # Seasonal adjustment: up more in winter (Dec-Feb), down more in spring (Mar-May)
        month = date.month
        if month in [12, 1, 2]:  # Winter
            seasonal_factor = 1.5
        elif month in [3, 4, 5]:  # Spring
            seasonal_factor = -1.5
        else:
            seasonal_factor = 1.0
        base_return *= seasonal_factor
        
        # Add noise: ±1% daily return
        noise = np.random.uniform(-0.01, 0.01)
        daily_return = base_return + noise
        
        new_price = prev_price * (1 + daily_return)
        prices.append(new_price)
    
    df = pd.DataFrame({'Date': dates, 'Price': prices})
    df['Return'] = df['Price'].pct_change()
    df['Return_Direction'] = (df['Return'] > 0).astype(int)  # 1 for up, 0 for down
    df = df.dropna().reset_index(drop=True)
    return df

# Prepare data for LSTM
def prepare_data(df, lookback=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Return']].values)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(df['Return_Direction'].iloc[i])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, scaler

# Train LSTM model
def train_model(X_train, y_train):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
    return model

# Generate plots
def generate_plots(df, predictions, test_start_idx):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Predictions vs Actual Price with background colors
    dates_all = df['Date']
    prices_all = df['Price']
    
    # Training and testing phases
    train_dates = dates_all[:test_start_idx]
    test_dates = dates_all[test_start_idx:]
    train_prices = prices_all[:test_start_idx]
    test_prices = prices_all[test_start_idx:]
    
    # Background for predictions: blue for predicted up, orange for predicted down
    for i in range(len(predictions)):
        if predictions[i] == 1:
            ax1.axvspan(test_dates.iloc[i], test_dates.iloc[i] + timedelta(days=1), color='blue', alpha=0.3)
        else:
            ax1.axvspan(test_dates.iloc[i], test_dates.iloc[i] + timedelta(days=1), color='orange', alpha=0.3)
    
    ax1.plot(dates_all, prices_all, color='black', label='Actual Price')
    ax1.set_title('Predictions vs Actual Price (Blue: Predicted Up, Orange: Predicted Down)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    
    # Plot 2: Capital development
    capital = [1000]  # Start with 1000
    for i in range(len(predictions)):
        if predictions[i] == 1 and df['Return'].iloc[test_start_idx + i] > 0:
            capital.append(capital[-1] * (1 + df['Return'].iloc[test_start_idx + i]))
        elif predictions[i] == 0 and df['Return'].iloc[test_start_idx + i] < 0:
            capital.append(capital[-1] * (1 - df['Return'].iloc[test_start_idx + i]))
        else:
            capital.append(capital[-1])
    
    ax2.plot(test_dates, capital[:-1], color='green', label='Capital')
    ax2.set_title('Capital Development Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Capital')
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Save plot to string buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return plot_data

@app.route('/')
def index():
    # Generate data
    df = generate_mock_data()
    
    # Split data: first 80% for training, last 20% for testing
    split_idx = int(0.8 * len(df))
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    
    # Prepare training data
    X_train, y_train, scaler = prepare_data(train_df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Prepare testing data
    full_scaled = scaler.transform(df[['Return']].values)
    X_test = []
    lookback = 60
    for i in range(split_idx, len(full_scaled)):
        X_test.append(full_scaled[i-lookback:i, 0])
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Make predictions
    predictions = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    
    # Generate plots
    plot_data = generate_plots(df, predictions, split_idx)
    
    # Calculate accuracy
    y_test = test_df['Return_Direction'].values
    accuracy = accuracy_score(y_test, predictions)
    
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>LSTM Prediction Results</title>
    </head>
    <body>
        <h1>LSTM Model for Return Direction Prediction</h1>
        <p>Accuracy on Test Set: {accuracy:.2%}</p>
        <img src="data:image/png;base64,{plot_data}" alt="Plots">
    </body>
    </html>
    '''
    return render_template_string(html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)