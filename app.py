import pandas as pd
import gdown
import os
from flask import Flask, render_template_string
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

app = Flask(__name__)

# Download the file from Google Drive
file_id = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
url = f'https://drive.google.com/uc?id={file_id}'
output_file = 'downloaded_data.csv'

try:
    gdown.download(url, output_file, quiet=False)
    print(f"File downloaded successfully as {output_file}")
except Exception as e:
    print(f"Error downloading file: {e}")
    exit(1)

# Load the data
try:
    df = pd.read_csv(output_file)
    # Assuming the CSV has columns for datetime and OHLCV; adjust if needed
    # Convert datetime column to datetime type if it exists
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        # If no datetime column, assume first column is datetime
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
    
    # Set datetime as index for resampling
    df.set_index('datetime', inplace=True)
    
    # Resample 1-minute data to daily OHLCV
    daily_df = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Filter dates from 2022 onwards
    daily_df = daily_df[daily_df.index >= '2022-01-01']
    
    # Add a binary column based on specified date ranges
    # Initialize the column with 0
    daily_df['binary_column'] = 0
    
    # Define the date ranges for setting to 1
    ranges = [
        ('2022-12-07', '2024-03-12'),
        ('2024-09-06', '2024-12-17'),
        ('2025-04-06', '2025-10-05')
    ]
    
    for start, end in ranges:
        daily_df.loc[start:end, 'binary_column'] = 1
    
    # Reset index to have datetime as a column for plotting
    daily_df.reset_index(inplace=True)
    
    # Prepare features and target for logistic regression
    # Use previous OHLCV data with lookback of 30 days
    lookback = 30
    features = []
    targets = []
    dates = []
    
    for i in range(lookback, len(daily_df)):
        # Extract OHLCV values for the past 'lookback' days
        feature_row = []
        for j in range(lookback):
            idx = i - lookback + j
            feature_row.extend([
                daily_df.iloc[idx]['open'],
                daily_df.iloc[idx]['high'],
                daily_df.iloc[idx]['low'],
                daily_df.iloc[idx]['close'],
                daily_df.iloc[idx]['volume']
            ])
        features.append(feature_row)
        targets.append(daily_df.iloc[i]['binary_column'])
        dates.append(daily_df.iloc[i]['datetime'])
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(targets)
    dates_array = np.array(dates)
    
    # Split data into 50-50 train-test split
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates_array, test_size=0.5, random_state=42, shuffle=False
    )
    
    # Train logistic regression model
    model = LogisticRegression(random_state=42, max_iter=400)
    model.fit(X_train, y_train)
    
    # Make predictions on the entire dataset (features from lookback onwards)
    y_pred_all = model.predict(X)
    
    # Create a DataFrame for predictions aligned with dates
    pred_df = pd.DataFrame({
        'datetime': dates_array,
        'predicted_binary': y_pred_all
    })
    
    print("Data processed and model trained successfully")
    
except Exception as e:
    print(f"Error processing data: {e}")
    exit(1)

@app.route('/')
def plot_data():
    # Create a plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot close price on the primary y-axis
    ax1.plot(daily_df['datetime'], daily_df['close'], label='Close Price', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)
    
    # Create a secondary y-axis for the binary column and predictions
    ax2 = ax1.twinx()
    ax2.plot(daily_df['datetime'], daily_df['binary_column'], label='Actual Binary', color='red', linestyle='--')
    ax2.plot(pred_df['datetime'], pred_df['predicted_binary'], label='Predicted Binary', color='green', linestyle='-.')
    ax2.set_ylabel('Binary Values', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(-0.1, 1.1)  # Set y-axis limits for binary values
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Daily OHLCV Data from 2022 with Actual and Predicted Binary Column')
    
    # Save plot to a bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    # HTML template to display the plot and download link
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>OHLCV Data Plot with Predictions</title>
    </head>
    <body>
        <h1>Daily OHLCV Data from 2022 with Actual and Predicted Binary Column</h1>
        <img src="data:image/png;base64,{{ plot_url }}" alt="Plot">
        <p>Data includes a binary column with values set based on specified date ranges and logistic regression predictions using 30-day OHLCV lookback.</p>
        <p><a href="/download">Download the processed dataset as CSV</a></p>
    </body>
    </html>
    '''
    return render_template_string(html_template, plot_url=plot_url)

@app.route('/download')
def download_data():
    # Convert the DataFrame to CSV and serve as a downloadable file
    csv_data = daily_df.to_csv(index=False)
    return csv_data, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': 'attachment; filename=daily_ohlcv_data.csv'
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)