import pandas as pd
import gdown
import os
from flask import Flask, render_template_string
import matplotlib.pyplot as plt
import io
import base64

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
    
    print("Data processed successfully")
    
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
    
    # Create a secondary y-axis for the binary column
    ax2 = ax1.twinx()
    ax2.plot(daily_df['datetime'], daily_df['binary_column'], label='Binary Column', color='red', linestyle='--')
    ax2.set_ylabel('Binary Column', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(-0.1, 1.1)  # Set y-axis limits for binary values
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Daily OHLCV Data from 2022 with Binary Column')
    
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
        <title>OHLCV Data Plot</title>
    </head>
    <body>
        <h1>Daily OHLCV Data from 2022 with Binary Column</h1>
        <img src="data:image/png;base64,{{ plot_url }}" alt="Plot">
        <p>Data includes a binary column with values set based on specified date ranges.</p>
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
