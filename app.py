import os
import pandas as pd
import matplotlib.pyplot as plt
import gdown
from flask import Flask, render_template_string
import io
import base64

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

# Function to generate plot as base64 image
def generate_plot(data, title):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['close'], label='Close Price')
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
        plots[tf_name] = generate_plot(data, f'BTC OHLCV - {tf_name}')
    
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>BTC OHLCV Timeframes</title>
    </head>
    <body>
        <h1>BTC OHLCV Data - Various Timeframes</h1>
        {% for tf, img in plots.items() %}
            <h2>{{ tf }}</h2>
            <img src="data:image/png;base64,{{ img }}" alt="{{ tf }} Plot">
        {% endfor %}
    </body>
    </html>
    '''
    return render_template_string(html_template, plots=plots)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)