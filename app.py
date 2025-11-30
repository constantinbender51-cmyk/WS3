import os
import gdown
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template_string

app = Flask(__name__)

# Download the CSV file if not exists
file_id = '1urEZZhBwT4df6-IDYq1eK-wlGWYz8nfs'
url = f'https://drive.google.com/uc?id={file_id}'
filename = 'data.csv'

if not os.path.exists(filename):
    try:
        gdown.download(url, filename, quiet=False)
        print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Error downloading file: {e}")

# Load and preprocess data
try:
    df = pd.read_csv(filename)
    # Ensure datetime column is parsed
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        raise KeyError("CSV must contain a 'datetime' column")
    # Check for required columns
    required_cols = ['datetime', 'open', 'high', 'low', 'close', 'sma_position', 'model_output']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"CSV must contain a '{col}' column")
    # Compute capital series
    if not df.empty:
        # Capital 1 based on SMA position
        capital1 = [1.0]  # Start with initial capital of 1
        for i in range(1, len(df)):
            prev_price = df['close'].iloc[i-1]
            curr_price = df['close'].iloc[i]
            position = df['sma_position'].iloc[i]  # Position is SMA position
            # Handle division by zero or missing data
            if prev_price == 0 or pd.isna(prev_price) or pd.isna(curr_price) or pd.isna(position):
                capital1.append(capital1[-1])  # Keep previous capital if invalid
            else:
                change = (curr_price - prev_price) / prev_price
                new_capital = capital1[-1] * (1 + position * change)
                capital1.append(new_capital)
        df['capital1'] = capital1
        
        # Capital 2 based on model output
        capital2 = [1.0]  # Start with initial capital of 1
        for i in range(1, len(df)):
            prev_price = df['close'].iloc[i-1]
            curr_price = df['close'].iloc[i]
            position = df['model_output'].iloc[i]  # Position is model output
            # Handle division by zero or missing data
            if prev_price == 0 or pd.isna(prev_price) or pd.isna(curr_price) or pd.isna(position):
                capital2.append(capital2[-1])  # Keep previous capital if invalid
            else:
                change = (curr_price - prev_price) / prev_price
                new_capital = capital2[-1] * (1 + position * change)
                capital2.append(new_capital)
        df['capital2'] = capital2
    else:
        df['capital1'] = []
        df['capital2'] = []
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame()  # Empty DataFrame as fallback

@app.route('/')
def index():
    if df.empty:
        return "Error: Data not available. Check server logs for details."
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Price (Close)', 'SMA Position', 'Model Output', 'Capital'),
        vertical_spacing=0.1
    )
    
    # Add price trace (using close column)
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['close'], mode='lines', name='Price'),
        row=1, col=1
    )
    
    # Add SMA position trace
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['sma_position'], mode='lines', name='SMA Position'),
        row=2, col=1
    )
    
    # Add model output trace
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['model_output'], mode='lines', name='Model Output'),
        row=3, col=1
    )
    
    # Add capital traces
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['capital1'], mode='lines', name='Capital 1 (SMA)'),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['capital2'], mode='lines', name='Capital 2 (Model)'),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(height=1000, title_text="Financial Data Visualization")
    fig.update_xaxes(title_text="Datetime", row=4, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="SMA Position", row=2, col=1)
    fig.update_yaxes(title_text="Model Output", row=3, col=1)
    fig.update_yaxes(title_text="Capital", row=4, col=1)
    
    # Convert plot to HTML
    plot_html = fig.to_html(include_plotlyjs='cdn')
    
    # HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Financial Data Dashboard</title>
    </head>
    <body>
        <h1>Financial Data Visualization</h1>
        {{ plot|safe }}
    </body>
    </html>
    """
    
    return render_template_string(html_template, plot=plot_html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)