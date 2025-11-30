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
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame()  # Empty DataFrame as fallback

@app.route('/')
def index():
    if df.empty:
        return "Error: Data not available. Check server logs for details."
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price (Close)', 'SMA Position', 'Model Output'),
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
    
    # Update layout
    fig.update_layout(height=800, title_text="Financial Data Visualization")
    fig.update_xaxes(title_text="Datetime", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="SMA Position", row=2, col=1)
    fig.update_yaxes(title_text="Model Output", row=3, col=1)
    
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