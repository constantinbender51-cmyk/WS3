import os
import gdown
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template_string
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

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
    
    # Train logistic regression model
    # Prepare features: close from day i-1, close from day i-2, |high-low| from day i-1, 7-day SMA from day i-1, 28-day SMA from day i-1
    df_model = df.copy()
    df_model['prev_close_1'] = df_model['close'].shift(1)
    df_model['prev_close_2'] = df_model['close'].shift(2)
    df_model['high_low_range'] = (df_model['high'] - df_model['low']).shift(1)
    df_model['sma_7'] = df_model['close'].rolling(window=7).mean().shift(1)
    df_model['sma_28'] = df_model['close'].rolling(window=28).mean().shift(1)
    
    # Filter data where sma_position is exactly 0 and exclude first 365 days
    start_date = df_model['datetime'].iloc[0] + pd.Timedelta(days=365)
    filtered_data = df_model[(df_model['sma_position'] == 0) & 
                            (df_model['datetime'] >= start_date) &
                            (~df_model['prev_close_1'].isna()) & 
                            (~df_model['prev_close_2'].isna()) &
                            (~df_model['high_low_range'].isna()) &
                            (~df_model['sma_7'].isna()) &
                            (~df_model['sma_28'].isna())].copy()
    
    # Prepare features and target (binary: 1 if close > prev_close, else 0)
    filtered_data['target'] = (filtered_data['close'] > filtered_data['close'].shift(1)).astype(int)
    X = filtered_data[['prev_close_1', 'prev_close_2', 'high_low_range', 'sma_7', 'sma_28']]
    y = filtered_data['target']
    
    # 60-40 train-test split
    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, shuffle=False)
        
        # Train logistic regression model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions (probabilities for class 1)
        train_predictions_proba = model.predict_proba(X_train)[:, 1]
        test_predictions_proba = model.predict_proba(X_test)[:, 1]
        train_predictions_binary = (train_predictions_proba >= 0.5).astype(int)
        test_predictions_binary = (test_predictions_proba >= 0.5).astype(int)
        
        # Store predictions back in dataframe
        df_model['train_pred'] = np.nan
        df_model['test_pred'] = np.nan
        df_model['train_color'] = 'gray'
        df_model['test_color'] = 'gray'
        
        train_indices = X_train.index
        test_indices = X_test.index
        
        df_model.loc[train_indices, 'train_pred'] = train_predictions_binary
        df_model.loc[test_indices, 'test_pred'] = test_predictions_binary
        
        # Color coding: green if prediction is 1 (above previous close), red if 0 (below)
        df_model.loc[train_indices, 'train_color'] = np.where(train_predictions_binary == 1, 'green', 'red')
        df_model.loc[test_indices, 'test_color'] = np.where(test_predictions_binary == 1, 'green', 'red')
    else:
        df_model['train_pred'] = np.nan
        df_model['test_pred'] = np.nan
        df_model['train_color'] = 'gray'
        df_model['test_color'] = 'gray'
    
    # Create subplots
    fig = make_subplots(
        rows=6, cols=1,
        subplot_titles=('Price (Close)', 'SMA Position', 'Model Output', 'Capital 1 (SMA-based)', 'Capital 2 (Model-based)', 'Logistic Regression Predictions'),
        vertical_spacing=0.08
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
    
    # Add capital 1 trace (SMA-based)
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['capital1'], mode='lines', name='Capital 1 (SMA)', line=dict(color='blue')),
        row=4, col=1
    )
    
    # Add capital 2 trace (Model-based)
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['capital2'], mode='lines', name='Capital 2 (Model)', line=dict(color='red')),
        row=5, col=1
    )
    
    # Add logistic regression predictions with color coding
    fig.add_trace(
        go.Scatter(x=df_model['datetime'], y=df_model['train_pred'], mode='markers', name='Train Predictions', marker=dict(color=df_model['train_color'], size=4)),
        row=6, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_model['datetime'], y=df_model['test_pred'], mode='markers', name='Test Predictions', marker=dict(color=df_model['test_color'], size=4)),
        row=6, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['close'], mode='lines', name='Actual Close', line=dict(color='blue', width=1)),
        row=6, col=1
    )
    
    # Update layout
    fig.update_layout(height=1400, title_text="Financial Data Visualization")
    fig.update_xaxes(title_text="Datetime", row=6, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="SMA Position", row=2, col=1)
    fig.update_yaxes(title_text="Model Output", row=3, col=1)
    fig.update_yaxes(title_text="Capital 1", row=4, col=1)
    fig.update_yaxes(title_text="Capital 2", row=5, col=1)
    fig.update_yaxes(title_text="Price", row=6, col=1)
    
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