import gdown
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string, send_file
import os

app = Flask(__name__)

# Constants
GOOGLE_SHEET_URL = 'https://docs.google.com/spreadsheets/d/1bjfIzg2_I_zN95v5oQac671_4AV0D74x/edit?usp=drivesdk&ouid=114372102418564925207&rtpof=true&sd=true'
LOCAL_CSV_FILENAME = 'ohlcv_data.csv'
FLAT_TRADE_THRESHOLD = 0.02  # 2% threshold

# --- Data Processing Function ---
def process_ohlcv_data():
    try:
        print(f"Attempting to download data from {GOOGLE_SHEET_URL}")
        # Use gdown to download the file
        gdown.download(url=GOOGLE_SHEET_URL, output=LOCAL_CSV_FILENAME, quiet=False, fuzzy=True)
        
        if not os.path.exists(LOCAL_CSV_FILENAME):
            raise Exception(f"Downloaded file '{LOCAL_CSV_FILENAME}' does not exist.")

        df = pd.read_csv(LOCAL_CSV_FILENAME)
        print("Data downloaded and loaded successfully.")

        # Ensure required columns are present
        required_columns = ['datetime', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV must contain a '{col}' column.")
        
        # Convert 'datetime' to datetime and set as index
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        # 1. Calculate daily returns
        df['daily_return'] = df['close'].pct_change()
        
        # 2. Add initial 'perfect_position' column: 1 for positive return, 0 for non-positive
        df['perfect_position'] = (df['daily_return'] > 0).astype(int)
        
        # 3. Calculate trade returns and identify flat positions
        df['trade_group'] = (df['perfect_position'] != df['perfect_position'].shift(1)).cumsum()
        
        # Calculate cumulative returns for each trade group
        trade_returns = df.groupby('trade_group')['daily_return'].sum()
        
        # Identify trade groups with returns below threshold
        flat_trade_groups = trade_returns[trade_returns.abs() < FLAT_TRADE_THRESHOLD].index
        
        # Set perfect_position to 2 for flat trade groups
        df.loc[df['trade_group'].isin(flat_trade_groups), 'perfect_position'] = 2
        
        # Drop temporary columns
        df = df.drop(columns=['trade_group'])
        
        print(f"Data processing complete. Found {len(flat_trade_groups)} flat trade periods.")
        return df
        
    except Exception as e:
        print(f"Error processing data: {e}")
        # Return sample data if download fails
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        sample_df = pd.DataFrame({
            'datetime': dates,
            'open': range(100, 200),
            'high': range(105, 205),
            'low': range(95, 195),
            'close': range(100, 200),
            'volume': [1000] * 100
        })
        sample_df['daily_return'] = sample_df['close'].pct_change()
        sample_df['perfect_position'] = (sample_df['daily_return'] > 0).astype(int)
        return sample_df

# --- Flask Routes ---
@app.route('/')
def index():
    df = process_ohlcv_data()
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot Close price with background colors for positions
    plt.subplot(2, 1, 1)
    
    # Add background colors for positions
    colors = ['red', 'green', 'gray']
    labels = ['Short (0)', 'Long (1)', 'Flat (2)']
    
    for i, position_value in enumerate([0, 1, 2]):
        mask = df['perfect_position'] == position_value
        position_dates = df.loc[mask, 'datetime']
        if not position_dates.empty:
            for date in position_dates:
                plt.axvspan(date, date + pd.Timedelta(days=1), color=colors[i], alpha=0.3)
    
    plt.plot(df['datetime'], df['close'], label='Close Price', linewidth=2)
    plt.title('OHLCV Data with Perfect Positions')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot perfect positions
    plt.subplot(2, 1, 2)
    colors = ['red', 'green', 'gray']
    labels = ['Short (0)', 'Long (1)', 'Flat (2)']
    
    for position_value in [0, 1, 2]:
        mask = df['perfect_position'] == position_value
        plt.scatter(df.loc[mask, 'datetime'], df.loc[mask, 'perfect_position'], 
                   c=colors[position_value], label=labels[position_value], alpha=0.7)
    
    plt.ylabel('Position')
    plt.xlabel('Date')
    plt.yticks([0, 1, 2], ['Short', 'Long', 'Flat'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert plot to base64 for HTML display
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Create HTML template
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>OHLCV Data Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .plot { margin: 20px 0; text-align: center; }
            .download-btn { 
                display: inline-block; 
                padding: 10px 20px; 
                background: #007cba; 
                color: white; 
                text-decoration: none; 
                border-radius: 5px; 
                margin: 20px 0;
            }
            .stats { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>OHLCV Data with Perfect Position Analysis</h1>
            
            <div class="stats">
                <h3>Data Statistics</h3>
                <p><strong>Total Records:</strong> {{ total_records }}</p>
                <p><strong>Date Range:</strong> {{ date_range }}</p>
                <p><strong>Position Distribution:</strong></p>
                <ul>
                    <li>Long Positions (1): {{ long_count }} ({{ long_pct }}%)</li>
                    <li>Short Positions (0): {{ short_count }} ({{ short_pct }}%)</li>
                    <li>Flat Positions (2): {{ flat_count }} ({{ flat_pct }}%)</li>
                </ul>
            </div>
            
            <div class="plot">
                <img src="data:image/png;base64,{{ plot_url }}" alt="OHLCV Data Plot">
            </div>
            
            <a href="/download" class="download-btn">Download Processed CSV</a>
            
            <div style="margin-top: 30px;">
                <h3>Sample Data (First 10 rows):</h3>
                {{ sample_table|safe }}
            </div>
        </div>
    </body>
    </html>
    '''
    
    # Calculate statistics
    total_records = len(df)
    date_range = f"{df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}"
    
    position_counts = df['perfect_position'].value_counts()
    long_count = position_counts.get(1, 0)
    short_count = position_counts.get(0, 0)
    flat_count = position_counts.get(2, 0)
    
    long_pct = round((long_count / total_records) * 100, 1)
    short_pct = round((short_count / total_records) * 100, 1)
    flat_pct = round((flat_count / total_records) * 100, 1)
    
    # Create sample table
    sample_table = df.head(10).to_html(classes='table table-striped', index=False)
    
    return render_template_string(html_template,
                                plot_url=plot_url,
                                total_records=total_records,
                                date_range=date_range,
                                long_count=long_count,
                                short_count=short_count,
                                flat_count=flat_count,
                                long_pct=long_pct,
                                short_pct=short_pct,
                                flat_pct=flat_pct,
                                sample_table=sample_table)

@app.route('/download')
def download_csv():
    df = process_ohlcv_data()
    
    # Create CSV in memory
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(output,
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name='ohlcv_with_positions.csv')

if __name__ == '__main__':
    print("Starting web server on http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, debug=True)