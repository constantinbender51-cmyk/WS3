import os
import io
import base64
import time
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, render_template_string
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# ==========================================
# 1. CONFIGURATION (Updated to match your specs)
# ==========================================
MODEL_FILENAME = 'lstm_optimized.pth'
PAIR = 'XBTUSD'
KRAKEN_INTERVAL = 1440   # Daily interval (more granular for 30-step history)
SEQ_LENGTH = 30          # As requested: 30 steps of history

# --- MODEL PARAMETERS (Updated) ---
INPUT_DIM = 1         
HIDDEN_DIM = 128         # As requested
NUM_LAYERS = 2           # As requested
DROPOUT = 0.4            # As requested
NUM_CLASSES = 3       

class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize LSTM with provided parameters
        self.lstm = nn.LSTM(
            input_size=INPUT_DIM, 
            hidden_size=HIDDEN_DIM, 
            num_layers=NUM_LAYERS, 
            batch_first=True, 
            dropout=DROPOUT if NUM_LAYERS > 1 else 0
        )
        # BatchNorm matches hidden_dim (128)
        self.bn = nn.BatchNorm1d(HIDDEN_DIM)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        
    def forward(self, x):
        # x shape: (Batch, Seq, Features)
        out, _ = self.lstm(x)
        # Take the output from the last time step
        out = out[:, -1, :]
        # Apply Batch Normalization
        out = self.bn(out)
        # Final Classification
        out = self.fc(out)
        return out

# ==========================================
# 2. DATA ENGINE
# ==========================================
def get_analysis_data():
    """Fetches Kraken data, processes it, and runs inference with optimized parameters."""
    print(f"Fetching {PAIR} data from Kraken...", flush=True)
    url = "https://api.kraken.com/0/public/OHLC"
    try:
        # Fetch Daily data to provide enough history for the 30-step sequence
        resp = requests.get(url, params={'pair': PAIR, 'interval': KRAKEN_INTERVAL}, timeout=15)
        data = resp.json()
        if data.get('error'): 
            return None, f"Kraken API Error: {data['error']}"
        
        # Extract OHLC data
        result_keys = list(data['result'].keys())
        target_key = [k for k in result_keys if k != 'last'][0]
        ohlc = data['result'][target_key]
        
        # Build DataFrame
        df = pd.DataFrame(ohlc, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'vol', 'count'])
        df['close'] = pd.to_numeric(df['close'])
        df['dt'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('dt', inplace=True)
        
        # Use log returns for stationary input
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df.dropna(inplace=True)
        
        # Standardize returns
        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(df['log_ret'].values.reshape(-1, 1))
        
        # Load Model
        device = torch.device('cpu')
        model = LSTMClassifier().to(device)
        
        if not os.path.exists(MODEL_FILENAME):
            return None, f"Checkpoint not found: {MODEL_FILENAME}. Please ensure the file is present in the directory."
            
        try:
            model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
            model.eval()
        except Exception as e:
            return None, f"Model State Dict Mismatch: {e}. Check if HIDDEN_DIM/LAYERS matches the saved file."
        
        # Run Inference Loop
        results = []
        # Start where we have a full sequence of 30
        for i in range(SEQ_LENGTH, len(df)):
            window = scaled_vals[i-SEQ_LENGTH : i]
            tensor_seq = torch.from_numpy(window).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(tensor_seq)
                probs = torch.nn.functional.softmax(logits, dim=1)
                _, pred_idx = torch.max(logits, 1)
                
            class_idx = pred_idx.item() # 0, 1, 2
            # Mapping class index to market signals
            signal = {0: -1, 1: 0, 2: 1}[class_idx] # SELL, HOLD, BUY
            confidence = probs[0][class_idx].item()
            
            results.append({
                'date': df.index[i],
                'price': df['close'].iloc[i],
                'signal': signal,
                'confidence': confidence,
                'logits': [round(x, 4) for x in logits.numpy()[0].tolist()]
            })
            
        return results, None
        
    except Exception as e:
        return None, f"Analysis Exception: {str(e)}"

# ==========================================
# 3. SCIENTIFIC VISUALIZATION
# ==========================================
def create_plot(results):
    dates = [r['date'] for r in results]
    prices = [r['price'] for r in results]
    signals = [r['signal'] for r in results]
    conf = [r['confidence'] for r in results]
    
    # 90s Academic Style Configuration
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(hspace=0.05)
    
    # Primary Plot: Price Action
    ax1.plot(dates, prices, color='black', linewidth=0.8, label='Asset Price (Index)')
    
    # Scatter plot signals
    buy_dates = [d for d, s in zip(dates, signals) if s == 1]
    buy_prices = [p for p, s in zip(prices, signals) if s == 1]
    sell_dates = [d for d, s in zip(dates, signals) if s == -1]
    sell_prices = [p for p, s in zip(prices, signals) if s == -1]
    
    ax1.scatter(buy_dates, buy_prices, marker='^', c='white', edgecolors='black', s=60, label='Long Entry (Model)', zorder=5)
    ax1.scatter(sell_dates, sell_prices, marker='v', c='black', edgecolors='black', s=60, label='Short Exit (Model)', zorder=5)
    
    ax1.set_ylabel('USD Price', fontsize=11, fontweight='bold')
    ax1.set_title(f'Market Dynamics Analysis: LSTM Optimized ({HIDDEN_DIM} Hidden Units)', fontsize=13, fontweight='bold', pad=20)
    ax1.grid(True)
    ax1.legend(loc='upper left', frameon=True, edgecolor='black', framealpha=1)
    
    # Secondary Plot: Probability Confidence
    ax2.plot(dates, conf, color='black', linewidth=1, alpha=0.7)
    ax2.fill_between(dates, conf, 0, color='gray', alpha=0.2)
    ax2.set_ylabel('Prob. Confidence', fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True)
    
    # X-Axis Formatting
    date_fmt = DateFormatter('%Y-%m')
    ax2.xaxis.set_major_formatter(date_fmt)
    fig.autofmt_xdate()
    
    # Output to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64

# ==========================================
# 4. WEB INTERFACE
# ==========================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LSTM Inference Dashboard</title>
    <style>
        body { font-family: "Georgia", serif; background: #e5e5e5; color: #1a1a1a; margin: 0; padding: 20px; }
        .container { background: white; max-width: 1100px; margin: 0 auto; padding: 30px; border: 1px solid #999; }
        .header { text-align: center; border-bottom: 3px double #333; margin-bottom: 25px; padding-bottom: 10px; }
        .header h1 { margin: 0; font-size: 28px; letter-spacing: 1px; }
        .stats-bar { display: flex; justify-content: space-around; background: #f9f9f9; padding: 10px; border: 1px solid #ccc; margin-bottom: 20px; font-size: 0.9em; }
        .plot-box { text-align: center; margin-bottom: 30px; }
        .plot-box img { max-width: 100%; border: 1px solid #000; }
        table { width: 100%; border-collapse: collapse; font-family: "Courier New", monospace; font-size: 13px; }
        th, td { border: 1px solid #666; padding: 6px; text-align: center; }
        th { background: #ddd; }
        .sig-buy { background: #e6fffa; font-weight: bold; }
        .sig-sell { background: #fff5f5; font-weight: bold; text-decoration: underline; }
        .error-msg { background: #fee2e2; border: 1px solid #ef4444; color: #b91c1c; padding: 15px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>QUANTITATIVE ANALYSIS REPORT</h1>
            <p>Automated LSTM Sequence Classification Engine</p>
        </div>

        <div class="stats-bar">
            <span><strong>Model:</strong> {{ model_name }}</span>
            <span><strong>Architecture:</strong> {{ layers }} Layers / {{ hidden }} Hidden</span>
            <span><strong>Context:</strong> {{ seq }} Time-steps</span>
            <span><strong>Clock:</strong> {{ now }}</span>
        </div>

        {% if error %}
            <div class="error-msg">
                <strong>System Fault:</strong> {{ error }}
            </div>
        {% else %}
            <div class="plot-box">
                <img src="data:image/png;base64,{{ plot_img }}" alt="Inference Chart">
            </div>

            <h3 style="border-bottom: 1px solid #333;">Temporal Log (Recent Sequence Data)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Terminal Price</th>
                        <th>Inference Signal</th>
                        <th>Confidence (%)</th>
                        <th>Vector Logits [Sell, Hold, Buy]</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in table_data %}
                    <tr>
                        <td>{{ row.date.strftime('%Y-%m-%d %H:%M') }}</td>
                        <td>${{ "{:,.2f}".format(row.price) }}</td>
                        <td class="{{ 'sig-buy' if row.signal == 1 else 'sig-sell' if row.signal == -1 else '' }}">
                            {{ "BUY" if row.signal == 1 else "SELL" if row.signal == -1 else "HOLD" }}
                        </td>
                        <td>{{ "{:.2f}".format(row.confidence * 100) }}%</td>
                        <td style="color: #666;">{{ row.logits }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    results, error = get_analysis_data()
    plot_img = None
    table_data = []
    
    if results:
        plot_img = create_plot(results)
        # Show most recent 15 results in the table
        table_data = results[-15:][::-1]
    
    return render_template_string(
        HTML_TEMPLATE,
        model_name=MODEL_FILENAME,
        layers=NUM_LAYERS,
        hidden=HIDDEN_DIM,
        seq=SEQ_LENGTH,
        now=time.strftime('%Y-%m-%d %H:%M:%S'),
        error=error,
        plot_img=plot_img,
        table_data=table_data
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # Run the server
    app.run(host='0.0.0.0', port=port)