import os
import io
import base64
import time
import json
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, render_template_string, Response
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# ==========================================
# 1. CONFIGURATION & MODEL
# ==========================================
MODEL_FILENAME = 'lstm_regularized.pth'
PAIR = 'XBTUSD'
KRAKEN_INTERVAL = 10080  # Weekly
SEQ_LENGTH = 60          # 5 Years of Monthly data

# UPDATED HYPERPARAMETERS TO MATCH CHECKPOINT
INPUT_DIM = 1
HIDDEN_DIM = 64       # Changed from 128 to 64
NUM_LAYERS = 1        # Changed from 2 to 1
DROPOUT = 0.2
NUM_CLASSES = 3

class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, 
                            batch_first=True, dropout=DROPOUT)
        # BatchNorm matches hidden_dim (64)
        self.bn = nn.BatchNorm1d(HIDDEN_DIM)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the output from the last time step
        out = out[:, -1, :]
        # Apply Batch Normalization
        out = self.bn(out)
        # Final Classification
        out = self.fc(out)
        return out

def get_action_name(idx):
    return {-1: "SELL", 0: "HOLD", 1: "BUY"}.get({0: -1, 1: 0, 2: 1}.get(idx))

# ==========================================
# 2. DATA ENGINE
# ==========================================
def get_analysis_data():
    """Fetches Kraken data, processes it, and runs inference on the whole timeline."""
    print("Fetching data from Kraken...", flush=True)
    url = "https://api.kraken.com/0/public/OHLC"
    try:
        resp = requests.get(url, params={'pair': PAIR, 'interval': KRAKEN_INTERVAL}, timeout=10)
        data = resp.json()
        if data.get('error'): return None, f"Kraken API Error: {data['error']}"
        
        # Extract Data
        result_keys = list(data['result'].keys())
        target_key = [k for k in result_keys if k != 'last'][0]
        ohlc = data['result'][target_key]
        
        df = pd.DataFrame(ohlc, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'vol', 'count'])
        df['close'] = pd.to_numeric(df['close'])
        df['dt'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('dt', inplace=True)
        
        # Resample Weekly -> Monthly
        monthly_df = df['close'].resample('ME').last().to_frame()
        monthly_df['log_ret'] = np.log(monthly_df['close'] / monthly_df['close'].shift(1))
        monthly_df.dropna(inplace=True)
        
        # Normalize
        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(monthly_df['log_ret'].values.reshape(-1, 1))
        
        # Load Model
        device = torch.device('cpu')
        model = LSTMClassifier().to(device)
        try:
            model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
            model.eval()
        except Exception as e:
            return None, f"Model Load Error: {e}"
        
        # Run Inference Loop
        results = []
        # Start where we have a full sequence
        for i in range(SEQ_LENGTH, len(monthly_df)):
            window = scaled_vals[i-SEQ_LENGTH : i]
            tensor_seq = torch.from_numpy(window).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(tensor_seq)
                probs = torch.nn.functional.softmax(logits, dim=1)
                _, pred_idx = torch.max(logits, 1)
                
            class_idx = pred_idx.item() # 0, 1, 2
            signal = {0: -1, 1: 0, 2: 1}[class_idx] # -1, 0, 1
            confidence = probs[0][class_idx].item()
            
            results.append({
                'date': monthly_df.index[i],
                'price': monthly_df['close'].iloc[i],
                'signal': signal,
                'confidence': confidence,
                'logits': logits.numpy()[0].tolist()
            })
            
        return results, None
        
    except Exception as e:
        return None, str(e)

# ==========================================
# 3. PLOTTING (90s Scientific Style)
# ==========================================
def create_plot(results):
    # Setup Data
    dates = [r['date'] for r in results]
    prices = [r['price'] for r in results]
    signals = [r['signal'] for r in results]
    conf = [r['confidence'] for r in results]
    
    # 90s Style Config
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.alpha'] = 0.6
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(hspace=0.1)
    
    # Plot 1: Price & Signals
    ax1.plot(dates, prices, color='black', linewidth=1, label='BTC/USD (Monthly Close)')
    
    # Extract signal points
    buy_dates = [d for d, s in zip(dates, signals) if s == 1]
    buy_prices = [p for p, s in zip(prices, signals) if s == 1]
    
    sell_dates = [d for d, s in zip(dates, signals) if s == -1]
    sell_prices = [p for p, s in zip(prices, signals) if s == -1]
    
    # Markers: Empty fill, black edge for that "printed on paper" look
    ax1.scatter(buy_dates, buy_prices, marker='^', c='white', edgecolors='black', s=80, label='BUY Signal', zorder=5)
    ax1.scatter(sell_dates, sell_prices, marker='v', c='black', edgecolors='black', s=80, label='SELL Signal', zorder=5)
    
    ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Figure 1: LSTM Model Inference on {PAIR} (Monthly)', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True)
    ax1.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=1)
    
    # Plot 2: Confidence
    ax2.plot(dates, conf, color='black', linewidth=0.8, linestyle='--')
    ax2.fill_between(dates, conf, 0, color='silver', alpha=0.3)
    ax2.set_ylabel('Model Confidence', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True)
    
    # Formatting
    date_fmt = DateFormatter('%Y-%m')
    ax2.xaxis.set_major_formatter(date_fmt)
    fig.autofmt_xdate()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64

# ==========================================
# 4. WEB SERVER
# ==========================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LSTM 32-Core Inference</title>
    <style>
        body { font-family: "Times New Roman", Serif; background: #f0f0f0; margin: 0; padding: 20px; }
        .paper { background: white; max-width: 1000px; margin: 0 auto; padding: 40px; border: 1px solid #000; box-shadow: 5px 5px 0px rgba(0,0,0,0.2); }
        h1 { text-align: center; border-bottom: 2px solid black; padding-bottom: 10px; text-transform: uppercase; letter-spacing: 2px; }
        .meta { text-align: center; font-style: italic; margin-bottom: 30px; }
        .plot-container { text-align: center; margin: 20px 0; border: 1px solid black; padding: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 30px; font-size: 14px; }
        th, td { border: 1px solid black; padding: 8px; text-align: center; }
        th { background: #eee; text-transform: uppercase; }
        .buy { font-weight: bold; }
        .sell { font-weight: bold; text-decoration: underline; }
        .hold { color: #555; }
    </style>
</head>
<body>
    <div class="paper">
        <h1>LSTM Market Analysis</h1>
        <div class="meta">
            Model: {{ model_file }} | Sequence: 60 Months | Regularized<br>
            Generated: {{ generated_time }}
        </div>

        {% if error %}
            <div style="color: red; border: 1px solid red; padding: 10px;">
                <strong>Error:</strong> {{ error }}
            </div>
        {% else %}
            <div class="plot-container">
                <img src="data:image/png;base64,{{ plot_img }}" style="max-width: 100%;">
            </div>

            <h3>Recent Inference Log (Last 12 Months)</h3>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Close Price</th>
                    <th>Signal</th>
                    <th>Confidence</th>
                    <th>Raw Logits</th>
                </tr>
                {% for row in table_data %}
                <tr>
                    <td>{{ row.date.strftime('%Y-%m-%d') }}</td>
                    <td>${{ "{:,.0f}".format(row.price) }}</td>
                    <td class="{{ 'buy' if row.signal == 1 else 'sell' if row.signal == -1 else 'hold' }}">
                        {{ "BUY (▲)" if row.signal == 1 else "SELL (▼)" if row.signal == -1 else "HOLD (●)" }}
                    </td>
                    <td>{{ "{:.1f}%".format(row.confidence * 100) }}</td>
                    <td style="font-family: monospace; font-size: 10px;">{{ row.logits }}</td>
                </tr>
                {% endfor %}
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
        # Generate Plot
        plot_img = create_plot(results)
        # Get last 12 entries for table, reversed
        table_data = results[-12:][::-1]
    
    return render_template_string(HTML_TEMPLATE, 
                                  model_file=MODEL_FILENAME,
                                  generated_time=time.strftime('%Y-%m-%d %H:%M:%S'),
                                  error=error,
                                  plot_img=plot_img,
                                  table_data=table_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Scientific Server on port {port}...")
    app.run(host='0.0.0.0', port=port)