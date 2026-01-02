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
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# ==========================================
# 1. CONFIGURATION (Updated for Binance)
# ==========================================
MODEL_FILENAME = 'lstm_optimized.pth'
SYMBOL = 'BTCUSDT'       # Binance format (No slash)
INTERVAL = '1d'          # Daily Klines
SEQ_LENGTH = 30          # 30 steps of history as requested

# --- MODEL PARAMETERS (As requested) ---
INPUT_DIM = 1         
HIDDEN_DIM = 128         
NUM_LAYERS = 2           
DROPOUT = 0.4            
NUM_CLASSES = 3       

class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_DIM, 
            hidden_size=HIDDEN_DIM, 
            num_layers=NUM_LAYERS, 
            batch_first=True, 
            dropout=DROPOUT if NUM_LAYERS > 1 else 0
        )
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

# ==========================================
# 2. DATA ENGINE (Binance Public API)
# ==========================================
def fetch_binance_data(symbol, interval, limit=500):
    """Fetches OHLCV data from Binance public API."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    # Simple retry logic for reliability
    for i in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                time.sleep(2) # Backoff
                continue
            resp.raise_for_status()
            return resp.json(), None
        except Exception as e:
            if i == 2: return None, f"Binance Connectivity Error: {e}"
            time.sleep(1)
    return None, "Max retries exceeded"

def get_analysis_data():
    """Processes Binance data and runs LSTM inference."""
    raw_data, error = fetch_binance_data(SYMBOL, INTERVAL)
    if error: return None, error
        
    try:
        # Binance klines format: [OpenTime, Open, High, Low, Close, Volume, CloseTime, ...]
        df = pd.DataFrame(raw_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'
        ])
        
        df['close'] = pd.to_numeric(df['close'])
        df['dt'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('dt', inplace=True)
        
        # Calculate Log Returns
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df.dropna(inplace=True)
        
        # Scaling
        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(df['log_ret'].values.reshape(-1, 1))
        
        # Load Model
        device = torch.device('cpu')
        model = LSTMClassifier().to(device)
        
        if not os.path.exists(MODEL_FILENAME):
            return None, f"Model file '{MODEL_FILENAME}' not found in directory."
            
        model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
        model.eval()
        
        results = []
        # Inference on sliding windows of SEQ_LENGTH
        for i in range(SEQ_LENGTH, len(df)):
            window = scaled_vals[i-SEQ_LENGTH : i]
            tensor_seq = torch.from_numpy(window).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(tensor_seq)
                probs = torch.nn.functional.softmax(logits, dim=1)
                _, pred_idx = torch.max(logits, 1)
                
            class_idx = pred_idx.item()
            signal = {0: -1, 1: 0, 2: 1}[class_idx]
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
        return None, f"Data processing failed: {str(e)}"

# ==========================================
# 3. SCIENTIFIC VISUALIZATION
# ==========================================
def create_plot(results):
    dates = [r['date'] for r in results]
    prices = [r['price'] for r in results]
    signals = [r['signal'] for r in results]
    conf = [r['confidence'] for r in results]
    
    plt.rcParams['font.family'] = 'serif'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(hspace=0.05)
    
    # Chart 1: Market Trends
    ax1.plot(dates, prices, color='black', linewidth=0.7, label=f'{SYMBOL} Daily Close')
    
    buys = [(d, p) for d, p, s in zip(dates, prices, signals) if s == 1]
    sells = [(d, p) for d, p, s in zip(dates, prices, signals) if s == -1]
    
    if buys:
        ax1.scatter(*zip(*buys), marker='^', c='white', edgecolors='black', s=50, label='Model Long', zorder=5)
    if sells:
        ax1.scatter(*zip(*sells), marker='v', c='black', edgecolors='black', s=50, label='Model Short', zorder=5)
    
    ax1.set_ylabel('USD Value', fontweight='bold')
    ax1.set_title(f'Figure 1: LSTM Neural Analysis on Binance {SYMBOL}', fontsize=12, pad=15)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', frameon=True, edgecolor='black')
    
    # Chart 2: Probability
    ax2.plot(dates, conf, color='black', linewidth=1, alpha=0.6)
    ax2.fill_between(dates, conf, 0, color='gray', alpha=0.15)
    ax2.set_ylabel('Conf. Score', fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=140)
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
    <title>Binance LSTM Dashboard</title>
    <style>
        body { font-family: "Garamond", serif; background: #fdfdfd; padding: 20px; color: #222; }
        .wrapper { max-width: 1050px; margin: 0 auto; background: #fff; padding: 30px; border: 1px solid #ccc; box-shadow: 2px 2px 8px #eee; }
        .hdr { border-bottom: 2px solid #222; text-align: center; margin-bottom: 20px; }
        .stats { display: flex; justify-content: space-between; font-size: 0.85em; background: #f4f4f4; padding: 8px 15px; border: 1px solid #ddd; margin-bottom: 15px; }
        .img-box { border: 1px solid #000; padding: 5px; margin-bottom: 25px; }
        table { width: 100%; border-collapse: collapse; font-size: 0.85em; font-family: "Consolas", monospace; }
        th, td { border: 1px solid #999; padding: 5px; text-align: center; }
        th { background: #eee; }
        .b { font-weight: bold; background: #eaffea; }
        .s { font-weight: bold; text-decoration: underline; background: #ffeaea; }
        .err { color: #d00; border: 1px solid #d00; padding: 15px; background: #fffafa; }
    </style>
</head>
<body>
    <div class="wrapper">
        <div class="hdr">
            <h1 style="margin-bottom:5px;">BINANCE ANALYTICAL ENGINE</h1>
            <p style="margin-top:0; font-style:italic;">LSTM Deep Learning Architecture v2.1</p>
        </div>
        <div class="stats">
            <span><strong>Source:</strong> Binance Data</span>
            <span><strong>Arch:</strong> {{ hidden }}H | {{ layers }}L</span>
            <span><strong>Seq:</strong> {{ seq }} Steps</span>
            <span><strong>Updated:</strong> {{ now }}</span>
        </div>

        {% if error %}
            <div class="err">
                <strong>System Fault:</strong> {{ error }}
            </div>
        {% else %}
            <div class="img-box">
                <img src="data:image/png;base64,{{ plot_img }}" style="width:100%;">
            </div>

            <h3 style="border-bottom: 1px solid #222;">Inference History (Trailing)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Observation Time</th>
                        <th>Close Price</th>
                        <th>Model Signal</th>
                        <th>Confidence</th>
                        <th>Logit Vector [Sell, Hold, Buy]</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in table_data %}
                    <tr>
                        <td>{{ row.date.strftime('%Y-%m-%d %H:%M') }}</td>
                        <td>${{ "{:,.2f}".format(row.price) }}</td>
                        <td class="{{ 'b' if row.signal == 1 else 's' if row.signal == -1 else '' }}">
                            {{ "BUY (LONG)" if row.signal == 1 else "SELL (SHORT)" if row.signal == -1 else "HOLD (NEUTRAL)" }}
                        </td>
                        <td>{{ "{:.1f}%".format(row.confidence * 100) }}</td>
                        <td style="color:#666;">{{ row.logits }}</td>
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
        table_data = results[-12:][::-1] # Last 12 rows, newest top
    
    return render_template_string(
        HTML_TEMPLATE,
        hidden=HIDDEN_DIM, layers=NUM_LAYERS, seq=SEQ_LENGTH,
        now=time.strftime('%H:%M:%S'),
        error=error, plot_img=plot_img, table_data=table_data
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)