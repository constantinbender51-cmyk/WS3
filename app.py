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
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_FILENAME = 'lstm_optimized.pth'
SYMBOL = 'BTCUSDT'       
INTERVAL = '1M'
SEQ_LENGTH = 30          

# --- MODEL PARAMETERS ---
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
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.fc(out)
        return out

# ==========================================
# 2. DATA ENGINE (Full History Fetch)
# ==========================================
def fetch_all_monthly_binance(symbol):
    """Fetches every monthly candle available on Binance for the symbol."""
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    start_time = 0 # Start from the beginning of time
    
    print(f"Starting full history fetch for {symbol}...", flush=True)
    
    while True:
        params = {
            'symbol': symbol,
            'interval': INTERVAL,
            'startTime': start_time,
            'limit': 1000
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            if not data:
                break
                
            all_data.extend(data)
            # Set start_time to the end of the last candle + 1ms
            start_time = data[-1][6] + 1
            
            # If we received fewer than 1000 candles, we've reached the present
            if len(data) < 1000:
                break
                
            time.sleep(0.1) # Respectful rate limiting
        except Exception as e:
            return None, f"Full History Fetch Failed: {e}"
            
    return all_data, None

def get_analysis_data():
    """Processes full history and runs LSTM inference."""
    raw_data, error = fetch_all_monthly_binance(SYMBOL)
    if error: return None, error
        
    try:
        df = pd.DataFrame(raw_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'
        ])
        
        df['close'] = pd.to_numeric(df['close'])
        df['dt'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('dt', inplace=True)
        
        # Calculate log returns
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df.dropna(inplace=True)
        
        if len(df) < SEQ_LENGTH:
            return None, f"Insufficient historical data. Found: {len(df)} months."

        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(df['log_ret'].values.reshape(-1, 1))
        
        # Load Model
        device = torch.device('cpu')
        model = LSTMClassifier().to(device)
        
        if not os.path.exists(MODEL_FILENAME):
            return None, f"Checkpoint '{MODEL_FILENAME}' not found."
            
        model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
        model.eval()
        
        results = []
        # Inference Loop across the entire history
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
                'logits': [round(float(x), 4) for x in logits.numpy()[0]]
            })
            
        return results, None
        
    except Exception as e:
        return None, f"Inference Loop Failed: {str(e)}"

# ==========================================
# 3. SCIENTIFIC VISUALIZATION
# ==========================================
def create_plot(results):
    dates = [r['date'] for r in results]
    prices = [r['price'] for r in results]
    signals = [r['signal'] for r in results]
    conf = [r['confidence'] for r in results]
    
    plt.rcParams['font.family'] = 'serif'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(hspace=0.08)
    
    # Price and Full History Signals
    ax1.plot(dates, prices, color='black', linewidth=1, label='BTC Monthly Price')
    
    # Filter signals for plotting
    buys = [(d, p) for d, p, s in zip(dates, prices, signals) if s == 1]
    sells = [(d, p) for d, p, s in zip(dates, prices, signals) if s == -1]
    
    if buys:
        ax1.scatter(*zip(*buys), marker='^', facecolors='none', edgecolors='green', s=80, linewidth=1.5, label='BUY Signal', zorder=5)
    if sells:
        ax1.scatter(*zip(*sells), marker='v', color='red', s=80, label='SELL Signal', zorder=5)
    
    ax1.set_yscale('log') # Log scale is essential for full BTC history
    ax1.set_ylabel('USD Price (Log Scale)', fontweight='bold')
    ax1.set_title(f'Full Historical Sequence Analysis: {SYMBOL} (Monthly)', fontsize=14, pad=20)
    ax1.grid(True, which="both", linestyle='--', alpha=0.3)
    ax1.legend(loc='upper left', frameon=True, edgecolor='black')
    
    # Confidence Score
    ax2.fill_between(dates, conf, 0, color='gray', alpha=0.2, label='Confidence')
    ax2.plot(dates, conf, color='black', linewidth=0.8)
    ax2.set_ylabel('Probability', fontsize=10)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
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
    <title>LSTM Full History Dashboard</title>
    <style>
        body { font-family: "Times New Roman", serif; background: #eee; padding: 20px; }
        .paper { max-width: 1200px; margin: 0 auto; background: #fff; padding: 40px; border: 1px solid #111; box-shadow: 15px 15px 0px rgba(0,0,0,0.1); }
        .hdr { text-align: center; border-bottom: 3px double #000; margin-bottom: 30px; }
        .meta-grid { display: grid; grid-template-columns: repeat(3, 1fr); border: 1px solid #000; margin-bottom: 25px; background: #f9f9f9; }
        .meta-item { padding: 10px; border: 1px solid #eee; font-size: 0.9em; }
        .chart-box { border: 1px solid #000; padding: 10px; margin-bottom: 30px; }
        table { width: 100%; border-collapse: collapse; font-family: "Courier New", monospace; font-size: 12px; }
        th, td { border: 1px solid #333; padding: 6px; text-align: center; }
        th { background: #eee; }
        .buy { background: #dcfce7; font-weight: bold; }
        .sell { background: #fee2e2; font-weight: bold; }
        .error { color: #b91c1c; border: 2px solid #b91c1c; padding: 20px; text-align: center; }
    </style>
</head>
<body>
    <div class="paper">
        <div class="hdr">
            <h1>ARCHIVAL MARKET INTELLIGENCE REPORT</h1>
            <p>Comprehensive Historical Inference Engine • Monthly Resolution</p>
        </div>

        <div class="meta-grid">
            <div class="meta-item"><strong>Instrument:</strong> {{ symbol }}</div>
            <div class="meta-item"><strong>LSTM Config:</strong> {{ hidden }}H / {{ layers }}L</div>
            <div class="meta-item"><strong>Window:</strong> {{ seq }} Months</div>
            <div class="meta-item"><strong>Sample Count:</strong> {{ total_samples }}</div>
            <div class="meta-item"><strong>Model File:</strong> {{ model }}</div>
            <div class="meta-item"><strong>Report Date:</strong> {{ now }}</div>
        </div>

        {% if error %}
            <div class="error">
                CRITICAL INITIALIZATION ERROR: {{ error }}
            </div>
        {% else %}
            <div class="chart-box">
                <img src="data:image/png;base64,{{ plot_img }}" style="width: 100%;">
            </div>

            <h3 style="border-bottom: 2px solid #000; padding-bottom: 5px;">Complete Inference Archive (Recent to Oldest)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Close (USD)</th>
                        <th>Model Signal</th>
                        <th>Confidence</th>
                        <th>Raw Logits [S, H, B]</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in table_data %}
                    <tr>
                        <td>{{ row.date.strftime('%Y-%m') }}</td>
                        <td>${{ "{:,.2f}".format(row.price) }}</td>
                        <td class="{{ 'buy' if row.signal == 1 else 'sell' if row.signal == -1 else '' }}">
                            {{ "BUY" if row.signal == 1 else "SELL" if row.signal == -1 else "HOLD" }}
                        </td>
                        <td>{{ "{:.2f}%".format(row.confidence * 100) }}</td>
                        <td style="color: #666; font-size: 10px;">{{ row.logits }}</td>
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
        # Display the full log, newest first
        table_data = results[::-1]
    
    return render_template_string(
        HTML_TEMPLATE,
        symbol=SYMBOL,
        model=MODEL_FILENAME, 
        hidden=HIDDEN_DIM, 
        layers=NUM_LAYERS, 
        seq=SEQ_LENGTH,
        total_samples=len(results) if results else 0,
        now=time.strftime('%Y-%m-%d %H:%M:%S'),
        error=error, 
        plot_img=plot_img, 
        table_data=table_data
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
