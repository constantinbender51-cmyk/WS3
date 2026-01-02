import os
import io
import base64
import time
import threading
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

# ==========================================
# 1. CONFIGURATION & MODEL
# ==========================================
MODEL_FILENAME = 'lstm_optimized.pth'
SYMBOL = 'BTCUSDT'       
SEQ_LENGTH = 30 # This script assumes the model expects a sequence of 30 log-returns
INPUT_DIM = 1         
HIDDEN_DIM = 128         
NUM_LAYERS = 2           
DROPOUT = 0.4            
NUM_CLASSES = 3       

# Global storage for pre-calculated results
cache = {
    "status": "initializing",
    "summary": None,
    "plot_img": None,
    "error": None,
    "timestamp": None
}

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
# 2. DATA UTILITIES
# ==========================================
def fetch_binance_data(symbol, interval, limit=1000, start_time=None):
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    if start_time: params['startTime'] = int(start_time)
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=['ot','o','h','l','close','v','ct','q','n','tb','tq','i'])
    df['close'] = pd.to_numeric(df['close'])
    df['dt'] = pd.to_datetime(df['ot'], unit='ms')
    return df[['dt', 'close']]

# ==========================================
# 3. BACKTEST ENGINE (BACKGROUND TASK)
# ==========================================
def run_startup_backtest():
    global cache
    cache["status"] = "processing"
    print(f"Startup: Beginning 1-Year Hourly Backtest for {SYMBOL}...")
    
    try:
        # 1. Fetch Context (Monthly) - Need context to form sequences
        # We need 31 months to get 30 log returns
        df_monthly = fetch_binance_data(SYMBOL, '1M', limit=60)
        
        # 2. Fetch 1 Year of Hourly Data
        all_hourly = []
        start_ts = (time.time() - (365 * 24 * 3600)) * 1000
        for _ in range(9):
            chunk = fetch_binance_data(SYMBOL, '1h', limit=1000, start_time=start_ts)
            if chunk.empty: break
            all_hourly.append(chunk)
            start_ts = (chunk['dt'].iloc[-1].timestamp() * 1000) + 1
        
        df_hourly = pd.concat(all_hourly).drop_duplicates().sort_values('dt')
        
        # 3. Model Loading
        device = torch.device('cpu')
        model = LSTMClassifier().to(device)
        if not os.path.exists(MODEL_FILENAME):
            raise FileNotFoundError(f"{MODEL_FILENAME} not found on Railway disk.")
        
        model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
        model.eval()

        # 4. Simulation
        capital = 1.0
        position = 0
        returns = []
        equity_curve = []
        scaler = StandardScaler()

        for i in range(len(df_hourly)):
            curr_row = df_hourly.iloc[i]
            curr_time, curr_price = curr_row['dt'], curr_row['close']

            if i > 0:
                h_ret = (curr_price - df_hourly.iloc[i-1]['close']) / df_hourly.iloc[i-1]['close']
                period_pnl = position * h_ret
                capital *= (1 + period_pnl)
                returns.append(period_pnl)
            
            # Sequence: last SEQ_LENGTH closed months + current price
            # We need SEQ_LENGTH + 1 price points to get SEQ_LENGTH returns
            context = df_monthly[df_monthly['dt'] < curr_time].tail(SEQ_LENGTH)
            prices_seq = np.append(context['close'].values, curr_price)
            
            # Log Returns & Scale
            log_rets = np.log(prices_seq[1:] / prices_seq[:-1])
            if len(log_rets) < SEQ_LENGTH: continue # Safety check
            
            scaled = scaler.fit_transform(log_rets.reshape(-1, 1))
            
            with torch.no_grad():
                tensor_seq = torch.from_numpy(scaled).float().unsqueeze(0).to(device)
                logits = model(tensor_seq)
                _, pred_idx = torch.max(logits, 1)
                position = {0: -1, 1: 0, 2: 1}[pred_idx.item()]

            if i % 24 == 0 or i == len(df_hourly) - 1:
                equity_curve.append({'date': curr_time, 'equity': capital, 'price': curr_price})

        # 5. Stats
        returns = np.array(returns)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(8760) if np.std(returns) > 0 else 0
        
        # 6. Plotting
        dates = [d['date'] for d in equity_curve]
        equity_vals = [d['equity'] for d in equity_curve]
        price_norm = [d['price'] / equity_curve[0]['price'] for d in equity_curve]

        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity_vals, color='#0ea5e9', linewidth=2, label='LSTM Strategy')
        plt.plot(dates, price_norm, color='#94a3b8', linestyle='--', alpha=0.6, label='BTC Buy & Hold')
        plt.title(f"Annual Strategy Performance: {SYMBOL}")
        plt.legend()
        plt.grid(alpha=0.2)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        cache.update({
            "status": "complete",
            "summary": {"sharpe": sharpe, "total_return": (capital - 1) * 100},
            "plot_img": img_base64,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        })
        print("Startup: Backtest complete. Cache populated.")

    except Exception as e:
        print(f"Startup Error: {e}")
        cache.update({"status": "error", "error": str(e)})

# ==========================================
# 4. WEB SERVER
# ==========================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LSTM Lab - Startup Report</title>
    <meta http-equiv="refresh" content="{{ '10' if status == 'processing' or status == 'initializing' else '3600' }}">
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; background: #f8fafc; color: #1e293b; padding: 40px; }
        .card { max-width: 1000px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 12px; box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1); border: 1px solid #e2e8f0; }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0; }
        .stat-box { background: #f1f5f9; padding: 20px; border-radius: 8px; text-align: center; }
        .val { font-size: 32px; font-weight: 800; display: block; }
        .label { font-size: 12px; text-transform: uppercase; color: #64748b; font-weight: 700; }
        .loader { border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 30px; height: 30px; animation: spin 2s linear infinite; display: inline-block; vertical-align: middle; margin-right: 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .badge { background: #e0f2fe; color: #0369a1; padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; }
    </style>
</head>
<body>
    <div class="card">
        <h1>LSTM Archival Analysis</h1>
        
        {% if status == 'processing' or status == 'initializing' %}
            <div style="padding: 100px 0; text-align: center;">
                <div class="loader"></div>
                <h2 style="display:inline-block;">Executing 1-Year Backtest...</h2>
                <p style="color: #64748b;">The engine is running ~8,760 inferences on startup. This page will refresh automatically.</p>
            </div>
        {% elif status == 'error' %}
            <div style="background: #fef2f2; color: #991b1b; padding: 20px; border-radius: 8px;">
                <strong>Simulation Failed:</strong> {{ error }}
            </div>
        {% else %}
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="badge">Last Computed: {{ timestamp }}</span>
                <span style="font-size: 14px; color: #64748b;">{{ symbol }} / Hourly Resolution</span>
            </div>

            <div class="stat-grid">
                <div class="stat-box">
                    <span class="label">Annualized Sharpe Ratio</span>
                    <span class="val">{{ "%.4f"|format(summary.sharpe) }}</span>
                </div>
                <div class="stat-box">
                    <span class="label">Total Strategy Return</span>
                    <span class="val" style="color: {{ '#16a34a' if summary.total_return > 0 else '#dc2626' }}">
                        {{ "%.2f"|format(summary.total_return) }}%
                    </span>
                </div>
            </div>

            <img src="data:image/png;base64,{{ plot_img }}" style="width: 100%; border-radius: 8px; border: 1px solid #e2e8f0;">
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE,
        status=cache["status"],
        summary=cache["summary"],
        plot_img=cache["plot_img"],
        error=cache["error"],
        timestamp=cache["timestamp"],
        symbol=SYMBOL
    )

if __name__ == "__main__":
    # Start the backtest thread before launching the web server
    threading.Thread(target=run_startup_backtest, daemon=True).start()
    
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)