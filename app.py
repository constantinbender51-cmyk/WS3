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
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION & MODEL SPECS
# ==========================================
# Updated based on 'gru_full_dataset.pth' training script
MODEL_FILENAME = 'gru_full_dataset.pth'
SYMBOL = 'BTCUSDT'       
SEQ_LENGTH = 20          # Updated from 30
INPUT_DIM = 2            # Updated from 1 (Feat 1: LogRet, Feat 2: Month)
HIDDEN_DIM = 256         # Updated from 128
NUM_LAYERS = 3           # Updated from 2
DROPOUT = 0.2            # Updated from 0.4
NUM_CLASSES = 3       

# Global storage for pre-calculated results
cache = {
    "status": "initializing",
    "summary": None,
    "plot_img": None,
    "error": None,
    "timestamp": None
}

class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=INPUT_DIM, 
            hidden_size=HIDDEN_DIM, 
            num_layers=NUM_LAYERS, 
            batch_first=True, 
            dropout=DROPOUT
        )
        self.bn = nn.BatchNorm1d(HIDDEN_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        
    def forward(self, x):
        # Input shape: (Batch, Seq, Feature)
        out, _ = self.gru(x)
        # Take the last time step
        out = out[:, -1, :] 
        out = self.bn(out)
        out = torch.relu(out)
        out = self.dropout(out)
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
    print(f"Startup: Beginning 1-Year Hourly Backtest for {SYMBOL} using GRU...")
    
    try:
        # 1. Fetch 1 Year of Hourly Data
        # We need enough history to fit the scaler reasonably well
        all_hourly = []
        start_ts = (time.time() - (365 * 24 * 3600)) * 1000
        for _ in range(9): # Fetch chunks to get ~8760 rows
            chunk = fetch_binance_data(SYMBOL, '1h', limit=1000, start_time=start_ts)
            if chunk.empty: break
            all_hourly.append(chunk)
            start_ts = (chunk['dt'].iloc[-1].timestamp() * 1000) + 1
        
        df_hourly = pd.concat(all_hourly).drop_duplicates().sort_values('dt').reset_index(drop=True)
        
        # 2. Pre-calculate Features for the whole history (to fit scalers)
        # Log Returns
        df_hourly['log_ret'] = np.log(df_hourly['close'] / df_hourly['close'].shift(1))
        # Month
        df_hourly['month'] = df_hourly['dt'].dt.month
        
        # Drop initial NaN from shift
        df_hourly = df_hourly.dropna().reset_index(drop=True)

        # 3. Fit Scalers (Global Fit based on history)
        # The training script used RobustScaler on Returns and Z-Score on Month
        scaler_ret = RobustScaler()
        scaler_ret.fit(df_hourly['log_ret'].values.reshape(-1, 1))
        
        month_mean = df_hourly['month'].mean()
        month_std = df_hourly['month'].std()
        if month_std == 0: month_std = 1.0

        # 4. Model Loading
        device = torch.device('cpu')
        model = GRUClassifier().to(device)
        
        # Check for model file
        if not os.path.exists(MODEL_FILENAME):
            print(f"Warning: {MODEL_FILENAME} not found. Running with initialized weights (random) for demo.")
        else:
            try:
                model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
                print(f"Loaded weights from {MODEL_FILENAME}")
            except Exception as e:
                 print(f"Error loading weights: {e}")

        model.eval()

        # 5. Simulation Loop
        capital = 1.0
        position = 0
        returns = []
        equity_curve = []
        
        # We start the loop after SEQ_LENGTH to ensure we have enough context
        # We simulate "walking forward" through the data
        start_index = SEQ_LENGTH
        
        for i in range(start_index, len(df_hourly)):
            curr_row = df_hourly.iloc[i]
            curr_time = curr_row['dt']
            curr_price = curr_row['close']

            # Calculate PnL from previous step
            if i > start_index:
                prev_price = df_hourly.iloc[i-1]['close']
                h_ret = (curr_price - prev_price) / prev_price
                period_pnl = position * h_ret
                capital *= (1 + period_pnl)
                returns.append(period_pnl)

            # Prepare Input Sequence
            # We need the LAST 20 rows of data up to this point
            window = df_hourly.iloc[i-SEQ_LENGTH+1 : i+1].copy()
            
            # 1. Get raw features
            raw_rets = window['log_ret'].values.reshape(-1, 1)
            raw_months = window['month'].values.reshape(-1, 1)
            
            # 2. Scale features (using the pre-fitted stats to prevent lookahead bias)
            scaled_rets = scaler_ret.transform(raw_rets)
            scaled_months = (raw_months - month_mean) / month_std
            
            # 3. Stack [LogRet, Month] -> Shape (20, 2)
            seq_data = np.hstack([scaled_rets, scaled_months])
            
            # Inference
            with torch.no_grad():
                tensor_seq = torch.from_numpy(seq_data).float().unsqueeze(0).to(device)
                logits = model(tensor_seq)
                _, pred_idx = torch.max(logits, 1)
                # Map 0->Short, 1->Neutral, 2->Long
                position = {0: -1, 1: 0, 2: 1}[pred_idx.item()]

            # Record Equity Curve
            if i % 24 == 0 or i == len(df_hourly) - 1:
                equity_curve.append({'date': curr_time, 'equity': capital, 'price': curr_price})

        # 6. Stats & Plotting
        returns = np.array(returns)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(8760) if len(returns) > 0 and np.std(returns) > 0 else 0
        
        dates = [d['date'] for d in equity_curve]
        equity_vals = [d['equity'] for d in equity_curve]
        # Normalize BTC price to start at 1.0 for comparison
        base_price = equity_curve[0]['price'] if equity_curve else 1.0
        price_norm = [d['price'] / base_price for d in equity_curve]

        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity_vals, color='#8b5cf6', linewidth=2, label='GRU Strategy')
        plt.plot(dates, price_norm, color='#94a3b8', linestyle='--', alpha=0.6, label='BTC Buy & Hold')
        plt.title(f"Annual Strategy Performance: {SYMBOL} (GRU)")
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
        import traceback
        traceback.print_exc()
        cache.update({"status": "error", "error": str(e)})

# ==========================================
# 4. WEB SERVER
# ==========================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>GRU Lab - Startup Report</title>
    <meta http-equiv="refresh" content="{{ '10' if status == 'processing' or status == 'initializing' else '3600' }}">
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; background: #f8fafc; color: #1e293b; padding: 40px; }
        .card { max-width: 1000px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 12px; box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1); border: 1px solid #e2e8f0; }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0; }
        .stat-box { background: #f1f5f9; padding: 20px; border-radius: 8px; text-align: center; }
        .val { font-size: 32px; font-weight: 800; display: block; }
        .label { font-size: 12px; text-transform: uppercase; color: #64748b; font-weight: 700; }
        .loader { border: 4px solid #f3f3f3; border-top: 4px solid #8b5cf6; border-radius: 50%; width: 30px; height: 30px; animation: spin 2s linear infinite; display: inline-block; vertical-align: middle; margin-right: 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .badge { background: #f3e8ff; color: #6b21a8; padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; }
    </style>
</head>
<body>
    <div class="card">
        <h1>GRU Archival Analysis</h1>
        
        {% if status == 'processing' or status == 'initializing' %}
            <div style="padding: 100px 0; text-align: center;">
                <div class="loader"></div>
                <h2 style="display:inline-block;">Executing 1-Year Backtest...</h2>
                <p style="color: #64748b;">The engine is running ~8,760 inferences on startup using the GRU model (2-Feature). This page will refresh automatically.</p>
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
