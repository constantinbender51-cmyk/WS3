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
import matplotlib.dates as mdates

# ==========================================
# 1. CONFIGURATION & MODEL SPECS
# ==========================================
MODEL_FILENAME = 'gru_full_dataset.pth'
SYMBOL = 'BTCUSDT'       
SEQ_LENGTH = 20          
INPUT_DIM = 2            # Feat 1: LogRet, Feat 2: Month
HIDDEN_DIM = 256         
NUM_LAYERS = 3           
DROPOUT = 0.2            
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
        out, _ = self.gru(x)
        out = out[:, -1, :] 
        out = self.bn(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ==========================================
# 2. DATA UTILITIES
# ==========================================
def fetch_binance_data(symbol, interval, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
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
    print(f"Startup: Beginning Full History Monthly Backtest for {SYMBOL}...")
    
    try:
        # 1. Fetch Monthly Data
        df = fetch_binance_data(SYMBOL, '1M', limit=1000)
        df = df.sort_values('dt').reset_index(drop=True)
        
        # 2. Pre-calculate Features
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['month'] = df['dt'].dt.month
        
        # Drop initial NaN
        df = df.dropna().reset_index(drop=True)

        if len(df) < SEQ_LENGTH + 1:
            raise ValueError(f"Not enough monthly data. Need > {SEQ_LENGTH} months, got {len(df)}.")

        # 3. Fit Scalers (Global Fit)
        scaler_ret = RobustScaler()
        scaler_ret.fit(df['log_ret'].values.reshape(-1, 1))
        
        month_mean = df['month'].mean()
        month_std = df['month'].std()
        if month_std == 0: month_std = 1.0

        # 4. Model Loading
        device = torch.device('cpu')
        model = GRUClassifier().to(device)
        
        if not os.path.exists(MODEL_FILENAME):
            print(f"Warning: {MODEL_FILENAME} not found. Using random weights.")
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
        
        # Start simulation after the first full sequence
        start_index = SEQ_LENGTH
        
        for i in range(start_index, len(df)):
            curr_row = df.iloc[i]
            curr_time = curr_row['dt']
            curr_price = curr_row['close']

            # Calculate PnL (Monthly)
            if i > start_index:
                prev_price = df.iloc[i-1]['close']
                m_ret = (curr_price - prev_price) / prev_price
                period_pnl = position * m_ret
                capital *= (1 + period_pnl)
                returns.append(period_pnl)

            # Prepare Input Sequence (Last 20 Months)
            window = df.iloc[i-SEQ_LENGTH+1 : i+1].copy()
            
            # Feature Prep
            raw_rets = window['log_ret'].values.reshape(-1, 1)
            raw_months = window['month'].values.reshape(-1, 1)
            
            scaled_rets = scaler_ret.transform(raw_rets)
            scaled_months = (raw_months - month_mean) / month_std
            
            seq_data = np.hstack([scaled_rets, scaled_months])
            
            with torch.no_grad():
                tensor_seq = torch.from_numpy(seq_data).float().unsqueeze(0).to(device)
                logits = model(tensor_seq)
                _, pred_idx = torch.max(logits, 1)
                
                # New Position Calculation
                # Map 0->Short, 1->Neutral, 2->Long
                new_position = {0: -1, 1: 0, 2: 1}[pred_idx.item()]

            # Store current state and the signal that will be ACTIVE for the next period
            equity_curve.append({
                'date': curr_time, 
                'equity': capital, 
                'price': curr_price,
                'signal': new_position 
            })
            
            # Update position for next iteration
            position = new_position

        # 6. Stats & Plotting
        returns = np.array(returns)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(12) if len(returns) > 0 and np.std(returns) > 0 else 0
        
        dates = [d['date'] for d in equity_curve]
        equity_vals = [d['equity'] for d in equity_curve]
        base_price = equity_curve[0]['price'] if equity_curve else 1.0
        price_norm = [d['price'] / base_price for d in equity_curve]
        signals = [d['signal'] for d in equity_curve]

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot Background Colors for Signals
        # Iterate through dates and color the span to the next date based on the current signal
        for k in range(len(dates) - 1):
            start_date = dates[k]
            end_date = dates[k+1]
            sig = signals[k]
            
            if sig == 1:
                ax.axvspan(start_date, end_date, color='#dcfce7', alpha=0.6, lw=0) # Green for Long
            elif sig == -1:
                ax.axvspan(start_date, end_date, color='#fee2e2', alpha=0.6, lw=0) # Red for Short
        
        ax.plot(dates, equity_vals, color='#8b5cf6', linewidth=2, label='GRU Strategy')
        ax.plot(dates, price_norm, color='#64748b', linestyle='--', alpha=0.6, label='BTC Buy & Hold')
        
        ax.set_title(f"Strategy Performance: {SYMBOL} (Monthly)")
        ax.legend(loc='upper left')
        ax.grid(alpha=0.2)
        
        # Formatting Date Axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

        cache.update({
            "status": "complete",
            "summary": {"sharpe": sharpe, "total_return": (capital - 1) * 100},
            "plot_img": img_base64,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        })
        print("Startup: Monthly Backtest complete.")

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
    <title>GRU Lab - Monthly Report</title>
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
        .legend-box { margin-top: 10px; display: flex; gap: 15px; justify-content: center; font-size: 13px; color: #64748b; }
        .dot { width: 10px; height: 10px; display: inline-block; border-radius: 2px; margin-right: 5px; }
    </style>
</head>
<body>
    <div class="card">
        <h1>GRU Monthly Analysis</h1>
        
        {% if status == 'processing' or status == 'initializing' %}
            <div style="padding: 100px 0; text-align: center;">
                <div class="loader"></div>
                <h2 style="display:inline-block;">Running Monthly Simulation...</h2>
                <p style="color: #64748b;">Processing monthly candles for {{ symbol }}.</p>
            </div>
        {% elif status == 'error' %}
            <div style="background: #fef2f2; color: #991b1b; padding: 20px; border-radius: 8px;">
                <strong>Simulation Failed:</strong> {{ error }}
            </div>
        {% else %}
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="badge">Last Computed: {{ timestamp }}</span>
                <span style="font-size: 14px; color: #64748b;">{{ symbol }} / Monthly Resolution</span>
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

            <div style="position: relative;">
                <img src="data:image/png;base64,{{ plot_img }}" style="width: 100%; border-radius: 8px; border: 1px solid #e2e8f0;">
                <div class="legend-box">
                    <span><span class="dot" style="background:#dcfce7;"></span>Long Zone</span>
                    <span><span class="dot" style="background:#fee2e2;"></span>Short Zone</span>
                </div>
            </div>
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
    threading.Thread(target=run_startup_backtest, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)