import os
import io
import sys
import time
import base64
import threading
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from flask import Flask, render_template_string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try importing gdown, handle if missing
try:
    import gdown
except ImportError:
    gdown = None

# ==========================================
# 1. CONFIGURATION (FROM CORRECT ORIGIN)
# ==========================================
# System
torch.set_num_threads(12)
DEVICE = torch.device('cpu')

# Data
FILE_ID = '1zmPWQo5MAxgyDyvaFTpf_NiqN2o6lswa'
DOWNLOAD_OUTPUT = 'market_data.csv'
SEQ_LENGTH = 20
TRAIN_SPLIT = 0.8

# Model
INPUT_DIM = 2          # Feature 1: Log Returns, Feature 2: Z-Scored Month
HIDDEN_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.2
NUM_CLASSES = 3

# Training
BATCH_SIZE = 4096
EPOCHS = 50 # Reduced from 300 for demo speed (User can adjust)
MAX_LR = 1e-2
WEIGHT_DECAY = 1e-4
MODEL_FILENAME = 'gru_full_dataset.pth'

# Global State for Web UI
cache = {
    "status": "initializing",
    "epoch": 0,
    "total_epochs": EPOCHS,
    "train_loss": 0.0,
    "val_acc": 0.0,
    "summary": None,
    "plot_img": None,
    "error": None,
    "timestamp": None,
    "logs": []
}

def log(msg):
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)
    cache["logs"].append(f"[{ts}] {msg}")
    # Keep logs trimmed
    if len(cache["logs"]) > 50:
        cache["logs"].pop(0)

# ==========================================
# 2. MODEL DEFINITION (GRU)
# ==========================================
class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, 
                          batch_first=True, dropout=DROPOUT)
        self.bn = nn.BatchNorm1d(HIDDEN_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        
    def forward(self, x):
        # Input shape: (Batch, Seq, Feature)
        out, _ = self.gru(x)
        # GRU returns (output, hidden). We take the last time step
        out = out[:, -1, :] 
        out = self.bn(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ==========================================
# 3. BACKGROUND TASK: DATA & TRAIN
# ==========================================
def download_data():
    if not os.path.exists(DOWNLOAD_OUTPUT):
        log(f"Downloading data from ID: {FILE_ID}...")
        try:
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            if gdown:
                gdown.download(url, DOWNLOAD_OUTPUT, quiet=False, fuzzy=True)
            else:
                # Fallback if gdown is not installed
                log("gdown not found. Attempting raw requests download...")
                resp = requests.get(url)
                with open(DOWNLOAD_OUTPUT, 'wb') as f:
                    f.write(resp.content)
        except Exception as e:
            raise RuntimeError(f"Download Error: {e}")

def run_pipeline():
    global cache
    cache["status"] = "processing"
    
    try:
        # --- 1. Data Loading ---
        download_data()
        df = pd.read_csv(DOWNLOAD_OUTPUT)
        df.columns = df.columns.str.strip().str.lower()
        log(f"Data Loaded: {len(df)} rows.")

        # Feature Engineering
        if 'value' in df.columns: price_col = 'value'
        elif 'close' in df.columns: price_col = 'close'
        else: raise ValueError("No price column found")

        # Log Returns
        df['log_ret'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Month Z-Score
        if 'month' not in df.columns:
            if 'datetime' in df.columns:
                df['month'] = pd.to_datetime(df['datetime']).dt.month
            else:
                df['month'] = 0
        
        month_mean = df['month'].mean()
        month_std = df['month'].std() if df['month'].std() != 0 else 1.0
        df['month_norm'] = (df['month'] - month_mean) / month_std
        
        # Labels
        label_map = {-1: 0, 0: 1, 1: 2}
        if 'signal' not in df.columns: raise ValueError("No signal column")
        df['target_class'] = df['signal'].map(label_map)
        
        # Cleanup
        df.dropna(subset=['log_ret', 'target_class', 'month_norm'], inplace=True)
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
        
        # Scaling
        log_ret_vals = df['log_ret'].values.reshape(-1, 1)
        scaler = RobustScaler()
        log_ret_scaled = scaler.fit_transform(log_ret_vals)
        month_scaled = df['month_norm'].values.reshape(-1, 1)
        
        # Stack Features (N, 2)
        data_val = np.hstack([log_ret_scaled, month_scaled])
        labels_val = df['target_class'].values.astype(int)
        raw_prices = df[price_col].values # Keep for backtest simulation
        
        # Sequence Generation
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(data_val, window_shape=SEQ_LENGTH, axis=0)
        X = windows.transpose(0, 2, 1) # (N, Seq, Feat)
        y = labels_val[SEQ_LENGTH-1:]
        
        # Align prices for backtest
        # The prediction at index i corresponds to the price movement AFTER index i (usually)
        # or the signal generated at that time. We align prices to the end of the sequence.
        prices_aligned = raw_prices[SEQ_LENGTH-1:]
        
        min_len = min(len(X), len(y))
        X, y = X[:min_len], y[:min_len]
        prices_aligned = prices_aligned[:min_len]
        
        # Split
        split_idx = int(len(X) * TRAIN_SPLIT)
        X_train = torch.tensor(X[:split_idx], dtype=torch.float32)
        y_train = torch.tensor(y[:split_idx], dtype=torch.long)
        X_test = torch.tensor(X[split_idx:], dtype=torch.float32)
        y_test = torch.tensor(y[split_idx:], dtype=torch.long)
        
        # Weights
        unique_classes = np.unique(y[:split_idx])
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y[:split_idx])
        weights_tensor = torch.zeros(NUM_CLASSES).to(DEVICE)
        for i, cls in enumerate(unique_classes):
            weights_tensor[cls] = class_weights[i]

        # --- 2. Training ---
        train_ds = TensorDataset(X_train, y_train)
        test_ds = TensorDataset(X_test, y_test)
        
        loader_bs = min(BATCH_SIZE, len(train_ds))
        train_loader = DataLoader(train_ds, batch_size=loader_bs, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=loader_bs, shuffle=False)
        
        model = GRUClassifier().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.3
        )
        
        log(f"Starting Training: {EPOCHS} Epochs")
        best_acc = 0
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_correct = 0
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    out = model(bx)
                    val_correct += (out.argmax(1) == by).sum().item()
            
            val_acc = 100 * val_correct / len(test_ds)
            avg_train_loss = train_loss / len(train_loader)
            
            # Update UI
            cache["epoch"] = epoch + 1
            cache["train_loss"] = avg_train_loss
            cache["val_acc"] = val_acc
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), MODEL_FILENAME)
                
            if (epoch + 1) % 5 == 0:
                log(f"Ep {epoch+1}: Loss {avg_train_loss:.4f} | Acc {val_acc:.2f}%")

        # --- 3. Backtest Simulation (on Test Set) ---
        log("Running Backtest on Test Split...")
        model.load_state_dict(torch.load(MODEL_FILENAME, map_location=DEVICE))
        model.eval()
        
        # Predict all test set
        all_preds = []
        with torch.no_grad():
            # Process in chunks to avoid memory issues
            for i in range(0, len(X_test), 1000):
                batch = X_test[i:i+1000].to(DEVICE)
                out = model(batch)
                preds = out.argmax(1).cpu().numpy()
                all_preds.extend(preds)
        
        all_preds = np.array(all_preds)
        test_prices = prices_aligned[split_idx:]
        
        # Simple Logic: 0 -> Short, 1 -> Neutral, 2 -> Long
        # Position mapping: 0 -> -1, 1 -> 0, 2 -> 1
        positions = np.vectorize({0: -1, 1: 0, 2: 1}.get)(all_preds)
        
        # Calculate Returns
        # Return at t is (Price_t - Price_t-1)/Price_t-1 * Position_t-1
        # aligned prices are raw prices corresponding to the sequence end.
        price_returns = np.diff(test_prices) / test_prices[:-1]
        strategy_returns = positions[:-1] * price_returns
        
        # Cumulative
        cum_strategy = (1 + strategy_returns).cumprod()
        cum_bnh = (1 + price_returns).cumprod()
        
        # Metrics
        total_ret = (cum_strategy[-1] - 1) * 100
        sharpe = (np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)) if np.std(strategy_returns) > 0 else 0

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(cum_strategy, label='GRU Strategy', color='#0ea5e9', linewidth=2)
        plt.plot(cum_bnh, label='Buy & Hold', color='#94a3b8', linestyle='--', alpha=0.6)
        plt.title(f"Out-of-Sample Performance (Acc: {best_acc:.2f}%)")
        plt.legend()
        plt.grid(alpha=0.2)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        cache.update({
            "status": "complete",
            "summary": {"sharpe": sharpe, "total_return": total_ret, "acc": best_acc},
            "plot_img": img_base64,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        })
        log("Pipeline Complete.")

    except Exception as e:
        log(f"Error: {e}")
        cache["status"] = "error"
        cache["error"] = str(e)
        import traceback
        traceback.print_exc()

# ==========================================
# 4. WEB SERVER
# ==========================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>GRU Deployment Dashboard</title>
    <meta http-equiv="refresh" content="{{ '3' if status == 'processing' or status == 'initializing' else '600' }}">
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #111827; color: #f3f4f6; margin: 0; padding: 40px; }
        .container { max-width: 900px; margin: 0 auto; }
        .card { background: #1f2937; padding: 30px; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); border: 1px solid #374151; }
        h1 { margin-top: 0; color: #60a5fa; font-size: 24px; }
        .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
        .metric { background: #374151; padding: 15px; border-radius: 8px; text-align: center; }
        .val { font-size: 24px; font-weight: bold; display: block; margin-top: 5px; }
        .label { font-size: 12px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; }
        .log-box { background: #000; color: #10b981; font-family: monospace; padding: 15px; border-radius: 8px; height: 150px; overflow-y: auto; font-size: 12px; margin-top: 20px; border: 1px solid #374151; }
        .status-badge { display: inline-block; padding: 4px 12px; border-radius: 99px; font-size: 12px; font-weight: bold; }
        .processing { background: #f59e0b; color: #fff; }
        .complete { background: #10b981; color: #fff; }
        .error { background: #ef4444; color: #fff; }
        img { width: 100%; border-radius: 8px; margin-top: 20px; border: 1px solid #374151; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h1>GRU Market Classifier</h1>
                <span class="status-badge {{ status }}">{{ status|upper }}</span>
            </div>

            {% if status == 'processing' or status == 'initializing' %}
                <div class="metric" style="margin: 20px 0; background: #262626;">
                    <span class="label">Training Progress</span>
                    <span class="val">Epoch {{ epoch }} / {{ total_epochs }}</span>
                    <div style="width: 100%; background: #4b5563; height: 4px; margin-top: 10px; border-radius: 2px;">
                        <div style="width: {{ (epoch/total_epochs)*100 }}%; background: #60a5fa; height: 100%; border-radius: 2px; transition: width 0.5s;"></div>
                    </div>
                </div>
                <div class="grid">
                    <div class="metric">
                        <span class="label">Current Loss</span>
                        <span class="val">{{ "%.4f"|format(train_loss) }}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Val Accuracy</span>
                        <span class="val">{{ "%.2f"|format(val_acc) }}%</span>
                    </div>
                </div>
            {% elif status == 'complete' %}
                <div class="grid">
                    <div class="metric">
                        <span class="label">Best Val Accuracy</span>
                        <span class="val" style="color: #60a5fa;">{{ "%.2f"|format(summary.acc) }}%</span>
                    </div>
                    <div class="metric">
                        <span class="label">Sharpe Ratio</span>
                        <span class="val">{{ "%.3f"|format(summary.sharpe) }}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Strategy Return</span>
                        <span class="val" style="color: {{ '#34d399' if summary.total_return > 0 else '#f87171' }}">
                            {{ "%.2f"|format(summary.total_return) }}%
                        </span>
                    </div>
                </div>
                <img src="data:image/png;base64,{{ plot_img }}">
            {% elif status == 'error' %}
                <div style="color: #ef4444; margin: 20px 0;">
                    <h3>Error Occurred</h3>
                    <pre>{{ error }}</pre>
                </div>
            {% endif %}

            <div class="log-box" id="logs">
                {% for l in logs[-8:] %}
                    <div>{{ l }}</div>
                {% endfor %}
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE,
        status=cache["status"],
        epoch=cache["epoch"],
        total_epochs=cache["total_epochs"],
        train_loss=cache["train_loss"],
        val_acc=cache["val_acc"],
        summary=cache["summary"],
        plot_img=cache["plot_img"],
        error=cache["error"],
        logs=cache["logs"]
    )

if __name__ == "__main__":
    # Launch Background Pipeline
    t = threading.Thread(target=run_pipeline, daemon=True)
    t.start()
    
    # Run Server
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)