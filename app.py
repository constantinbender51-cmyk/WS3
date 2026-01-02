import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import requests
import time
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_FILENAME = 'lstm_regularized.pth'
PAIR = 'XBTUSD'        # Bitcoin/USD
KRAKEN_INTERVAL = 10080 # 10080 min = 1 Week (Fetch Weekly to build Monthly)
SEQ_LENGTH = 60        # 60 Months (5 Years)

# Model Hyperparameters
INPUT_DIM = 1         
HIDDEN_DIM = 128      
NUM_LAYERS = 2        
DROPOUT = 0.5
NUM_CLASSES = 3       

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, 
                            batch_first=True, dropout=DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def get_action_name(class_index):
    """Maps class indices (0,1,2) back to signals (-1,0,1)."""
    mapping = {0: -1, 1: 0, 2: 1}
    return mapping.get(class_index, "Unknown")

# ==========================================
# 3. DATA PIPELINE
# ==========================================
def fetch_and_process_data():
    print(f"1. Fetching Weekly data from Kraken for {PAIR}...", flush=True)
    url = "https://api.kraken.com/0/public/OHLC"
    params = {'pair': PAIR, 'interval': KRAKEN_INTERVAL}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('error'):
            print(f"Kraken Error: {data['error']}", flush=True)
            return None, None

        result_keys = list(data['result'].keys())
        target_key = [k for k in result_keys if k != 'last'][0]
        ohlc = data['result'][target_key]
        
        df = pd.DataFrame(ohlc, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'vol', 'count'])
        df['close'] = pd.to_numeric(df['close'])
        df['dt'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('dt', inplace=True)
        
        print(f"   > Received {len(df)} weekly candles.", flush=True)

        print("2. Resampling Weekly -> Monthly...", flush=True)
        monthly_df = df['close'].resample('ME').last().to_frame() 
        
        monthly_df['log_ret'] = np.log(monthly_df['close'] / monthly_df['close'].shift(1))
        monthly_df.dropna(inplace=True)
        
        print(f"   > Generated {len(monthly_df)} monthly return points.", flush=True)
        
        print("3. Normalizing (StandardScaler)...", flush=True)
        scaler = StandardScaler()
        data_val = monthly_df['log_ret'].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(data_val)
        
        return scaled_data, monthly_df

    except Exception as e:
        print(f"Data Processing Error: {e}", flush=True)
        return None, None

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("-" * 65, flush=True)
    print(" MONTHLY BITCOIN PREDICTION (Historical Timeline)", flush=True)
    print("-" * 65, flush=True)

    # 1. Load Model
    device = torch.device('cpu')
    model = LSTMClassifier().to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
        model.eval()
        print(f"Model weights loaded from {MODEL_FILENAME}", flush=True)
    except FileNotFoundError:
        print(f"Error: {MODEL_FILENAME} not found.", flush=True)
        exit()

    # 2. Get Full Data
    scaled_data, monthly_df = fetch_and_process_data()
    
    if scaled_data is not None:
        total_len = len(scaled_data)
        
        # Start at 59 so we include the very first valid sequence (indices 0-59)
        start_loop = SEQ_LENGTH - 1

        print("\n" + "="*80, flush=True)
        print(f"{'DATE':<12} | {'ACTION':<6} | {'CONF':<8} | {'RAW LOGITS'}", flush=True)
        print("="*80, flush=True)

        # 3. Loop through EVERY valid month
        for i in range(start_loop, total_len):
            window_end = i + 1
            window_start = window_end - SEQ_LENGTH
            
            sequence = scaled_data[window_start:window_end]
            tensor_seq = torch.from_numpy(sequence).float().unsqueeze(0)
            
            with torch.no_grad():
                logits = model(tensor_seq)
                probs = torch.nn.functional.softmax(logits, dim=1)
                _, pred_idx = torch.max(logits, 1)
                action_code = pred_idx.item()
                action = get_action_name(action_code)

            date_str = monthly_df.index[i].strftime('%Y-%m-%d')
            prob_val = probs[0][action_code].item() * 100
            
            if action == 1:
                act_str = "BUY"
            elif action == -1:
                act_str = "SELL"
            else:
                act_str = "HOLD"

            # CRITICAL: flush=True ensures line sends immediately
            print(f"{date_str:<12} | {act_str:<6} | {prob_val:>6.2f}%  | {logits.numpy()[0]}", flush=True)
            
            # CRITICAL: Sleep ensures log system preserves order
            time.sleep(0.1)

        print("="*80, flush=True)