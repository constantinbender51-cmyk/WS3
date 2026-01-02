import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_FILENAME = 'lstm_32core.pth'
PAIR = 'XBTUSD'        # Bitcoin/USD
KRAKEN_INTERVAL = 10080 # 10080 min = 1 Week (Fetch Weekly to build Monthly)
SEQ_LENGTH = 60        # 60 Months (5 Years)

# Model Hyperparameters (Must match training exactly)
INPUT_DIM = 1         
HIDDEN_DIM = 128      
NUM_LAYERS = 2        
DROPOUT = 0.2     
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
    print(f"1. Fetching Weekly data from Kraken for {PAIR}...")
    url = "https://api.kraken.com/0/public/OHLC"
    # Fetch weekly data (Interval=10080). 
    # Kraken usually returns ~720 candles (approx 14 years of weekly data), 
    # which is plenty to build a 60-month history.
    params = {'pair': PAIR, 'interval': KRAKEN_INTERVAL}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('error'):
            print(f"Kraken Error: {data['error']}")
            return None, None

        # Dynamic key extraction (e.g., XXBTZUSD vs XBTUSD)
        result_keys = list(data['result'].keys())
        target_key = [k for k in result_keys if k != 'last'][0]
        ohlc = data['result'][target_key]
        
        # Create DataFrame
        df = pd.DataFrame(ohlc, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'vol', 'count'])
        df['close'] = pd.to_numeric(df['close'])
        df['dt'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('dt', inplace=True)
        
        print(f"   > Received {len(df)} weekly candles.")

        # --- RESAMPLING TO MONTHLY ---
        print("2. Resampling Weekly -> Monthly...")
        # We take the 'last' close of every month to simulate Monthly candles
        monthly_df = df['close'].resample('ME').last().to_frame() # 'ME' is Month End
        
        # --- CALCULATE LOG RETURNS ---
        # Same formula as training: ln(current / prev)
        monthly_df['log_ret'] = np.log(monthly_df['close'] / monthly_df['close'].shift(1))
        monthly_df.dropna(inplace=True)
        
        print(f"   > Generated {len(monthly_df)} monthly return points.")
        
        # --- NORMALIZATION ---
        # CRITICAL: We fit the scaler on the ENTIRE available history 
        # to best approximate the global mean/std the model expects.
        print("3. Normalizing (StandardScaler)...")
        scaler = StandardScaler()
        data_val = monthly_df['log_ret'].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(data_val)
        
        # --- SEQUENCE CREATION ---
        if len(scaled_data) < SEQ_LENGTH:
            print(f"ERROR: Not enough monthly data. Need {SEQ_LENGTH}, got {len(scaled_data)}")
            return None, None
            
        # Grab exactly the last 60 months
        final_sequence = scaled_data[-SEQ_LENGTH:]
        
        # Convert to PyTorch Tensor: (1, 60, 1)
        tensor_seq = torch.from_numpy(final_sequence).float().unsqueeze(0)
        
        # Get date range for display
        start_date = monthly_df.index[-SEQ_LENGTH]
        end_date = monthly_df.index[-1]
        
        return tensor_seq, (start_date, end_date)

    except Exception as e:
        print(f"Data Processing Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("-" * 50)
    print(" MONTHLY BITCOIN PREDICTION (Kraken -> Resample)")
    print("-" * 50)

    # 1. Load Model
    device = torch.device('cpu')
    model = LSTMClassifier().to(device)
    
    try:
        print(f"Loading weights from {MODEL_FILENAME}...")
        model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"Error: {MODEL_FILENAME} not found.")
        exit()

    # 2. Get Data
    input_tensor, date_info = fetch_and_process_data()
    
    if input_tensor is not None:
        # 3. Inference
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            _, pred_idx = torch.max(logits, 1)
            action = get_action_name(pred_idx.item())

        # 4. Output
        print("\n" + "="*40)
        print(" PREDICTION RESULTS")
        print("="*40)
        print(f"Timeframe:     Monthly (Resampled from Weekly)")
        print(f"Range Used:    {date_info[0].date()} to {date_info[1].date()}")
        print("-" * 40)
        print(f"Raw Logits:    {logits.numpy()[0]}")
        print(f"Probabilities: {probs.numpy()[0]}")
        print("-" * 40)
        print(f"MODEL DECISION: {action}")
        print("="*40)
        
        if action == 1:
            print(">>> BUY (Long Term)")
        elif action == -1:
            print(">>> SELL (Long Term)")
        else:
            print(">>> HOLD")