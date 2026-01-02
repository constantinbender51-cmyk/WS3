import torch
import torch.nn as nn
import numpy as np

# ==========================================
# 1. DEFINE THE MODEL ARCHITECTURE
# ==========================================
# Must match the training configuration exactly
INPUT_DIM = 1         
HIDDEN_DIM = 128      
NUM_LAYERS = 2        
DROPOUT = 0.2
NUM_CLASSES = 3       

class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, 
                            batch_first=True, dropout=DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        out, _ = self.lstm(x)
        # We only care about the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

def get_action_name(class_index):
    """
    Maps class indices back to original signals based on training logic:
    Original Map: {-1: 0, 0: 1, 1: 2}
    Reverse Map: {0: -1, 1: 0, 2: 1}
    """
    mapping = {0: -1, 1: 0, 2: 1}
    return mapping.get(class_index, "Unknown")

if __name__ == "__main__":
    # ==========================================
    # 2. SETUP & MOCK DATA
    # ==========================================
    # Device configuration
    device = torch.device('cpu') 
    
    # Instantiate the model
    model = LSTMClassifier().to(device)

    # Load weights
    model_path = 'model.pth'
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set to evaluation mode (turns off dropout)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Ensure the file exists.")
        exit()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # Generate Mock Data
    # Shape: (Batch Size=1, Sequence Length=60, Features=1)
    # Using randn to simulate normalized data (mean 0, std 1)
    print("\nGenerating mock sequence (60 data points)...")
    mock_data = torch.randn(1, 60, 1).to(device)
    
    # print first few points just to show data exists
    print(f"First 5 data points: {mock_data[0, :5, 0].numpy()}")

    # ==========================================
    # 3. INFERENCE
    # ==========================================
    with torch.no_grad():
        # Get raw logits
        outputs = model(mock_data)
        
        # Apply Softmax to get probabilities (optional, for display)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get predicted class (index with highest value)
        _, predicted_class = torch.max(outputs, 1)
        
        pred_idx = predicted_class.item()
        action = get_action_name(pred_idx)
        
        print("\n" + "="*30)
        print(f"PREDICTION RESULTS")
        print("="*30)
        print(f"Raw Logits:     {outputs.numpy()}")
        print(f"Probabilities:  {probs.numpy()}")
        print(f"Predicted Class: {pred_idx}")
        print(f"Final Action:    {action}")
        print("="*30)