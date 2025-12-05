import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure we can import agents
sys.path.append(os.getcwd())

# Use the new ResNet class (Make sure your file is named HexPolicyNet.py and class is HexResNet)
# If your class is named HexPolicyNet in the file, change HexResNet to HexPolicyNet below.
from agents.PolicyNetwork.HexPolicyNet import HexResNet

# --- Loss Function (Based on your provided code) ---
class AlphaZeroLoss(nn.Module):
    def __init__(self):
        super(AlphaZeroLoss, self).__init__()
        # 1. For the Value Head (Regression)
        self.mse_loss = nn.MSELoss()

        # 2. For the Policy Head (Classification)
        # We do NOT use CrossEntropyLoss directly because it expects 'Class Indices'.
        # We have 'Probabilities' (MCTS visits). So we handle the math manually.

    def forward(self, pred_policy_logits, pred_value, target_probs, target_values):
        """
        pred_policy_logits: The raw output from the Policy Head (before Softmax)
        pred_value:         The raw output from the Value Head (between -1 and 1)
        target_probs:       The 'True' probabilities from MCTS (N, 121)
        target_values:      The 'True' winner (+1 or -1) (N, 1)
        """
        # --- A. Calculate Value Loss ---
        # Simple distance between Prediction and Reality
        # Ensure target_values has the same shape as pred_value (N, 1)
        value_loss = self.mse_loss(pred_value.view(-1), target_values.view(-1))

        # --- B. Calculate Policy Loss ---
        # 1. Convert Logits to Log-Probabilities (more numerically stable than log(softmax))
        log_probs = torch.log_softmax(pred_policy_logits, dim=1)

        # 2. Multiply by Target Probabilities (Cross Entropy Formula)
        # Formula: - sum( target * log(prediction) )
        policy_loss = -torch.mean(torch.sum(target_probs * log_probs, dim=1))

        # --- C. Combine ---
        total_loss = policy_loss + value_loss

        return total_loss

# --- Configuration ---
DATA_FILE = "data/mcts_advanced_games.pt"
MODEL_PATH = "models/hex_resnet_v1.pth"
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

class HexDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # data item: (board, policy_target, value_target)
        return self.data[idx]

def train():
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file {DATA_FILE} not found.")
        print("Please run generate_mohex_data.py first.")
        return

    print("Loading data...")
    try:
        raw_data = torch.load(DATA_FILE)
        print(f"Loaded {len(raw_data)} samples.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if len(raw_data) == 0:
        print("Data file is empty!")
        return

    dataset = HexDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Initialize ResNet
    # Note: Assuming board_size=11, in_channels=4 (as in your board_set.py logic presumably)
    model = HexResNet(board_size=11, in_channels=4, num_blocks=4, width=128).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = AlphaZeroLoss()
    
    print(f"Starting training for {EPOCHS} epochs...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (boards, target_policies, target_values) in enumerate(dataloader):
            boards = boards.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device).float() # Ensure float for MSE
            
            optimizer.zero_grad()
            
            # Forward pass returns TWO values: policy_logits and value
            pred_policy_logits, pred_value = model(boards)
            
            # Calculate combined AlphaZero loss
            loss = criterion(pred_policy_logits, pred_value, target_policies, target_values)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    if not os.path.exists("models"): os.makedirs("models")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()