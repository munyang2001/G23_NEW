import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from agents.Group23.Network.HexPolicyNet import HexResNet

# --- CONFIG ---
DATA_PATH = "DATA_GEN1_60K.pt"  # Rename your file to this
BATCH_SIZE = 512  # RTX 4090 can handle big batches
EPOCHS = 15  # Quick training
LR = 0.001  # Standard Adam LR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"--- TRAINING GEN 2 MODEL ---")

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    print("Loading dataset...")
    raw_data = torch.load(DATA_PATH)
    # raw_data is list of (input, pi, z)

    # Convert to Tensors
    inputs = torch.stack([item[0] for item in raw_data])
    policies = torch.stack([torch.from_numpy(item[1]) for item in raw_data])
    values = torch.tensor([item[2] for item in raw_data], dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(inputs, policies, values)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(dataset)} samples.")

    # 2. Initialize NEW Model (8 Blocks)
    # Ensure you updated HexPolicyNet.py to default num_blocks=8!
    model = HexResNet(num_blocks=8).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # 3. Training Loop
    print("Starting Training...")
    model.train()

    for epoch in range(EPOCHS):
        total_p_loss = 0
        total_v_loss = 0

        for batch_idx, (x, pi, z) in enumerate(loader):
            x, pi, z = x.to(DEVICE), pi.to(DEVICE), z.to(DEVICE)

            optimizer.zero_grad()

            pred_pi, pred_v = model(x)

            # Loss Calculation
            # Policy: Cross Entropy between target prob (pi) and logit output (pred_pi)
            # We use CrossEntropyLoss but targets are probabilities, so we need:
            # -sum(target * log_softmax(pred))
            log_probs = torch.log_softmax(pred_pi, dim=1)
            policy_loss = -torch.sum(pi * log_probs) / x.size(0)

            # Value: MSE
            value_loss = F.mse_loss(pred_v, z)

            total_loss = policy_loss + value_loss
            total_loss.backward()
            optimizer.step()

            total_p_loss += policy_loss.item()
            total_v_loss += value_loss.item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Policy Loss: {total_p_loss / len(loader):.4f} | Value Loss: {total_v_loss / len(loader):.4f}")

    # 4. Save
    save_path = "../../../hex_model_gen2.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Training Complete. Saved to {save_path}")
    print("Next Step: Update 'generate_gen1.py' to load this new model and run it for the rest of the day!")


import torch.nn.functional as F

if __name__ == "__main__":
    main()