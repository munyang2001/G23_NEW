import torch
import torch.optim as optim
from HexPolicyNet import HexResNet  # Assuming your file is named this
from Loss import AlphaZeroLoss  # Assuming you saved the Loss class here

# 1. Setup the Network and Components
# -----------------------------------
board_size = 11
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the ResNet (The Brain)
model = HexResNet(board_size=board_size).to(device)

# Initialize the Loss Function (The Scoreboard)
criterion = AlphaZeroLoss()

# Initialize the Optimizer (The Driver)
# The paper suggests Adam is superior for Hex
# lr=0.001 is a standard starting point.
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


def train_step(batch_data):
    """
    Performs a single update of the weights.
    batch_data: A tuple containing (boards, target_pis, target_vs)
    """
    # Unpack the data
    # boards: (Batch, 4, 11, 11)
    # target_pis: (Batch, 121) -> Probabilities from MCTS
    # target_vs: (Batch, 1)    -> Winner (+1 or -1)
    boards, target_pis, target_vs = batch_data

    # Move data to GPU (if available)
    boards = boards.to(device)
    target_pis = target_pis.to(device)
    target_vs = target_vs.to(device)

    # --- Step 1: Zero the Gradients ---
    # PyTorch accumulates gradients by default. We must reset them.
    optimizer.zero_grad()

    # --- Step 2: Forward Pass ---
    # Ask the model for predictions
    out_pi, out_v = model(boards)

    # --- Step 3: Calculate Loss ---
    # Compare predictions to MCTS targets
    total_loss = criterion(out_pi, out_v, target_pis, target_vs)

    # --- Step 4: Backward Pass ---
    # Calculate how much every weight contributed to the error
    total_loss.backward()

    # --- Step 5: Update Weights ---
    # Adjust the weights slightly to reduce error next time
    optimizer.step()

    return total_loss.item()


# --- Simulation Exercise ---
if __name__ == "__main__":
    print(f"Training on device: {device}")

    # Let's create Fake MCTS Data to test the loop
    batch_size = 8

    # 1. Fake Boards (Random 0s and 1s)
    fake_boards = torch.randn(batch_size, 4, 11, 11)

    # 2. Fake MCTS Probabilities (Random vectors that sum to 1)
    fake_pis = torch.rand(batch_size, 121)
    fake_pis = fake_pis / fake_pis.sum(dim=1, keepdim=True)  # Normalize to sum to 1

    # 3. Fake Winners (Random +1 or -1)
    fake_vs = torch.randn(batch_size, 1)

    # Run one training step
    loss = train_step((fake_boards, fake_pis, fake_vs))

    print(f"Success! The network trained on one batch.")
    print(f"Total Error (Loss): {loss:.4f}")