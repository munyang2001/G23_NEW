import torch
import numpy as np
from agents.Group23.Network.HexPolicyNet import HexResNet

# CONFIG
MODEL_PATH = "./hex_model_epoch_10.pth"  # Ensure this matches your actual file name
BOARD_SIZE = 11


def run_sanity_check():
    # 1. Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {MODEL_PATH} to {device}...")

    model = HexResNet()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.to(device)
    model.eval()  # Important: Disables Dropout/BatchNorm randomness

    # 2. Create an Empty Board Input
    # Shape: (Batch=1, Channels=3, Height=11, Width=11)
    # Channels: [My Stones, Opponent Stones, Empty Stones]
    input_tensor = torch.zeros((1, 3, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)

    # Set Channel 2 (Empty Stones) to 1.0 everywhere
    input_tensor[0, 2, :, :] = 1.0

    # Move to GPU
    input_tensor = input_tensor.to(device)

    # 3. Inference
    with torch.no_grad():
        pi_logits, v = model(input_tensor)

    # 4. Interpret Results
    # Value: Convert scalar to float
    win_value = v.item()

    # Policy: Convert logits to probabilities (Softmax)
    pi_probs = torch.softmax(pi_logits, dim=1).cpu().numpy().flatten()

    # 5. Visualization
    print(f"\n--- RESULTS FOR EMPTY BOARD ---")
    print(f"Predicted Value (v): {win_value:.4f}")
    print(f"(Expected: Close to 0.0 or slightly positive, as First Player usually wins)")

    print("\n--- TOP 5 RECOMMENDED MOVES ---")
    # Get indices of top 5 probabilities
    top_indices = np.argsort(pi_probs)[::-1][:5]

    for rank, idx in enumerate(top_indices):
        r, c = divmod(idx, BOARD_SIZE)
        prob = pi_probs[idx]
        print(f"#{rank + 1}: ({r}, {c}) -> Probability: {prob:.4f} ({prob * 100:.2f}%)")

    # Quick ASCII Heatmap of the center
    print("\n--- CENTER 3x3 PROBABILITIES ---")
    center_probs = pi_probs.reshape(11, 11)[4:7, 4:7]
    print(center_probs)


if __name__ == "__main__":
    run_sanity_check()