import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from agents.Group23.Network.HexPolicyNet import HexResNet

# Import your AlphaZeroLoss

# CONFIG
MIXED_MODEL_PATH = "hex_model_final_mixed.pt"  # The one that failed the test
HIGH_QUALITY_DATA = "DATA_GEN2_60K.pt"  # Your best data (Gen 2)
BATCH_SIZE = 512
EPOCHS = 3  # VERY SHORT! We just want to nudge the weights, not overwrite them.
LR = 0.0001  # VERY LOW! 10x smaller than before.


def main():
    print("--- FINE-TUNING FOR TACTICAL RECOVERY ---")

    # 1. Load the Failed Mixed Model
    model = HexResNet(num_blocks=8).to("cuda")
    model.load_state_dict(torch.load(MIXED_MODEL_PATH))

    # 2. Load ONLY High-Quality Data
    data = torch.load(HIGH_QUALITY_DATA, weights_only=False)

    # Convert to Tensors
    inputs = torch.stack([torch.from_numpy(item[0]).float() for item in data])
    policies = torch.stack([torch.from_numpy(item[1]).float() for item in data])
    values = torch.tensor([item[2] for item in data], dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(inputs, policies, values)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Training with LOW LR
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()

    print(f"Fine-tuning for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, pi, z in loader:
            x, pi, z = x.to("cuda"), pi.to("cuda"), z.to("cuda")
            optimizer.zero_grad()
            pred_pi, pred_v = model(x)

            # Loss (Standard Code)
            log_probs = torch.log_softmax(pred_pi, dim=1)
            p_loss = -torch.sum(pi * log_probs) / x.size(0)
            v_loss = torch.nn.functional.mse_loss(pred_v, z)
            loss = p_loss + v_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss {total_loss / len(loader):.4f}")

    # 4. Save as the TRUE FINAL model
    torch.save(model.state_dict(), "hex_model_final_tuned.pt")
    print("Saved hex_model_final_tuned.pt")


if __name__ == "__main__":
    main()