import torch
from torch.utils.data import Dataset, DataLoader
from agents.Group23.Network.HexPolicyNet import *
from agents.Group23.Network.Loss import AlphaZeroLoss

class HexDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = torch.load(data_path, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert len(self.data[idx]) == 3, "Wrong data format!"

        board_state, pi, v = self.data[idx]

        board_state = torch.tensor(board_state, dtype=torch.float32)
        pi = torch.tensor(pi, dtype=torch.float32)
        v = torch.tensor(v, dtype=torch.float32)

        sample = (board_state, pi, v,)

        return sample


if __name__ == "__main__":
    # For splitting the training and validation datasets
    from torch.utils.data import random_split

    # ---- Data Loader ----
    full_dataset = HexDataset("./MASTER_DATASET.pt")

    train_size = int(len(full_dataset) * 0.9)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training on {train_size} samples | Validating on {val_size} samples")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # ----Constants----
    LR = 0.001
    WEIGHT_DECAY = 0.0001
    EPOCHS = 10

    # ----Device Setup----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"On device: {device}")

    # ----Model Initialisation----
    model = HexResNet()
    model.to(device)  # Move the model to GPU

    # ----Optimiser Initialisation----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ----Learning Rate Scheduler----
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # Lowers the lr by 10 times every 3 steps.

    # ---- Loss Function ----
    criterion = AlphaZeroLoss()

    # ---- Training Loop ----
    model.train()  # Switch to training mode (enables Dropout/BatchNorm)

    for epoch in range(EPOCHS):
        print(f"---- Epoch {epoch + 1}/{EPOCHS} ----")

        # We track running loss to see average performance over the epoch
        running_loss = 0.0

        for batch_idx, (boards, pi_targets, v_targets) in enumerate(train_loader):
            # 1. Move data to device
            boards = boards.to(device)
            pi_targets = pi_targets.to(device)
            v_targets = v_targets.to(device)

            # 2. Zero Gradients
            optimizer.zero_grad()

            # 3. Forward Pass
            # Output will be: pred_pi, pred_v
            pred_pi, pred_v = model(boards)

            # 4. Calculate Loss
            # It returns: total_loss, loss_pi, loss_v
            total_loss, loss_pi, loss_v = criterion(pred_pi, pred_v, pi_targets, v_targets)

            # 5. Backward Pass
            total_loss.backward()

            # 6. Optimizer Step
            optimizer.step()

            running_loss += total_loss.item()

            # 7. Print status every 50 batches
            if batch_idx % 50 == 0:
                print(
                    f"Batch: {batch_idx} | Total Loss: {total_loss.item():.4f} | policy_loss: {loss_pi.item():.4f}, value_loss: {loss_v.item():.4f}"
                )

        # ---- Training Loss ----
        avg_loss = running_loss / len(train_loader)
        print(f"====> Epoch {epoch + 1} Complete | Average Loss: {avg_loss:.4f}")

        # ---- Validation Step ----
        model.eval()  # Switch to evaluation mode (turns off Dropout/BatchNorm updates)
        val_loss = 0.0
        val_pi_loss = 0.0
        val_v_loss = 0.0

        # We don't need gradients for validation (saves VRAM and speed)
        with torch.no_grad():
            for boards, pi_targets, v_targets in val_loader:
                boards = boards.to(device)
                pi_targets = pi_targets.to(device)
                v_targets = v_targets.to(device)

                pred_pi, pred_v = model(boards)

                # Sum up the losses
                total_l, l_pi, l_v = criterion(pred_pi, pred_v, pi_targets, v_targets)
                val_loss += total_l.item()
                val_pi_loss += l_pi.item()
                val_v_loss += l_v.item()

        # Calculate averages
        avg_val_loss = val_loss / len(val_loader)
        avg_val_pi = val_pi_loss / len(val_loader)
        avg_val_v = val_v_loss / len(val_loader)

        print(f"====> Validation: Avg Loss: {avg_val_loss:.4f} (Pi: {avg_val_pi:.4f}, V: {avg_val_v:.4f})")

        model.train()  # CRITICAL: Switch back to train mode for the next epoch!

        # Save the weights.
        save_path = f"./hex_model_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)

        # Step the scheduler
        scheduler.step()

    print("\nTraining Complete!")
