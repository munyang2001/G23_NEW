import torch
from torch.utils.data import Dataset, DataLoader, random_split
from agents.Group23.Network.HexPolicyNet import *
from agents.Group23.Network.Loss import AlphaZeroLoss


class HexDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        # Load data (weights_only=False required for older formats/tuples)
        self.data = torch.load(data_path, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_state, pi, v = self.data[idx]

        # Convert to float32 tensors
        board_state = torch.tensor(board_state, dtype=torch.float32)
        pi = torch.tensor(pi, dtype=torch.float32)
        v = torch.tensor(v, dtype=torch.float32)

        return board_state, pi, v


if __name__ == "__main__":
    # ----Constants----
    LR = 0.001
    WEIGHT_DECAY = 0.0001
    EPOCHS = 10
    BATCH_SIZE = 256

    # ----Device Setup----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # ----Data Loading & Splitting----
    full_dataset = HexDataset("./MASTER_DATASET.pt")

    # 90% Train, 10% Validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Dataset Loaded: {len(full_dataset)} total samples.")
    print(f"Training on {train_size} | Validating on {val_size}")

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ----Model & Setup----
    model = HexResNet()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    criterion = AlphaZeroLoss()

    # ---- Main Loop ----
    for epoch in range(EPOCHS):
        print(f"\n==== Epoch {epoch + 1}/{EPOCHS} ====")

        # --- A. TRAINING PHASE ---
        model.train()
        train_loss = 0.0

        for batch_idx, (boards, pi_targets, v_targets) in enumerate(train_loader):
            boards, pi_targets, v_targets = boards.to(device), pi_targets.to(device), v_targets.to(device)

            optimizer.zero_grad()
            pred_pi, pred_v = model(boards)
            total_loss, _, _ = criterion(pred_pi, pred_v, pi_targets, v_targets)

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

            if batch_idx % 100 == 0:
                print(f" [Batch {batch_idx}] Train Loss: {total_loss.item():.4f}")

        # --- B. VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_pi_loss = 0.0
        val_v_loss = 0.0

        with torch.no_grad():
            for boards, pi_targets, v_targets in val_loader:
                boards, pi_targets, v_targets = boards.to(device), pi_targets.to(device), v_targets.to(device)

                pred_pi, pred_v = model(boards)
                total_l, l_pi, l_v = criterion(pred_pi, pred_v, pi_targets, v_targets)

                val_loss += total_l.item()
                val_pi_loss += l_pi.item()
                val_v_loss += l_v.item()

        # --- C. METRICS & SAVING ---
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_pi = val_pi_loss / len(val_loader)
        avg_val_v = val_v_loss / len(val_loader)

        print(
            f"--> Result: Train Loss: {avg_train_loss:.4f} || Val Loss: {avg_val_loss:.4f} (Pi: {avg_val_pi:.4f}, V: {avg_val_v:.4f})")

        # Save Checkpoint
        save_path = f"./hex_model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), save_path)

        # Step Scheduler
        scheduler.step()

    print("\nTraining Complete!")