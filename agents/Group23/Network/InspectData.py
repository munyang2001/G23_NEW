import torch
import numpy as np

# Path to combined dataset
DATA_PATH = "MASTER_DATASET.pt"  # Or whatever your file is named


def check_data_quality():
    print(f"Loading {DATA_PATH}...")
    try:
        data = torch.load(DATA_PATH, weights_only=False)
    except FileNotFoundError:
        print("File not found. Please check the path.")
        return

    print(f"Loaded {len(data)} samples.")

    # Check the first few samples and some random ones
    indices = [0, 10, 100, len(data) // 2]

    print("\n--- DATASET INSPECTION ---")
    for idx in indices:
        if idx >= len(data): continue

        # Unpack sample
        board, pi, val = data[idx]

        # Analysis
        max_prob = np.max(pi)
        argmax = np.argmax(pi)
        entropy = -np.sum(pi * np.log(pi + 1e-9))

        print(f"Sample {idx}:")
        print(f"  - Value Target (z): {val}")
        print(f"  - Max Policy Prob:  {max_prob:.4f} ({max_prob * 100:.1f}%)")
        print(f"  - Entropy:          {entropy:.4f} (High=Flat, Low=Sharp)")

        if max_prob < 0.05:
            print("    [WARNING] This sample is extremely flat!")
        else:
            print("    [OK] This sample is sharp.")


if __name__ == "__main__":
    check_data_quality()