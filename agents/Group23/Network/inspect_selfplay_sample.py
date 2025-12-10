# inspect_selfplay_sample.py

import os
import sys
import glob
import random
import torch

# --- PATH SETUP (same style as generate_data.py) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.Colour import Colour


def find_any_selfplay_file():
    candidates = sorted(glob.glob("self_play_data_*.pt"))
    if not candidates:
        raise FileNotFoundError("No self_play_data_*.pt files found in current directory.")
    return candidates[0]


def main():
    filename = find_any_selfplay_file()
    print(f"Loading: {filename}")
    data = torch.load(filename, map_location="cpu", weights_only=False)

    print(f"Total samples: {len(data)}")
    if len(data) == 0:
        return

    # Pick a random sample
    idx = random.randrange(len(data))
    sample = data[idx]

    board_tensor = sample[0]
    pi = sample[1]
    z = sample[2]

    print(f"\nRandom sample index: {idx}")
    print(f"board_tensor type: {type(board_tensor)}, shape: {getattr(board_tensor, 'shape', None)}")
    print(f"pi type:            {type(pi)}, shape: {getattr(pi, 'shape', None)}")

    if isinstance(z, torch.Tensor):
        z_val = float(z.item())
    else:
        z_val = float(z)
    print(f"z (outcome):        {z_val}")

    # Basic sanity checks
    if isinstance(board_tensor, torch.Tensor):
        print(f"board_tensor dtype: {board_tensor.dtype}")
    if isinstance(pi, torch.Tensor):
        print(f"pi dtype:           {pi.dtype}")
        print(f"pi sum:             {float(pi.sum().item()):.6f}")
        print(f"pi min/max:         {float(pi.min().item()):.6f} / {float(pi.max().item()):.6f}")

    # Optional: check if 'my stones' and 'opp stones' are 0/1-ish
    if isinstance(board_tensor, torch.Tensor) and board_tensor.ndim == 3:
        my_plane = board_tensor[0].flatten()
        opp_plane = board_tensor[1].flatten()
        print(f"\nMy plane unique values (first 10): {my_plane.unique()[:10]}")
        print(f"Opp plane unique values (first 10): {opp_plane.unique()[:10]}")

    # Optional: guess perspective by counting stones
    if isinstance(board_tensor, torch.Tensor):
        my_count = int(board_tensor[0].sum().item())
        opp_count = int(board_tensor[1].sum().item())
        print(f"\nMy stones count:  {my_count}")
        print(f"Opp stones count: {opp_count}")
        print("Note: 'my stones' plane corresponds to the player whose z=+1 "
              "if your sign convention is 'current player win = +1'.")


if __name__ == "__main__":
    main()
