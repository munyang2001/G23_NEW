import torch
import numpy as np
import time
import os
import uuid
import random
from tqdm import tqdm

# --- IMPORTS ---
# Adjust these if your folder structure is slightly different
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from agents.Group23.OptimisedBoardV2 import Board_Optimized
from agents.Group23.Network.NeuralAgent import NeuralMCTS
from agents.Group23.Network.HexPolicyNet import HexResNet

# --- CONFIGURATION ---
MODEL_PATH = "agents/Group23/Network/hex_model_epoch_10.pth"  # Ensure correct path
NUM_GAMES = 1000  # Goal for Gen-1
TIME_LIMIT = 0.4  # Fast simulations (~200-400 sims per move)
TEMP_THRESHOLD = 12  # Number of moves to use temperature exploration

# Setup Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 1. Load Model (Global)
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = HexResNet().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.eval()
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
        return

    # 2. Storage
    all_training_samples = []
    print(f"Starting Generation of {NUM_GAMES} games on {DEVICE}...")

    # 3. Main Loop
    for i in tqdm(range(NUM_GAMES)):
        board = Board(board_size=11)

        # Initialize MCTS trees for both players sharing the same model
        # (We reset trees every game for clean data)
        mcts_red = NeuralMCTS(model)
        mcts_blue = NeuralMCTS(model)

        turn = 1
        current_colour = Colour.RED
        game_history = []  # List of (Input, Policy, Player)

        while True:
            # Select active MCTS tree
            active_mcts = mcts_red if current_colour == Colour.RED else mcts_blue

            # Create Optimized Board for Search
            opt_board = Board_Optimized.from_game_board(board, current_colour)

            # --- MCTS SEARCH ---
            # This populates the tree with visit counts based on Value Head guidance
            best_move = active_mcts.search(opt_board, current_colour, time_limit=TIME_LIMIT)

            # --- EXTRACT POLICY (PI) ---
            # 1. Get visit counts from the root
            root_children = active_mcts.root.children
            visits = {move: node.visits for move, node in root_children.items()}
            sum_visits = sum(visits.values())

            # 2. Create Probability Distribution
            pi = np.zeros(121, dtype=np.float32)

            if sum_visits > 0:
                for (r, c), v in visits.items():
                    # CRITICAL: Handle Rotation for Blue
                    # If Blue, the neural net sees a transposed board.
                    # So the output policy must also be transposed (c*11 + r)
                    if current_colour == Colour.BLUE:
                        idx = c * 11 + r
                    else:
                        idx = r * 11 + c
                    pi[idx] = v / sum_visits

            # 3. Save Sample (Input, Policy, Player)
            # We don't know Z (Outcome) yet, so we store Player to assign Z later
            input_tensor = opt_board.to_nn_input(current_colour)
            game_history.append([input_tensor, pi, current_colour])

            # --- SELECT MOVE (Temperature) ---
            if turn <= TEMP_THRESHOLD:
                # Exploration Mode: Sample from the distribution
                # We must map the flattened pi back to moves or sample visits directly
                moves = list(visits.keys())
                counts = np.array([visits[m] for m in moves])

                # Safety check for empty visits (should not happen with search)
                if len(moves) == 0:
                    legal = list(board.get_legal_moves())
                    move_to_play = legal[0]
                else:
                    probs = counts / counts.sum()
                    choice_idx = np.random.choice(len(moves), p=probs)
                    move_to_play = moves[choice_idx]
            else:
                # Competitive Mode: Argmax (Best Move)
                move_to_play = best_move

            # --- PLAY MOVE ---
            board.set_tile_colour(move_to_play[0], move_to_play[1], current_colour)

            # Update Trees (Keep the relevant subtree, discard the rest)
            mcts_red.update_root(move_to_play)
            mcts_blue.update_root(move_to_play)

            # Check End
            if board.has_ended(current_colour):
                winner = current_colour
                break

            # Switch Turn
            current_colour = Colour.BLUE if current_colour == Colour.RED else Colour.RED
            turn += 1

        # --- ASSIGN REWARDS (Z) ---
        # If Red Won: Red samples get +1, Blue samples get -1
        # If Blue Won: Blue samples get +1, Red samples get -1
        for sample in game_history:
            inp, pi_target, player = sample

            if player == winner:
                z = 1.0
            else:
                z = -1.0

            all_training_samples.append((inp, pi_target, z))

        # Optional: Save every 100 games to avoid data loss on crash
        if (i + 1) % 100 == 0:
            temp_name = f"gen1_checkpoint_{i + 1}.pt"
            torch.save(all_training_samples, temp_name)

    # 4. Final Save
    run_id = uuid.uuid4().hex[:8]
    final_filename = f"gen1_data_FULL_{run_id}.pt"
    torch.save(all_training_samples, final_filename)

    print("=" * 40)
    print(f"GENERATION COMPLETE")
    print(f"Total Games: {NUM_GAMES}")
    print(f"Total Samples: {len(all_training_samples)}")
    print(f"Saved to: {final_filename}")
    print("=" * 40)


if __name__ == "__main__":
    main()