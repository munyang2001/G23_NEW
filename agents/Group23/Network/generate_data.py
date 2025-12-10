import uuid
import random
import torch
import numpy as np
import concurrent.futures
from tqdm import tqdm
import os
import sys

# --- PATH SETUP (To fix ModuleNotFoundError) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)

# --- IMPORTS ---
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.AgentBase import AgentBase
from agents.Group23.board_sets import Board_Optimized
from agents.Group23.Agent_Parallel import MCTS

# --- CONFIGURATION ---
NUM_GAMES = 625  # Set this to however many games you want (e.g., 1000)
TIME_LIMIT = 0.5  # 0.5s is a good balance for data generation
WORKERS = 6
TEMPERATURE_EXP = 3.0  # Sharpening factor (Higher = more decisive targets)


# --- 1. THE WORKER ---
def mcts_worker(board, time_limit, colour, seed):
    random.seed(seed)
    mcts = MCTS()
    return mcts.search(board, colour, time_limit)


# --- 2. THE AGENT ---
class DataCollectorAgent(AgentBase):
    def __init__(self, colour: Colour, executor):
        super().__init__(colour)
        self.executor = executor
        self.game_history = []

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        # --- SWAP HEURISTIC ---
        if turn == 2 and opp_move is not None:
            if 2 <= opp_move.x <= 8 and 2 <= opp_move.y <= 8:
                return Move(-1, -1)

        optimized_board = Board_Optimized.from_game_board(board, self.colour)
        legal_moves = list(optimized_board.get_legal_moves())

        if len(legal_moves) == 1:
            return Move(legal_moves[0][0], legal_moves[0][1])

        seeds = [random.getrandbits(64) for _ in range(WORKERS)]
        futures = []
        for i in range(WORKERS):
            futures.append(self.executor.submit(
                mcts_worker, optimized_board, TIME_LIMIT, self.colour, seeds[i]
            ))

        aggregated_stats = {}

        for f in concurrent.futures.as_completed(futures):
            try:
                result, sims = f.result()
                for move, visits in result.items():
                    aggregated_stats[move] = aggregated_stats.get(move, 0) + visits
            except Exception:
                continue

        if aggregated_stats:
            # --- DIRICHLET NOISE (Exploration) ---
            # We add noise to the visit counts to encourage the agent to try
            # different moves during self-play, preventing it from playing
            # the exact same game 1000 times.

            # Extract moves and counts
            moves = list(aggregated_stats.keys())
            counts = np.array([aggregated_stats[m] for m in moves], dtype=np.float32)
            total_visits = counts.sum()

            # Normalize to get probability distribution (pi)
            pi_clean = counts / total_visits

            # Generate Dirichlet Noise
            # alpha=0.3 is standard for Chess (average ~35 moves).
            # For Hex 11x11 (average ~60 moves), 0.3 is a good starting point.
            alpha = 0.3
            noise = np.random.dirichlet([alpha] * len(moves))

            # Mix: 75% Truth + 25% Noise
            epsilon = 0.25
            pi_noisy = (1 - epsilon) * pi_clean + epsilon * noise

            # Update aggregated_stats with noisy counts
            # We scale back up to 'total_visits' so the sharpening logic below works mathematically
            for i, m in enumerate(moves):
                aggregated_stats[m] = pi_noisy[i] * total_visits

            # --- SHARPENING LOGIC ---
            total_powered = 0
            powered_stats = {}

            for move, visits in aggregated_stats.items():
                # Apply temperature to distinguish good moves from noise
                powered_val = visits ** TEMPERATURE_EXP
                powered_stats[move] = powered_val
                total_powered += powered_val

            # --- POLICY TARGET CREATION ---
            pi_target = np.zeros(121, dtype=np.float32)

            for (r, c), p_val in powered_stats.items():
                if self.colour == Colour.BLUE:
                    # CANONICAL ROTATION:
                    # If I am Blue, the NN sees a transposed board (3 channels).
                    # So the label must ALSO be transposed (c, r)
                    idx = c * 11 + r
                else:
                    # Standard Red Perspective
                    idx = r * 11 + c

                pi_target[idx] = p_val / total_powered

            # Save the input tensor (3 channels, rotated if Blue)
            input_tensor = optimized_board.to_nn_input(self.colour)
            self.game_history.append((input_tensor, pi_target))

            # --- SELECT MOVE ---
            # We select based on the sharpened probabilities to encourage good play
            moves = list(aggregated_stats.keys())

            # Map probabilities back to the moves list order
            probs = []
            for m in moves:
                if self.colour == Colour.BLUE:
                    idx = m[1] * 11 + m[0]
                else:
                    idx = m[0] * 11 + m[1]
                probs.append(pi_target[idx])

            probs = np.array(probs)
            probs /= probs.sum()  # Re-normalize to prevent float errors

            choice_idx = np.random.choice(len(moves), p=probs)
            r, c = moves[choice_idx]
            return Move(r, c)

        return Move(legal_moves[0][0], legal_moves[0][1])


# --- 3. MAIN LOOP ---
def main():
    print(f"Generating {NUM_GAMES} games with {WORKERS} workers...")
    all_samples = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:

        for i in tqdm(range(NUM_GAMES)):
            board = Board(board_size=11)

            red_agent = DataCollectorAgent(Colour.RED, executor)
            blue_agent = DataCollectorAgent(Colour.BLUE, executor)

            current_colour = Colour.RED
            turn = 1
            opp_move = None
            winner = None

            while winner is None:
                active_agent = red_agent if current_colour == Colour.RED else blue_agent

                # 1. Move
                move = active_agent.make_move(turn, board, opp_move)

                # 2. Swap Handling
                if move.x == -1:
                    if turn != 2: raise ValueError("Illegal swap")
                    if active_agent.colour != Colour.BLUE: raise ValueError("Illegal swap")

                    # Swap roles
                    red_agent, blue_agent = blue_agent, red_agent
                    # Swap internal colours
                    red_agent.colour = Colour.RED
                    blue_agent.colour = Colour.BLUE

                    # Swap the game_history
                    red_agent.game_history, blue_agent.game_history = blue_agent.game_history, red_agent.game_history

                    # Blue continues to play
                    current_colour = Colour.BLUE
                else:
                    board.set_tile_colour(move.x, move.y, current_colour)
                    current_colour = Colour.BLUE if current_colour == Colour.RED else Colour.RED

                # 3. Check End
                if move.x != -1 and board.has_ended(active_agent.colour):
                    # FIX: Fetch the winner immediately before breaking!
                    winner = board.get_winner()
                    break

                opp_move = move
                turn += 1
                winner = board.get_winner()

            # --- LABEL & SAVE ---
            z_red = 1.0 if winner == Colour.RED else -1.0
            z_blue = 1.0 if winner == Colour.BLUE else -1.0  # aka -z_red

            # For the Red agent, z=1 means Win, z=-1 means Loss
            for tensor, pi in red_agent.game_history:
                all_samples.append((tensor, pi, z_red))

            # For the Blue agent, z=1 means Win (Blue Won), z=-1 means Loss
            for tensor, pi in blue_agent.game_history:
                all_samples.append((tensor, pi, z_blue))

            # Periodic Save
            if i % 10 == 0:
                temp_filename = f"self_play_data_temp.pt"
                torch.save(all_samples, temp_filename)

    # Save with the unique ID
    # Generate a unique 8-character ID for this specific terminal run
    run_id = uuid.uuid4().hex[:8]
    final_filename = f"self_play_data_{run_id}.pt"
    torch.save(all_samples, final_filename)

    print(f"Success! Saved {len(all_samples)} samples to {final_filename}.")


if __name__ == "__main__":
    main()