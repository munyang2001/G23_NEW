import random
import numpy as np
import concurrent.futures

# --- IMPORTS ---
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.AgentBase import AgentBase
from agents.Group23.board_sets import Board_Optimized
from agents.Group23.Agent_Parallel import MCTS

# --- CONFIGURATION ---
NUM_GAMES = 1  # Single debug game
TIME_LIMIT = 1.0
WORKERS = 4


# --- UPDATE THIS HELPER ---
def print_top_moves(pi_target, top_k=5):
    """Prints top K moves regardless of how small the probability is."""
    indexed_probs = [(i, prob) for i, prob in enumerate(pi_target)]
    indexed_probs.sort(key=lambda x: x[1], reverse=True)

    print(f"   MCTS Beliefs (Top {top_k}):")
    for i in range(min(top_k, len(indexed_probs))):
        idx, prob = indexed_probs[i]
        r, c = idx // 11, idx % 11
        # Print with higher precision to see the 0.8% vs 0.9% difference
        print(f"   - ({r}, {c}): {prob * 100:.2f}%")

# def print_top_moves(pi_target, top_k=5):
#     """Prints the top K moves from the policy vector in a human-readable format."""
#     indexed_probs = [(i, prob) for i, prob in enumerate(pi_target)]
#     indexed_probs.sort(key=lambda x: x[1], reverse=True)
#
#     print(f"   MCTS Beliefs (Top {top_k}):")
#     for i in range(min(top_k, len(indexed_probs))):
#         idx, prob = indexed_probs[i]
#         if prob < 0.01: break
#         r, c = idx // 11, idx % 11
#         print(f"   - ({r}, {c}): {prob * 100:.1f}%")


def mcts_worker(board, time_limit, colour, seed):
    random.seed(seed)
    mcts = MCTS()
    return mcts.search(board, colour, time_limit)


class DataCollectorAgent(AgentBase):
    def __init__(self, colour: Colour, executor):
        super().__init__(colour)
        self.executor = executor
        self.game_history = []

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
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
        total_sims = 0

        for f in concurrent.futures.as_completed(futures):
            try:
                result, sims = f.result()
                total_sims += sims
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
            # We raise visits to a power (e.g., 2 or 3) to emphasize the best moves.
            # This is standard practice in AlphaZero to create cleaner training targets.
            temperature_exponent = 3.0

            total_powered = 0
            powered_stats = {}

            for move, visits in aggregated_stats.items():
                powered_val = visits ** temperature_exponent
                powered_stats[move] = powered_val
                total_powered += powered_val

            # Create Policy Target
            pi_target = np.zeros(121, dtype=np.float32)

            for (r, c), p_val in powered_stats.items():
                if self.colour == Colour.BLUE:
                    # Rotation for Blue
                    idx = c * 11 + r
                else:
                    idx = r * 11 + c

                pi_target[idx] = p_val / total_powered

            input_tensor = optimized_board.to_nn_input(self.colour)
            self.game_history.append((input_tensor, pi_target))

            print(f"   Simulations: {sum(aggregated_stats.values())}")
            print_top_moves(pi_target)  # Print the sharpened stats

            # Pick move based on SHARPENED probs
            moves = list(aggregated_stats.keys())  # Original keys
            # We map probabilities back to the moves correctly
            probs = []
            for m in moves:
                # Retrieve the sharpened probability we just calculated
                if self.colour == Colour.BLUE:
                    idx = m[1] * 11 + m[0]
                else:
                    idx = m[0] * 11 + m[1]
                probs.append(pi_target[idx])

            # Normalize probs again just to be safe for random.choice (floating point errors)
            probs = np.array(probs)
            probs /= probs.sum()

            choice_idx = np.random.choice(len(moves), p=probs)
            r, c = moves[choice_idx]
            return Move(r, c)

        return Move(legal_moves[0][0], legal_moves[0][1])


def main():
    print(f"--- STARTING DEBUG MATCH ({NUM_GAMES} Game) ---")

    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:

        board = Board(board_size=11)
        red_agent = DataCollectorAgent(Colour.RED, executor)
        blue_agent = DataCollectorAgent(Colour.BLUE, executor)

        current_colour = Colour.RED
        turn = 1
        opp_move = None
        winner = None

        while winner is None:
            active_agent = red_agent if current_colour == Colour.RED else blue_agent
            player_name = "RED" if current_colour == Colour.RED else "BLUE"

            print(f"\n=== TURN {turn}: {player_name} Thinking... ===")
            move = active_agent.make_move(turn, board, opp_move)

            if move.x == -1:
                print(f"!!! {player_name} SWAPS !!!")
                red_agent, blue_agent = blue_agent, red_agent
                red_agent.colour = Colour.RED
                blue_agent.colour = Colour.BLUE
            else:
                print(f"   Action: Placed at ({move.x}, {move.y})")
                board.set_tile_colour(move.x, move.y, current_colour)
                current_colour = Colour.BLUE if current_colour == Colour.RED else Colour.RED

            if move.x != -1 and board.has_ended(active_agent.colour):
                # FIX: Fetch the winner immediately before breaking!
                winner = board.get_winner()
                break

            opp_move = move
            turn += 1
            winner = board.get_winner()

        winner_name = "RED" if winner == Colour.RED else "BLUE"
        print(f"\n\n=== GAME OVER ===")
        print(f"Winner: {winner_name}")

        # Check Tensor Shape
        if len(red_agent.game_history) > 0:
            sample_tensor = red_agent.game_history[0][0]
            print(f"Input Tensor Shape: {sample_tensor.shape} (Should be (3, 11, 11))")


if __name__ == "__main__":
    main()