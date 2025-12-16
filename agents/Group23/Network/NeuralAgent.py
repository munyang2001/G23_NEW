import torch
import math
import time
import numpy as np
import random
import os

from src.Colour import Colour
from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from agents.Group23.OptimisedBoardV2 import Board_Optimized
from agents.Group23.Network.HexPolicyNet import HexResNet

# --- CONFIGURATION ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hex_model_final_tuned.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Exploration (PUCT):
# High exploration for Data Generation to find new tactical ideas.
# (Set to 1.25 for competitive play, 3.0 - 4.0 for training data gen)
CPU_CT = 1.25


class NeuralNode:
    __slots__ = ['parent', 'move', 'player', 'visits', 'value_sum', 'prior', 'children']

    def __init__(self, parent=None, move=None, player=None, prior=0.0):
        self.parent = parent
        self.move = move
        self.player = player
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}

    def is_leaf(self):
        return len(self.children) == 0

    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class NeuralMCTS:
    def __init__(self, model):
        self.model = model
        self.root = NeuralNode(player=None)

    def update_root(self, move):
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            self.root = NeuralNode(player=None)

    def search(self, board, color, time_limit):
        time_start = time.time()

        if self.root.player is None:
            self.root.player = 1 if color == Colour.RED else 2

        if self.root.is_leaf():
            self.expand_and_evaluate(self.root, board)

        # SIMULATION LOOP
        # We can add a max_simulations break here if needed for consistent speed
        while time.time() - time_start < time_limit:
            node = self.root
            search_board = board.copy()
            play_func = search_board.play

            # 1. SELECTION
            while not node.is_leaf():
                node = self.select_child(node)
                mv = node.move
                move_colour = Colour.RED if (3 - node.player) == 1 else Colour.BLUE
                play_func(mv[0], mv[1], move_colour)

            # 2. CHECK TERMINAL / EXPANSION
            winner = search_board.winner
            if winner is not None:
                if winner == Colour.RED:
                    v = 1.0 if node.player == 1 else -1.0
                else:
                    v = 1.0 if node.player == 2 else -1.0
            else:
                v = self.expand_and_evaluate(node, search_board)

            # 3. BACKPROPAGATION
            self.backpropagate(node, v)

        # Fallback
        if not self.root.children:
            return list(board.get_legal_moves())[0]

        # Select best move
        best_move = max(self.root.children.items(), key=lambda item: item[1].visits)[0]
        return best_move

    def select_child(self, node):
        best_score = -float('inf')
        best_child = None
        sqrt_parent_visits = math.sqrt(node.visits)

        for child in node.children.values():
            q_val = -child.value()
            u_val = CPU_CT * child.prior * sqrt_parent_visits / (1 + child.visits)
            score = q_val + u_val

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand_and_evaluate(self, node, board):
        current_colour = Colour.RED if node.player == 1 else Colour.BLUE

        # 1. Inference
        input_tensor = torch.from_numpy(board.to_nn_input(current_colour)).unsqueeze(0).float().to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            pi_logits, v = self.model(input_tensor)

        value = v.item()

        # 2. Temperature/Sharpening
        # Reduced from 3.0 to 1.5 to allow for more creative exploration in Gen 2
        sharpening_factor = 1.0
        pi_probs = torch.softmax(pi_logits * sharpening_factor, dim=1).cpu().numpy().flatten()

        # 3. Mask Illegal Moves & Renormalize
        legal_moves = list(board.get_legal_moves())
        policy_sum = 0.0

        # Helper to calculate index based on rotation
        def get_index(r, c):
            if current_colour == Colour.BLUE:
                return c * 11 + r
            else:
                return r * 11 + c

        for r, c in legal_moves:
            policy_sum += pi_probs[get_index(r, c)]

        # 4. Expand Children
        for r, c in legal_moves:
            idx = get_index(r, c)

            # Safe Renormalization
            if policy_sum > 0:
                prior = pi_probs[idx] / policy_sum
            else:
                prior = 1.0 / len(legal_moves)

            next_player = 3 - node.player
            child = NeuralNode(parent=node, move=(r, c), player=next_player, prior=prior)
            node.children[(r, c)] = child

        return value

    def backpropagate(self, node, value):
        curr = node
        current_val = value
        while curr is not None:
            curr.visits += 1
            curr.value_sum += current_val
            current_val = -current_val
            curr = curr.parent


class Agent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)

        # --- CRITICAL UPDATE: 8 Blocks ---
        self.model = HexResNet(num_blocks=8)

        try:
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        except FileNotFoundError:
            print(f"ERROR: Model not found at {MODEL_PATH}")
            exit()
        except RuntimeError as e:
            print(f"ERROR: Architecture Mismatch. Did you train 8 blocks? {e}")
            exit()

        self.model.to(DEVICE)
        print(f"Successfully loaded model to {DEVICE}")
        self.mcts = NeuralMCTS(self.model)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        # Standard Pie Rule Check
        if turn == 2 and opp_move is not None:
            if 2 <= opp_move.x <= 8 and 2 <= opp_move.y <= 8:
                return Move(-1, -1)

        if opp_move is not None:
            self.mcts.update_root((opp_move.x, opp_move.y))

        # Use efficient board
        optimized_board = Board_Optimized.from_game_board(board, self.colour)

        # For Competitive Play: 4.0s.
        # For Generation: This parameter is usually overridden by the generator script.
        best_move = self.mcts.search(optimized_board, self.colour, time_limit=8.0)

        self.mcts.update_root(best_move)
        return Move(best_move[0], best_move[1])