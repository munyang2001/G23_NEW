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
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hex_model_epoch_10.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Exploration (PUCT):
# INCREASED to 3.0. Since our priors are weak, we need high exploration
# to force the MCTS to consult the Value Head on many different moves.
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

        sim_count = 0
        while time.time() - time_start < time_limit:
            node = self.root
            search_board = board.copy()
            play_func = search_board.play

            # 1. SELECTION
            while not node.is_leaf():
                node = self.select_child(node)

                mv = node.move
                # Determine move color based on who made the move (prev player)
                move_colour = Colour.RED if (3 - node.player) == 1 else Colour.BLUE
                play_func(mv[0], mv[1], move_colour)

            # 2. CHECK TERMINAL / EXPANSION
            winner = search_board.winner
            if winner is not None:
                # Value relative to node.player
                if winner == Colour.RED:
                    v = 1.0 if node.player == 1 else -1.0
                else:
                    v = 1.0 if node.player == 2 else -1.0
            else:
                v = self.expand_and_evaluate(node, search_board)

            # 3. BACKPROPAGATION
            self.backpropagate(node, v)
            sim_count += 1

        # Fallback
        if not self.root.children:
            return list(board.get_legal_moves())[0]

        # Select best move
        best_move = max(self.root.children.items(), key=lambda item: item[1].visits)[0]

        return best_move

    def select_child(self, node):
        best_score = -float('inf')
        best_child = None

        # Pre-calculate sqrt for speed
        sqrt_parent_visits = math.sqrt(node.visits)

        for child in node.children.values():
            # Q-value: Negate because we want to choose the move that is BAD for opponent
            q_val = -child.value()

            # PUCT Formula
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

        # 2. Sharpening (Optional but helpful for weak Gen-0 policies)
        # We multiply logits by 2.0 or 3.0 to make the "best" moves stand out more
        sharpening_factor = 3.0
        pi_probs = torch.softmax(pi_logits * sharpening_factor, dim=1).cpu().numpy().flatten()

        # 3. Mask Illegal Moves & Renormalize
        legal_moves = list(board.get_legal_moves())
        policy_sum = 0.0

        for r, c in legal_moves:
            if current_colour == Colour.BLUE:
                idx = c * 11 + r
            else:
                idx = r * 11 + c
            policy_sum += pi_probs[idx]

        # 4. Expand Children
        for r, c in legal_moves:
            if current_colour == Colour.BLUE:
                idx = c * 11 + r
            else:
                idx = r * 11 + c

            # Renormalize: P(move) = P(raw) / Sum(legal)
            if policy_sum > 0:
                prior = pi_probs[idx] / policy_sum
            else:
                # Fallback (Safety): If network predicts 0.0 for ALL legal moves
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
        self.model = HexResNet()
        try:
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        except FileNotFoundError:
            print(f"ERROR: Model not found at {MODEL_PATH}")
            exit()

        self.model.to(DEVICE)
        self.mcts = NeuralMCTS(self.model)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        if turn == 2 and opp_move is not None:
            if 2 <= opp_move.x <= 8 and 2 <= opp_move.y <= 8:
                return Move(-1, -1)

        if opp_move is not None:
            self.mcts.update_root((opp_move.x, opp_move.y))

        # OPTIMIZATION: Convert board once here
        optimized_board = Board_Optimized.from_game_board(board, self.colour)

        # Run Search
        best_move = self.mcts.search(optimized_board, self.colour, time_limit=4.0)

        self.mcts.update_root(best_move)

        return Move(best_move[0], best_move[1])