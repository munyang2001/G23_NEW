import math
import time
import random
from src.Colour import Colour
from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from agents.Group23.board_set import Board_Optimized
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from agents.Group23 import heuristics
class Node:
    __slots__ = ['parent', 'move', 'player', 'visits', 'wins', 
                 'rave_visits', 'rave_wins', 'children', 'allowed_moves']

    def __init__(self, parent=None, move=None, player=None):
        self.parent = parent
        self.move = move
        self.player = player
        self.visits = 0
        self.wins = 0
        self.rave_visits = 0
        self.rave_wins = 0
        self.children = {}
        self.allowed_moves = None

    def is_fully_expanded(self):
        return self.allowed_moves is not None and len(self.allowed_moves) == 0

    def has_children(self):
        return len(self.children) > 0

class MCTS:
    def __init__(self):
        self._C = math.sqrt(2)
        self._RAVE = 300
        self.root = Node(parent=None, move=None, player=None)
        self.transposition_table = {}

    def update_root(self, move):
        if move in self.root.children:
            self.root = self.root.children[move]
        else:
            self.root = Node(parent=None, move=None, player=None)
        self.root.parent = None
    

    def search(self, board, color, turn, time_limit):
        time_start = time.time()
        
        random_randrange = random.randrange
        
        # 1. Initialize Root
        if self.root.allowed_moves is None:
            self.root.allowed_moves = list(board.get_legal_moves())
            if board.hash not in self.transposition_table:
                self.transposition_table[board.hash] = self.root

        # Convention: Root player is the one who needs to move NOW
        if self.root.player is None:
            self.root.player = 1 if color == Colour.RED else 2
        while time.time() - time_start < time_limit:
            node = self.root
            board_copy = board.copy()
            board_play = board_copy.play
            path = [node]
            red_moves = set()
            blue_moves = set()

            # --- SELECTION ---
            while node.has_children() and node.is_fully_expanded():
                child = self.child_selection(node)
                if child is None:
                    break
                node = child
                path.append(node)

                # RE-PLAY LOGIC:
                # 'node' is the state reached by 'node.move'.
                # 'node.player' is the one to move NEXT.
                # So 'node.move' was made by the PREVIOUS player (3 - node.player).
                mv = node.move
                prev_player = 3 - node.player 
                move_color = Colour.RED if prev_player == 1 else Colour.BLUE
                board_play(mv[0], mv[1], move_color)

                if move_color == Colour.RED:
                    red_moves.add(mv)
                else:
                    blue_moves.add(mv)

                if board_copy.winner is not None:
                    break
            
            # --- EXPANSION ---
            # Dead Cell Pruning
            pruned_moves = []
            for move in self.empty_spots:
                if not heuristics.is_dead_cell(board, color, move[0], move[1]):
                    pruned_moves.append(move)
            if board_copy.winner is None and node.allowed_moves:
                # Pick move for CURRENT player (node.player)
                if pruned_moves:
                    # HEURISTICS SCORING
                    best_move = None
                    best_score = -(10**9)
                    for move in pruned_moves:
                        h = heuristics.heuristic_scoring(board_copy, color, move[0], move[1], turn)
                        if h > best_score:
                            best_score = h
                            best_move = move
                    print(f"best move {best_move}")
                    print(f"best score {best_score}")
                    input("...")

                    node.allowed_moves.remove(best_move)
                    move_to_expand = best_move
                else:
                    # Select random move
                    moves_len = len(node.allowed_moves)
                    rand_idx = random_randrange(moves_len)
                    node.allowed_moves[rand_idx], node.allowed_moves[-1] = node.allowed_moves[-1], node.allowed_moves[rand_idx]
                    move_to_expand = node.allowed_moves.pop()

                current_p = node.player
                play_color = Colour.RED if current_p == 1 else Colour.BLUE
                board_play(move_to_expand[0], move_to_expand[1], play_color)

                if play_color == Colour.RED:
                    red_moves.add(move_to_expand)
                else:
                    blue_moves.add(move_to_expand)

                # The child node will be for the NEXT player
                next_player = 3 - current_p
                
                board_hash = board_copy.hash
                if board_hash in self.transposition_table:
                    candidate = self.transposition_table[board_hash]
                    if candidate.parent is None:
                        child_node = candidate
                        child_node.parent = node
                        child_node.move = move_to_expand
                        child_node.player = next_player
                    else:
                        child_node = Node(parent=node, move=move_to_expand, player=next_player)
                else:
                    child_node = Node(parent=node, move=move_to_expand, player=next_player)
                    self.transposition_table[board_hash] = child_node

                if child_node.allowed_moves is None:
                    child_node.allowed_moves = list(board_copy.get_legal_moves())

                node.children[move_to_expand] = child_node
                node = child_node
                path.append(node)
                
                # Rollout starts with the player whose turn it is at the NEW node
                rollout_player = next_player
            else:
                rollout_player = node.player

            # --- SIMULATION ---
            winner = self.rollout(board_copy, rollout_player, red_moves, blue_moves)
            
            # --- BACKPROPAGATION ---
            self.backpropagate(path, winner, red_moves, blue_moves)

        best_child = max(self.root.children.values(), key=lambda c: c.visits, default=None)
        if best_child is None:
            return random.choice(list(board.get_legal_moves()))
        return best_child.move

    def child_selection(self, node):
        best_score = -float('inf')
        best_node = None
        
        parent_visits = node.visits if node.visits > 0 else 1
        log_parent = math.log(parent_visits)
        
        exploration_numerator = self._C * math.sqrt(log_parent)
        rave_const = self._RAVE

        for child in node.children.values():
            if child.visits == 0:
                return child

            q = child.wins / child.visits
            amaf = (child.rave_wins / child.rave_visits) if child.rave_visits > 0 else 0.0

            beta = rave_const / (rave_const + child.visits + 1e-9)
            combined = (1.0 - beta) * q + beta * amaf

            exploration = exploration_numerator / math.sqrt(1 + child.visits)
            score = combined + exploration

            if score > best_score:
                best_score = score
                best_node = child

        return best_node

    def rollout(self, board, next_player, red_moves, blue_moves):
        current_player = next_player
        sim_moves = list(board.get_legal_moves())
        play_func = board.play_rollout
        random_randrange = random.randrange
        
        while sim_moves and board.winner is None:
            idx = random_randrange(len(sim_moves))
            sim_moves[idx], sim_moves[-1] = sim_moves[-1], sim_moves[idx]
            move = sim_moves.pop()
            
            colour = Colour.RED if current_player == 1 else Colour.BLUE
            play_func(move[0], move[1], colour)
            
            if current_player == 1:
                red_moves.add(move)
            else:
                blue_moves.add(move)
            current_player = 3 - current_player

        if board.winner == Colour.RED:
            return 1
        if board.winner == Colour.BLUE:
            return 2
        return 0

    def backpropagate(self, path, winner, red_moves, blue_moves):
        for node in reversed(path):
            node.visits += 1
            
            # UCT: Did the move leading to 'node' result in a win?
            # The move was made by (3 - node.player).
            if winner != 0 and (3 - node.player) == winner:
                node.wins += 1
            
            # RAVE: Update children of 'node'.
            # These children represent moves made by 'node.player'.
            if node.player == 1:
                current_moves = red_moves
            else:
                current_moves = blue_moves

            for mv in current_moves:
                if mv in node.children:
                    child = node.children[mv]
                    child.rave_visits += 1
                    if winner == node.player:
                        child.rave_wins += 1

class Agent(AgentBase):
    _board_size: int = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.mcts = MCTS()

    def __deepcopy__(self, memo):
        return Agent(self.colour)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        if turn == 2:
            opp_r, opp_c = -1, -1
            for r in range(self._board_size):
                for c in range(self._board_size):
                    if board.tiles[r][c].colour is not None:
                        opp_r, opp_c = r, c
            if 2 <= opp_r <= 8 and 2 <= opp_c <= 8:
                return Move(-1, -1)

        if opp_move is not None:
            opp_move_tuple = (opp_move._x, opp_move._y)
            self.mcts.update_root(opp_move_tuple)

        optimized_board = Board_Optimized.from_game_board(board, self.colour)

        legal_moves = list(optimized_board.get_legal_moves())
        if len(legal_moves) == 1:
            return Move(legal_moves[0][0], legal_moves[0][1])

        row, col = self.mcts.search(optimized_board, self.colour, turn, time_limit=8.5)
        my_move_tuple = (row, col)
        self.mcts.update_root(my_move_tuple)
        return Move(row, col)
