import math
import time
import random
import concurrent.futures
from src.Colour import Colour
from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from agents.Group23.board_set import Board_Optimized

TIME = 8.5
WORKERS = 8

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

    def search(self, board, color, time_limit):
        time_start = time.time()
        random_randrange = random.randrange
        
        if self.root.allowed_moves is None:
            self.root.allowed_moves = list(board.get_legal_moves())
            if board.hash not in self.transposition_table:
                self.transposition_table[board.hash] = self.root

        if self.root.player is None:
            self.root.player = 1 if color == Colour.RED else 2
        simulations =0
        END = time_start + time_limit
        while time.time() < END:
            simulations+=1
            node = self.root
            board_copy = board.copy()
            board_play = board_copy.play
            path = [node]
            red_moves = set()
            blue_moves = set()

            while node.has_children() and node.is_fully_expanded():
                child = self.child_selection(node)
                if child is None:
                    break
                node = child
                path.append(node)

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

            if board_copy.winner is None and node.allowed_moves:
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
                
                rollout_player = next_player
            else:
                rollout_player = node.player

            winner = self.rollout(board_copy, rollout_player, red_moves, blue_moves)
            self.backpropagate(path, winner, red_moves, blue_moves)

        stats = {}
        for move, child in self.root.children.items():
            stats[move] = child.visits
        return stats,simulations

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
            #beta = math.sqrt(rave_const / (3*child.visits + rave_const+1e-9))

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
            
            if winner != 0 and (3 - node.player) == winner:
                node.wins += 1
            
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
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS)

    def __deepcopy__(self, memo):
        return Agent(self.colour)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        if turn == 2 and opp_move is not None:
             if 2 <= opp_move._x <= 8 and 2 <= opp_move._y <= 8:
                return Move(-1, -1)

        optimized_board = Board_Optimized.from_game_board(board, self.colour)

        legal_moves = list(optimized_board.get_legal_moves())
        if len(legal_moves) == 1:
            return Move(legal_moves[0][0], legal_moves[0][1])

        seed = []
        for x in range(WORKERS):
            seed.append(random.getrandbits(64))
        
        futures = []
        for i in range(WORKERS):
            futures.append(self.executor.submit(
                mcts_worker, 
                optimized_board,
                TIME, 
                self.colour,  
                seed[i]))
                
        statistics = {}
        sims=0
        for process in concurrent.futures.as_completed(futures):
            try:
                result,simulations = process.result()
                sims+=simulations
                for move, visits in result.items():
                    statistics[move] = statistics.get(move, 0) + visits
            except Exception:
                continue 
                
        if not statistics:
            return Move(legal_moves[0][0], legal_moves[0][1])
        print(sims)
        row, col = max(statistics, key=statistics.get)
        return Move(row, col)

def mcts_worker(board, time_limit, colour, seed):
    random.seed(seed)
    mcts = MCTS()
    return mcts.search(board, colour, time_limit)

