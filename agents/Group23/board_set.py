import numpy as np
import random
from src.Colour import Colour
from src.Tile import Tile

ZOBRIST_TABLE = [[[random.getrandbits(64) for _ in range(2)] for _ in range(11)] for _ in range(11)]
TURN_HASH = random.getrandbits(64)

class Board_Optimized:
    RED_INT = 1
    BLUE_INT = 2
    EMPTY = 0

    def __init__(self, player, skip_hash_init=False):
        self.turn = player
        self.grid = np.zeros((11, 11), dtype=int)
        self.empty_spots = set((r, c) for r in range(11) for c in range(11))
        self.winner = None

        self._dsu_size = 125 
        self.parent = np.arange(self._dsu_size)
        self.rank = np.zeros(self._dsu_size, dtype=int)

        self.TOP_RED = 121
        self.BOTTOM_RED = 122
        self.LEFT_BLUE = 123
        self.RIGHT_BLUE = 124

        self.hash = 0
        if not skip_hash_init and player == Colour.RED:
            self.hash ^= TURN_HASH

    @classmethod
    def with_seed(cls, seed, player):
        random.seed(seed)
        global ZOBRIST_TABLE, TURN_HASH
        ZOBRIST_TABLE = [[[random.getrandbits(64) for _ in range(2)] for _ in range(11)] for _ in range(11)]
        TURN_HASH = random.getrandbits(64)
        return cls(player)

    def _index(self, row, col):
        return row * 11 + col

    def find(self, i):
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        
        curr = i
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
        return root

    def union(self, i, j):
        ri = self.find(i)
        rj = self.find(j)
        if ri == rj:
            return
        if self.rank[ri] > self.rank[rj]:
            self.parent[rj] = ri
        elif self.rank[rj] > self.rank[ri]:
            self.parent[ri] = rj
        else:
            self.parent[rj] = ri
            self.rank[ri] += 1

    @classmethod
    def from_game_board(cls, heavy_board, player):
        b = cls(player, skip_hash_init=True)
        for r in range(11):
            for c in range(11):
                tile = heavy_board.tiles[r][c]
                if tile.colour == Colour.RED:
                    b._place_direct(r, c, Colour.RED)
                elif tile.colour == Colour.BLUE:
                    b._place_direct(r, c, Colour.BLUE)
        if player == Colour.RED:
            b.hash ^= TURN_HASH
        return b

    def _place_direct(self, row, col, colour):
        colour_int = self.RED_INT if colour == Colour.RED else self.BLUE_INT
        self.grid[row, col] = colour_int
        
        if (row, col) in self.empty_spots:
            self.empty_spots.remove((row, col))
            
        current = self._index(row, col)
        hash_id = 0 if colour == Colour.RED else 1
        self.hash ^= ZOBRIST_TABLE[row][col][hash_id]

        for k in range(Tile.NEIGHBOUR_COUNT):
            nr = row + Tile.I_DISPLACEMENTS[k]
            nc = col + Tile.J_DISPLACEMENTS[k]
            if 0 <= nr < 11 and 0 <= nc < 11:
                if self.grid[nr, nc] == colour_int:
                    neighbor = self._index(nr, nc)
                    self.union(current, neighbor)

        if colour == Colour.RED:
            if row == 0: self.union(current, self.TOP_RED)
            if row == 10: self.union(current, self.BOTTOM_RED)
            if self.find(self.TOP_RED) == self.find(self.BOTTOM_RED):
                self.winner = Colour.RED

        elif colour == Colour.BLUE:
            if col == 0: self.union(current, self.LEFT_BLUE)
            if col == 10: self.union(current, self.RIGHT_BLUE)
            if self.find(self.LEFT_BLUE) == self.find(self.RIGHT_BLUE):
                self.winner = Colour.BLUE

    def play(self, row, col, colour):
        if self.winner is not None:
            return False
            
        if self.grid[row, col] != self.EMPTY:
            return False

        colour_int = self.RED_INT if colour == Colour.RED else self.BLUE_INT
        self.grid[row, col] = colour_int
        
        if (row, col) in self.empty_spots:
            self.empty_spots.remove((row, col))
            
        current = self._index(row, col)
        hash_id = 0 if colour == Colour.RED else 1
        self.hash ^= ZOBRIST_TABLE[row][col][hash_id]
        self.hash ^= TURN_HASH
        
        for k in range(Tile.NEIGHBOUR_COUNT):
            nr = row + Tile.I_DISPLACEMENTS[k]
            nc = col + Tile.J_DISPLACEMENTS[k]
            if 0 <= nr < 11 and 0 <= nc < 11:
                if self.grid[nr, nc] == colour_int:
                    neighbor = self._index(nr, nc)
                    self.union(current, neighbor)

        if colour == Colour.RED:
            if row == 0: self.union(current, self.TOP_RED)
            if row == 10: self.union(current, self.BOTTOM_RED)
            if self.find(self.TOP_RED) == self.find(self.BOTTOM_RED):
                self.winner = Colour.RED

        elif colour == Colour.BLUE:
            if col == 0: self.union(current, self.LEFT_BLUE)
            if col == 10: self.union(current, self.RIGHT_BLUE)
            if self.find(self.LEFT_BLUE) == self.find(self.RIGHT_BLUE):
                self.winner = Colour.BLUE

        self.turn = Colour.BLUE if self.turn == Colour.RED else Colour.RED
        return True
    
    def play_rollout(self, row, col, colour):
        """
        A stripped-down play function just for random simulations.
        Removes hashing, turn flipping, and set updates to run faster.
        """
        colour_int = self.RED_INT if colour == Colour.RED else self.BLUE_INT
        self.grid[row, col] = colour_int
        
        current = self._index(row, col)
        
        # Fast Union-Find update
        for k in range(Tile.NEIGHBOUR_COUNT):
            nr = row + Tile.I_DISPLACEMENTS[k]
            nc = col + Tile.J_DISPLACEMENTS[k]
            if 0 <= nr < 11 and 0 <= nc < 11:
                if self.grid[nr, nc] == colour_int:
                    neighbor = self._index(nr, nc)
                    self.union(current, neighbor)

        # Fast Winner Check
        if colour == Colour.RED:
            if row == 0: self.union(current, self.TOP_RED)
            if row == 10: self.union(current, self.BOTTOM_RED)
            if self.find(self.TOP_RED) == self.find(self.BOTTOM_RED):
                self.winner = Colour.RED
        else:
            if col == 0: self.union(current, self.LEFT_BLUE)
            if col == 10: self.union(current, self.RIGHT_BLUE)
            if self.find(self.LEFT_BLUE) == self.find(self.RIGHT_BLUE):
                self.winner = Colour.BLUE    

    def get_legal_moves(self):
        return list(self.empty_spots)

    def copy(self):
        new_board = object.__new__(Board_Optimized)
        new_board.turn = self.turn
        new_board.grid = np.copy(self.grid)
        new_board.empty_spots = self.empty_spots.copy()
        new_board.winner = self.winner
        new_board._dsu_size = self._dsu_size
        new_board.parent = np.copy(self.parent)
        new_board.rank = np.copy(self.rank)
        new_board.TOP_RED = self.TOP_RED
        new_board.BOTTOM_RED = self.BOTTOM_RED
        new_board.LEFT_BLUE = self.LEFT_BLUE
        new_board.RIGHT_BLUE = self.RIGHT_BLUE
        new_board.hash = self.hash
        return new_board

    def to_nn_input(self, player_perspective):
        tensor = np.zeros((3, 11, 11), dtype=np.float32)
        my_int = self.RED_INT if player_perspective == Colour.RED else self.BLUE_INT
        opp_int = self.BLUE_INT if my_int == self.RED_INT else self.RED_INT

        tensor[0] = (self.grid == my_int).astype(np.float32)
        tensor[1] = (self.grid == opp_int).astype(np.float32)
        
        if player_perspective == Colour.BLUE:
            tensor[2, :, :] = 1.0
            
        return tensor

    def hash_value(self):
        return self.hash
