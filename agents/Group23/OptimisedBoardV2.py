import numpy as np
import random
from src.Colour import Colour
from src.Tile import Tile

ZOBRIST_TABLE = [[[random.getrandbits(64) for _ in range(2)] for _ in range(11)] for _ in range(11)]
TURN_HASH = random.getrandbits(64)

NEIGHBOR_OFFSETS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))


class Board_Optimized:
    __slots__ = (
        'turn', 'grid', 'empty_spots', 'winner',
        '_dsu_size', 'parent', 'rank',
        'TOP_RED', 'BOTTOM_RED', 'LEFT_BLUE', 'RIGHT_BLUE',
        'hash'
    )

    RED_INT = 1
    BLUE_INT = 2
    EMPTY = 0

    def __init__(self, player, skip_hash_init=False):
        self.turn = player
        # Keep grid as NumPy for compatibility with NN input, but could be list for pure speed
        self.grid = np.zeros((11, 11), dtype=int)
        self.empty_spots = set((r, c) for r in range(11) for c in range(11))
        self.winner = None

        self._dsu_size = 125
        # OPTIMIZATION: Use Python lists for DSU, not NumPy.
        # NumPy scalar access is slow; lists are fast.
        self.parent = list(range(self._dsu_size))
        self.rank = [0] * self._dsu_size

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
        # Path compression with iterative approach (fastest in Python)
        path = []
        root = i
        while self.parent[root] != root:
            path.append(root)
            root = self.parent[root]

        for node in path:
            self.parent[node] = root
        return root

    def union(self, i, j):
        ri = self.find(i)
        rj = self.find(j)
        if ri != rj:
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
        # Manually populate to avoid overhead
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

        current = row * 11 + col
        hash_id = 0 if colour == Colour.RED else 1
        self.hash ^= ZOBRIST_TABLE[row][col][hash_id]

        for dr, dc in NEIGHBOR_OFFSETS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 11 and 0 <= nc < 11:
                if self.grid[nr, nc] == colour_int:
                    self.union(current, nr * 11 + nc)

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
        if self.winner is not None or self.grid[row, col] != self.EMPTY:
            return False

        colour_int = self.RED_INT if colour == Colour.RED else self.BLUE_INT
        self.grid[row, col] = colour_int

        if (row, col) in self.empty_spots:
            self.empty_spots.remove((row, col))

        current = row * 11 + col
        hash_id = 0 if colour == Colour.RED else 1
        self.hash ^= ZOBRIST_TABLE[row][col][hash_id]
        self.hash ^= TURN_HASH

        for dr, dc in NEIGHBOR_OFFSETS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 11 and 0 <= nc < 11:
                if self.grid[nr, nc] == colour_int:
                    self.union(current, nr * 11 + nc)

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

        self.turn = Colour.BLUE if self.turn == Colour.RED else Colour.RED
        return True

    def play_rollout(self, row, col, colour):
        """
        Streamlined for random rollouts:
        - No hashing
        - No turn flipping
        - No empty_spots update (handled by MCTS list)
        """
        colour_int = self.RED_INT if colour == Colour.RED else self.BLUE_INT
        self.grid[row, col] = colour_int

        current = row * 11 + col

        # Fast arithmetic loop
        for dr, dc in NEIGHBOR_OFFSETS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 11 and 0 <= nc < 11:
                if self.grid[nr, nc] == colour_int:
                    self.union(current, nr * 11 + nc)

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
        # np.copy is fast for the grid
        new_board.grid = np.copy(self.grid)
        new_board.empty_spots = self.empty_spots.copy()
        new_board.winner = self.winner
        new_board._dsu_size = self._dsu_size
        # List slicing [:] is faster than np.copy for small arrays
        new_board.parent = self.parent[:]
        new_board.rank = self.rank[:]

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
        tensor[2] = (self.grid == self.EMPTY).astype(np.float32)

        if player_perspective == Colour.BLUE:
            # Canonical Rotation: Swap spatial axes (H, W) only
            # (Channels, Height, Width) -> (Channels, Width, Height)
            tensor = np.transpose(tensor, (0, 2, 1))

        return tensor

    def hash_value(self):
        return self.hash