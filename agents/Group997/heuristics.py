from src.Board import Board
from src.Move import Move
from src.Colour import Colour
from src.Tile import Tile
from collections import deque

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

#Weights
weight_shortest_path = 10
weight_opponent_block = 7
weight_connectivity = 3
weight_bridge = 5
weight_dead_cell = -50
weight_edge_bias = 1
weight_center_bias = 0.5
weight_local_friendly = 0.5

turns_for_edge_bias = 15
turns_for_center_bias = 10
turns_for_local_friendly = 20

# Shortest path
def shortest_path_length(board: Board, colour: Colour) -> int:
    size = board.size
    INF = 10**9

    dist = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(INF)
        dist.append(row)

    dq = deque()

    # Red connect top to bottom
    if colour == Colour.RED:
        for y in range(size):
            cell = board.tiles[0][y].colour
            if cell == Colour.RED:
                dist[0][y] = 0
                dq.appendleft((0, y))
            elif cell is None:
                dist[0][y] = 1
                dq.append((0, y))

    # Blue connect left to right
    else:
        for x in range(size):
            cell = board.tiles[x][0].colour
            if cell == Colour.BLUE:
                dist[x][0] = 0
                dq.append((x, 0))
            elif cell is None:
                dist[x][0] = 1
                dq.append((x, 0))
    
    # BFS
    while dq:
        x, y = dq.popleft()
        current_cost = dist[x][y]

        # Explore all 6 neighbours
        for k in range(Tile.NEIGHBOUR_COUNT):
            neighbour_x = x + Tile.I_DISPLACEMENTS[k]
            neighbour_y = y + Tile.J_DISPLACEMENTS[k]

            # skip if neighbour is outside the board
            if not (0 <= neighbour_x < size and 0 <= neighbour_y < size):
                continue

            cell = board.tiles[neighbour_x][neighbour_y].colour

            # determine movement cost
            if cell == colour:
                step = 0
            elif cell is None:
                step = 1
            else:
                continue # opponent tile, can't step here

            new_cost = current_cost + step

            if new_cost < dist[neighbour_x][neighbour_y]:
                dist[neighbour_x][neighbour_y] = new_cost
                # 0 cost goes first, 1 cost go later
                if step == 0:
                    dq.appendleft((neighbour_x, neighbour_y))
                else:
                    dq.append((neighbour_x, neighbour_y))

    if colour == Colour.RED:
        best = INF
        for y in range(size):
            value = dist[size - 1][y]
            if value < best:
                best = value
        return best
    else:
        best = INF
        for x in range(size):
            value = dist[x][size - 1]
            if value < best:
                best = value
        return best

# Union-Find connectivity score
def connectivity_score(board: Board, colour: Colour, x: int, y: int) -> int:
    size = board.size
    total = size * size
    uf = UnionFind(total)

    # convert (i,j) to union-find index
    def uf_index(i, j):
        return i * size + j

    # Union all same-colour neighbours on the board
    for i in range(size):
        for j in range(size):
            if board.tiles[i][j].colour != colour:
                continue
            for k in range(Tile.NEIGHBOUR_COUNT):
                neighbour_i = i + Tile.I_DISPLACEMENTS[k]
                neighbour_j = j + Tile.J_DISPLACEMENTS[k]
                if 0 <= neighbour_i < size and 0 <= neighbour_j < size:
                    if board.tiles[neighbour_i][neighbour_j].colour == colour:
                        uf.union(uf_index(i, j), uf_index(neighbour_i, neighbour_j))

    # find how many distinct friendly groups new move (x, y) touches
    touched_roots = set()

    for k in range(Tile.NEIGHBOUR_COUNT):
        neighbour_i = x + Tile.I_DISPLACEMENTS[k]
        neighbour_j = y + Tile.J_DISPLACEMENTS[k]
        if 0 <= neighbour_i < size and 0 <= neighbour_j < size:
            if board.tiles[neighbour_i][neighbour_j].colour == colour:
                touched_roots.add(uf.find(uf_index(neighbour_i, neighbour_j)))

    return len(touched_roots)

# Bridge pattern detection
def is_bridge_move(board:Board, colour: Colour, x: int, y: int) -> bool:
    size = board.size

    bridge_pairs = [
        ((x+1, y),   (x,   y+1)),
        ((x-1, y),   (x,   y-1)),
        ((x+1, y+1), (x,   y-1)),
        ((x+1, y-1), (x,   y+1)),
        ((x-1, y+1), (x,   y)),     # mirrored
        ((x-1, y-1), (x,   y)),
    ]

    for (ax, ay), (bx, by) in bridge_pairs:

        # check if it's outside the board
        if not (0 <= ax < size and 0 <= ay < size):
            continue
        if not (0 <= bx < size and 0 <= by < size):
            continue

        # check if both stones are same colour
        if (board.tiles[ax][ay].colour == colour and board.tiles[bx][by].colour == colour):
            return True

    return False

# Dead cell detection
def is_dead_cell(board: Board, colour: Colour, x: int, y: int) -> bool:
    size = board.size
    opponent = Colour.opposite(colour)

    def bfs_reaches_side(side: str) -> bool:
        visited = []
        for i in range(size):
            row = []
            for j in range(size):
                row.append(False)
            visited.append(row)

        dq = deque()
        dq.append((x, y))
        visited[x][y] = True

        while dq:
            cell_x, cell_y = dq.popleft()

            if colour == Colour.RED:
                if side == "top" and cell_x == 0:
                    return True
                if side == "bottom" and cell_x == size - 1:
                    return True
            else:
                if side == "left" and cell_y == 0:
                    return True
                if side == "right" and cell_y == size - 1:
                    return True
            
            for k in range(Tile.NEIGHBOUR_COUNT):
                neighbour_x = cell_x + Tile.I_DISPLACEMENTS[k]
                neighbour_y = cell_y + Tile.J_DISPLACEMENTS[k]

                if 0 <= neighbour_x < size and 0 <= neighbour_y < size:
                    if not visited[neighbour_x][neighbour_y]:
                        if board.tiles[neighbour_x][neighbour_y].colour != opponent:
                            visited[neighbour_x][neighbour_y] = True
                            dq.append((neighbour_x, neighbour_y))
        return False
    
    if colour == Colour.RED:
        reach_top = bfs_reaches_side("top")
        reach_bottom = bfs_reaches_side("bottom")

        if reach_top and reach_bottom:
            return False
        else:
            return True
    else:
        reach_left = bfs_reaches_side("left")
        reach_right = bfs_reaches_side("right")

        if reach_left and reach_right:
            return False
        else:
            return True

def edge_bias(x: int, y: int, size: int):
    if x == 0 or x == size - 1 or y == 0 or y == size - 1:
        return weight_edge_bias
    return 0

def center_bias(x: int, y: int, size: int) -> float:
    mid = size // 2
    dist = abs(x - mid) + abs(y - mid)
    max_dist = 2 * mid
    return (max_dist - dist) / max_dist

def local_friendly_adjacency(board: Board, colour: Colour, x: int, y: int) -> int:
    count = 0
    for k in range(Tile.NEIGHBOUR_COUNT):
        neighbour_x = x + Tile.I_DISPLACEMENTS[k]
        neighbour_y = y + Tile.J_DISPLACEMENTS[k]
        if 0 <= neighbour_x < board.size and 0 <= neighbour_y < board.size:
            if board.tiles[neighbour_x][neighbour_y].colour == colour:
                count += 1
    return count

# Combined score of all heuristics
def heuristic_scoring(board:Board, colour: Colour, x: int, y: int, turn: int) -> float:
    score = 0 

    # bias to influence early game move decisions
    if turn < turns_for_edge_bias:
        score += edge_bias(x, y, board.size) * weight_edge_bias
    if turn < turns_for_center_bias:
        score += center_bias(x, y, board.size) * weight_center_bias
    if turn < turns_for_local_friendly:
        score += local_friendly_adjacency(board, colour, x, y) * weight_local_friendly

    # shortest path improvement 
    old = shortest_path_length(board, colour)
    board.set_tile_colour(x, y, colour)
    new = shortest_path_length(board, colour)
    board.set_tile_colour(x, y, None)
    score += (old - new) * weight_shortest_path

    # opponent shortest path block
    opponent = Colour.opposite(colour)
    if opponent is not None:
        opponent_old = shortest_path_length(board, opponent)
        board.set_tile_colour(x, y, colour)
        opponent_new = shortest_path_length(board, opponent)
        board.set_tile_colour(x, y, None)
        score += (opponent_new - opponent_old) * weight_opponent_block

    # connectivity
    score += connectivity_score(board, colour, x, y) * weight_connectivity

    # bridge
    if is_bridge_move(board, colour, x, y):
        score += weight_bridge

    # dead cell penalty
    if is_dead_cell(board, colour, x, y):
        score += weight_dead_cell

    return score