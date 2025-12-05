import torch
from src.Board import Board
from src.Colour import Colour

def encode_board_to_tensor(board:Board, my_colour:Colour) -> torch.Tensor:
    """
    Encode the Hex board into a tensor of shape (1, C, H, W).
    Channels:
      0: my stones       (1 where my stone is, 0 elsewhere)
      1: opp stones      (1 where opponent stone is, 0 elsewhere)
      2: empty cells     (1 where cell is empty, 0 elsewhere)
      3: my turn plane   (all 1 if it's my move, else all 0)
    my_colour: Colour
    The colour this agent is playing (e.g. Colour.RED or Colour.BLUE).
    """
    assert isinstance(my_colour, Colour)

    # 1) Get size and tiles from the Board.
    size = board.size
    tiles = board.tiles

    # 2) Create empty planes as float32 tensors
    # torch.zeros((H, W), dtype=torch.float32) creates a 2D tensor filled with 0.0
    my_plane = torch.zeros((size, size), dtype=torch.float32)
    opp_plane = torch.zeros((size, size), dtype=torch.float32)
    empty_plane = torch.zeros((size, size), dtype=torch.float32)

    # Get the opponent's colour
    opp_colour = Colour.opposite(my_colour)

    # Maintain two counters to infer the next player.
    count_red = 0
    count_blue = 0

    # Traverse through each tile (efficiency could be improved)
    for r in range(size):
        for c in range(size):
            tile = tiles[r][c]
            colour = tile.colour
            if colour == Colour.RED:
                count_red += 1
            elif colour == Colour.BLUE:
                count_blue += 1

            if colour == my_colour:
                my_plane[r, c] = 1.0
            elif colour == opp_colour:
                opp_plane[r, c] = 1.0
            else:
                empty_plane[r, c] = 1.0

    count_total = count_red + count_blue
    if count_total % 2 == 0:
        next_player = Colour.RED
    else:
        next_player = Colour.BLUE

    my_turn_value = 1.0 if next_player == my_colour else 0.0

    my_turn_plane = torch.full(
        (size, size),
        fill_value=my_turn_value,
        dtype=torch.float32,
    )

    # 4) Stack into channels: (C, H, W)
    stacked = torch.stack(
        [my_plane, opp_plane, empty_plane, my_turn_plane],
        dim=0,
    )

    # 5) Add batch dimension: (1, C, H, W)
    return stacked.unsqueeze(0)