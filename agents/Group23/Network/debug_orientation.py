import torch
import numpy as np
from src.Board import Board
from src.Colour import Colour
from agents.Group23.OptimisedBoardV2 import Board_Optimized


def main():
    print("--- ORIENTATION DEBUG ---")
    board = Board(board_size=11)

    # Place a line of RED stones along the TOP ROW (Row 0)
    # In Hex, Red connects Top-to-Bottom.
    # So stones at (0,0), (0,1), (0,2)... should be at the "Start" of Red's path.
    print("Placing Red stones on Row 0 (Top Edge)...")
    for col in range(5):
        # Assuming set_tile_colour(row, col) - we will verify this!
        board.set_tile_colour(0, col, Colour.RED)

    # Convert to Tensor (Red Perspective)
    opt_board = Board_Optimized.from_game_board(board, Colour.RED)
    input_tensor = opt_board.to_nn_input(Colour.RED)
    # Shape is (3, 11, 11) -> (Channels, Height, Width)

    print(f"\nTensor Shape: {input_tensor.shape}")

    # Extract Channel 0 (My Stones = Red)
    red_channel = input_tensor[0]

    print("\nVisualizing 'My Stones' Channel (Input to Net):")
    # We print the matrix. 1.0 = Stone, 0.0 = Empty.
    for r in range(11):
        row_str = ""
        for c in range(11):
            val = red_channel[r][c]
            char = "R" if val > 0.5 else "."
            row_str += f" {char}"
        print(f"Row {r}: {row_str}")

    print("\n--- ANALYSIS ---")
    print("If you see 'R R R R R' on 'Row 0', the input matches the board logic.")
    print("If you see 'R' down the first column (Row 0, 1, 2...), the input is TRANSPOSED.")
    print("If Transposed: The Network thinks 'Top' is 'Left'. This breaks everything.")


def check_blue_perspective():
    print("\n--- BLUE PERSPECTIVE DEBUG ---")
    board = Board(board_size=11)

    # Place Blue stones on LEFT COLUMN (Column 0)
    # Blue connects Left-Right.
    print("Placing Blue stones on Col 0 (Left Edge)...")
    for row in range(5):
        board.set_tile_colour(row, 0, Colour.BLUE)

    # Convert to Tensor (Blue Perspective)
    # This should trigger the ".T" (Transpose) logic in your code
    opt_board = Board_Optimized.from_game_board(board, Colour.BLUE)
    input_tensor = opt_board.to_nn_input(Colour.BLUE)

    # Extract Channel 0 (My Stones = Blue)
    blue_channel = input_tensor[0]

    print("Visualizing 'My Stones' (Blue) - Should be rotated to Top:")
    for r in range(11):
        row_str = ""
        for c in range(11):
            val = blue_channel[r][c]
            char = "B" if val > 0.5 else "."
            row_str += f" {char}"
        print(f"Row {r}: {row_str}")

    print("\n--- EXPECTATION ---")
    print("If you see 'B B B B B' on 'Row 0', the rotation is PERFECT.")
    print("This means the network always sees a 'Top-to-Bottom' game, regardless of colour.")
if __name__ == "__main__":
    # main()
    check_blue_perspective()