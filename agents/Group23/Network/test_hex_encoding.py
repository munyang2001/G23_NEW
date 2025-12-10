# test_hex_encoding.py

import os
import sys
import unittest
import numpy as np

# --- PATH SETUP (same style as generate_data.py) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.Board import Board
from src.Colour import Colour
from agents.Group23.board_sets import Board_Optimized


class HexEncodingTests(unittest.TestCase):
    BOARD_SIZE = 11

    def test_red_and_blue_planes_and_transpose(self):
        """Check:
        - Red sees RED stones in channel 0, BLUE in channel 1
        - Blue sees BLUE in channel 0, RED in channel 1
        - Blue perspective uses transpose (r, c) -> (c, r)
        """
        board = Board(board_size=self.BOARD_SIZE)

        # Place one Red and one Blue stone
        red_r, red_c = 2, 4
        blue_r, blue_c = 5, 1

        board.set_tile_colour(red_r, red_c, Colour.RED)
        board.set_tile_colour(blue_r, blue_c, Colour.BLUE)

        # --- Red perspective ---
        opt_red = Board_Optimized.from_game_board(board, Colour.RED)
        x_red = opt_red.to_nn_input(Colour.RED)  # (3, 11, 11)

        self.assertEqual(x_red.shape, (3, self.BOARD_SIZE, self.BOARD_SIZE))

        # Red sees its own stone in channel 0 at (2, 4)
        self.assertAlmostEqual(x_red[0, red_r, red_c], 1.0)
        self.assertAlmostEqual(x_red[1, red_r, red_c], 0.0)

        # Red sees Blue's stone in channel 1 at (5, 1)
        self.assertAlmostEqual(x_red[1, blue_r, blue_c], 1.0)
        self.assertAlmostEqual(x_red[0, blue_r, blue_c], 0.0)

        # --- Blue perspective ---
        opt_blue = Board_Optimized.from_game_board(board, Colour.BLUE)
        x_blue = opt_blue.to_nn_input(Colour.BLUE)  # (3, 11, 11)

        # Red stone (2,4) becomes (4,2) after transpose
        red_r_canon, red_c_canon = red_c, red_r  # (4, 2)
        self.assertAlmostEqual(x_blue[1, red_r_canon, red_c_canon], 1.0)
        self.assertAlmostEqual(x_blue[0, red_r_canon, red_c_canon], 0.0)

        # Blue stone (5,1) becomes (1,5) after transpose and is "my stone"
        blue_r_canon, blue_c_canon = blue_c, blue_r  # (1, 5)
        self.assertAlmostEqual(x_blue[0, blue_r_canon, blue_c_canon], 1.0)
        self.assertAlmostEqual(x_blue[1, blue_r_canon, blue_c_canon], 0.0)

    def test_blue_policy_index_matches_transpose_math(self):
        """Check that your Blue index formula: idx = c*11 + r
        matches 'flattened index' on the transposed (canonical) board.
        """
        B = self.BOARD_SIZE

        # Two arbitrary coordinates (r, c) in physical board space
        coords = [(2, 4), (5, 1)]

        for (r, c) in coords:
            idx_blue = c * B + r       # your generate_data.py formula

            # Canonical board for Blue is transposed: (r_c, c_c) = (c, r)
            r_canon, c_canon = c, r
            canon_idx = r_canon * B + c_canon  # row-major flatten

            self.assertEqual(
                idx_blue, canon_idx,
                msg=f"Blue index mismatch for (r={r}, c={c}): "
                    f"{idx_blue} vs {canon_idx}"
            )

    def test_blue_plane_flat_index_matches_policy_index(self):
        """End-to-end: place a single BLUE stone and check that the
        position of '1' in the my-stones plane matches c*11 + r.
        """
        B = self.BOARD_SIZE
        board = Board(board_size=B)

        # Single blue stone at (r, c)
        r, c = 5, 1
        board.set_tile_colour(r, c, Colour.BLUE)

        opt_blue = Board_Optimized.from_game_board(board, Colour.BLUE)
        x_blue = opt_blue.to_nn_input(Colour.BLUE)  # (3, 11, 11)

        my_plane = x_blue[0]  # (11, 11) for BLUE's stones
        coords = np.argwhere(my_plane == 1.0)

        # There should be exactly one BLUE stone
        self.assertEqual(coords.shape[0], 1)

        r_canon, c_canon = coords[0]  # coordinates in canonical (transposed) view
        flat_idx = int(r_canon * B + c_canon)

        expected_idx = c * B + r  # your generate_data.py Blue formula

        self.assertEqual(
            flat_idx, expected_idx,
            msg=(
                f"Flat index from NN plane ({flat_idx}) does not match "
                f"expected Blue label index ({expected_idx})."
            )
        )


if __name__ == "__main__":
    unittest.main()
