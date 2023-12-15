"""
Author: Benhao Huang
Date: 11/24/2023
Description: this is the utils functions used to implement conv method
"""
import numpy as np

def split_board_into_parts(cur_state, part_size=8):
    """
    Splits the current state into five parts and returns their positions in the original board.

    :param cur_state: The current state of the board, assumed to be of shape (1, 4, board_size, board_size)
    :param part_size: Size of each part to split into, default is 8
    :return: A list of parts, each of shape (1, 4, part_size, part_size), and a list of lists of positions
    """
    board_size = cur_state.shape[2]  # Assuming the board is square
    center = (board_size) // 2 + board_size % 2 - (part_size // 2 + part_size % 2)

    coords = {
        "left_up": (0, part_size, 0, part_size),
        "right_up": (0, part_size, board_size - part_size, board_size),
        "left_down": (board_size - part_size, board_size, 0, part_size),
        "right_down": (board_size - part_size, board_size, board_size - part_size, board_size),
        "center": (center, center + part_size, center, center + part_size)
    }

    parts = []
    positions = []

    for key, (x1, x2, y1, y2) in coords.items():
        part = cur_state[:, :, x1:x2, y1:y2]
        parts.append(part)

        # Record the positions
        part_positions = []
        for j in range(y1, y2):
            for i in range(x1, x2):
                part_positions.append((i, j))
        positions.append(part_positions)

    return parts, positions