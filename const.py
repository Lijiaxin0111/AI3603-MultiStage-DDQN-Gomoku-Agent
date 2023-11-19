"""
FileName: const.py
Author: Benhao Huang
Create Date: 2023/11/19
Description: Some const value for Demo
"""

import numpy as np

_BOARD_SIZE = 8
_BOARD_SIZE_1D = _BOARD_SIZE * _BOARD_SIZE
_BLANK = 0
_BLACK = 1
_WHITE = 2
_PLAYER_SYMBOL = {
    _WHITE: "⚪",
    _BLANK: "➕",
    _BLACK: "⚫",
}
_PLAYER_COLOR = {
    _WHITE: "AI",
    _BLANK: "Blank",
    _BLACK: "YOU HUMAN",
}
_HORIZONTAL = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)
_VERTICAL = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ]
)
_DIAGONAL_UP_LEFT = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
)
_DIAGONAL_UP_RIGHT = np.array(
    [
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ]
)

_ROOM_COLOR = {
    True: _BLACK,
    False: _WHITE,
}
