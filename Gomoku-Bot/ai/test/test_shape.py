## Passed

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from board import Board
from minmax import minmax, cache_hits
from eval import FIVE, FOUR
import config
from shape import get_shape_fast as getShapeFast
from shape import get_all_shapes_of_point as getAllShapesOfPoint
from time import time
def test():
    board = Board(9)
    # 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0
    # 2 1 1 1 1 0 0 0 0
    # 0 0 0 1 0 0 0 0 0
    # 0 0 1 0 0 0 0 0 0
    # 0 0 2 2 2 0 0 0 0
    # 0 0 2 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0
    steps = [[3, 1], [3, 0], [3, 2], [6, 2], [3, 3], [6, 3], [4, 3], [6, 4], [5, 2], [7, 2], [3, 4]]
    for step in steps:
        x, y = step
        board.put(x, y)
    assert getShapeFast(board.evaluator.board, 2, 5, -1, 1, 1)[0] == 4


def test_chongsi():
    board = Board(15)
    steps = [[7, 7], [8, 6], [7, 5], [7, 6], [6, 6], [5, 5], [6, 7], [5, 7], [5, 6], [7, 8], [6, 4], [6, 5], [8, 7], [9, 7],
             [3, 4], [4, 5], [3, 5], [4, 4], [4, 3], [9, 6], [3, 6], [3, 3], [5, 2], [2, 5], [3, 7], [3, 8], [4, 6], [2, 6],
             [6, 1], [7, 0], [2, 4], [5, 9], [0, 2], [1, 3], [9, 8], [2, 8], [6, 2], [6, 9], [4, 9], [9, 9], [6, 3], [6, 0]]
    for step in steps:
        x, y = step
        board.put(x, y)
    print(board.display())
    print(getShapeFast(board.evaluator.board, 8, 9, 1, 0, -1))
    print(getAllShapesOfPoint(board.evaluator.shapeCache, 8, 9, -1))
    print(getAllShapesOfPoint(board.evaluator.shapeCache, 10, 6, -1))

def test_huo_si():
    board = Board(10)
    steps = [[4, 4], [5, 3], [4, 5], [5, 4], [5, 5], [6, 4]]
    for step in steps:
        x, y = step
        board.put(x, y)
    print(board.display())
    assert getShapeFast(board.evaluator.board, 6, 5, 1, 0, 1)[0] == 3


def test_shi_zhan_huo_si():
    board = Board(15)
    steps = [
        [7, 7], [8, 6], [7, 6], [7, 5], [9, 7], [8, 7], [8, 5], [9, 4], [8, 8], [7, 9], [6, 6], [5, 5], [10, 10],
        [9, 9], [5, 8], [6, 7], [6, 9], [8, 4], [4, 7], [7, 10], [3, 6], [2, 5], [6, 4], [9, 3], [10, 2], [10, 5],
        [11, 4], [10, 3], [8, 3], [8, 9], [4, 6], [5, 6], [7, 8], [6, 8], [10, 9], [11, 6], [9, 5], [12, 7], [13, 8],
        [12, 5], [12, 6], [9, 11], [8, 10], [10, 7], [9, 8], [14, 6], [4, 4], [4, 5], [3, 4], [5, 4], [5, 3],
    ]
    for step in steps:
        x, y = step
        board.put(x, y)
    shape = getShapeFast(board.evaluator.board, 3, 5, 1, 0, -1)
    assert shape[0] == 4
    print(shape)
    shape2 = getShapeFast(board.evaluator.board, 6, 5, 1, 0, -1)
    assert shape2[0] == 40

if __name__ == '__main__':
    test()
    test_chongsi()
    test_huo_si()
    test_shi_zhan_huo_si()