import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from board import Board
from minmax import minmax, cache_hits
from eval import FOUR

def test_minmax_1():
    board = Board(15)
    steps = [[7, 7], [8, 6], [8, 8], [7, 8], [9, 7], [7, 9], [9, 9], [6, 6], [10, 10], [11, 11], [8, 7], [10, 7], [9, 8], [9, 10], [9, 6], [9, 5], [10, 8], [10, 9], [6, 7], [5, 7], [11, 8], [12, 8], [8, 10], [11, 7], [10, 6], [11, 5], [10, 5], [11, 4], [11, 6], [10, 4], [7, 11], [6, 12], [9, 4], [8, 3], [8, 9], [8, 11], [7, 6], [8, 5], [6, 5], [5, 4], [12, 6], [13, 6], [12, 7], [13, 8]]
    for step in steps:
        x, y = step
        board.put(x, y)
    print(board.display())
    print('moves', board.getValuableMoves(1, 0, False, False))
    print('evaluate', board.evaluate(1))
    print('score', minmax(board, 1, 4))

def test_minmax_2():
    board = Board(15, 1)
    steps = [[7, 7], [7, 8], [8, 6], [9, 5], [6, 6], [8, 8], [7, 5], [6, 8], [5, 8], [9, 8], [10, 8], [9, 7], [9, 6], [7, 6], [5, 7], [8, 4], [10, 6], [8, 7], [6, 5]]
    for step in steps:
        x, y = step
        board.put(x, y)
    print(board.display())
    print('moves', board.getValuableMoves(-1, 0, False, False))
    print('evaluate', board.evaluate(1))
    print('score', minmax(board, -1, 4))

test_minmax_1()
test_minmax_2()