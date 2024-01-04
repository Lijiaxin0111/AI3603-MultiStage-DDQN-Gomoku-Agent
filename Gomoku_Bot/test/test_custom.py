import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from board import Board
from minmax import minmax, cache_hits
from eval import FOUR

def test_minmax():
    board = Board(10)
    # 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 1 1 0 0 0 0
    # 0 0 0 2 2 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 0
    steps = [[4, 4], [5, 3], [4, 5], [5, 4]]
    for step in steps:
        x, y = step
        board.put(x, y)
    print(board.display())
    print(board.getValuableMoves(1, 0, False, False))

test_minmax()