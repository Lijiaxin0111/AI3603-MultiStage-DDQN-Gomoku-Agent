# Assuming the required classes and functions have been imported
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from board import Board
from minmax import minmax, cache_hits
from eval import FIVE, FOUR
import config
from time import time



enableVCT = True

def test_first():
    board = Board(6)

    steps = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4]]
    for step in steps:
        x, y = step
        board.put(x, y)

    score = minmax(board, 1, 4, enableVCT)
    assert score[0] == FIVE
    print('minmax score1', score)
    print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'], 'hit rate',
          cache_hits['hit'] / cache_hits['total'])


def test_second():
    board = Board(9)
    """
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    2 1 1 1 0 0 0 0 0
    0 0 0 1 0 0 0 0 0
    0 0 1 0 0 0 0 0 0
    0 0 2 2 2 0 0 0 0
    0 0 2 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    """

    steps = [[3, 1], [3, 0], [3, 2], [6, 2], [3, 3], [6, 3], [4, 3], [6, 4], [5, 2], [7, 2]]
    for step in steps:
        x, y = step
        board.put(x, y)


    score = minmax(board, 1, 6, enableVCT)
    print('minmax score2', score)
    print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'], 'hit rate',
          cache_hits['hit'] / cache_hits['total'])
    print('evaluateTime:', board.evaluateTime / 1000)
    assert score[0] == FIVE


def test_third():
    board = Board(10)

    steps = [[4, 4], [5, 3], [4, 5], [5, 4]]
    for step in steps:
        x, y = step
        board.put(x, y)

    score = minmax(board, 1, 6, enableVCT)
    assert score[0] < FOUR
    print('minmax score3', score)
    print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'], 'hit rate',
          cache_hits['hit'] / cache_hits['total'])
    print('evaluateTime:', board.evaluateTime / 1000)


# Run the tests
# test_first()
# print(cache_hits)
# cache_hits = {
#     "search": 0,
#     "total": 0,
#     "hit": 0
# }
test_second()
test_third()