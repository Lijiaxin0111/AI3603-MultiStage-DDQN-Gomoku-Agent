import unittest


import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from board import Board
from minmax import vct, cache_hits, minmax

enableVCT = True
from eval import FIVE, FOUR, performance


class TestMinMax(unittest.TestCase):
    def test_no_kill_moves_2(self):
        board = Board(15)
        steps = [[7, 7], [8, 6], [7, 6], [7, 5], [9, 7], [8, 7], [8, 5], [9, 4], [8, 8], [7, 9], [6, 6], [5, 5],
                 [10, 10], [9, 9], [5, 8], [6, 7], [6, 9], [8, 4], [4, 7], [7, 10], [3, 6], [2, 5], [6, 4], [9, 3],
                 [10, 2], [10, 5], [11, 4], [10, 3], [8, 3], [8, 9], [4, 6], [5, 6], [7, 8], [6, 8], [10, 9], [11, 6],
                 [9, 5], [12, 7], [13, 8], [12, 5], [12, 6], [9, 11], [8, 10], [10, 7], [9, 8], [14, 6]]
        for step in steps:
            x, y = step
            board.put(x, y)

        print(board.display())
        score = vct(board, 1, 14)
        print('minmax score8', score)
        print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'], 'hit rate',
              cache_hits['hit'] / cache_hits['total'])
        print('evaluateTime:', board.evaluateTime / 1000)
        print('update point time', performance['updateTime'])
        print('get point time', performance['getPointsTime'])
        self.assertLess(score[0], FIVE)


if __name__ == '__main__':
    unittest.main()