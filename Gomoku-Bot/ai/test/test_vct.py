import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from board import Board
from minmax import vct, cache_hits, minmax

enableVCT = True
from eval import FIVE, FOUR, performance


class VCTTestCase(unittest.TestCase):
    def test连五胜(self):
        board = Board(6)
        steps = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4]]
        for step in steps:
            x, y = step
            board.put(x, y)
        score = vct(board, 1, 4)
        self.assertEqual(score[0], FIVE)
        print('minmax score1', score)
        print('cache: total', cache_hits['total'], 'hit', cache_hits['hit'], 'hit rate',
              cache_hits['hit'] / cache_hits['total'])

    def test冲四活三胜利(self):
        board = Board(9)
        steps = [[3, 1], [3, 0], [3, 2], [6, 2], [3, 3], [6, 3], [4, 3], [6, 4], [5, 2], [7, 2]]
        for step in steps:
            x, y = step
            board.put(x, y)
        score = vct(board, 1, 10)
        print('minmax score2', score)
        print('cache: total', cache_hits['total'], 'hit', cache_hits['hit'], 'hit rate',
              cache_hits['hit'] / cache_hits['total'])
        print('evaluateTime:', board.evaluateTime / 1000)
        self.assertEqual(score[0], FIVE)

    def test开局(self):
        board = Board(10)
        steps = [[4, 4], [5, 3], [4, 5], [5, 4], [5, 5], [6, 4]]
        for step in steps:
            x, y = step
            board.put(x, y)
        score = vct(board, 1, 8)
        self.assertLess(score[0], FOUR)
        print('minmax score3', score)
        print('cache: total', cache_hits['total'], 'hit', cache_hits['hit'], 'hit rate',
              cache_hits['hit'] / cache_hits['total'])
        print('evaluateTime:', board.evaluateTime / 1000)

    def test从零开局(self):
        board = Board(9)
        score = vct(board, 1, 8)
        self.assertLess(score[0], FOUR)
        print('minmax score4', score)
        print('cache: total', cache_hits['total'], 'hit', cache_hits['hit'], 'hit rate',
              cache_hits['hit'] / cache_hits['total'] if cache_hits['total'] != 0 else 0)
        print('evaluateTime:', board.evaluateTime / 1000)
        print('update point time', performance['updateTime'])
        print('get point time', performance['getPointsTime'])

    def test连续冲四活三胜(self):
        board = Board(15)
        steps = [[7, 7], [8, 6], [7, 5], [7, 6], [6, 6], [5, 5], [6, 7], [5, 7], [5, 6], [7, 8], [6, 4], [6, 5], [8, 7],
                 [9, 7], [3, 4], [4, 5], [3, 5], [4, 4], [4, 3], [9, 6], [3, 6], [3, 3], [5, 2], [2, 5], [3, 7], [3, 8],
                 [4, 6], [2, 6], [6, 1], [7, 0], [2, 4], [5, 9], [0, 2], [1, 3], [9, 8], [2, 8], [6, 2], [6, 9], [4, 9],
                 [9, 9], [6, 3], [6, 0]]
        for step in steps:
            x, y = step
            board.put(x, y)
        print(board.display())
        score = vct(board, -1, 8)
        print('minmax score5', score)
        print('cache: search', cache_hits['search'], ', total', cache_hits['total'], 'hit', cache_hits['hit'],
              'hit rate', cache_hits['hit'] / cache_hits['total'])
        print('evaluateTime:', board.evaluateTime / 1000)
        self.assertEqual(score[0], FIVE)

    def test_防守连续冲四活三(self):
        board = Board(15)
        steps = [[7, 7], [8, 6], [7, 5], [7, 6], [6, 6], [5, 5], [6, 7], [5, 7], [5, 6], [7, 8], [6, 4], [6, 5], [8, 7],
                 [9, 7], [3, 4], [4, 5], [3, 5], [4, 4], [4, 3], [9, 6], [3, 6], [3, 3], [5, 2], [2, 5], [3, 7], [3, 8],
                 [4, 6], [2, 6], [6, 1], [7, 0], [2, 4], [5, 9], [0, 2], [1, 3], [9, 8], [2, 8], [6, 2], [6, 9], [4, 9],
                 [9, 9], [6, 3], [6, 0]]
        for x, y in steps:
            board.put(x, y)
        print(board.display())
        score = vct(board, 1, 12)
        print('##防守连续冲四活三')
        print('minmax score:', score)
        print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'],
              'hit rate',
              cache_hits['hit'] / cache_hits['total'])
        print('evaluateTime:', board.evaluateTime / 1000)
        print('update point time', performance['updateTime'])
        print('get point time', performance['getPointsTime'])
        self.assertLess(score[0], FIVE)

    def test_无杀棋(self):
        board = Board(15)
        steps = [[7, 7], [8, 6], [8, 8], [6, 6], [7, 8], [6, 8], [6, 7], [8, 7], [5, 6], [8, 9]]
        for x, y in steps:
            board.put(x, y)
        score = vct(board, 1, 10)
        print('minmax score6:', score)
        print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'],
              'hit rate',
              cache_hits['hit'] / cache_hits['total'])
        print('evaluateTime:', board.evaluateTime / 1000)
        print('update point time', performance['updateTime'])
        print('get point time', performance['getPointsTime'])
        self.assertLess(score[0], FIVE)

    def test_算对面杀棋(self):
        board = Board(15)
        steps = [[7, 7], [6, 7], [8, 6], [6, 6], [6, 8], [5, 9], [9, 5], [10, 4], [9, 7], [6, 4], [6, 5], [8, 5],
                 [10, 6], [7, 6], [9, 4], [9, 6], [11, 7], [8, 4], [12, 8], [13, 9], [10, 8], [5, 8], [4, 9], [7, 5],
                 [5, 7], [10, 2], [9, 3], [10, 3], [10, 1], [10, 7], [7, 4]]
        for x, y in steps:
            board.put(x, y)
        print(board.display())
        score = vct(board, 1, 10)
        print('minmax score7:', score)
        print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'],
              'hit rate',
              cache_hits['hit'] / cache_hits['total'])
        print('evaluateTime:', board.evaluateTime / 1000)
        print('update point time', performance['updateTime'])
        print('get point time', performance['getPointsTime'])
        self.assertEqual(score[0], FIVE)

    def test_无杀棋2(self):
        #FIXME:  时间太久，减少递归深度！
        board = Board(15)
        steps = [[7, 7], [8, 6], [7, 6], [7, 5], [9, 7], [8, 7], [8, 5], [9, 4], [8, 8], [7, 9], [6, 6], [5, 5],
                 [10, 10], [9, 9], [5, 8], [6, 7], [6, 9], [8, 4], [4, 7], [7, 10], [3, 6], [2, 5], [6, 4], [9, 3],
                 [10, 2], [10, 5], [11, 4], [10, 3], [8, 3], [8, 9], [4, 6], [5, 6], [7, 8], [6, 8], [10, 9], [11, 6],
                 [9, 5], [12, 7], [13, 8], [12, 5], [12, 6], [9, 11], [8, 10], [10, 7], [9, 8], [14, 6]]
        for x, y in steps:
            board.put(x, y)
        print(board.display())
        score = vct(board, 1, 14)
        print('minmax score8', score)
        print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'],
              'hit rate', cache_hits['hit'] / cache_hits['total'])
        print('evaluateTime:', board.evaluateTime / 1000)
        print('update point time', performance['updateTime'])
        print('get point time', performance['getPointsTime'])
        self.assertLess(score[0], FIVE)

    def test_无杀棋3(self):
        board = Board(15)
        steps = [[7, 7], [8, 6], [6, 6], [8, 8], [7, 5], [7, 6], [8, 7], [6, 7], [8, 5], [9, 7], [9, 5], [10, 5]]
        for x, y in steps:
            board.put(x, y)
        score = vct(board, 1, 10)
        print('minmax score8', score)
        print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'],
              'hit rate', cache_hits['hit'] / cache_hits['total'])
        print('evaluateTime:', board.evaluateTime / 1000)
        print('update point time', performance['updateTime'])
        print('get point time', performance['getPointsTime'])
        self.assertLess(score[0], FIVE)

    def test_无杀棋4(self):
        board = Board(15)
        steps = [[7, 7], [6, 6], [6, 8], [8, 6], [5, 7], [5, 9], [7, 9], [4, 6]]
        for x, y in steps:
            board.put(x, y)
        score = vct(board, 1, 10)
        print('minmax score9', score)
        print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'],
              'hit rate', cache_hits['hit'] / cache_hits['total'])
        print('evaluateTime:', board.evaluateTime / 1000)
        print('update point time', performance['updateTime'])
        print('get point time', performance['getPointsTime'])
        self.assertLess(score[0], FIVE)

    def test_防守(self):
        board = Board(15)
        steps = [[7, 6], [7, 5], [8, 5], [8, 6], [9, 4], [6, 7], [10, 3], [11, 2], [10, 5], [8, 3], [11, 4], [10, 4],
                 [11, 6], [7, 7], [12, 7], [13, 8], [9, 5]]
        for x, y in steps:
            board.put(x, y)
        score = vct(board, -1, 10)
        print('#####test 防守#####')
        print(board.display())
        print('minmax score', score)
        print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'],
              'hit rate', cache_hits['hit'] / cache_hits['total'])
        print('evaluateTime:', board.evaluateTime / 1000)
        print('update point time', performance['updateTime'])
        print('get point time', performance['getPointsTime'])
        self.assertEqual(score[0], -FIVE)

    def test_实战1(self):
        # 应该防守活四
        board = Board(15)
        steps = [[7, 7], [8, 6], [6, 6], [8, 8], [7, 5], [7, 6], [8, 7], [6, 7], [8, 5], [9, 6], [8, 4], [9, 3],
                 [11, 6], [10, 5], [9, 7], [10, 7], [5, 5], [6, 5], [10, 6], [7, 3], [8, 3], [8, 2], [11, 5], [7, 8],
                 [11, 4], [11, 3], [5, 6], [5, 7], [3, 3], [4, 4], [11, 8], [11, 7], [12, 4], [13, 3], [13, 6], [12, 6],
                 [10, 3], [12, 5], [12, 7], [10, 9], [10, 4], [9, 4], [9, 2], [9, 10], [7, 0], [8, 1], [13, 4], [14, 4],
                 [10, 8], [10, 10], [8, 9], [10, 11], [13, 8], [12, 8], [13, 7], [13, 5], [8, 10], [10, 12], [10, 13],
                 [11, 11], [12, 11], [11, 12], [13, 9], [13, 10], [11, 10], [9, 12]]
        for x, y in steps:
            board.put(x, y)
        print(board.display())
        score = minmax(board, 1, 4, enableVCT)
        print('minmax score1', score)
        print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'],
              'hit rate', cache_hits['hit'] / cache_hits['total'])
        self.assertIn(score[1], [(8, 12), (12, 12)])

        def test_无杀棋5(self):
            board = Board(15)
            steps = [[7, 7], [8, 6], [9, 6], [8, 5], [8, 7], [7, 8], [9, 7], [10, 7], [8, 8], [9, 9], [6, 7], [5, 7],
                     [9, 4], [9, 5], [10, 5], [11, 4], [6, 6], [10, 10], [5, 5], [4, 4], [6, 4], [6, 5], [7, 2], [8, 3],
                     [10, 6], [7, 9], [8, 4], [7, 4]]
            for x, y in steps:
                board.put(x, y)
            score = vct(board, 1, 10)
            print('##无杀棋5')
            print('minmax score', score)
            print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'],
                  'hit rate', cache_hits['hit'] / cache_hits['total'])
            print('evaluateTime:', board.evaluateTime / 1000)
            print('update point time', performance['updateTime'])
            print('get point time', performance['getPointsTime'])
            self.assertLess(score[0], FIVE)

        def test_有杀棋6(self):
            board = Board(15)
            steps = [[7, 7], [8, 6], [9, 6], [9, 5], [7, 5], [6, 6], [7, 6], [7, 4], [8, 7], [7, 8], [6, 7], [5, 7],
                     [10, 7], [9, 7], [10, 9], [9, 8], [11, 8], [12, 9], [5, 8], [4, 9], [9, 4], [8, 5], [12, 7],
                     [9, 10]]
            for x, y in steps:
                board.put(x, y)
            score = vct(board, 1, 12)
            print('##有杀棋6')
            print('minmax score', score)
            print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'],
                  'hit rate', cache_hits['hit'] / cache_hits['total'])
            print('evaluateTime:', board.evaluateTime / 1000)
            print('update point time', performance['updateTime'])
            print('get point time', performance['getPointsTime'])
            self.assertEqual(score[0], FIVE)

    def test_无杀棋5(self):
        board = Board(15)
        steps = [[7, 7], [8, 6], [9, 6], [8, 5], [8, 7], [7, 8], [9, 7], [10, 7], [8, 8], [9, 9], [6, 7], [5, 7],
                 [9, 4], [9, 5], [10, 5], [11, 4], [6, 6], [10, 10], [5, 5], [4, 4], [6, 4], [6, 5], [7, 2], [8, 3],
                 [10, 6], [7, 9], [8, 4], [7, 4]]
        for x, y in steps:
            board.put(x, y)
        score = vct(board, 1, 10)
        print('##无杀棋5')
        print('minmax score', score)
        print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'],
              'hit rate', cache_hits['hit'] / cache_hits['total'])
        print('evaluateTime:', board.evaluateTime / 1000)
        print('update point time', performance['updateTime'])
        print('get point time', performance['getPointsTime'])
        self.assertLess(score[0], FIVE)

    def test_有杀棋6(self):
        board = Board(15)
        steps = [[7, 7], [8, 6], [9, 6], [9, 5], [7, 5], [6, 6], [7, 6], [7, 4], [8, 7], [7, 8], [6, 7], [5, 7],
                 [10, 7], [9, 7], [10, 9], [9, 8], [11, 8], [12, 9], [5, 8], [4, 9], [9, 4], [8, 5], [12, 7], [9, 10]]
        for x, y in steps:
            board.put(x, y)
        score = vct(board, 1, 12)
        print('##有杀棋6')
        print('minmax score', score)
        print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'],
              'hit rate', cache_hits['hit'] / cache_hits['total'])
        print('evaluateTime:', board.evaluateTime / 1000)
        print('update point time', performance['updateTime'])
        print('get point time', performance['getPointsTime'])
        self.assertEqual(score[0], FIVE)

    def test_无杀棋7(self):
        board = Board(15)
        steps = [[7, 7], [6, 6], [5, 6], [6, 7], [6, 5], [7, 5], [5, 7], [8, 4], [6, 8], [5, 8], [7, 6], [8, 8], [7, 9],
                 [8, 10], [7, 10], [7, 8], [8, 9], [9, 3], [10, 2], [8, 6], [7, 4], [8, 3]]
        for x, y in steps:
            board.put(x, y)
        score = vct(board, 1, 12)
        print('##无杀棋7')
        print(board.display())
        print('minmax score', score)
        print('cache: search', cache_hits['search'], ', total ', cache_hits['total'], 'hit', cache_hits['hit'],
              'hit rate', cache_hits['hit'] / cache_hits['total'])
        print('evaluateTime:', board.evaluateTime / 1000)
        print('update point time', performance['updateTime'])
        print('get point time', performance['getPointsTime'])
        self.assertLess(score[0], FIVE)


if __name__ == '__main__':
    unittest.main()
