import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from board import Board
from minmax import minmax, cache_hits
from time import time

def test_performance():
    board = Board(15)
    start = time()
    role = 1
    while not board.isGameOver():
        score, move, _ = minmax(board, role, 6)
        board.put(move[0], move[1], role)
        role *= -1
        print('move', move, 'score', score)
        print(board.display())
        print([[h['i'], h['j']] for h in board.history])
    elapsed_time = (time() - start) / 1000
    print('Performance of self-play in 30 steps: Total time elapsed', elapsed_time, 's, Average time per step', elapsed_time / 30, 's')

test_performance()