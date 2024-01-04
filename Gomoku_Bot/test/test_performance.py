import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from board import Board
from minmax import minmax, cache_hits
from time import time
import random

def test_performance():
    board = Board(15)
    start = time()
    role = 1
    print(board.display())
    while not board.isGameOver():
        score, move, _ = minmax(board, role, 4)
        if move is None:
            move = random.choice(board.getValidMoves())
        board.put(move[0], move[1], role)

        role *= -1
        print('move', move, 'score', score)
        print(board.display())
        print([[h['i'], h['j']] for h in board.history])
    elapsed_time = (time() - start) / 1000
    print('Performance of self-play in 30 steps: Total time elapsed', elapsed_time, 's, Average time per step', elapsed_time / 30, 's')

for i in range(10):
    test_performance()