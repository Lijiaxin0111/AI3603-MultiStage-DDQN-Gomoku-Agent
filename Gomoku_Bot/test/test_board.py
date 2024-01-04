import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from board import Board
from board_manuls import wins, validMoves

def test_board_init():
    board = Board(15)
    for i in range(15):
        for j in range(15):
            assert board.board[i][j] == 0

def test_board_put():
    board = Board(15)
    board.put(1, 1)
    assert board.board[1][1] == 1
    assert len(board.history) == 1

def test_board_get_valid_moves():
    board = Board(15)
    board.put(1, 1)
    valid_moves = board.getValidMoves()
    assert [1, 1] not in valid_moves

def test_board_is_game_over():
    board = Board(6)
    assert not board.isGameOver()
    steps = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4], [4, 4]]
    for step in steps:
        x, y = step
        board.put(x, y)
    assert board.isGameOver()

def test_board_undo():
    board = Board(15)
    board.put(1, 1)
    assert board.board[1][1] == 1
    board.undo()
    assert board.board[1][1] == 0
    assert board.role == 1

def test_board_get_winner():
    for win in wins:
        board = Board(win[0])
        for move in win[1]:
            i, j = board.position2coordinate(move)
            board.put(i, j)
        print(board.getWinner())
        assert board.getWinner() == win[2]

# Add more tests for win condition and other situations

def run_tests():
    test_board_init()
    test_board_put()
    test_board_get_valid_moves()
    test_board_is_game_over()
    test_board_undo()
    test_board_get_winner()

run_tests()