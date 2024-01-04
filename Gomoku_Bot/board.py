from .zobrist import ZobristCache as Zobrist
from .cache import Cache
from .eval import Evaluate, FIVE
from scipy import signal
import pickle
import os
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_data/data', 'train_data.pkl')

if 'numpy' not in globals():
    import numpy as np


class Board:
    def __init__(self, size=15, firstRole=1):
        self.size = size
        self.board = [[0] * self.size for _ in range(self.size)]
        self.firstRole = firstRole  # 1 for black, -1 for white
        self.role = firstRole  # 1 for black, -1 for white
        self.history = []
        self.zobrist = Zobrist(self.size)
        self.winnerCache = Cache()
        self.gameoverCache = Cache()
        self.evaluateCache = Cache()
        self.valuableMovesCache = Cache()
        self.evaluateTime = 0
        self.evaluator = Evaluate(self.size)
        self.available = [(i, j) for i in range(self.size) for j in range(self.size)]
        self.patterns = [np.ones((1, 5)), np.ones((5, 1)), np.eye(5), np.fliplr(np.eye(5))]
        self.train_data = {1:[], -1: []}
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                self.train_data = pickle.load(f)

    def isGameOver(self):
        # Checked
        hash = self.hash()
        if self.gameoverCache.get(hash):
            return self.gameoverCache.get(hash)
        if self.getWinner() != 0:
            self.gameoverCache.put(hash, True)
            # save train data
            # with open(save_path, 'wb') as f:
            #     pickle.dump(self.train_data, f)
            return True  # Someone has won
        # Game is over when there is no empty space on the board or someone has won
        if len(self.history) == self.size ** 2:
            self.gameoverCache.put(hash, True)
            return True
        else:
            self.gameoverCache.put(hash, False)
            return False

    def getWinner(self):
        # Checked
        hash = self.hash()
        flag = True
        if self.winnerCache.get(hash):
            return self.winnerCache.get(hash)
        directions = [[1, 0], [0, 1], [1, 1], [1, -1]]  # Horizontal, Vertical, Diagonal
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    flag = False
                    continue
                for direction in directions:
                    count = 0
                    while (
                            0 <= i + direction[0] * count < self.size and
                            0 <= j + direction[1] * count < self.size and
                            self.board[i + direction[0] * count][j + direction[1] * count] == self.board[i][j]
                    ):
                        count += 1
                    if count >= 5:
                        self.winnerCache.put(hash, self.board[i][j])
                        return self.board[i][j]
        if flag:
            print("tie!!!")
            return 0
        self.winnerCache.put(hash, 0)
        return 0

    def getValidMoves(self):
        return self.available

    def put(self, i, j, role=None):
        # Checked
        if role is None:
            role = self.role
        if not isinstance(i, int) or not isinstance(j, int):
            print("Invalid move: Not Number!", i, j)
            return False
        if self.board[i][j] != 0:
            print("Invalid move!", i, j)
            return False
        self.board[i][j] = role
        self.available.remove((i, j))
        self.history.append({"i": i, "j": j, "role": role})
        self.zobrist.togglePiece(i, j, role)
        self.evaluator.move(i, j, role)
        self.role *= -1  # Switch role
        return True

    def undo(self):
        # Checked
        if len(self.history) == 0:
            print("No moves to undo!")
            return False

        lastMove = self.history.pop()
        self.board[lastMove['i']][lastMove['j']] = 0  # Remove the piece from the board
        self.role = lastMove['role']  # Switch back to the previous player
        self.zobrist.togglePiece(lastMove['i'], lastMove['j'], lastMove['role'])
        self.evaluator.undo(lastMove['i'], lastMove['j'])
        self.available.append((lastMove['i'], lastMove['j']))
        return True

    def position2coordinate(self, position):
        # checked
        row = position // self.size
        col = position % self.size
        return [row, col]

    def coordinate2position(self, coordinate):
        # Checked
        return coordinate[0] * self.size + coordinate[1]

    def getValuableMoves(self, role, depth=0, onlyThree=False, onlyFour=False):
        # Checked
        hash = self.hash()
        prev = self.valuableMovesCache.get(hash)
        if prev:
            if (prev["role"] == role and
                    prev["depth"] == depth and
                    prev["onlyThree"] == onlyThree
                    and prev["onlyFour"] == onlyFour):
                return prev["moves"]

        moves, train_data = self.evaluator.getMoves(role, depth, onlyThree, onlyFour)
        self.train_data[self.role].append(train_data)
        # Handle a special case, if the center point is not occupied, add it by default

        # 开局的时候随机走一步，增加开局的多样性
        if not onlyThree and not onlyFour:
            center = self.size // 2
            if self.board[center][center] == 0:
                moves.append((center, center))

            # x_step = np.random.randint(-self.size // 2, self.size // 2)
            # y_step = np.random.randint(-self.size // 2, self.size // 2)
            # x = center + x_step
            # y = center + y_step
            # if 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == 0:
            #     moves.append((x, y))

        self.valuableMovesCache.put(hash, {
            "role": role,
            "moves": moves,
            "depth": depth,
            "onlyThree": onlyThree,
            "onlyFour": onlyFour
        })
        return moves

    def display(self, extraPoints=[]):
        # Checked
        extraPosition = [self.coordinate2position(point) for point in extraPoints]
        result = ""
        for i in range(self.size):
            for j in range(self.size):
                position = self.coordinate2position([i, j])
                if position in extraPosition:
                    result += "? "
                    continue
                value = self.board[i][j]
                if value == 1:
                    result += "B " # Black
                elif value == -1:
                    result += "W " # White
                else:
                    result += "- "
            result += "\n"
        return result

    def hash(self):
        # Checked
        return self.zobrist.getHash()  # Return the hash value of the current board, used for caching

    def evaluate(self, role):
        # Checked
        hash_key = self.hash()
        prev = self.evaluateCache.get(hash_key)
        if prev:
            if prev["role"] == role:
                return prev["score"]

        winner = self.getWinner()
        score = 0
        if winner != 0:
            score = FIVE * winner * role
        else:
            score = self.evaluator.evaluate(role)

        self.evaluateCache.put(hash_key, {"role": role, "score": score})
        return score

    def reverse(self):
        # Checked
        new_board = Board(self.size, -self.firstRole)
        for move in self.history:
            x, y, role = move['i'], move['j'], move['role']
            new_board.put(x, y, -role)
        return new_board

    def toString(self):
        # Checked
        return ''.join([''.join(map(str, row)) for row in self.board])
