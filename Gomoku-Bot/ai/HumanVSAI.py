import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from board import Board
from minmax import vct, cache_hits, minmax
from eval import FIVE, FOUR, performance


class Game():
    def __init__(self, firstRole=1):
        self.board = Board(15, firstRole)
        self.steps = []
        self.step = 0
        self.enableVCT = True  # 是否开启算杀, 算杀会在某些leaf节点加深搜索, 但是不一定会增加搜索时间

    def human_input(self):
        x, y = map(int, input('Your move: ').split())
        return x, y

    def start_play(self, human_first=False):
        if not human_first:
            while not self.board.isGameOver():
                print(self.board.display())
                if self.step % 2 == 1:
                    x, y = self.human_input()
                    while not self.board.put(x, y):
                        x, y = self.human_input()
                else:
                    score = minmax(self.board, 1, 4, enableVCT=self.enableVCT)
                    print(score)
                    x, y = score[1]
                    self.board.put(x, y)
                self.step += 1
        else:
            while not self.board.isGameOver():
                print(self.board.display())
                if self.step % 2 == 0:
                    x, y = self.human_input()
                    while not self.board.put(x, y):
                        x, y = self.human_input()
                else:
                    score = minmax(self.board, -1, 4, enableVCT=self.enableVCT)
                    print(score)
                    x, y = score[1]
                    self.board.put(x, y)
                self.step += 1
        print(self.board.display())


if __name__ == '__main__':
    game = Game()
    game.start_play(False)
