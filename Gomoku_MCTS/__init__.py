from .mcts_pure import MCTSPlayer as MCTSpure
from .mcts_alphaZero import MCTSPlayer as alphazero
from .dueling_net import PolicyValueNet
import numpy as np


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.last_move = None
        self.availables = None
        self.current_player = None
        self.width = int(kwargs.get('width', 8))  # if no width, default 8
        self.height = int(kwargs.get('height', 8))
        self.board_map = np.zeros(shape=(self.width, self.height), dtype=int)
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = kwargs.get('players', [1, 2])  # player1 and player2
        self.init_board(0)

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move: int):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """
        return the board state from the perspective of the current player.
        state shape: 4*width*height
        这个状态数组具有四个通道：
        第一个通道表示当前玩家的棋子位置，第二个通道表示对手的棋子位置，第三个通道表示最后一步移动的位置。
        第四个通道是一个指示符，用于表示当前轮到哪个玩家（如果棋盘上的总移动次数是偶数，那么这个通道的所有元素都为1，表示是第一个玩家的回合；否则，所有元素都为0，表示是第二个玩家的回合）。
        每个通道都是一个 width x height 的二维数组，代表着棋盘的布局。对于第一个和第二个通道，如果一个位置上有当前玩家或对手的棋子，那么该位置的值为 1，否则为0。
        对于第三个通道，只有最后一步移动的位置是1，其余位置都为0。对于第四个通道，如果是第一个玩家的回合，那么所有的位置都是1，否则都是0。
        最后，状态数组在垂直方向上翻转，以匹配棋盘的实际布局。
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        # get (x,y) of this move
        x, y = self.move_to_location(move)
        self.board_map[x][y] = self.current_player

        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player
