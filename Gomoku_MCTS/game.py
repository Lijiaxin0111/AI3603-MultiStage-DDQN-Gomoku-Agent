"""
FileName: game.py
Author: Jiaxin Li
Create Date: yyyy/mm/dd
Description: to be completed
Edit History:
- 2023/11/18, Sat,  Edited by Hbh (hbh001098hbh@sjtu.edu.cn)
    - added some comments and optimize import and some structures
- 2023/11/19, Sun,  Edited by Hbh (hbh001098hbh@sjtu.edu.cn)
    - added an API for retrieving simulation time
"""

import numpy as np
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_pure import Human_Player
from mcts_alphaZero import MCTSPlayer as MCST_AlphaZero
from collections import defaultdict
from policy_value_net_pytorch import PolicyValueNet
# from dueling_net import PolicyValueNet

class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.last_move = None
        self.availables = None
        self.current_player = None
        self.width = int(kwargs.get('width', 8))  # if no width, default 8
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

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


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board
        self.pure_mcts_playout_num = 200  # simulation time

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 f1irst)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
     
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """
        start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
        
    def start_play_collect(self, player1_train, player2_high, is_shown = 0, temp = 1e-3,start_player = 0):
        """
        start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
            
        """start a game between two players, store the self-play data: (state, mcts_probs, z) for training"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 f1irst)')
        
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1_train.set_player_ind(p1)
        player2_high.set_player_ind(p2)
        players = {p1: player1_train, p2: player2_high}
        states, mcts_probs, current_players = [], [], []
        while True:
            current_player = self.board.get_current_player()
         
            player_in_turn = players[current_player]

            if current_player == p2:
                move = player_in_turn.get_action(self.board)
                current_players.append(self.board.current_player)
                
             

            elif current_player == p1:
                move,move_probs = player_in_turn.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
                states.append(self.board.current_state())
                mcts_probs.append(move_probs)
                current_players.append(self.board.current_player)
            
          
                
                # print(self.board.availables)

            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()

            if end:
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                    
                player1_train.reset_player()
                winners_z = winners_z[np.array(current_players) == p1]

                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)

    def start_parser(self, out_file,is_shown = 0):
        """
        move by the out_file, get the data
        """
        self.board.init_board()
        p1, p2 = self.board.players

        with open(out_file,"r") as file:
            moves = file.readlines()
        # print("out::", moves)
   
        cnt = 2

        states, mcts_probs, current_players = [], [], []

        while True:
            move_probs = np.zeros(self.board.width * self.board.height)
            move = moves[cnt]
            cnt += 1


            move = move.split(',')
     
            # print(out_file)
            if move[0] == "0\n":
                # print(move)
                winner = -1
                end = True
            else:
            
          
            
 
        
                move = int(move[0]) + int(move[1])* self.board.width 

        
                move_probs[move] = 1

                

                # store the data
                states.append(self.board.current_state())
                mcts_probs.append(move_probs)
                current_players.append(self.board.current_player)
                # perform a move
                self.board.do_move(move)
                if is_shown:
                    self.graphic(self.board, p1, p2)
                end, winner = self.board.game_end()

            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
           
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
        
    def start_play_collect(self, player1_train, player2_high, is_shown = 0, temp = 1e-3,start_player = 0):
        """
        start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
            
        """start a game between two players, store the self-play data: (state, mcts_probs, z) for training"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 f1irst)')
        
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1_train.set_player_ind(p1)
        player2_high.set_player_ind(p2)
        players = {p1: player1_train, p2: player2_high}
        states, mcts_probs, current_players = [], [], []
        while True:
            current_player = self.board.get_current_player()
         
            player_in_turn = players[current_player]

            if current_player == p2:
                move = player_in_turn.get_action(self.board)
                current_players.append(self.board.current_player)
                
             

            elif current_player == p1:
                move,move_probs = player_in_turn.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
                states.append(self.board.current_state())
                mcts_probs.append(move_probs)
                current_players.append(self.board.current_player)
            
          
                
                # print(self.board.availables)

            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()

            if end:
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                    
                player1_train.reset_player()
                winners_z = winners_z[np.array(current_players) == p1]

                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)



    # 多了下面这一串测试代码

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTS_Pure(c_puct=5,
                                        n_playout=self.pure_mcts_playout_num)

        pi_eval = PolicyValueNet(self.board.width, self.board.height,
                                 model_file=r'Gomoku_MCTS\checkpoint\test_Alphazero_high_collect_epochs=1000_size=9\best_policy.model')
        current_mcts_player = MCST_AlphaZero(pi_eval.policy_value_fn,
                                             c_puct=5,
                                             n_playout=self.pure_mcts_playout_num,
                                             is_selfplay=0)
        # pure_mcts_player = MCTS_Pure(c_puct=5,
        #                              n_playout=self.pure_mcts_playout_num)

        pure_mcts_player = Human_Player()
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.start_play(current_mcts_player,
                                     pure_mcts_player,
                                     start_player=i % 2,
                                     is_shown=1)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio


if __name__ == '__main__':
    board_width = 8
    board_height = 8
    n_in_row = 5
    board = Board(width=board_width,
                  height=board_height,
                  n_in_row=n_in_row)
    task = Game(board)
    task.policy_evaluate(n_games=10)
