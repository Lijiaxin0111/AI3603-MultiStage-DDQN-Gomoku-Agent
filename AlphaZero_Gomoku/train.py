# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras
# import joblib
from config.options import *
import sys
from config.utils import *
from tqdm import tqdm


class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = opts.board_width
        self.board_height = opts.board_width
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)  # !
        self.game = Game(self.board)  # !
        # training params 
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = opts.n_playout  # num of simulations for each move 400 -> 200
        self.c_puct = 5
        self.buffer_size = opts.buffer_size
        self.batch_size = opts.batch_size  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)  # !
        self.play_batch_size = opts.play_batch_size
        self.epochs = opts.epochs  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = opts.check_freq
        self.game_batch_num = opts.game_batch_num
        self.best_win_ratio = 0.0  # !
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 200  # 1000 -> 200
        print("Current Board Size: {}*{}".format(self.board_width, self.board_height))
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)


        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        print("collectind datas")
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        print("====updating policy=====")
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return kl, loss, entropy, explained_var_old, explained_var_new

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        # pure_mcts_player = MCTS_Pure(c_puct=5,
        #                              n_playout=self.pure_mcts_playout_num)

        # use former trained model as opponent
        pi_eval = PolicyValueNet(self.board.width, self.board.height,
                                 model_file=opts.preload_model)
        pure_mcts_player = MCTSPlayer(pi_eval.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.pure_mcts_playout_num,
                                      is_selfplay=0)
        win_cnt = defaultdict(int)
        for i in tqdm(range(n_games), desc="policy_evaluate", ncols=100):
            winner = self.game.start_play(
                pure_mcts_player, current_mcts_player,
                start_player=i % 2,
                is_shown=0)
            win_cnt[winner] += 1
            print(f" {i}_th winner:", winner)
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            batch_bar = tqdm(range(self.game_batch_num), desc="total epochs", ncols=100)
            for i in batch_bar:
                self.collect_selfplay_data(self.play_batch_size)

                if len(self.data_buffer) > self.batch_size:
                    kl, loss, entropy, explained_var_old, explained_var_new = self.policy_update()

                    writer.add_scalar("policy_update/kl", kl, i)
                    writer.add_scalar("policy_update/loss", loss, i)
                    writer.add_scalar("policy_update/entropy", entropy, i)
                    writer.add_scalar("policy_update/explained_var_old", explained_var_old, i)
                    writer.add_scalar("policy_update/explained_var_new ", explained_var_new, i)

                batch_bar.set_description(f"game batch num {i}")

                # check the performance of the current model,
                # and save the model params
                if (i + 1) % self.check_freq == 0:
                    win_ratio = self.policy_evaluate()

                    batch_bar.set_description(f"game batch num {i + 1}")
                    writer.add_scalar("evaluate/explained_var_new ", win_ratio, i)
                    batch_bar.set_postfix(loss=loss, entropy=entropy, win_ratio=win_ratio)

                    save_model(self.policy_value_net.policy_value_net, "current_policy.model")
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        save_model(self.policy_value_net.policy_value_net, "best_policy.model")
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    writer = visualizer()
    training_pipeline = TrainPipeline(init_model=opts.preload_model)
    training_pipeline.run()
    training_pipeline.policy_evaluate()
