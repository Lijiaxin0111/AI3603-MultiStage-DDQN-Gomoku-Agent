from __future__ import print_function

import os.path
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from mcts_Gumbel_Alphazero import Gumbel_MCTSPlayer
import torch.optim as optim
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from dueling_net import PolicyValueNet as dueling_PolicyValueNet  # Pytorch
from policy_value_net_pytorch_new import PolicyValueNet as new_PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras
# import joblib
from torch.autograd import Variable
import torch.nn.functional as F

from config.options import *
import sys
from config.utils import *
from torch.backends import cudnn

import torch
import json

from tqdm import *
# from torch.utils.tensorboard import SummaryWriter

from multiprocessing import Pool


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def std_log():
    if get_rank() == 0:
        save_path = make_path()
        makedir(config['log_base'])
        sys.stdout = open(os.path.join(config['log_base'], "{}.txt".format(save_path)), "w")


def init_seeds(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


class MainWorker():
    def __init__(self, device):

        # --- init the set of pipeline -------
        self.board_width = opts.board_width
        self.board_height = opts.board_height
        self.n_in_row = opts.n_in_row
        self.learn_rate = opts.learn_rate
        self.lr_multiplier = opts.lr_multiplier
        self.temp = opts.temp
        self.n_playout = opts.n_playout
        self.c_puct = opts.c_puct
        self.buffer_size = opts.buffer_size
        self.batch_size = opts.batch_size
        self.play_batch_size = opts.play_batch_size
        self.epochs = opts.epochs
        self.kl_targ = opts.kl_targ
        self.check_freq = opts.check_freq
        self.game_batch_num = opts.game_batch_num
        self.pure_mcts_playout_num = opts.pure_mcts_playout_num
        self.m = opts.action_m

        self.device = device
        # self.use_gpu = opts.use_gpu
        self.use_gpu = device == "cuda"

        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)

        # The data collection of the history of games
        self.data_buffer = deque(maxlen=self.buffer_size)

        # The best win ratio of the training agent
        self.best_win_ratio = 0.0

        if opts.preload_model:
            if opts.model_type == "duel":
                print("preload duel model")
                # start training from an initial policy-value net
                self.policy_value_net = dueling_PolicyValueNet(self.board_width,
                                                               self.board_height,
                                                               model_file=opts.preload_model,
                                                               use_gpu=(self.device == "cuda"))
            elif opts.model_type == "biased":
                print("preload biased model")
                self.policy_value_net = new_PolicyValueNet(self.board_width,
                                                           self.board_height,
                                                           model_file=opts.preload_model,
                                                           use_gpu=(self.device == "cuda"),
                                                           bias=True)
            elif opts.model_type == "normal" or "gumbel":
                print("preload normal/gumbel model")
                self.policy_value_net = new_PolicyValueNet(self.board_width,
                                                           self.board_height,
                                                           model_file=opts.preload_model,
                                                           use_gpu=(self.device == "cuda"),
                                                           bias=False)
            else:
                raise ValueError("illegal model type")


        else:
            # start training from a new policy-value net
            if opts.model_type == "duel":
                self.policy_value_net = dueling_PolicyValueNet(self.board_width,
                                                               self.board_height,
                                                               use_gpu=(self.device == "cuda"))
            elif opts.model_type == "biased":
                self.policy_value_net = new_PolicyValueNet(self.board_width,
                                                           self.board_height,
                                                           use_gpu=(self.device == "cuda"),
                                                           bias=True)
            elif opts.model_type == "normal" or "gumbel":
                self.policy_value_net = new_PolicyValueNet(self.board_width,
                                                           self.board_height,
                                                           use_gpu=(self.device == "cuda"),
                                                           bias=False)
            else:
                raise ValueError("illegal model type")

        if opts.model_type != "gumbel":
            self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                          c_puct=self.c_puct,
                                          n_playout=self.n_playout,
                                          is_selfplay=1)
            print("[Now] The MCTS PLATER: Alphazero ")
        elif opts.model_type == "gumbel":
            self.mcts_player = Gumbel_MCTSPlayer(self.policy_value_net.policy_value_fn,
                                                 c_puct=self.c_puct,
                                                 n_playout=self.n_playout,
                                                 is_selfplay=1,
                                                 m_action=self.m)
            print("[Now] The MCTS PLATER: Gumbel_Alphazero ")

        if opts.data_collect == 1:
            self.high_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                          c_puct=self.c_puct,
                                          n_playout=self.n_playout)

        # The set of optimizer
        self.optimizer = optim.Adam(self.policy_value_net.policy_value_net.parameters(),
                                    weight_decay=opts.l2_const)
        # set learning rate
        set_learning_rate(self.optimizer, self.learn_rate * self.lr_multiplier)

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

    def job(self, i):
        game = self.game
        player = self.mcts_player
        winner, play_data = game.start_self_play(player,
                                                 temp=self.temp)

        play_data = list(play_data)[:]

        play_data = self.get_equi_data(play_data)

        return play_data

    def job_highplay(self, i):
        game = self.game
        player = self.mcts_player

        high_player = self.high_player

        winner, play_data = game.start_play_collect(player, high_player, start_player=i % 2,
                                                    temp=self.temp)

        play_data = list(play_data)[:]

        play_data = self.get_equi_data(play_data)

        return play_data

    def parser_output(self, outflies, n_games):

        ignore_opening_random = 3

        for i in range(n_games):
            winner, play_data = self.game.start_parser(outflies[i])
            print("[DATA] get_data from ", outflies[i])

            play_data = list(play_data)[ignore_opening_random:]

            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

        # return play_data  

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        # print("[STAGE] Collecting self-play data for training")

        # collection_bar = tqdm( range(n_games))
        collection_bar = range(n_games)
        if n_games <= 4:
            for i in collection_bar:
                self.data_buffer.extend(self.job(i))
        else:
            with Pool(4) as p:
                play_datas = p.map(self.job, collection_bar)
            for play_data in play_datas:
                self.data_buffer.extend(play_data)
        # print('\n', 'data buffer size:', len(self.data_buffer))

    def collect_highplay_data(self, n_games=1):
        collection_bar = range(n_games)
        if n_games <= 4:
            for i in collection_bar:
                self.data_buffer.extend(self.job_highplay(i))
        else:
            with Pool(4) as p:
                play_datas = p.map(self.job_highplay, collection_bar)
            for play_data in play_datas:
                self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        epoch_bar = tqdm(range(self.epochs))

        for i in epoch_bar:
            """perform a training step"""
            # wrap in Variable
            if self.use_gpu:

                state_batch = Variable(torch.FloatTensor(state_batch).cuda())
                mcts_probs = Variable(torch.FloatTensor(mcts_probs_batch).cuda())
                winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
            else:
                state_batch = Variable(torch.FloatTensor(state_batch))
                mcts_probs = Variable(torch.FloatTensor(mcts_probs_batch))
                winner_batch = Variable(torch.FloatTensor(winner_batch))

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            log_act_probs, value = self.policy_value_net.policy_value_net(state_batch)

            # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
            # Note: the L2 penalty is incorporated in optimizer
            value_loss = F.mse_loss(value.view(-1), winner_batch)
            policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))

            # if the player is Gumbel player, policy loss is
            # if opts.Player ==1 :
            #     policy_loss = torch.mean( (torch.sum(mcts_probs * (
            #         np.log(mcts_probs + 1e-10) - log_act_probs),
            #         axis=1)))

            loss = value_loss + policy_loss

            # backward and optimize
            loss.backward()
            self.optimizer.step()
            # calc policy entropy, for monitoring only
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
            )
            loss = loss.item()
            entropy = entropy.item()

            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break

            epoch_bar.set_description(f"training epoch {i}")
            epoch_bar.set_postfix(new_v=new_v[0], kl=kl)

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

        return kl, loss, entropy, explained_var_old, explained_var_new, value_loss, policy_loss

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """

        if opts.model_type != "gumbel":
            current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                             c_puct=self.c_puct,
                                             n_playout=self.n_playout)
            print("[TEST] The MCTS PLATER: Alphazero ")
        elif opts.model_type == "gumbel":
            current_mcts_player = Gumbel_MCTSPlayer(self.policy_value_net.policy_value_fn,
                                                    c_puct=self.c_puct,
                                                    n_playout=self.n_playout,
                                                    m_action=self.m)
            print("[TEST] The MCTS PLATER: Gumbel_Alphazero ")

        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)

        if opts.split == "test":
            #  Alphazero Vs MCTS_Pure
            if opts.mood == 0:

                current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                                 c_puct=self.c_puct,
                                                 n_playout=self.n_playout)

                pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
                print("[TEST] Alphazero  Vs MCTS_Pure")

            #  Gumbel_Alphazero Vs MCTS_Pure
            elif opts.mood == 1:
                current_mcts_player = Gumbel_MCTSPlayer(self.policy_value_net.policy_value_fn,
                                                        c_puct=self.c_puct,
                                                        n_playout=self.n_playout,
                                                        m_action=self.m)

                pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
                print("[TEST] Gumbel_Alphazero  Vs MCTS_Pure")

            # Alphazero Vs  Gumbel_Alphazero
            elif opts.mood == 2:

                current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                                 c_puct=self.c_puct,
                                                 n_playout=self.n_playout)

                pure_mcts_player = Gumbel_MCTSPlayer(self.policy_value_net.policy_value_fn,
                                                     c_puct=self.c_puct,
                                                     n_playout=self.n_playout,
                                                     m_action=self.m)

                print("[TEST] Alphazero Vs  Gumbel_Alphazero ")
            else:
                print("> error: illegal mood num")

        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player, pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=True)
            win_cnt[winner] += 1
            print(f" {i}_th winner:", winner)
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games

        if (opts.split == "test"):
            if opts.mood == 0:
                print("[TEST] Alphazero  Vs MCTS_Pure")
            elif opts.mood == 1:
                print("[TEST] Gumbel_Alphazero  Vs MCTS_Pure")
            elif opts.mood == 2:
                print("[TEST] Alphazero Vs  Gumbel_Alphazero ")

        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""

        print("training start....")
        print("model_type: ", opts.model_type)
        try:

            batch_bar = tqdm(range(self.game_batch_num))
            for i in batch_bar:
                if opts.data_collect == 1:
                    self.mcts_player._is_selfplay = 0
                    self.collect_highplay_data(self.play_batch_size)
                    # print("[Done] collect high")
                elif opts.data_collect == 2:
                    # get absolute path
                    dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    output_dir = os.path.join(dirname, "generate_data", "10_thousand_data")

                    files = os.listdir(output_dir)[1:]  # 这里需要把json 跳掉
                    # random_files = random.sample(files, self.play_batch_size)
                    # random_files = [os.path.join(output_dir,file) for file in random_files]

                    with open(os.path.join(output_dir, "data_split.json")) as json_file:
                        split = json.load(json_file)

                    # files = os.listdir(output_dir)
                    # print(split)
                    files = split['1']  # 这里按照需要设置 胜者 1, 2 ; 平局 -1 , 可以平均随机一下

                    files = random.sample([os.path.join(output_dir, file) for file in files],
                                          self.play_batch_size)  # 建议把这个play_batch_size 调大一些

                    self.parser_output(files, self.play_batch_size)  # 建议把这个play_batch_size 调大一些

                else:
                    self.collect_selfplay_data(self.play_batch_size)

                # print("Done")
                if len(self.data_buffer) > self.batch_size:
                    kl, loss, entropy, explained_var_old, explained_var_new, value_loss, policy_loss = self.policy_update()

                    writer.add_scalar("policy_update/kl", kl, i)
                    writer.add_scalar("policy_update/loss", loss, i)
                    writer.add_scalar("policy_update/value_loss", value_loss, i)
                    writer.add_scalar("policy_update/policy_loss", policy_loss, i)
                    writer.add_scalar("policy_update/entropy", entropy, i)
                    writer.add_scalar("policy_update/explained_var_old", explained_var_old, i)
                    writer.add_scalar("policy_update/explained_var_new ", explained_var_new, i)

                batch_bar.set_description(f"game batch num {i}")

                # check the performance of the current model,
                # and save the model params
                print(self.board.availables)
                if (i + 1) % self.check_freq == 0:
                    win_ratio = self.policy_evaluate()

                    batch_bar.set_description(f"game batch num {i + 1}")
                    writer.add_scalar("evaluate/win_ratio ", win_ratio, i)
                    batch_bar.set_postfix(loss=loss, entropy=entropy, win_ratio=win_ratio)

                    save_model(self.policy_value_net.policy_value_net, "current_policy.model")

                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        print("best win_ratio: ", self.best_win_ratio)
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        save_model(self.policy_value_net.policy_value_net, "best_policy.model")
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == "__main__":
    print("START train....")

    # ------init set-----------

    if opts.std_log:
        std_log()
    writer = visualizer()

    if opts.distributed:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        init_seeds(opts.seed + local_rank)

    else:
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = opts.device
        init_seeds(opts.seed)

    print("seed: ", opts.seed)
    print("device:", device)

    if opts.split == "train":
        training_pipeline = MainWorker(device)
        training_pipeline.run()

    if get_rank() == 0 and opts.split == "test":
        training_pipeline = MainWorker(device)
        training_pipeline.policy_evaluate()
