# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0

@author: Junxiao Song
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import numpy as np
import scipy.stats as stats



class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        board_height = board_width = self.board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def reshape_linear(self):
        """
        if board_width and board_height != 8, then reshape self.act_fc1 and self.val_fc1, this is because the model we load is for board_size 8 * 8
        use some methods which is simlar to enlarge an image wihout losing much information (Super High Resolution)
        for example, before reshaping, self.act_fc1 = nn.Linear(4*8*8, 8*8)
        after reshaping, it becomes nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        the same for self.val_fc1
        """
        if (self.board_width == 8) and (self.board_height == 8):
            return
        # 假设原始尺寸是 8x8
        original_size = 8 * 8
        new_size = self.board_width * self.board_height

        # 调整 self.act_fc1
        original_weight = self.act_fc1.weight.data.view(1, 1, 64, 256)
        new_weight = F.interpolate(original_weight, size=( 4 * new_size, new_size), mode='bilinear', align_corners=False)
        self.act_fc1 = nn.Linear(4 * new_size, new_size)
        self.act_fc1.weight.data = new_weight.view(new_size, 4 * new_size)

        # 调整 self.val_fc1
        original_weight = self.val_fc1.weight.data.view(1, 1, 2 * 64, 64)
        new_weight = F.interpolate(original_weight, size=(2 * new_size, 64), mode='bilinear', align_corners=False)
        self.val_fc1 = nn.Linear(2 * new_size, 64)
        self.val_fc1.weight.data = new_weight.view( 64,2 * new_size)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act))
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
    
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)
            # self.policy_value_net.reshape_linear()

    def apply_normal_bias(self, tensor, mean=0, std=1):
        bsize = tensor.shape[0]
        x, y = np.meshgrid(np.linspace(-1, 1, bsize), np.linspace(-1, 1, bsize))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 1.0, 0.0
        gauss = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        # Applying the bias only to non-zero elements
        biased_tensor = tensor - (tensor != 0) * gauss
        return biased_tensor

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board, bias=False):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        """
        数组具有四个通道:通道一表示当前玩家的棋子位置，
        第二个通道表示对手的棋子位置，第三个通道表示最后一步移动的位置.通道四是一个指示符，用于表示当前轮到哪个玩家(如果棋盘上的总移动次数是偶数，那么这个通道的所有元素都为1，表示是第一个玩道都是一个 width x height 的二维数组，代表着棋盘的布局。对于第一个和第二个通道，如果一个位置上有当前玩家或对手的棋三个通道，只有最后一步移动的位置是1，其余位置都为8。对于第四个通道，如果是第一个玩家的回合，那么所有的位置都是1，否状态数组在垂直方向上翻转，以匹配棋盘的实际布局。
        """
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        if bias:
            current_state[0][1] = self.apply_normal_bias(current_state[0][1])

        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value
    

# 搬到main_worker

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
    
        # self.use_gpu = True
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        # return loss.data[0], entropy.data[0]
        #for pytorch version >= 0.5 please use the following line instead.
        return loss.item(), entropy.item()

    # def get_policy_param(self):
    #     net_params = self.policy_value_net.state_dict()
    #     return net_params

    # def save_model(self, model_file):
    #     """ save model params to file """
    #     net_params = self.get_policy_param()  # get model params
    #     torch.save(net_params, model_file)
