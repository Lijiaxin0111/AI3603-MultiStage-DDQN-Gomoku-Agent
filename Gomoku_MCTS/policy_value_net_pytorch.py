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
from conv_utils import split_board_into_parts



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
                 model_file=None, use_gpu=False, bias = False):
        self.use_gpu = use_gpu
        self.l2_const = 1e-4  # coef of l2 penalty
        self.use_conv = False
        self.board_width = board_width
        self.board_height = board_height
        self.bias = bias

        if model_file:
            net_params = torch.load(model_file, map_location='cpu' if not use_gpu else None)
            
            

            # Infer board dimensions from the loaded model
            inferred_width, inferred_height = self.infer_board_size_from_model(net_params)

            if inferred_width and inferred_height:
                self.policy_value_net = Net(inferred_width, inferred_height).cuda() if use_gpu else Net(
                    inferred_width, inferred_height)
                self.policy_value_net.load_state_dict(net_params)
            else:
                raise Exception("The model file does not contain the board dimensions")

            if inferred_width < board_width:
                self.use_conv = True
            elif inferred_width > board_width:
                raise Exception("The model file has a larger board size than the current board size!!")
        else:
            # the policy value net module
            if self.use_gpu:
                self.policy_value_net = Net(board_width, board_height).cuda()
            else:
                self.policy_value_net = Net(board_width, board_height)

        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                        weight_decay=self.l2_const)

    def infer_board_size_from_model(self, model):
        # Use the size of the act_fc1 layer to infer board dimensions
        for name in model.keys():
            if name == 'act_fc1.weight':
                # Assuming the weight shape is [board_width * board_height, 4 * board_width * board_height]
                c, _ = model[name].shape
                print(f"act_fc1.weight shape: {model[name].shape}")
                board_size = int(c ** 0.5)  # Extracting board_width/height assuming they are the same
                print(f"Board size inferred from model: {board_size}x{board_size}")
                return board_size, board_size
        return None, None

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
            print(" !!",torch.FloatTensor(state_batch).shape)
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        bias = self.bias
        """
        数组具有四个通道:通道一表示当前玩家的棋子位置，
        第二个通道表示对手的棋子位置，第三个通道表示最后一步移动的位置.通道四是一个指示符，用于表示当前轮到哪个玩家(如果棋盘上的总移动次数是偶数，那么这个通道的所有元素都为1，表示是第一个玩道都是一个 width x height 的二维数组，代表着棋盘的布局。对于第一个和第二个通道，如果一个位置上有当前玩家或对手的棋三个通道，只有最后一步移动的位置是1，其余位置都为8。对于第四个通道，如果是第一个玩家的回合，那么所有的位置都是1，否状态数组在垂直方向上翻转，以匹配棋盘的实际布局。
        """
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        if bias:
            current_state[0][1] = self.apply_normal_bias(current_state[0][1])

        if self.use_gpu:
            if self.use_conv:
                act_probs, value = self.conv_method(current_state, self.policy_value_net)
            else:
                log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
                act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            if self.use_conv:
                act_probs, value = self.conv_method(current_state, self.policy_value_net)
            else:
                log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
                act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        if self.use_conv:
            return act_probs, value
        value = value.data[0][0]
        return act_probs, value

    def conv_method(self, cur_state, net):
        """
        since net's boardsize is smaller than current one(for example, a checkpoint trained on 8 * 8 board,
        we want to use it on 9 * 9 or large board, then we need to do some tricks:
        we split the board into five part(with overlap), leftup corner, rightup corner, leftdown corner, rightdown corner and center part
        then we use net to predict the probability of each part, and finally we merge them together.
        The merge method is:
        - for those without overlap, we just use the prob and val derived from the net
        - for those overlapped, we pick the max prob and proper cope with val
        the output of the net is decideded by:
            x_act = (1, board_width * board_height)
            x_val = (1, 1), the wining "probability"
            return x_act, x_val
        In this case, we need to:
        1. split the board into five part, by doing index operation on cur_state, which has input size of (1,4,board_width,board_height)
        2. do the forward pass on each part, and get corresponding prob and val
        3. merge
            3.1 for acts, we need to know its position mapping to the original board; we initialize a board[][], then fill the value derived from
                5 parts, which should be taken good care of. We should figure it out how to fill the board[][], according to the position of these five parts, as well as taking into considerations that the return of net is a list of array with len = board_width * board_height(board_width <= current board width)
            3.2 for vals, we have a corresponding value for each part; we could simply pick the max one as val first
        """

        # 1. split the board into five part, by doing index operation on cur_state, which has input size of (1,4,board_width,board_height)
        parts, positions = split_board_into_parts(cur_state, part_size=8)
        # 2. do the forward pass on each part, and get corresponding prob and val (prob1, prob2, prob3, prob4, prob5) (val1, val2, val3, val4, val5)
        probs = []
        vals = []
        for part in parts:
            if self.use_gpu:
                part = Variable(torch.FloatTensor(part).cuda())
            else:
                part = Variable(torch.FloatTensor(part))
            log_act_probs, value = net(part)
            probs.append(log_act_probs)
            vals.append(value.data[0][0])
        # 3. merge probs. For prob1, it should be remapped to the board[][] according to the position of probs1, namely the left up corner; others is the same. Note that when visiting board[][], if it already has a value, then it should be compared with the current probs, and pick the max one. (board[i][j] = max(board[i][j], probs1[i][j]))
        new_results = torch.Tensor(np.zeros((self.board_width, self.board_height)))
        for k in range(len(probs)):
            prob = probs[k].data.cpu().numpy().flatten()
            prob = np.exp(prob)
            position = positions[k]

            # 现在你可以将这个值赋给 new_results[i][j]

            for index in range(len(prob)):
                i, j = position[index]
                prob_value_as_tensor = torch.tensor(prob[index], dtype=torch.float32)
                new_results[i][j] = torch.max(new_results[i][j], prob_value_as_tensor)
        # flatten the new_results, to 1 * board_width * board_height
        new_results = new_results.flatten()
        # 4. merge vals, make the final return val the max of (val1, val2, val3, val4, val5)
        new_val = max(vals)
        # nomarlize the new_results
        # print(sum(new_results))
        new_results = new_results / sum(new_results)

        return new_results, new_val


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
