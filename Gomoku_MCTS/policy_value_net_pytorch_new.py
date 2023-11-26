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


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class Net(nn.Module):
    """Policy-Value network module for AlphaZero Gomoku."""
    def __init__(self, board_width, board_height, num_residual_blocks=5):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.res_layers = nn.Sequential(*[ResidualBlock(32) for _ in range(num_residual_blocks)])

        # Action Policy layers
        self.act_conv1 = nn.Conv2d(32, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)

        # State Value layers
        self.val_conv1 = nn.Conv2d(32, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_layers(x)

        # Action Policy head
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # State Value head
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))

        return x_act, x_val


class PolicyValueNet():
    """policy-value network """

    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=False):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.use_gpu = use_gpu
        self.l2_const = 1e-4  # coef of l2 penalty
        self.board_width = board_width
        self.board_height = board_height

        if model_file:
            net_params = torch.load(model_file, map_location='cpu' if not use_gpu else None)

            # Infer board dimensions from the loaded model
            inferred_width, inferred_height = self.infer_board_size_from_model(net_params)
            if inferred_width and inferred_height:
                self.policy_value_net = Net(inferred_width, inferred_height).to(self.device) if use_gpu else Net(
                    inferred_width, inferred_height)
                self.policy_value_net.load_state_dict(net_params)
                print("Use model file to initialize the policy value net")
            else:
                raise Exception("The model file does not contain the board dimensions")

            if inferred_width < board_width:
                self.use_conv = True
            elif inferred_width > board_width:
                raise Exception("The model file has a larger board size than the current board size!!")
        else:
            # the policy value net module
            if self.use_gpu:
                self.policy_value_net = Net(board_width, board_height).to(self.device)
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
        return None

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
            state_batch = Variable(torch.FloatTensor(state_batch).to(self.device))
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
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, 4, self.board_width, self.board_height))
        if bias:
            current_state[0][1] = self.apply_normal_bias(current_state[0][1])

        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                Variable(torch.from_numpy(current_state)).to(self.device).float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""

        # self.use_gpu = True
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).to(self.device))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).to(self.device))
            winner_batch = Variable(torch.FloatTensor(winner_batch).to(self.device))
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
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        )

        # for pytorch version >= 0.5 please use the following line instead.
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)


if __name__ == "__main__":
    import torch
    import torch.onnx

    # 假设您的 Net 模型已经定义好了
    model = Net(board_width=9, board_height=9)  # 使用适当的参数初始化模型
    dummy_input = torch.randn(1, 4, 9, 9)  # 创建一个示例输入

    # 将模型导出到 ONNX 格式
    torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)
