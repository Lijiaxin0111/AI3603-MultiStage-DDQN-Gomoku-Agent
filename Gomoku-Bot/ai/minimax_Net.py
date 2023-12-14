import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
from tensorboardX import SummaryWriter
from tqdm import tqdm
import datetime
from torch.utils.data import DataLoader, TensorDataset

date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


class BoardEvaluationNet(nn.Module):
    def __init__(self, board_size):
        super(BoardEvaluationNet, self).__init__()
        self.board_size = board_size

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * board_size * board_size, 256)
        self.fc2 = nn.Linear(256, board_size * board_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * self.board_size * self.board_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, self.board_size, self.board_size)


def normalize(t):
    return t


if __name__ == "__main__":

    writer = SummaryWriter(os.path.join(dir_path, 'train_data/log', date), comment='BoardEvaluationNet')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best = np.Inf
    loss_fn = nn.CrossEntropyLoss()

    # Example usage
    BS = 15

    net_for_black = BoardEvaluationNet(BS).to(device)
    net_for_white = BoardEvaluationNet(BS).to(device)

    net_for_black.load_state_dict(torch.load(os.path.join(dir_path, 'train_data/model', 'best_loss=680.5813717259707.pth')))

    optimizer = torch.optim.Adam(net_for_black.parameters(), lr=1e-5, betas=(0.9, 0.99),
                                 eps=1e-8)

    data_path = os.path.join(dir_path, 'train_data/data', 'train_data.pkl')
    with open(data_path, 'rb') as f:
        datas = pickle.load(f)

    train_data_for_black = datas[1][:int(len(datas[1]) * 1)]
    test_data_for_black = datas[1][int(len(datas[1]) * 0.8):]
    train_data_for_white = datas[-1]
    epochs = 500
    batch_size = 32
    train_dataset = TensorDataset(torch.stack([torch.tensor(item['state'], dtype=torch.float) for item in train_data_for_black]),
                                  torch.stack([normalize(torch.tensor(item['scores'], dtype=torch.float)) for item in train_data_for_black]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        print('Epoch:', epoch)
        for i, (states, scores) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            states = states.to(device)
            scores = scores.to(device)

            # print(input_tensor.shape)
            infer_start = datetime.datetime.now()
            output_tensor = net_for_black(states)
            infer_end = datetime.datetime.now()
            loss = loss_fn(output_tensor, scores)
            print(loss.item())
            exit(0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

            writer.add_scalar('train/infer_time', (infer_end - infer_start).microseconds,
                              i + epoch * len(train_dataloader))

        epoch_loss /= len(train_dataloader)
        writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
        # test
        with torch.no_grad():
            test_loss = 0
            net_for_black.eval()
            for j, item in tqdm(enumerate(test_data_for_black), total=len(test_data_for_black)):
                scores = normalize(torch.tensor(item['scores'], dtype=torch.float).to(device).unsqueeze(0))  # 将数据类型设为float
                state = item['state']
                input_tensor = torch.tensor(state, dtype=torch.float).to(device).unsqueeze(0)  # 将数据类型设为float，并转移到设备上
                output_tensor = net_for_black(input_tensor).to(device)
                loss = loss_fn(output_tensor, scores)
                test_loss += loss.item()
            test_loss /=len(test_data_for_black)
            writer.add_scalar('test/loss', test_loss, epoch)
            if best > test_loss:
                best = test_loss
                model_path = os.path.join(dir_path, 'train_data/model')
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                torch.save(net_for_black.state_dict(),
                           os.path.join(model_path, f'best_loss={best}.pth'))
        net_for_black.train()