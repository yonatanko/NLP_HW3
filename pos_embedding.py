import torch
from torch import nn
from torch.utils.data import DataLoader


class AutoencoderNN(torch.nn.Module):
    def __init__(self):
        super(AutoencoderNN, self).__init__()
        self.loss = nn.MSELoss()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(45, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 45)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, self.loss(decoded, x)


def train(model, train_data, optimizer, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        decoded, loss = model(train_data)
        loss.backward()
        optimizer.step()
        if epoch == 999:
            return decoded


def build_one_hot_tensors(pos_list):
    tensor_dim = len(pos_list)
    one_hot_tensors = []
    for pos in pos_list:
        one_hot_tensor = torch.zeros(tensor_dim)
        one_hot_tensor[pos_list.index(pos)] = 1
        one_hot_tensors.append(one_hot_tensor)

    return torch.stack(one_hot_tensors)


def transform(pos_train):
    model = AutoencoderNN()
    one_hot_data = build_one_hot_tensors(pos_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    decoded_pos = train(model, one_hot_data, optimizer, 1000)
    pos_to_vec = {}
    for i, pos in enumerate(pos_train):
        pos_to_vec[pos] = decoded_pos[i]

    return pos_to_vec


