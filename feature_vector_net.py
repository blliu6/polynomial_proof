import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureVectorNet(nn.Module):
    def __init__(self, input_size, dense=1, hidden_size=500, output_size=500):
        super().__init__()
        self.seq = nn.Sequential()
        s = input_size
        for i in range(dense):
            self.seq.add_module(f'dense_{i}', nn.Linear(s, hidden_size))
            self.seq.add_module(f'relu_{i}', nn.ReLU())
            s = hidden_size
        self.seq.add_module(f'dense_{dense}', nn.Linear(s, output_size))

    def forward(self, x):
        # torch.max(self.seq(x), dim=0)[0]
        return self.seq(x).max(axis=1)[0]


if __name__ == '__main__':
    # torch.manual_seed(2024)
    # net = FeatureVectorNet(6)
    # x1 = torch.randn(10, 6)
    # x2 = torch.randn(20, 6)
    # x3 = torch.randn(30, 6)
    # x = torch.tensor([x1, x2, x3])
    # print(x)
    # print(net(x).size())
    # print(net(x))
    pass
