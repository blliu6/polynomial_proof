import torch
import torch.nn as nn


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

    def forward(self, x, prefix):
        # torch.max(self.seq(x), dim=0)[0]
        # 将数据预处理，形成一个N * input_size的矩阵
        res = []
        x = self.seq(x)
        for i in range(len(prefix) - 1):
            l, r = prefix[i], prefix[i + 1]
            # print(x[l:r, :].max(0)[0])
            res.append(x[l:r, :].max(0)[0].reshape(1, -1))
        return torch.cat(res, 0)


if __name__ == '__main__':
    x = torch.randn(10, 6)
    pre = [0, 4, 6, 8, 10]
    net = FeatureVectorNet(6, hidden_size=5, output_size=5)
    print(net(x, pre))
