import torch
import torch.nn as nn
import numpy as np


class HN(nn.Module):
    def __init__(self, input_size):
        super(HN, self).__init__()
        self.t_gate = nn.Sequential(nn.Linear(input_size, input_size), nn.Sigmoid())  # 好像用什么激活函数都可以
        self.h_layer = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())  # 好像用什么激活函数都可以
        self.t_gate[0].bias.data.fill_(0)  # 这行不知道干啥的

    def forward(self, x):
        t = self.t_gate(x)  # T=sigmoid(wx+b) 忽略bias
        h = self.h_layer(x)  # H=ReLU(wx+b)
        z = torch.mul(1 - t, x) + torch.mul(t, h)  # x*C+H*T (C=1-T)

        return z


class LM(nn.Module):
    def __init__(self, wtoken, ctoken, max_len, embed_dim, channels, kernels, hidden_size):
        super(LM, self).__init__()

        self.embedding = nn.Embedding(ctoken, embed_dim, padding_idx=0)

        self.conv_layers = nn.ModuleList()
        for kernel in kernels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(1, channels * kernel, kernel_size=(kernel, embed_dim)),  # channels*kernel?
                nn.Tanh(),
                nn.MaxPool2d((max_len - kernel + 1, 1))
            ))

        input_size = int(channels * np.sum(kernels))
        self.hw = HN(input_size)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.5)

        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, wtoken)
        )

    def forward(self, x, h):
        batch_size = x.shape[0]

        seq_len = x.shape[1]

        x = x.view(-1, x.shape[2])

        x = self.embedding(x.long())

        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])

        y = [cnn(x).squeeze() for cnn in self.conv_layers]

        w = torch.cat(y, 1)

        w = self.hw(w)

        w = w.view(batch_size, seq_len, -1)
        out, h = self.lstm(w, h)

        out = out.contiguous().view(batch_size * seq_len, -1)

        out = self.linear(out)

        return out, h
