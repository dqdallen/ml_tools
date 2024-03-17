import torch
from torch import nn


class TweedieNet(nn.Module):
    def __init__(self, input_dim, layers, is_bn, activation, dropout_rate=None):
        self.input_dim = input_dim
        self.layers = layers
        self.is_bn = is_bn
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.net = nn.Sequential()
        emb_dim = self.input_dim
        for i in range(len(self.layers)):
            self.net.add_module(nn.Linear(emb_dim, self.layers[i]))
            if is_bn:
                self.net.add_module(nn.BatchNorm1d(1))
            if i < len(self.layers)-1:
                self.net.add_module(self.activation)
            if dropout_rate is not None:
                self.net.add_module(nn.Dropout(self.dropout_rate))
            emb_dim = self.layers[i]
    
    def forward(self, inputs):
        mu = self.net(inputs)
        return mu, torch.exp(mu)