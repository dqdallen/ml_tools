import torch
from torch import nn


class Expert(nn.Module):
    def __init__(self, input_dim, layers, is_bn, activation, dropout_rate=None):
        """

        Args:
            input_dim (int): inputs emb dim
            layers (list): a list of hidden layer size
            is_bn (bool): if use BatchNorm
            activation (nn.Module): activation function like nn.ReLU
            dropout_rate (float, optional): dropout rate, None indicates no droupout
        """
        super(Expert, self).__init__()
        self.input_dim = input_dim
        self.layers = layers
        self.is_bn = is_bn
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.expert = nn.Sequential()
        emb_dim = self.input_dim
        for i in range(len(self.layers)):
            self.expert.add_module(nn.Linear(emb_dim, self.layers[i]))
            if is_bn:
                self.expert.add_module(nn.BatchNorm1d(1))
            self.expert.add_module(self.activation)
            if dropout_rate is not None:
                self.expert.add_module(nn.Dropout(self.dropout_rate))
            emb_dim = self.layers[i]
    
    def forward(self, inputs):
        output = self.expert(inputs)
        return output


class ExpertGroup(nn.Module):
    def __init__(self, input_dim, layers, expert_num, is_bn, activation, dropout_rate=None):
        """A group of experts

        Args:
            input_dim (int): inputs emb dim
            layers (list): a list of hidden layer size
            expert_num (int): the number of experts
            is_bn (bool): if use BatchNorm
            activation (nn.Module): activation function like nn.ReLU
            dropout_rate (float, optional): dropout rate, None indicates no droupout
        """
        super(Expert, self).__init__()
        self.input_dim = input_dim
        self.layers = layers
        self.is_bn = is_bn
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.expert_num = expert_num

        self.expert_group = nn.ModuleList([
            Expert(input_dim, layers, is_bn, activation, dropout_rate)
            for _ in range(self.expert_num)
        ])
    

    def forward(self, inputs):
        output = self.expert_group(inputs)
        output = torch.cat([out.unsqueeze(dim=1) for out in output], dim=1)
        return output


class Gate(nn.Module):
    def __init__(self, input_dim, layers, is_bn, activation, dropout_rate=None):
        """gate network

        Args:
            input_dim (int): inputs emb dim
            layers (list): a list of hidden layer size
            is_bn (bool): if use BatchNorm
            activation (nn.Module): activation function like nn.ReLU
            dropout_rate (float, optional): dropout rate, None indicates no droupout
        """
        super(Gate, self).__init__()
        self.input_dim = input_dim
        self.layers = layers
        self.is_bn = is_bn
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.gate = Expert(self.input_dim, self.layers, self.is_bn, self.activation, self.dropout_rate)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, inputs):
        gate_emb = self.gate(inputs)
        output = self.softmax(gate_emb)
        return output