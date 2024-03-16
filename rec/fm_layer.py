import torch
from torch import nn

class FMLayer(nn.Module):
    def __init__(self):
        super(FMLayer, self).__init__()
    
    def forward(self, inputs):
        sum_square = torch.pow(torch.sum(inputs, dim=1), 2)
        square_sum = torch.sum(torch.pow(inputs, 2), dim=1)
        return sum_square - square_sum