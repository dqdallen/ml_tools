import torch
from torch import nn


def tweedie_loss(pred, label, rho=1.7):
    a = torch.exp(pred * (2 - rho)) / (2 - rho)
    b = -label * torch.exp(pred * (1 - rho)) / (1 - rho)
    return torch.mean(a + b)


class TweedieLoss(nn.Module):
    def __init__(self, rho=1.7):
        super(TweedieLoss, self).__init__()
        self.rho = rho

    def forward(self, pred, label):
        return tweedie_loss(pred, label, self.rho)
