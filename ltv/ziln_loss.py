import torch
from torch import nn
from torch.distributions.log_normal import LogNormal


def ziln_loss(mu, sigma, cls_pred, label):
    cls_label = torch.where(
        cls_label > 0, torch.ones_like(label), torch.zeros_like(label)
    )
    ln_label = cls_label * label + (1 - cls_label) * torch.ones_like(label)
    lognormal = LogNormal(mu, sigma)
    ce_loss = nn.BCELoss()(cls_pred, cls_label)
    log_loss = -torch.mean(lognormal.log_prob(ln_label))
    return log_loss + ce_loss


class ZILNLoss(nn.Module):
    def __init__(self):
        super(ZILNLoss, self).__init__()

    def forward(self, mu, sigma, cls_label, label):
        return ziln_loss(mu, sigma, cls_pred, label)
