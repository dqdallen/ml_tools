import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch.distributions.log_normal import LogNormal
import sys
from mtl.share_bottom import ShareBottom
import numpy as np
import random

class Single(nn.Module):
    def __init__(self, config):
        super(Single, self).__init__()
        self.emb_dim = config['emb_dim']
        self.dropout_rate = config['dropout_rate']
        self.num_fea_ids = config['num_fea_ids']
        self.device = config['device']
        self.cate_fea_num = config['cate_fea_num'] # 每个cate特征对应的id个数，是一个数组
        
        self.cate_emb_tables = nn.ModuleList([nn.Embedding(cate_ids, self.emb_dim) for cate_ids in self.cate_fea_num])
        self.num_emb_linear = nn.ModuleList([nn.Linear(1, self.emb_dim) for _ in range(self.num_fea_ids)])
        self.output_dim = config['output_dim']
        self.hidden_units = [self.emb_dim * 2] + config['hidden_units']
        layers = []
        for i in range(len(self.hidden_units)-1):
            layers.append(nn.Linear(self.hidden_units[i], self.hidden_units[i+1]))
            layers.append(nn.BatchNorm1d(self.hidden_units[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        self.mlp = nn.Sequential(**layers)
        self.predict_layer = nn.Linear(self.hidden_units[-1], self.output_dim)
    
    def forward(self, cate_feat, num_feat):
        cate_fea_embs = [emb_table(cate_feat[:, idx]).unsqueeze(dim=1) for idx, emb_table in enumerate(self.cate_emb_tables)]
        cate_fea_emb_cat = torch.cat(cate_fea_embs, dim=1)
        num_fea_embs = [linear(num_feat[:, idx:idx+1]) for idx, linear in enumerate(self.num_emb_linear)]
        num_fea_emb_cat = torch.cat(num_fea_embs, dim=1)
        fea_emb = torch.cat([cate_fea_emb_cat, num_fea_emb_cat], dim=1)
        fea_emb = self.mlp(fea_emb)
        output = torch.sigmoid(self.predict_layer(fea_emb))
        return {
            'pred': output
        }
    
    def create_loss(self, y_pred, y_true):
        loss = nn.BCELoss()(y_pred.reshape(-1), y_true.reshape(-1))
        return loss
    
class CustomShareBottom(nn.Module):
    def __init__(self, config):
        super(CustomShareBottom, self).__init__()
        self.emb_dim = config['emb_dim']
        self.dropout_rate = config['dropout_rate']
        self.num_fea_ids = config['num_fea_ids']
        self.device = config['device']
        self.cate_fea_num = config['cate_fea_num']

        self.share_units = config['share_units'] # [512, 512]
        self.task_units = config['task_units'] # [[512, 256], [512, 256]]
        self.task_num = len(self.task_units)

        self.cate_emb_tables = nn.ModuleList([nn.Embedding(cate_ids, self.emb_dim) for cate_ids in self.cate_fea_num])
        self.num_emb_linear = nn.ModuleList([nn.Linear(1, self.emb_dim) for _ in range(self.num_fea_ids)])
        self.output_dim = config['output_dim']
        self.sharebottom = ShareBottom(self.emb_dim*2, self.share_units, self.task_units, is_bn=True, activation=nn.ReLU(), dropout_rate=self.dropout_rate)
        self.predict_layers = nn.ModuleList([nn.Linear(256, self.output_dim) for _ in range(self.task_num)])

    def forward(self, cate_feat, num_feat):
        cate_fea_embs = [emb_table(cate_feat[:, idx]).unsqueeze(dim=1) for idx, emb_table in enumerate(self.cate_emb_tables)]
        cate_fea_emb_cat = torch.cat(cate_fea_embs, dim=1)
        num_fea_embs = [linear(num_feat[:, idx:idx+1]) for idx, linear in enumerate(self.num_emb_linear)]
        num_fea_emb_cat = torch.cat(num_fea_embs, dim=1)
        fea_emb = torch.cat([cate_fea_emb_cat, num_fea_emb_cat], dim=1)
        fea_emb = self.sharebottom(fea_emb)
        outputs = [torch.sigmoid(self.predict_layers[i](fea_emb)) for i in range(self.task_num)]
        return {
            'pred': outputs
        }
    
    def create_loss(self, y_pred, y_true):
        for i in range(self.task_num):
            loss = nn.BCELoss()(y_pred[i].reshape(-1), y_true[i].reshape(-1))
        return loss