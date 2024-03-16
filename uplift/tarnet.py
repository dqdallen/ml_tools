import torch
from torch import nn


class TarNet(nn.Module):
    def __init__(self, input_dim, share_layers, treatment_layers, is_bn, activation, last_activation, treatments, dropout_rate=None):
        super(TarNet, self).__init__()
        self.input_dim = input_dim
        self.share_layers = share_layers
        self.treatment_layers = treatment_layers
        self.is_bn = is_bn
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.treatments  = treatments
        self.last_activation = last_activation

        self.share_layer = nn.Sequential()
        emb_dim = self.input_dim
        for i in range(len(self.share_layers)):
            self.share_layer.add_module(nn.Linear(emb_dim, self.share_layers[i]))
            if is_bn:
                self.share_layer.add_module(nn.BatchNorm1d(1))
            self.share_layer.add_module(self.activation)
            if dropout_rate is not None:
                self.share_layer.add_module(nn.Dropout(self.dropout_rate))
            emb_dim = self.share_layers[i]
        
        treatment_tower = []
        for t in range(self.treatments):
            treatment_layer = nn.Sequential()
            emb_dim = self.share_layers[-1]
            for i in range(len(self.treatment_layers)):
                treatment_layer.add_module(nn.Linear(emb_dim, self.treatment_layers[i]))
                if is_bn:
                    treatment_layer.add_module(nn.BatchNorm1d(1))
                if i == len(self.treatment_layers)-1:
                    if self.last_activation is not None:
                        treatment_layer.add_module(self.last_activation)
                else:
                    treatment_layer.add_module(self.activation)
                if dropout_rate is not None:
                    treatment_layer.add_module(nn.Dropout(self.dropout_rate))
                emb_dim = self.treatment_layers[i]
            treatment_tower.append(treatment_layer)
        self.treatment_towers = nn.ModuleList(treatment_tower)
    
    def forward(self, inputs):
        share_out = self.share_layer(inputs)
        outputs = []
        for treat_layer in self.treatment_towers:
            t_out = treat_layer(share_out)
            outputs.append(t_out)
        return outputs