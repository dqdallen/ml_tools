from component import Expert
from torch import nn

class ShareBottom(nn.Module):
    def __init__(self, input_dim, share_layers, task_layers, is_bn, activation, dropout_rate=None):
        """share bottom

        Args:
            input_dim (int): inputs emb dim
            share_layers (list): a list of hidden share layer size, [a, b, c]
            task_layers (list): a list of hidden task layer size, [a, b, c]
            is_bn (bool): _description_
            activation (nn.Module): _description_
            dropout_rate (float, optional): _description_. Defaults to None.
        """
        super(ShareBottom, self).__init__()
        self.input_dim = input_dim
        self.share_layers = share_layers
        self.task_layers = task_layers
        self.is_bn = is_bn
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        self.share = self.Expert(self.input_dim, self.share_layers, self.is_bn, self.activation, self.dropout_rate)
        self.task = nn.ModuleList([
            self.Expert(self.input_dim, self.task_layers[i], self.is_bn, self.activation, self.dropout_rate)
            for i in range(len(self.task_layers))
        ])
    
    def forward(self, inputs):
        share_emb = self.share(inputs)
        task_outputs = [t_layer(share_emb) for t_layer in enumerate(self.task)]
        return task_outputs
