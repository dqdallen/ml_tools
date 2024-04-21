from transformers import BertModel, BertConfig, BertForMaskedLM
import torch
from torch import nn

class CustomBertModel(nn.Module):
    def __init__(self, config):
        super(CustomBertModel, self).__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.fea_len = config.fea_len
        self.mask_ratio = config.mask_ratio
        self.mask_len = int(self.mask_ratio * self.fea_len)
        self.num_attention_heads = config.num_attention_heads

        self.device =  config.device
        self.cls_token = torch.tensor([[2]]).to(self.device)
        self.bert_config = BertConfig(vocab_size=self.vocab_size, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers, num_attention_heads=self.num_attention_heads)
        
        self.bert_model = BertForMaskedLM(self.bert_config)
    
    def forward(self, input_seq):
        input_seq = input_seq + 3
        num_shape = input_seq.shape
        mask_random = torch.randn(num_shape[0], self.fea_len).to(self.device)
        mask_ind = torch.argsort(mask_random, dim=1).int()
        return None


a = torch.randn(2,3)
b = torch.randn(2,3)
c = torch.stack((a,b), dim=1)
print(c.shape)