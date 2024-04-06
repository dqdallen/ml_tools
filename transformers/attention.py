import torch
from torch import nn
import numpy as np


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_units, head_num, mask=None, dropout_rate=0.5):
        '''
        input_dim: int, input feature dimension
        hidden_units: int, hidden units of attention layer
        head_num: int, number of attention heads
        mask: tensor, mask tensor for padding
        dropout_rate: float, dropout rate
        '''
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.head_num = head_num
        self.d_k = hidden_units // head_num
        self.mask = mask
        self.dropout_rate = dropout_rate

        assert self.d_k * self.head_num == self.hidden_units

        self.qkv_linears = nn.ModuleList(
            [nn.Linear(self.input_dim, self.hidden_units) for _ in range(3)]
        )
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inputs):
        q, k, v = [
            linear(inputs).reshape(
                inputs.shape[0], inputs.shape[1], self.head_num, self.d_k
            )
            for linear in self.qkv_linears
        ]
        q = q.permute((0, 2, 1, 3))
        k = k.permute((0, 2, 3, 1))
        v = v.permute((0, 2, 1, 3))
        qk = torch.matmul(q, k) / torch.sqrt(self.d_k * 1.0)
        if self.mask:
            qk += torch.mul(self.mask, -1e-9)
        attn_score = torch.softmax(qk, dim=-1)
        attn_score = self.dropout(attn_score)  # b,head,len,d
        agg_emb = torch.matmul(attn_score, v)
        return agg_emb.reshape((-1, inputs.shape[1], self.head_num * self.d_k))


def positional_encoding(
    input_len, hidden_units, min_timescale=1.0, max_timescale=1.0e4
):
    '''
    input_len: int, length of input sequence
    hidden_units: int, hidden units of attention layer
    min_timescale: float, minimum timescale
    max_timescale: float, maximum timescale
    '''
    assert hidden_units % 2 == 0
    position = torch.arange(0, input_len, dtype=torch.float32)
    num_timescales = hidden_units // 2
    log_timescale_increment = torch.log(torch.tensor(max_timescale / min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    return signal


def learned_positional_encoding(input_len, hidden_units):
    ''' 
    input_len: int, length of input sequence
    hidden_units: int, hidden units of attention layer
    '''
    position_index = torch.arange(input_len).unsqueeze(1)
    position_tensor = torch.LongTensor(position_index)
    emb_table = nn.Embedding(input_len, hidden_units)
    position_emb = emb_table(position_tensor)
    return position_emb.squeeze(1)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        input_len,
        hidden_units,
        min_timescale=1.0,
        max_timescale=1.0e4,
        pos_type="fixed",
        add_pos=True
    ):
        ''' 
        input_len: int, length of input sequence
        in_shape: tuple, input shape
        min_timescale: float, minimum timescale
        max_timescale: float, maximum timescale
        pos_type: str, position encoding type, fixed or learned
        seed: int, random seed
        add_pos: bool, whether to add positional encoding to input
        hidden_units: int, hidden units of attention layer
        '''
        super(PositionalEncoding, self).__init__()
        self.input_len = input_len
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.pos_type = pos_type
        self.add_pos = add_pos
        self.hidden_units = hidden_units

        if self.pos_type == "fixed":
            self.pos_encoding = positional_encoding(
                self.input_len,
                self.hidden_units,
                self.min_timescale,
                self.max_timescale,
            )
        elif self.pos_type == "learned":
            self.pos_encoding = learned_positional_encoding(
                self.input_len, self.hidden_units
            )
        else:
            raise ValueError("Positional encoding type not supported")

    def forward(self, inputs):
        if self.add_pos:
            b = inputs.shape[0]
            signal = self.pos_encoding.unsqueeze(0).repeat(b, 1, 1)
            output = inputs + signal
            return output
        else:
            return self.pos_encoding
