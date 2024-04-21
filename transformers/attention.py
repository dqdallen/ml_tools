import torch
from torch import nn
import numpy as np



class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return self.norm2(x)


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
