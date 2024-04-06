import torch 
from torch import nn 
from torch.nn import functional as F 
from torch.nn import TransformerEncoder, TransformerEncoderLayer 
from attention import PositionalEncoding

class TJEPA(nn.Module):
    def __init__(self, config):
        super(TJEPA, self).__init__()
        self.d_model = config.d_model
        self.seq_len = config.seq_len
        self.src_encoder = config.src_encoder
        self.tag_encoder = config.tag_encoder

        self.pred_layer = config.pred_layer
        self.seq_len = config.seq_len
        self.mask_len = config.mask_len

        self.pos_encoder = PositionalEncoding(self.seq_len, self.d_model, add_pos=False)

        self.num_emb_layer = nn.Conv1d(1, self.d_model, 1)
        self.cate_emb_tables = nn.ModuleList([
            nn.Embedding(config.cate_feature_num[i], self.d_model)
            for i in range(len(config.cate_feature_num))
        ])

        self.mask_token_emb = nn.Parameter(torch.randn(1, 1, self.d_model))
        torch.nn.init.normal_(self.mask_token_emb, std=0.02)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, inputs):
        num_emb = self.num_emb_layer(inputs[:, 9:].unsqueeze(1))
        num_emb = torch.permute(num_emb, (0, 2, 1))
        cate_emb = [self.cate_emb_tables[i](inputs[:, i].long()).unsqueeze(1) for i in range(0, 9)]
        emb = torch.cat([torch.cat(cate_emb, dim=1), num_emb], dim=1)
        pos_emb = self.pos_encoder(None).to(inputs.device)
        emb_addpos_masked, mask_emb_addpos_masked, emb_addpos_invmasked, emb_unmask, mask = apply_mask(emb.shape, self.mask_len, self.seq_len, pos_emb, emb, self.mask_token_emb)
        se = self.src_encoder(emb_addpos_invmasked)
        te = self.tag_encoder(emb_unmask)
        emb_masked = torch.gather(te, 1, mask.unsqueeze(-1).expand(-1, -1, self.d_model))
        pred_result  = []
        loss = 0
        
        for i in range(self.mask_len):
            tmp = self.pred_layer(torch.cat([se, emb_addpos_masked[:, i:i+1, :]], dim=1))
            pred_result.append(tmp)
            loss += self.create_loss(tmp[:, -1, :], te[:, i, :])
        return pred_result, loss / self.mask_len
    
    def create_loss(self, pred_result, target):
        loss = F.smooth_l1_loss(pred_result, target)
        return torch.mean(loss)
    

class TEncoder(nn.Module):
    def __init__(self, config):
        super(TEncoder, self).__init__()
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.num_layers = config.num_layers

        self.encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, self.num_layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs):
        output = self.transformer_encoder(inputs)
        return output


class Predictors(nn.Module):
    def __init__(self, config):
        super(Predictors, self).__init__()
        self.input_dim = config.input_dim
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.num_layers = config.num_layers
        self.output_dim = config.output_dim

        self.proj = nn.Linear(self.input_dim, self.d_model)
        self.encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, self.num_layers)
        self.pred_layer = nn.Linear(self.input_dim, self.output_dim)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  
            nn.init.constant_(m.weight, 1.0)
    

    def forward(self, inputs):
        inputs_proj = self.proj(inputs)
        output = self.transformer_encoder(inputs_proj)
        output = self.pred_layer(output)
        return output

# jepa没有cls
def apply_mask(num_shape, mask_len, seq_len, pos_emb, emb, mask_token_emb):
    mask_random = torch.randn(num_shape[0], seq_len).to(emb.device)
    mask_ind = torch.argsort(mask_random, dim=1)

    mask = mask_ind[:, 0: mask_len]

    pos_emb = pos_emb.unsqueeze(0).repeat(num_shape[0], 1, 1)
    pos_emb_masked = torch.gather(pos_emb, 1, mask.unsqueeze(-1).expand(-1, -1, num_shape[-1]))
    emb_masked = torch.gather(emb, 1, mask.unsqueeze(-1).expand(-1, -1, num_shape[-1]))
    
    invmask = mask_ind[:, mask_len:]
    pos_emb_invmasked = torch.gather(pos_emb, 1, invmask.unsqueeze(-1).expand(-1, -1, num_shape[-1]))
    emb_invmasked = torch.gather(emb, 1, invmask.unsqueeze(-1).expand(-1, -1, num_shape[-1]))
    mask_emb = mask_token_emb.repeat(num_shape[0], num_shape[1], 1)
    mask_emb = torch.gather(mask_emb, 1, mask.unsqueeze(-1).expand(-1, -1, num_shape[-1]))

    emb_addpos_masked = pos_emb_masked + emb_masked
    mask_emb_addpos_masked = pos_emb_masked + mask_emb
    emb_addpos_invmasked = pos_emb_invmasked + emb_invmasked

    emb_unmask = emb + pos_emb
    return emb_addpos_masked, mask_emb_addpos_masked, emb_addpos_invmasked, emb_unmask, mask


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.is_bn = config.is_bn
        self.dropout_rate = config.dropout_rate
        self.num_layers = config.num_layers

        self.mlp = nn.Sequential()
        emb_dim = self.input_dim
        for i in range(self.num_layers):
            self.mlp.add_module(nn.Linear(emb_dim, self.num_layers[i]))
            self.mlp.add_module(nn.ReLU())
            if self.is_bn:
                self.mlp.add_module(nn.BatchNorm1d(self.input_dim))
            if self.dropout_rate is not None and self.dropout_rate > 0:
                self.mlp.add_module(nn.Dropout(self.dropout_rate))
            emb_dim = self.num_layers[i]
        
        self.pred_layer = nn.Linear(self.num_layers[-1], self.output_dim)
