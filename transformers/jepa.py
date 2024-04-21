import torch 
from torch import nn 
from torch.nn import functional as F 
from attention import PositionalEncoding, Block
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


class TJEPA(nn.Module):
    def __init__(self, config):
        super(TJEPA, self).__init__()
        self.src_encoder = config.src_encoder
        self.tag_encoder = config.tag_encoder
        self.pred_layer = config.pred_layer
        self.seq_len = config.seq_len
        self.mask_len = config.mask_len
        self.device = config.device

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
        X = inputs
        num_shape = X.shape
        mask_random = torch.randn(num_shape[0], self.seq_len).to(self.device)
        mask_ind = torch.argsort(mask_random, dim=1)
        mask = mask_ind[:, 0: self.mask_len]
        invmask = mask_ind[:, self.mask_len:]

        src_emb = self.src_encoder(X, invmask)
        tag_emb = self.tag_encoder(X, None)
        pred_emb = self.pred_layer(src_emb, mask, invmask)
        te_mask = apply_mask(mask, tag_emb)

        loss = self.create_loss(pred_emb, te_mask)
        return tag_emb, loss
    
    def create_loss(self, pred_result, target):

        loss = F.smooth_l1_loss(pred_result, target)
        return torch.mean(loss)
    

class TEncoder(nn.Module):
    def __init__(self, feature_map, config):
        super(TEncoder, self).__init__()
        self.embedding_layer = FeatureEmbedding(feature_map, config.d_model)
        self.feature_map = feature_map
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.num_layers = config.num_layers
        self.seq_len = config.seq_len
        self.device = config.device

        self.pos_encoder = PositionalEncoding(self.seq_len, self.d_model, add_pos=False)

        self.transformer_encoder = nn.ModuleList(
            [
                Block(self.d_model, self.nhead)
                for _ in range(self.num_layers)
            ]
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  
            nn.init.constant_(m.weight, 1.0)

    def get_inputs(self, inputs, feature_source=None):
        if feature_source and type(feature_source) == str:
            feature_source = [feature_source]
        X_dict = dict()
        for feature, spec in self.feature_map.features.items():
            if (feature_source is not None) and (spec['source'] not in feature_source):
                continue
            if spec['type'] == 'meta':
                continue
            X_dict[feature] = inputs[:, self.feature_map.get_column_index(feature)].to(self.device)
        return X_dict

    def forward(self, inputs):
        inputs = self.get_inputs(inputs)
        emb = self.embedding_layer(inputs)
        pos_emb = self.pos_encoder(None).to(self.device).unsqueeze(0).repeat(emb.shape[0], 1, 1)
        emb_add_pos = emb + pos_emb
        if masks is not None:
            emb_add_pos = apply_masks(masks, emb_add_pos)
        output = emb_add_pos
        for layer in self.transformer_encoder:
            output = layer(output)
        return output


class Predictors(nn.Module):
    def __init__(self, config):
        super(Predictors, self).__init__()
        self.input_dim = config.input_dim
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.num_layers = config.num_layers
        self.output_dim = config.output_dim
        self.seq_len = config.seq_len
        self.mask_len = config.mask_len
        self.device = config.device

        self.proj = nn.Linear(self.input_dim, self.d_model)
        self.transformer_encoder = nn.ModuleList(
            [
                Block(self.d_model, self.nhead)
                for _ in range(self.num_layers)
            ]
        )
        self.pred_layer = nn.Linear(self.input_dim, self.output_dim)

        self.pos_encoder = PositionalEncoding(self.seq_len, self.d_model, add_pos=False)
        self.mask_token_emb = nn.Parameter(torch.randn(1, 1, self.d_model))
        torch.nn.init.normal_(self.mask_token_emb, mean=0, std=0.02)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  
            nn.init.constant_(m.weight, 1.0)
    

    def forward(self, inputs, masks, inv_masks):
        invmask_emb = self.proj(inputs)
        pos_emb = self.pos_encoder(None).to(inputs.device)
        num_shape = inputs.shape
        mask_emb = self.mask_token_emb.repeat(num_shape[0], self.seq_len, 1)
        masked_emb = apply_masks(masks, mask_emb)
        masked_pos_emb = apply_masks(masks, pos_emb.unsqueeze(0).repeat(num_shape[0], 1, 1))
        mask_emb_pos = masked_pos_emb + masked_emb

        invmask_pos = apply_masks(inv_masks, pos_emb.unsqueeze(0).repeat(num_shape[0], 1, 1))
        invmask_emb_pos = invmask_pos + invmask_emb
        invmask_emb_pos = invmask_emb_pos.unsqueeze(1).repeat(1, self.mask_len, 1, 1)
        mask_emb_pos = mask_emb_pos.unsqueez(1).repeat(1, self.mask_len, 1, 1)
        emb4pred = torch.cat([invmask_emb_pos, mask_emb_pos], dim=2)
        output = emb4pred.reshape(emb4pred.shape[0]*emb4pred.shape[1], emb4pred.shape[2], emb4pred.shape[3])
        for layer in self.transformer_encoder:
            output = layer(output)
        output = self.pred_layer(output[:, -1, :])
        return output

def apply_masks(mask, x):
    x_masked = torch.gather(x, 1, mask.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    return x_masked


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
