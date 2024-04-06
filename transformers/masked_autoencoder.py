import torch
from torch import nn
from attention import PositionalEncoding

class MaskedAutoencoder(nn.Module):
    def __init__(self, embed_dim, len, mask_ratio, head_num, encode_layer_num, decode_layer_num):
        super(MaskedAutoencoder, self).__init__()
        self.embed_dim = embed_dim
        self.len = len
        self.mask_ratio = mask_ratio
        self.head_num = head_num
        self.encode_layer_num = encode_layer_num
        self.decode_layer_num = decode_layer_num
        self.mask_len = int(self.len * self.mask_ratio)
        self.invmask_len = self.len - self.mask_len
        # mask掉对应的emb
        self.mask_token_emb = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        torch.nn.init.normal_(self.mask_token_emb, mean=0, std=0.02)
        # 序列开头[cls]
        self.cls_token_emb = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        torch.nn.init.normal_(self.cls_token_emb, std=0.02)
        self.encoders = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.head_num), num_layers=self.encode_layer_num)
        self.decoders = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.head_num), num_layers=self.decode_layer_num)
        self.pos_encoder = nn.Embedding(self.len, self.embed_dim)
        self.pos_decoder = nn.Embedding(self.len, self.embed_dim)
        self.num2emb = nn.Conv1d(1, self.embed_dim, 1)

        self.pred_mlp = nn.Linear(self.embed_dim, 1)
        self.enc_norm = nn.LayerNorm(self.embed_dim)
        self.dec_norm = nn.LayerNorm(self.embed_dim)
        self.decode_embed = nn.Linear(self.embed_dim, self.embed_dim)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  
            nn.init.constant_(m.weight, 1.0)

    def forward(self, number_feats_tensor):
        number_feats_tensor = number_feats_tensor.unsqueeze(1)
        num_shape = number_feats_tensor.shape
        mask_random = torch.randn(num_shape[0], self.len)
        mask_ind = torch.argsort(mask_random, dim=1)
        num_emb, mask_emb, mask, invmask, restored, en_cls = self.encoder(number_feats_tensor, num_shape, mask_ind)
        dec_emb, dec_cls = self.decode(num_emb, mask_emb, mask, invmask, restored, num_shape)
        return dec_emb, dec_cls, enc_emb, en_cls

    def encoder(self, number_feats_tensor, num_shape, mask_ind):
        num_emb = self.num2emb(number_feats_tensor)
        num_emb = torch.permute(num_emb, (0, 2, 1))
        mask = mask_ind[:, 0: self.mask_len+1].to(self.device)
        num_pos_emb = self.pos_encoder(None).to(self.device).detach()
        num_pos_emb = num_pos_emb.unsqueeze(1).repeat(1, num_shape[0], 1)
        num_pos_emb_masked = torch.gather(num_pos_emb[:,1:,:], 1, mask.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        num_emb_masked = torch.gather(num_emb, 1, mask.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        mask_emb = self.mask_token_emb.repeat(num_shape[0], num_shape[2], 1)
        invmask = mask_ind[:, self.mask_len+1:]
        mask_pos_emb_invmask = torch.gather(num_pos_emb[:,1:,:], 1, invmask.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        mask_emb = torch.gather(mask_emb, 1, invmask.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        num_emb = torch.cat([self.cls_token_emb.repeat(num_shape[0], 1, 1)+num_pos_emb[:,:1,:], num_emb_masked+num_pos_emb_masked], dim=1)
        num_emb = self.encoders(num_emb)
        restored = torch.zeros_like(num_emb)
        return num_emb, mask_emb, mask, invmask, restored, num_emb[:,:1,:]
    
    def decode(self, num_emb, mask, invmask, restored, num_shape):
        unmasked_emb = self.decode_embed(num_emb)
        restored.scatter_(1, mask.unsqueeze(-1).expand(-1, -1, self.embed_dim), unmasked_emb[:,1:,:])
        restored.scatter_(1, invmask.unsqueeze(-1).expand(-1, -1, self.embed_dim), mask_emb)
        pos = self.pos_decoder(None).to(self.device).detach()
        pos = pos.unsqueeze(1).repeat(1, num_shape[0], 1)
        dec_input = torch.cat([unmasked_emb[:,:1,:]+pos[:,:1,:],restored+pos[:,1:,:]],dim=1)
        dec_input = self.decoders(dec_input)
        pred_dec_emb = self.pred_mlp(dec_input[:,1:,:])
        dec_emb = torch.reshape(torch.sigmoid(pred_dec_emb), (num_shape[0], -1))
        return dec_emb, dec_input[:,:1,:]