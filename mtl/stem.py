from component import Expert, Gate, ExpertGroup
from torch import nn
import torch


class STEM(nn.Module):
    def __init__(self, expert_config, gate_config, task_config, is_last):
        super(CGC, self).__init__()
        self.expert_config = expert_config
        self.gate_config = gate_config
        self.task_config = task_config
        self.is_last = is_last

        self.expert_group = nn.ModuleList(
            [
                ExpertGroup(
                    self.expert_config.input_dim,
                    self.expert_config.layers,
                    self.expert_config.expert_num,
                    self.expert_config.is_bn,
                    self.expert_config.activation,
                    self.expert_config.dropout_rate,
                )
                for _ in range(self.expert_config.task_num+1)
            ]
        )
        
        self.task_gates = nn.ModuleList(
            [
                Gate(
                    self.gate_config.input_dim,
                    self.gate_config.layers,
                    self.gate_config.is_bn,
                    self.gate_config.activation,
                    self.gate_config.dropout_rate,
                ) for _ in range(self.task_config.task_num)
            ]
        )       
        share_gate_layers =  self.gate_config.layers
        share_gate_layers[-1] = self.self.expert_config.expert_num * self.task_config.task_num
        self.share_gate = Gate(
                        self.gate_config.input_dim,
                        share_gate_layers,
                        self.gate_config.is_bn,
                        self.gate_config.activation,
                        self.gate_config.dropout_rate,
                    ) 
        if self.is_last:
            self.towers = nn.ModuleList(
                [
                    Expert(
                        self.expert_config.layers[-1],
                        self.task_config.layers,
                        self.task_config.is_bn,
                        self.task_config.activation,
                        self.task_config.dropout_rate,
                    ) for _ in range(self.task_config.task_num)
                ]
            )
        else:
            self.towers = Expert(
                        self.expert_config.layers[-1]*(self.task_config.task_num+1),
                        self.task_config.layers,
                        self.task_config.is_bn,
                        self.task_config.activation,
                        self.task_config.dropout_rate,
                    )

    def forward(self, inputs):
        '''
        inputs: [batch_size, task_num+1, input_dim], idx=0 is the shared input
        '''
        expert_group_out = [exp(inputs[:, idx, :]).unsqueeze(dim=1) for idx, exp in enumerate(self.expert_group)]
        gate_weight = [gate(inputs) for gate in self.task_gates]
        gate_weight = []
        for i, gate in enumerate(self.task_gates):
            g = gate(inputs[:, i+1, :] + inputs[:, 0, :])
            gate_weight.append(g)
        emb_cat = torc.cat(expert_group_out, dim=1)
        agg_emb = []
        add_ind = F.one_hot(torch.tensor([0] * emb_cat.shape[0]), self.expert_config.task_num+1).to(emb_cat.device)
        add_ind = add_ind.reshape(emb_cat.shape[0], self.expert_config.task_num+1, 1).repeat(1, 1, self.expert_config.expert_num)
        
        output_arr = []
        for i in range(1, self.expert_config.task_num+1):
            ind = torch.tensor([i] * emb_cat.shape[0]).to(emb_cat.device)
            ont_hot_ind = F.one_hot(ind, self.task_config.task_num+1).reshape(emb_cat.shape[0], self.task_config.task_num+1, 1, 1).repeat(1,1,self.expert_config.expert_num, self.expert_config.layers[-1]) + add_ind
            agg_t = torch.where(one_hot_ind == 1, emb_cat, emb_cat.detach())
            agg_t = agg_t.reshape(agg_t.shape[0], -1, agg_t.shape[-1])
            gated_agg = torch.bmm(gate_weight[i-1].unsqueeze(dim=1), agg_t)
            agg_emb.append(gated_agg.reshape(-1, self.expert_config.layers[-1]))
            agg_emb = torch.matmul(gate_weight[i-1].unsqueeze(dim=1), torch.cat([expert_group_out[0], expert_group_out[i]], dim=1))
        res_arr = []
        for i, tower in enumerate(self.towers):
            t_emb = tower(agg_emb[i])
            res_arr.append(t_emb)
        return torch.cat(res_arr, dim=1)
        