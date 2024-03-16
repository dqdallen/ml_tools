from component import Expert, Gate, ExpertGroup
from torch import nn
import torch


class CGC(nn.Module):
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
        expert_group_out = [experts(inputs) for experts in self.expert_group]
        gate_weight = [gate(inputs) for gate in self.task_gates]
        
        if not self.is_last:
            
            output_arr = []
            for i in range(1, len(expert_group_out)):
                agg_emb = torch.matmul(gate_weight[i-1].unsqueeze(dim=1), torch.cat([expert_group_out[0], expert_group_out[i]], dim=1))
                output_arr.append(agg_emb.squeeze(dim=1))
            cgc_out = torch.cat(output_arr, dim=1)
            return self.towers(cgc_out)
        else:
            share_gate_weight = self.share_gate(inputs)
            share_exp_emb = torch.cat(expert_group_out, dim=1)
            output_arr = [torch.matmul(share_gate_weight.unsqueeze(dim=1), share_exp_emb)]
            for i in range(1, len(expert_group_out)):
                agg_emb = torch.matmul(gate_weight[i-1].unsqueeze(dim=1), torch.cat([expert_group_out[0], expert_group_out[i]], dim=1))
                output_arr.append(self.towers[i-1](agg_emb.squeeze(dim=1)))
            return output_arr
