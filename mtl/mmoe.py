from component import Expert, Gate, ExpertGroup
from torch import nn
import torch


class MMoE(nn.Module):
    def __init__(self, expert_config, gate_config, task_config):
        super(ShareBottom, self).__init__()
        self.expert_config = expert_config
        self.gate_config = gate_config
        self.task_config = task_config

        self.experts = ExpertGroup(
            self.expert_config.input_dim,
            self.expert_config.layers,
            self.expert_config.expert_num,
            self.expert_config.is_bn,
            self.expert_config.activation,
            self.expert_config.dropout_rate,
        )
        self.gates = nn.ModuleList(
            [
                Gate(
                    self.gate_config.input_dim,
                    self.gate_config.layers,
                    self.gate_config.is_bn,
                    self.gate_config.activation,
                    self.gate_config.dropout_rate,
                )
                for _ in range(self.task_config.task_num)
            ]
        )
        self.tasks = nn.ModuleList(
            [
                Expert(
                    self.expert_config.layers[-1],
                    self.task_config.layers,
                    self.task_config.is_bn,
                    self.task_config.activation,
                    self.task_config.dropout_rate,
                )
                for _ in range(self.task_config.task_num)
            ]
        )

    def forward(self, inputs):
        expert_out = self.experts(inputs)
        gate_weights = [gate(inputs) for gate in self.gates]
        output_arr = []
        for i in range(len(gate_weights)):
            agg_emb = torch.matmul(gate_weights[i].unsqueeze(dim=1), expert_out)
            output_arr.append(self.tasks[i](agg_emb.squeeze(dim=1)))
        return output_arr
