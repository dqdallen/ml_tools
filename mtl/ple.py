from component import Expert, Gate, ExpertGroup
from torch import nn
import torch


class CGC(nn.Module):
    def __init__(self, expert_config, gate_config, mid_config, task_config, num_level):
        super(CGC, self).__init__()
        self.expert_config = expert_config
        self.gate_config = gate_config
        self.task_config = task_config
        self.mid_config = mid_config
        self.num_level = num_level

        self.cgcs = nn.ModuleList([
            CGC(self.expert_config, self.gate_config, self.mid_config, False)
            for _ in range(num_level - 1)
        ])
        self.final_cgc = CGC(self.expert_config, self.gate_config, self.task_config, True)

    def forward(self, inputs):
        input_emb = inputs
        for cgc in self.cgcs:
            cgc_output = cgc(input_emb)
            input_emb = cgc_output
        output = self.final_cgc(input_emb)
        return output