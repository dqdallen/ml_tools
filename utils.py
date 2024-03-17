import torch
import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 使用确定性卷积算法
    torch.backends.cudnn.benchmark = False  # 关闭cudnn自适应算法，确保可复现性


