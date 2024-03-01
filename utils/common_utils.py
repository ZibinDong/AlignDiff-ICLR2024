import numpy as np
import torch
import random


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_grid_indices(x, x_min, x_max, num_bins):
    step_size = (x_max - x_min) / num_bins
    if isinstance(x, np.ndarray):
        indices = np.floor((x-x_min)/step_size)
        indices = np.clip(indices, 0, num_bins-1).astype(np.int64)
    elif isinstance(x, torch.Tensor):
        indices = torch.floor((x-x_min)/step_size)
        indices = torch.clamp(indices, 0, num_bins-1).long()
    return indices

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)