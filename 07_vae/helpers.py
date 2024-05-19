import torch
import numpy as np
import os


def select_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')

    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_used = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
    ii = np.arange(len(memory_used))
    mask = memory_used < 4000
    if mask.any():
        mask_index = np.argmin(memory_used[mask])
        index = (ii[mask])[mask_index]
        my_gpu = torch.device(index)
    else:
        my_gpu = torch.device('cpu')
    return my_gpu


def flatten_batch(x: torch.Tensor):
    return x.flatten(start_dim=1)
