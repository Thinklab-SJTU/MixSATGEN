from tkinter import FALSE
import torch
import numpy as np
import random
from torch_geometric.loader import DataLoader


def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _collate_fn(data_list, args):
    print(f'pass the collate fn')
    if len(data_list) == 1:
        return data_list
    zipped = list(zip(*data_list))
    if args.matching == 'rrwm':
        return data_list
    elif args.matching == 'satbench':
        data1, data2, idx = zipped
        return data1, data2, idx


def get_dataloader(dataset, args):
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False, #(args.mode=='train'),
        worker_init_fn=_worker_init_fn,
        collate_fn=lambda x: _collate_fn(x, args),
        pin_memory=False,
    )