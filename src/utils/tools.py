import numpy as np
import matplotlib.pyplot as plt
import torch
from satbench.gnn import GNN


def load_pretrain(args, path=None):
    if args.matching == 'satbench':
        model = GNN(args).to(args.device)
        path = path if path else args.pretrain
        print(f'Loading model {args.matching} pretrain from {path}..')
        if args.device == 'cpu':
            pretrain = torch.load(path, map_location='cpu')
        else:
            pretrain = torch.load(path)
        model.load_state_dict(pretrain['state_dict'], strict=False)
        model.to(args.device)
        return model
    return None
    

def load_finetune(args, pair_idx, dataset):
    if args.init:
        return load_pretrain(args)
    pair_idx_each = dataset.pair_id_list[pair_idx]
    if dataset.switch[pair_idx]:
        pair_idx_each = (pair_idx_each[1], pair_idx_each[0])
    model = load_pretrain(args, path=f'data/{args.dataset}/satbench_{args.satbench_model}_{dataset.name_list[pair_idx_each[0]]}_{dataset.name_list[pair_idx_each[1]]}_tau{args.sinkhorn_tau}iter{args.sinkhorn_iter}.pt')
    return model.to(args.device)

