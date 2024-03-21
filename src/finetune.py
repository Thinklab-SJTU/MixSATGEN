import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import pygmtools as pygm

from match import sinkhorn
from utils.gmtools import model_feat_to_X
from utils.tools import load_pretrain

def finetune(args, idx, data, dataset, path):
    # finetune the nsnet model by each specific question pair, using xKx for supervise
    data1, data2, pair_idx = data
    pair_idx_each = dataset.pair_id_list[pair_idx]
    data1 = data1.to(args.device)
    data2 = data2.to(args.device)
    model = load_pretrain(args)
    model.train()
    optimer = eval("torch.optim." + args.optim)
    optim = optimer(model.parameters(), lr=args.lr)
    best_loss = 0
    print(f'===== {args.matching} {dataset.name_list[pair_idx_each[0]]} {dataset.name_list[pair_idx_each[1]]} =====')
    assert args.matching in ['satbench']
    t1 = time.time()
    X = None
    X_unsinkhorn = None
    for epoch in range(args.epoch):
        feat1 = model(data1)
        feat2 = model(data2)
        X_unsinkhorn, _ = model_feat_to_X(feat1, feat2)
        
        X = sinkhorn(X_unsinkhorn, args, backend='pytorch')

        # self-supervised loss: x K x
        K = dataset.get_K(pair_idx).to(args.device)
        loss = torch.sparse.mm(K, X.reshape((-1, 1)))
        loss = torch.mm(X.reshape((1, -1)), loss)
        loss = - torch.sum(loss)
        
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        loss = loss.cpu().detach()
        if epoch % 5 == 0:
            print(f'------------- epoch {epoch} loss: {loss}')
            with open(f'{path}/loss.txt', 'a') as f:
                f.write(f'{dataset.name_list[pair_idx_each[0]]}-{dataset.name_list[pair_idx_each[1]]},finetune epoch {epoch}, {loss}\n')
        if epoch > args.epoch / 5 * 4 and loss < best_loss:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch, 
                'optimizer': optim.state_dict()}, f'data/{args.dataset}/satbench_{args.satbench_model}_{dataset.name_list[pair_idx_each[0]]}_{dataset.name_list[pair_idx_each[1]]}_tau{args.sinkhorn_tau}iter{args.sinkhorn_iter}.pt')
    print(f'finetune time: {time.time() - t1}')


