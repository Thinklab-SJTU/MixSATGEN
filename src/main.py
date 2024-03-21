import numpy as np
import os
import random

import pygmtools as pygm
import torch
import torch.nn.functional as F

from finetune import finetune
from match import sort_and_match_pairs, sinkhorn
from mix import mix_clause, mix_clause_random_match

from data.dataloader import get_dataloader
from data.dataset import GraphMatchDataset
from utils.gmtools import model_feat_to_X

from utils.formula import VCG_to_sat
from utils.args import make_args
from utils.tools import load_finetune

import matplotlib.pyplot as plt

def main_train(args, path):
    ###################################
    # part 1: Dataset & model loading #
    ###################################
    dataset = GraphMatchDataset(args)
    dataloader =  get_dataloader(dataset, args)
    for idx, data in enumerate(dataloader):
        ######################
        # part 2: Finetuning #
        ######################
        finetune(args, idx, data, dataset, path)


def main_test(args, path):
    ###################################
    # part 1: Dataset & model loading #
    ###################################
    dataset = GraphMatchDataset(args)
    dataloader =  get_dataloader(dataset, args)
    
    for idx, data in enumerate(dataloader):
        data1, data2, pair_idx = data
        pair_idx_each = dataset.pair_id_list[pair_idx]
        
        ####################
        # part 2: Matching #
        ####################
        if args.matching == 'rrwm':
            K = dataset.get_K(pair_idx, sparse=False)
            X = pygm.rrwm(K, data1, data2)
            argX = None
            X = sinkhorn(X, args)
            print(X.sum())
            
            K = dataset.get_K(pair_idx).to(args.device)
            loss = torch.sparse.mm(K, X.reshape((-1, 1)))
            loss = torch.mm(X.reshape((1, -1)), loss)
            loss = loss.cpu().detach()
            with open(f'{path}/loss.txt', 'a') as f:
                f.write(f'{dataset.name_list[pair_idx_each[0]]}-{dataset.name_list[pair_idx_each[1]]}, rrwm, {loss}\n')
            continue
        elif args.matching == 'satbench':
            if dataset.switch[pair_idx]:
                data1, data2 = data2, data1
            model = load_finetune(args, pair_idx, dataset)
            model.eval()
            data1 = data1.to(args.device)
            data2 = data2.to(args.device)
            feat1 = model(data1)
            feat2 = model(data2)

            X, argX = model_feat_to_X(feat1, feat2)
        
        X = X.cpu().detach()
        X = sinkhorn(X, args, backend='pytorch')
        X = X.numpy()
        match_pair, match_pair_2, match_belief, match_arg = sort_and_match_pairs(X, args.matching_threshold, argX=argX, switch=dataset.switch[pair_idx])
        
        # softmax entropy
        if args.entropy:
            X1 = np.array([a / np.sum(a) for a in X])
            X1 = np.array([np.exp(a) / np.sum(np.exp(a)) for a in X1])
            row = np.mean(np.array([-np.sum(p * np.log(p + 1e-8)) for p in X1]))
            X2 = X.transpose()
            X2 = np.array([a / np.sum(a) for a in X2])
            X2 = np.array([np.exp(a) / np.sum(np.exp(a)) for a in X2])
            col = np.mean(np.array([-np.sum(p * np.log(p + 1e-8)) for p in X2]))
            info_entropy = row + col
            with open(f'{path}/entropy.txt', 'a') as f:
                f.write(f'{dataset.name_list[pair_idx_each[0]]}-{dataset.name_list[pair_idx_each[1]]}, {row}, {col}, {info_entropy}\n')
            continue
        
        data1, data2, pair_idx = data
        pair_idx_each = dataset.pair_id_list[pair_idx]
        VCG1 = dataset.VCG_graph_list[pair_idx_each[0]]
        VCG2 = dataset.VCG_graph_list[pair_idx_each[1]]

        ##################
        # part 3: Mixing #
        ##################
        for repeat_idx in range(args.repeat):
            VCG3, outer_var_ptr = mix_clause(VCG1, VCG2, match_pair.copy(), match_pair_2.copy(), match_belief.copy(), match_arg.copy(), args)
            print(f'match_belief from {np.max(match_belief)} to {np.min(match_belief)}')
            
            # save
            VCG_to_sat(VCG3, f'{path}/{dataset.name_list[pair_idx_each[0]]}_{dataset.name_list[pair_idx_each[1]]}_repeat{repeat_idx}.cnf', match_pair.shape[0], outer_var_ptr=outer_var_ptr)


def main_random(args, path):
    ###################################
    # part 1: Dataset & model loading #
    ###################################
    dataset = GraphMatchDataset(args)
    dataloader =  get_dataloader(dataset, args)
    
    for idx, data in enumerate(dataloader):
        num_var1, num_var2, pair_idx = data
        pair_idx_each = dataset.pair_id_list[pair_idx]
        VCG1 = dataset.VCG_graph_list[pair_idx_each[0]]
        VCG2 = dataset.VCG_graph_list[pair_idx_each[1]]

        ##################
        # part 2: Mixing #
        ##################
        for repeat_idx in range(args.repeat):
            VCG3, outer_var_ptr = mix_clause_random_match(VCG1, VCG2, num_var1, num_var2, repeat_idx, args)

            VCG_to_sat(VCG3, f'{path}/{dataset.name_list[pair_idx_each[0]]}_{dataset.name_list[pair_idx_each[1]]}_repeat{repeat_idx}.cnf', num_var1, outer_var_ptr)



if __name__ == '__main__':
    args = make_args()
    pygm.BACKEND = "pytorch"
    args.dtype_torch = eval("torch." + args.dtype)
    args.dtype_np = eval("np." + args.dtype)
    args.device = args.device if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print(args)
    print('------------------------------------------------------------------')
    
    path = f'./data/{args.dataset}'
    if not os.path.isdir(path):
        os.makedirs(path)
    path = f'./generated/{args.dataset}'
    if not os.path.isdir(path):
        os.makedirs(path)

    if args.mode == 'random':
        path = f'./generated/random_{args.dataset}/random_{args.mixing_threshold}'
        if not os.path.isdir(f'./generated/{args.dataset}'): os.makedirs(f'./generated/random_{args.dataset}')
        if not os.path.isdir(path): os.makedirs(path)
        
        main_random(args, path)
    else:
        if args.init and not args.enable_gumbel:
            path = f'./generated/{args.dataset}/{args.matching}_{args.satbench_model}_{args.mixing_threshold}_init_iter{args.n_iterations}'
        elif not args.init and not args.enable_gumbel:
                path = f'./generated/{args.dataset}/{args.matching}_{args.satbench_model}_{args.mixing_threshold}_iter{args.n_iterations}'
        elif not args.init and args.enable_gumbel:
                path = f'./generated/{args.dataset}/{args.matching}_{args.satbench_model}_{args.mixing_threshold}_iter{args.n_iterations}_gumbel_factor{args.gumbel_factor}_temp{args.gumbel_temp}'
        else:
            path = f'./generated/{args.dataset}/{args.matching}_{args.satbench_model}_{args.mixing_threshold}_init{args.init}_gumbel{args.enable_gumbel}_factor{args.gumbel_factor}_temp{args.gumbel_temp}'
        if not os.path.isdir(path): os.makedirs(path)
        
        if args.mode == 'train':
            main_train(args, path)
        elif args.mode == 'test':
            main_test(args, path)