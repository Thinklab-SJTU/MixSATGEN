import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pygmtools as pygm
from pygmtools.linear_solvers import hungarian


def sort_X(X, argX=None):
    row_order = np.argsort(X, axis=1)
    row_order = np.flip(row_order, axis=1)
    ptr = np.zeros(X.shape[0], dtype=np.int32)
    length = X.shape[0] * X.shape[1]
    result = np.zeros((length, 2), dtype=np.int32)
    argresult = np.zeros((length, 1), dtype=np.bool)
    resptr = 0
    head_order = row_order[range(X.shape[0]), ptr]
    head_item = X[range(X.shape[0]), head_order]
    if argX is not None:
        head_arg = argX[range(X.shape[0]), head_order]
    while resptr < length:
        maxrow = np.argmax(head_item)
        result[resptr] = [maxrow, head_order[maxrow]]
        if argX is not None:
            argresult[resptr] = bool(head_arg[maxrow])
        
        if ptr[maxrow] == X.shape[1] - 1:
            head_item[maxrow] = -1
        else:
            ptr[maxrow] += 1
            head_order[maxrow] = row_order[maxrow, ptr[maxrow]]
            head_item[maxrow] = X[maxrow, head_order[maxrow]]
            if argX is not None:
                head_arg[maxrow] = argX[maxrow, head_order[maxrow]]
        resptr += 1
    if argX is not None:
        return result, argresult
    return result


def sort_and_match_pairs(X, match_threshold, n1=None, n2=None, argX=None, switch=False):
    GREEDY = False
    if GREEDY:
        res =  sort_and_match_pairs_greedy(X, match_threshold, n1, n2, argX)
        return res
    # hungarian
    match_pair, match_pair_2, match_belief, match_arg = sort_and_match_pairs_hungarian(X, match_threshold, n1, n2, argX)
    if switch:
        match_pair_2, match_pair, match_belief_2, match_arg_2 = match_pair, match_pair_2, match_belief, match_arg
        n = match_pair.shape[0]
        match_belief = np.zeros(n)
        match_arg = np.zeros(n) if argX is not None else None
        for idx in range(n):
            if match_pair[idx] != -1:
                idx2 = match_pair[idx]
                match_belief[idx] = match_belief_2[idx2]
                if argX is not None:
                    match_arg[idx] = match_arg_2[idx2]
    return match_pair, match_pair_2, match_belief, match_arg


def sort_and_match_pairs_greedy(X, match_threshold, n1=None, n2=None, argX=None):
    n1 = X.shape[0] if n1 is None else n1
    n2 = X.shape[1] if n2 is None else n2
    match_pair = -np.ones(n1, dtype=np.int32)
    match_belief = np.zeros(n1)
    match_pair_2 = -np.ones(n2, dtype=np.int32)
    
    if argX is not None:
        sorted_pair, sorted_arg = sort_X(X, argX)
        match_arg = np.zeros(n1)
    else:
        sorted_pair = sort_X(X)
        match_arg = None
    
    cur_idx = 0
    matched_num = 0
    while True:
        cur_pair = sorted_pair[cur_idx]
        cur_belief = X[cur_pair[0], cur_pair[1]]
        if cur_belief < match_threshold:
            break
        if match_pair[cur_pair[0]] == -1 and match_pair_2[cur_pair[1]] == -1:
            match_pair[cur_pair[0]] = cur_pair[1]
            match_pair_2[cur_pair[1]] = cur_pair[0]
            match_belief[cur_pair[0]] = cur_belief
            if argX is not None:
                match_arg[cur_pair[0]] = sorted_arg[cur_idx]
            matched_num += 1
        cur_idx += 1
        if cur_idx >= len(sorted_pair) or matched_num >= min(n1, n2):
            break
    return match_pair, match_pair_2, match_belief, match_arg

def sort_and_match_pairs_hungarian(X, match_threshold, n1=None, n2=None, argX=None):
    n1 = X.shape[0] if n1 is None else n1
    n2 = X.shape[1] if n2 is None else n2
    match_pair = -np.ones(n1, dtype=np.int32)
    match_belief = np.zeros(n1)
    match_pair_2 = -np.ones(n2, dtype=np.int32)
    match_arg = None if argX is None else np.zeros(n1)
    
    permutation = hungarian(X, n1, n2, backend='numpy')
    idx1, idx2 = np.where(permutation == 1)
    for i, j in zip(idx1, idx2):
        match_pair[i] = j
        match_pair_2[j] = i
        match_belief[i] = X[i, j]
        if argX is not None:
            match_arg[i] = argX[i, j]
    return match_pair, match_pair_2, match_belief, match_arg

def gumbel_noise(sizes, eps=1e-20):
    noise = torch.rand(*sizes)
    noise = -torch.log(-torch.log(noise + eps) + eps)
    return noise

def sinkhorn(X, args, backend='numpy'):
    if args.enable_gumbel:
        noise = gumbel_noise(X.shape)
        X = (X + noise * args.gumbel_factor) / args.gumbel_temp
    X = pygm.sinkhorn(X, backend=backend, tau=args.sinkhorn_tau, max_iter=args.sinkhorn_iter)
    return X