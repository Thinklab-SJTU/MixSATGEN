import numpy as np
import random
import time
import copy
import networkx as nx

def mix_clause(G1, G2, match_pair, match_pair_2, match_belief, match_arg=None, args=None):
    if args.matching in ['satbench'] or args.graph == 'lig':
        assert match_arg is not None, "Need match_arg for LIG matching"
    else:
        assert match_arg is None, "Shouldn't get match_arg for VIG matching"
    
    if args.mixing == 'clause':
        return mix_clause_greedy(G1, G2, match_pair, match_pair_2, match_belief, args.mixing_threshold, match_arg)
    else:
        assert False, f"Unsupported mixing method {args.mixing}!"


def mix_clause_greedy(G1, G2, match_pair, match_pair_2, match_belief, mix_threshold, match_arg=None):
    # aware that Graph node start from 1, alogn match_pair +1 too
    match_pair += 1
    match_pair_2 += 1
    G3 = copy.deepcopy(G1)
    outer_var_ptr = -2    # start from -2, because -1 and 0 is signals of "no matched node"
    debug_outer_point = set()
    
    t1 = time.time()
    num_vars_1 = len(match_pair)
    num_clause_1 = len(G1.nodes()) - num_vars_1

    # 1. sort clause order with variable belief
    clause_order = []
    for clause in range(num_vars_1 + 1, num_vars_1 + num_clause_1 + 1):
        belief = np.sum([match_belief[var-1] if var != 0 else 0 for var in G1[clause]])
        clause_order.append((int(clause), belief))
    clause_order = sorted(clause_order, key=(lambda itm: itm[1]), reverse=True)
    clause_order = list(np.array(clause_order, dtype=int)[:, 0])

    # 2. find most suitable clause2 for each clause1
    shifted_clause_1 = [] # (G1_clause, G2_clause) pair, document former clause node idx
    shifted_clause_2 = []
    remain_clause_1 = [x for x in range(num_vars_1 + 1, num_vars_1 + num_clause_1 + 1)]
    
    for clause1 in clause_order:
        if (len(shifted_clause_1)/float(num_clause_1) >= mix_threshold) or len(remain_clause_1) == 0:
            break

        remain_clause_1.remove(clause1)
        var1_list = np.array(G1[clause1])
        var2_list = match_pair[var1_list-[1]]
        var2_list = list(set(var2_list).difference(set([0])))
        
        # 2.1 clause2 -> 1. maximize same variable belief
        clause2_list = []
        for var2 in var2_list:
            surround_clause2 = list(G2[var2])
            for clause2 in surround_clause2:
                if int(clause2) in shifted_clause_2 or int(clause2) in clause2_list:
                    continue
                surround_var1_list = [match_pair_2[surround_var2-1] for surround_var2 in G2[clause2]]
                surround_var1_list = list(set(surround_var1_list).intersection(set(var1_list)))
                sum_belief = np.sum([match_belief[surround_var1-1] if surround_var1 != 0 else 0 for surround_var1 in surround_var1_list])
                clause2_list.append((int(clause2), sum_belief))
        clause2_list = sorted(clause2_list, key=(lambda itm: itm[1]), reverse=True)
        if len(clause2_list) == 0:
            continue
        belief_max = clause2_list[0][1]
        tmp = []
        for itm in clause2_list:
            if itm[1] != belief_max:
                break
            tmp.append(itm[0])
        clause2_list = tmp
        
        # 2.2 maximize all inner point belief
        tmp = []
        for clause2 in clause2_list:
            surround_var1_list = [match_pair_2[surround_var2-1] for surround_var2 in G2[clause2]]
            surround_var1_list = list(set(surround_var1_list).difference(set(var1_list)))
            try:
                sum_belief = np.sum([match_belief[surround_var1-1] if surround_var1 != 0 else 0 for surround_var1 in surround_var1_list])
            except:
                print(surround_var1_list)
                print(match_pair_2)
                print(match_belief)
            tmp.append((int(clause2), sum_belief))
        clause2_list = sorted(tmp, key=(lambda itm: itm[1]), reverse=True)
        if len(clause2_list) == 0:
            continue
        belief_max = clause2_list[0][1]
        tmp = []
        for itm in clause2_list:
            if itm[1] != belief_max:
                break
            tmp.append(itm[0])
        clause2_list = tmp
        clause2 = clause2_list[0]
        clause2 = random.choice(clause2_list) # if still multi clause, random choose
        
        # 2.3 get var2_list -> new_var1_list
        new_var1_list = []
        new_var1_w_list = []
        for var2 in G2[clause2]:
            if match_pair_2[var2-1] == 0:
                # outer point, using negative node to record, remap to pos in saving phase
                G3.add_node(outer_var_ptr)
                assert outer_var_ptr not in debug_outer_point
                debug_outer_point.add(outer_var_ptr)
                match_pair_2[var2-1] = outer_var_ptr
                outer_var_ptr -= 1
            new_var1_list.append(match_pair_2[var2-1])
            if match_arg is not None:
                sig_arg = int(match_arg[match_pair_2[var2-1]-1])
                new_var1_w_list.append(G2[clause2][var2]['weight'] * (-1)**sig_arg)
            else:
                new_var1_w_list.append(G2[clause2][var2]['weight'])
        
        
        # 3. modify G3 with clause1 & clause2
        var1_list = np.array(G3[clause1])
        for i, var1 in enumerate(var1_list):
            G3.remove_edge(clause1, var1)
        for i, new_var1 in enumerate(new_var1_list):
            G3.add_edge(clause1, new_var1, weight=new_var1_w_list[i])
        
        # preserve shifted info
        shifted_clause_1.append(clause1)
        shifted_clause_2.append(clause2)
    print(f'mix done: {len(shifted_clause_1)} clauses mix, time consumed {time.time() - t1}')
    return G3, outer_var_ptr


def mix_clause_centroid(G1, G2, match_pair, match_pair_2, match_belief, threshold, match_arg=None):
    # aware that Graph node start from 1, alogn match_pair +1 too
    match_pair += 1
    match_pair_2 += 1
    G3 = copy.deepcopy(G1)
    match_order = np.flip(np.argsort(match_belief) + 1)
    
    t1 = time.time()
    
    empty_clause_slot = []
    outer_var_ptr = -2    # start from -2, because -1 and 0 is a signal of "no matched node"
    removed_clause_num = 0
    added_clause_num = 0
    new_clause1_set = set() # a set for all new clause, items are string with strictly bigger order
    # to avoid duplicate clause
    debug_outer_point = set()
    
    for sort_idx, var1 in enumerate(match_order):
        print(f'=== sort {sort_idx} ===  {var1} <-> {match_pair[var1-1]} with {match_belief[var1-1]}')
        if match_pair[var1-1] == 0 or match_belief[var1-1] <= threshold:
            break
        var2 = match_pair[var1-1]
        # change: all clauses in G1 containing node1 -> clauses in G2 containing node2
        # but the clause nodes in G2 will be redirected to the matched nodes in G1
        
        # 1: delete original clause1 in G1
        clause1_list = list(G1[var1])
        for clause1 in clause1_list:
            try:
                G3.remove_node(clause1)
                removed_clause_num += 1
            except nx.exception.NetworkXError:
                pass    # already removed
            empty_clause_slot.append(clause1)
        
        # 2. transfer clause2 in G2 to G1
        clause2_list = list(G2[var2])
        for clause2 in clause2_list:
            clause_var2_list = list(G2[clause2])
            new_clause_var1_list = []
            # 2.1 get needed var2->1 of current clause1
            for clause_var2 in clause_var2_list:
                if match_pair_2[clause_var2-1] != 0:
                    new_clause_var1_list.append(match_pair_2[clause_var2-1] * G2.edges[clause2, clause_var2]['weight'])
                else:
                    # outer point, using negative node to record, remap to pos in saving phase
                    G3.add_node(outer_var_ptr)
                    assert outer_var_ptr not in debug_outer_point
                    debug_outer_point.add(outer_var_ptr)
                    match_pair_2[clause_var2-1] = outer_var_ptr
                    new_clause_var1_list.append((outer_var_ptr -0.5) * G2.edges[clause2, clause_var2]['weight'])
                    outer_var_ptr -= 1
            new_clause_var1_list.sort()
            new_clause1_set.add(' '.join(map(str, new_clause_var1_list)))
    
    # 2.2 For all new clause2->1, added them and the corresponding var2->1 edges
    added_clause_num  = len(new_clause1_set)
    new_clause1_set = [list(map(float, x.split(' '))) for x in new_clause1_set]
    for new_clause_var1_list in new_clause1_set:
        if empty_clause_slot:
            new_clause1 = empty_clause_slot[-1]
            empty_clause_slot = empty_clause_slot[:-1]
        else:
            new_clause1 = G3.number_of_nodes()
        G3.add_node(new_clause1)
        for new_clause_var1 in new_clause_var1_list:
            if not new_clause_var1 % 1:
                G3.add_edge(new_clause1, abs(int(new_clause_var1)), weight=abs(new_clause_var1)/new_clause_var1)
            else:
                # outer point
                G3.add_edge(new_clause1, -abs(int(new_clause_var1)), weight=-abs(new_clause_var1)/new_clause_var1)
    
    print(f'mix done: {removed_clause_num} clauses removed, {added_clause_num} clauses added, time consumed {time.time() - t1}')
    
    return G3, outer_var_ptr

def mix_clause_random(G1, num_var1, idx, args):
    seed = args.seed + idx
    np.random.seed(seed)
    random.seed(seed)
    G3 = copy.deepcopy(G1)
    num_clause1 = nx.number_of_nodes(G1) - num_var1*2
    sub_thresh = args.mixing_threshold
    
    shifted_clause_1 = []
    remain_clause_1 = [x for x in range(num_var1 + 1, num_var1 + num_clause1 + 1)]
    while (len(shifted_clause_1)/float(num_clause1) < sub_thresh) and len(remain_clause_1) != 0:
        clause1 = np.random.choice(remain_clause_1)
        remain_clause_1.remove(clause1)
        vars1 = np.array(G1[clause1])
        # permutation randomly
        action = random.randint(1, 2) # 1- add edge, 2 - delete edge
        if action == 1:
            node  = vars1[0]
            while node in vars1:
                node = random.randint(1, num_var1)
            vars1 = np.append(vars1, [node])
            sign = random.randint(1, 2)
            if sign == 2:
                sign = -1
            G3.add_edge(clause1, node, weight=sign)
        else:
            node = np.random.choice(vars1)
            G3.remove_edge(clause1, node)
    return G3


def mix_clause_random_match(G1, G2, num_var1, num_var2, idx, args):
    seed = args.seed + idx
    np.random.seed(seed)
    random.seed(seed)
    G3 = copy.deepcopy(G1)
    num_var1 = int(num_var1)
    num_var2 = int(num_var2)

    # randomize a match pair
    match_belief = np.ones(num_var1)
    if num_var1 <= num_var2:
        match_pair = np.linspace(0, num_var2-1, num_var2, dtype=np.int32)
        match_pair = np.random.permutation(match_pair)
        match_pair = match_pair[:num_var1]
        match_pair_2 = -np.ones(num_var2, dtype=np.int32)
        for i in range(num_var1):
            match_pair_2[match_pair[i]] = i
    else:
        match_pair_2 = np.linspace(0, num_var1-1, num_var1, dtype=np.int32)
        match_pair_2 = np.random.permutation(match_pair_2)
        match_pair_2 = match_pair_2[:num_var2]
        match_pair = -np.ones(num_var1, dtype=np.int32)
        for i in range(num_var2):
            match_pair[match_pair_2[i]] = i
    
    G3, outer_var_ptr = mix_clause_greedy(G1, G2, match_pair, match_pair_2, match_belief, args.mixing_threshold)
    return G3, outer_var_ptr