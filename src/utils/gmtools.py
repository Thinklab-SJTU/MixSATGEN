import numpy as np
import torch
import time


def build_batch(input, return_ori_dim=False):
    """
    pygmtools: numpy implementation of building a batched np.ndarray
    """
    assert type(input[0]) == np.ndarray
    it = iter(input)
    t = next(it)
    max_shape = list(t.shape)
    ori_shape = tuple([[_] for _ in max_shape])
    while True:
        try:
            t = next(it)
            for i in range(len(max_shape)):
                max_shape[i] = int(max(max_shape[i], t.shape[i]))
                ori_shape[i].append(t.shape[i])
        except StopIteration:
            break
    max_shape = np.array(max_shape)

    padded_ts = []
    for t in input:
        pad_pattern = np.zeros((len(max_shape), 2), dtype=np.int64)
        pad_pattern[:, 1] = max_shape - np.array(t.shape)
        padded_ts.append(np.pad(t, pad_pattern, 'constant', constant_values=0))

    if return_ori_dim:
        return np.stack(padded_ts, axis=0), ori_shape
    else:
        return np.stack(padded_ts, axis=0)

def dense_to_sparse(dense_adj, batch=False):
    """
    pygmtools: torch input/output of converting a dense adjacency matrix to a sparse 
    matrix
    """
    dense_adj_np = dense_adj.detach().numpy() if torch.is_tensor(dense_adj) else dense_adj.copy()
    if not batch:
        dense_adj_np = np.expand_dims(dense_adj_np, 0)
    batch_size = dense_adj_np.shape[0]
    conn, ori_dim = build_batch([np.unique(np.stack(np.nonzero(a)[:2], axis=1), axis=0) for a in dense_adj_np], return_ori_dim=True)
    nedges = ori_dim[0]
    edge_weight = build_batch([dense_adj_np[b][(conn[b, :nedges[b], 0], conn[b, :nedges[b], 1])] for b in range(batch_size)])
    if len(edge_weight.shape) == 2:
        edge_weight = np.expand_dims(edge_weight, axis=-1)
    if batch:
        return torch.from_numpy(conn), torch.from_numpy(edge_weight), nedges
    return torch.from_numpy(conn).squeeze(0), torch.from_numpy(edge_weight).squeeze(0), nedges


def Kidx(x, y, n2):
    return int(x * n2 + y)

def aff_mat_from_node_edge_aff(node_aff, edge_aff, conn1, conn2):
    device = node_aff.device
    dtype = node_aff.dtype
    n1 = node_aff.shape[0]
    n2 = node_aff.shape[1]

    indices = []
    values = []
    # edge-wise affinity
    if edge_aff is not None:
        t1 = time.time()
        for e1, (x, y) in enumerate(conn1):
            for e2, (i, j) in enumerate(conn2):
                indices.append(torch.tensor([Kidx(x, i, n2), Kidx(y, j, n2)], dtype=torch.int, device=device))
                indices.append(torch.tensor([Kidx(x, j, n2), Kidx(y, i, n2)], dtype=torch.int, device=device))
                indices.append(torch.tensor([Kidx(y, i, n2), Kidx(x, j, n2)], dtype=torch.int, device=device))
                indices.append(torch.tensor([Kidx(y, j, n2), Kidx(x, i, n2)], dtype=torch.int, device=device))
                values += [edge_aff[e1, e2]] * 4
    # node-wise affinity
    if node_aff is not None:
        node_aff_b = node_aff.reshape(-1)
        indices += [torch.tensor([idx, idx], dtype=torch.int, device=device) for idx in range(len(node_aff_b))]
        values += node_aff_b.tolist()
    
    indices = torch.stack(indices)
    values = torch.tensor(values)
    K = torch.sparse_coo_tensor(indices.T, values, [int(n1*n2), int(n1*n2)])
    return K

def inner_prod_aff_fn(feat1, feat2):
    return torch.matmul(feat1, feat2.transpose(0, 1))

def build_aff_mat_sparse(node_feat1, edge_feat1, connectivity1, node_feat2, edge_feat2, connectivity2):
    node_aff = inner_prod_aff_fn(node_feat1, node_feat2) if node_feat1 is not None else None
    edge_aff = inner_prod_aff_fn(edge_feat1, edge_feat2) if edge_feat1 is not None else None
    result = aff_mat_from_node_edge_aff(node_aff, edge_aff, connectivity1, connectivity2)
    return result

def model_feat_to_X(feat_1, feat_2):
    n1, n2 = feat_1.shape[0]//2, feat_2.shape[0]//2
    L1 = feat_1[:n1]
    nL1 = feat_1[n1:]
    L2 = feat_2[:n2]
    nL2 = feat_2[n2:]
    
    X1 = inner_prod_aff_fn(L1, L2) + inner_prod_aff_fn(nL1, nL2)
    X2 = inner_prod_aff_fn(L1, nL2) + inner_prod_aff_fn(nL1, L2)
    X1 = X1.unsqueeze(0)
    X2 = X2.unsqueeze(0)
    X = torch.cat((X1, X2), dim=0).max(dim=0)
    return X.values, X.indices