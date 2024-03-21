import os
import time
import numpy as np
import pygmtools as pygm
import networkx as nx
import torch
from torch.utils.data import Dataset
from satbench.dataset import SATDataset
from utils.formula import sat_to_graph
from utils.gmtools import dense_to_sparse, build_aff_mat_sparse

class GraphMatchDataset(Dataset):
    def __init__(self, args):
        self.dtype_torch = args.dtype_torch
        self.dtype_np = args.dtype_np
        self.device = args.device
        self.args = args
        self.satbenchDataset = None
        
        self.graph_list = []
        self.name_list = []
        self.pair_id_list = []
        self.switch = []
        self.all_file_list = set()
        self.load_data(args, get_VCG=True)
        
        self.aff_list = [None] * len(self.pair_id_list) # affinity matrix in rrwm and loss_check
        self.attr_list = [None] * len(self.graph_list)  # attribute matrix in rrwm
        self.feat_list = [None] * len(self.graph_list)  # lit feature in nsnet

    def __len__(self):
        return len(self.pair_id_list)

    def __getitem__(self, idx):
        func = eval(f'self.get_{self.args.matching}_item')
        return func(idx)

    def load_data(self, args, get_VCG=False):
        self.VCG_graph_list = [] if get_VCG else None
        formula_list = [] if args.matching in ['satbench'] else None
        num_vars_list = [] if args.matching in ['satbench'] else None
        
        input_path = f'./dataset/{args.dataset}/'
        pair_name_file = f'./dataset/{args.dataset}/pair_name.txt'
        content = open(pair_name_file).readlines()
        
        for line in content:
            if line[-1] == '\n' or line[-1] == '\0':
                line = line[:-1]
            pair_name = line.strip().split(' ')
            assert len(pair_name) == 2
            self.pair_id_list.append((pair_name[0], pair_name[1]))
            self.all_file_list.add(pair_name[0])
            self.all_file_list.add(pair_name[1])
        
        for filename in sorted(os.listdir(input_path)):
            if not filename[-4:] == '.cnf' or filename[:-4] not in self.all_file_list:
                continue
            input_file = input_path + filename
            if args.matching in ['satbench']:
                formula, num_vars = sat_to_graph(input_file, None)
                formula_list.append(formula)
                num_vars_list.append(num_vars)
            graph = sat_to_graph(input_file, args.graph)
            self.graph_list.append(graph)
            self.name_list.append(filename[:-4])
            if get_VCG:
                VCG, _, _ = sat_to_graph(input_file, 'vcg')
                self.VCG_graph_list.append(VCG)
        
        for i, _ in enumerate(self.pair_id_list):
            self.pair_id_list[i] = (self.name_list.index(self.pair_id_list[i][0]), self.name_list.index(self.pair_id_list[i][1]))
        
        self.switch = [False] * len(self.pair_id_list)
        self.all_file_list = list(self.all_file_list)
        
        if args.matching == 'satbench':
            self.satbenchDataset = SATDataset(formula_list, num_vars_list, self.name_list, args)
    
    
    def get_rrwm_item(self, idx):
        idx1, idx2 = self.pair_id_list[idx]
        node1, _, _ = self.get_attr(idx1)
        node2, _, _ = self.get_attr(idx2)
        n1 = node1.shape[0]
        n2 = node2.shape[0]
        if not self.args.mode == 'random':
            K = self.get_aff(idx, sparse=False)
        return n1, n2, idx
    
    def get_nsnet_item(self, idx):
        idx1, idx2 = self.pair_id_list[idx]
        data1 = self.NSNetDataset[idx1]
        data2 = self.NSNetDataset[idx2]
        return data1, data2, idx
    
    def get_satbench_item(self, idx):
        idx1, idx2 = self.pair_id_list[idx]
        data1 = self.satbenchDataset[idx1]
        data2 = self.satbenchDataset[idx2]
        return data1, data2, idx
    
    def get_K(self, idx, sparse=True):
        return self.get_aff(idx, sparse=sparse)
    
    def get_attr(self, idx):
        if self.attr_list[idx] is None:
            graph = self.graph_list[idx]
            A = nx.attr_matrix(graph)[0].astype(self.dtype_np)
            node = np.ones((graph.number_of_nodes(), 1), dtype=self.dtype_np)
            if self.args.graph == 'vig':
                edge1 = np.array(nx.attr_matrix(graph, edge_attr='weight1')[0])
                edge2 = np.array(nx.attr_matrix(graph, edge_attr='weight2')[0])
                edge = np.stack([edge1, edge2], axis=2).astype(self.dtype_np)
            else:
                edge = np.array(nx.attr_matrix(graph, edge_attr='weight')[0]).astype(self.dtype_np)
            edge = edge / np.max(edge)
            node = torch.from_numpy(node)
            edge = torch.from_numpy(edge)
            A = torch.from_numpy(A)
            self.attr_list[idx] = [node, edge, A]
        return self.attr_list[idx]
    
    def get_aff(self, idx, sparse=True):
        sparse=True
        idx1, idx2 = self.pair_id_list[idx]
        path = f'data/{self.args.dataset}/K_{self.name_list[idx1]}_{self.name_list[idx2]}_sparse{sparse}.pt'
        path2 = f'data/{self.args.dataset}/K_{self.name_list[idx2]}_{self.name_list[idx1]}_sparse{sparse}.pt'
        if self.aff_list[idx] is None:
            path_exist = os.path.exists(path) or os.path.exists(path2)
            if self.args.load_K and path_exist:
                if not os.path.exists(path) and os.path.exists(path2):
                    path = path2
                    self.switch[idx] = True
                t1 = time.time()
                K = torch.load(path)
                print(f'loaded K[{idx}] in {time.time() - t1:.2f}s')
                self.aff_list[idx] = True
            else:
                node1, _, A1 = self.get_attr(idx1)
                node2, _, A2 = self.get_attr(idx2)
                conn1, edge1, _ = dense_to_sparse(A1, batch=False)
                conn2, edge2, _ = dense_to_sparse(A2, batch=False)
                print(f'start K[{idx}] building...')
                t1 = time.time()
                if sparse:
                    K = build_aff_mat_sparse(node1, edge1, conn1, node2, edge2, conn2)
                else:
                    if self.args.dtype == 'float16':
                        pygm.BACKEND = 'numpy'
                        K = pygm.utils.build_aff_mat(
                            node1.unsqueeze(0).numpy(), edge1.unsqueeze(0).numpy(),
                            conn1.unsqueeze(0).numpy(), node2.unsqueeze(0).numpy(),
                            edge2.unsqueeze(0).numpy(), conn2.unsqueeze(0).numpy(),
                            edge_aff_fn=pygm.utils.inner_prod_aff_fn, backend='numpy')
                        pygm.BACKEND = 'pytorch'
                        K = torch.from_numpy(K).squeeze(0)
                    else:
                        K = pygm.utils.build_aff_mat(
                            node1.unsqueeze(0), edge1.unsqueeze(0),
                            conn1.unsqueeze(0), node2.unsqueeze(0),
                            edge2.unsqueeze(0), conn2.unsqueeze(0),
                            edge_aff_fn=pygm.utils.inner_prod_aff_fn).squeeze(0)
                self.aff_list[idx] = True
                torch.save(K, path)
                print(f'K[{idx}] built, time: {time.time() - t1:.2f}s')
        else:
            if self.switch[idx]:
                path = path2
            K = torch.load(path)
        return K

