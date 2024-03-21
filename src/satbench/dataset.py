import os
import glob
import torch
import pickle
import itertools

from torch_geometric.data import Dataset
from satbench.data import construct_lcg, construct_vcg


class SATDataset(Dataset):
    def __init__(self, formula_list, num_vars_list, name_list, args):
        self.args = args
        self.formula_list = formula_list
        self.num_vars_list = num_vars_list
        self.name_list = name_list
        self.all_labels = [None] * len(self.formula_list)
        
        self.graph = self.args.satbench_graph
        super().__init__(f'./dataset/{self.args.dataset}')
    
    @property
    def processed_file_names(self):
        return [f'{name}_{self.graph}.pt' for name in self.name_list]

    def _save_data(self, num_vars, formula, name):
        file_name = f'{name}_{self.graph}.pt'
        saved_path = os.path.join(self.processed_dir, file_name)
        print(f'save path: {saved_path}')
        if os.path.exists(saved_path):
            return
        
        if self.args.satbench_graph == 'lcg':
            data = construct_lcg(num_vars, formula)
        elif self.args.satbench_graph == 'vcg':
            data = construct_vcg(num_vars, formula)

        torch.save(data, saved_path)
    
    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)        
        for formula, num_vars, name in zip(self.formula_list, self.num_vars_list, self.name_list):
            self._save_data(num_vars, formula, name)

    def len(self):
        return len(self.formula_list)

    def get(self, idx):
        file_name = f'{self.name_list[idx]}_{self.graph}.pt'
        data = torch.load(os.path.join(self.processed_dir, file_name))
        data.y = None
        return data
