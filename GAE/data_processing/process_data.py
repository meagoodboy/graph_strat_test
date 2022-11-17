import pandas as pd
import networkx as nx
import numpy as np
import torch
from tqdm.auto import trange, tqdm
from torch_geometric.data import Data
## networkx 2.6 preferred


def get_edgelist(network_path):
    print("Loading the Network : ", network_path)
    nx_obj = nx.read_adjlist(network_path)
    B = nx.adjacency_matrix(nx_obj)
    adjmat = B.todense()
    nodes = list(nx_obj.nodes())
    mat = np.array(adjmat)
    lst = []
    for i in trange(mat.shape[0]):
        for j in range(i, mat.shape[0]):
            if mat[i][j] == 1:
                lst.append([i,j])
    edgelist = torch.Tensor(lst).long()
    print("Loaded the Network ")
    return nodes,edgelist.T


class SingleOmicData(torch.utils.data.Dataset):

    def __init__(self, network_file, omics_file, output_node_dim, norm = False):
        super(SingleOmicData, self).__init__()
        
        node_label, edgelist = get_edgelist(network_file)
        self.edge_list = edgelist
        self.edge_index = edgelist
        self.mat = pd.read_csv(omics_file, index_col=0)
        self.raw = self.mat
        if norm:
            normalized_df=(self.mat-self.mat.min())/(self.mat.max()-self.mat.min())
            print("Normalizsed the input data")
            self.mat = normalized_df.dropna(axis=1)
        self.mat = self.mat[node_label]
        self.patients = list(self.mat.index)
        self.node_order = node_label
        self.node_vec_len = output_node_dim
    
    def __len__(self):
        return len(self.mat)

    def indexes(self):
        return self.mat.index

    def __getitem__(self, idx):
        a = list(self.mat.iloc[idx,:])
        a = torch.Tensor(a)
        a = torch.reshape(a,(len(self.node_order),self.node_vec_len))
        mydata = Data(x=a, edge_index=self.edge_list)
        return mydata