import torch
from torch import nn

class L2Linear(torch.nn.Module):
    def __init__(self, out_channels, num_nodes, batch_size):
        super(L2Linear, self).__init__()
        self.batch = batch_size
        self.num_nodes = num_nodes
        self.feat_lin = torch.nn.Sequential(
            torch.nn.Linear(num_nodes*out_channels, num_nodes),
            torch.nn.BatchNorm1d(num_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(num_nodes, num_nodes)
        )

    def forward(self, x):
        shape = x.shape
        batch = int(x.shape[0]/self.num_nodes)
        # print(x.shape, flush=True)
        # print(batch, flush=True)
        x = x.view(batch, -1)
        # print(x.shape, flush=True)
        y = self.feat_lin(x)
        
        y  = torch.reshape(y, shape)
        # print(y.shape, flush=True)        
        return y