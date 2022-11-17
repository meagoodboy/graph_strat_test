import torch
from torch import nn
from torch.nn import Dropout as dropout
from torch_geometric.nn import SAGEConv, GCNConv, GraphConv, GatedGraphConv, GATv2Conv, GravNetConv, GATConv, ChebConv, TAGConv, AGNNConv, ResGatedGraphConv , LGConv
from torch_geometric.nn import TransformerConv, ARMAConv, SGConv, MFConv, CGConv, FeaStConv, LEConv, ClusterGCNConv, GENConv, PANConv, FiLMConv, EGConv , GeneralConv

MAX_CHANNELS = 100

class L3GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3GCNConv, self).__init__()
        self.conv1 = GCNConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = GCNConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = GCNConv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x


class L3SAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3SAGEConv, self).__init__()
        self.conv1 = SAGEConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = SAGEConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = SAGEConv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x


class L3GraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3GraphConv, self).__init__()
        self.conv1 = GraphConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = GraphConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = GraphConv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L3GatedGraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3GatedGraphConv, self).__init__()
        self.conv1 = GatedGraphConv(in_channels, out_channels)
        self.conv2 = GatedGraphConv(out_channels,  out_channels)
        self.conv3 = GatedGraphConv( out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L3GATv2Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3GATv2Conv, self).__init__()
        self.conv1 = GATv2Conv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = GATv2Conv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = GATv2Conv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x
    
class L3GravNetConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3GravNetConv, self).__init__()
        self.conv1 = GravNetConv(in_channels, MAX_CHANNELS * out_channels, 4,3,3)
        self.conv2 = GravNetConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels, 4,3,3)
        self.conv3 = GravNetConv(int(MAX_CHANNELS/2)  * out_channels, out_channels, 4,3,3)
        self.drop  = dropout(p=0.1)
        print("Passed edge index will not be used while learning")

    def forward(self, x, edge_index):
        x = self.conv1(x).relu()
        x = self.drop(x)
        x = self.conv2(x).relu()
        x = self.drop(x)
        x = self.conv3(x).relu()
        return x
    
class L2GravNetConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2GravNetConv, self).__init__()
        self.conv1 = GravNetConv(in_channels, MAX_CHANNELS * out_channels, 4,3,3)
        self.conv2 = GravNetConv(MAX_CHANNELS * out_channels, out_channels, 4,3,3)
        self.drop  = dropout(p=0.1)
        print("Passed edge index will not be used while learning")

    def forward(self, x, edge_index):
        x = self.conv1(x).relu()
        x = self.drop(x)
        x = self.conv2(x).relu()
        return x
    
class L2GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2GCNConv, self).__init__()
        self.conv1 = GCNConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = GCNConv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x


class L2SAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2SAGEConv, self).__init__()
        self.conv1 = SAGEConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = SAGEConv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x


class L2GraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2GraphConv, self).__init__()
        self.conv1 = GraphConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = GraphConv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

class L2GatedGraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2GatedGraphConv, self).__init__()
        self.conv1 = GatedGraphConv(in_channels,  out_channels)
        self.conv2 = GatedGraphConv( out_channels,  out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

class L2GATv2Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2GATv2Conv, self).__init__()
        self.conv1 = GATv2Conv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = GATv2Conv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

## Chebconv
class L3ChebConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3ChebConv, self).__init__()
        self.conv1 = ChebConv(in_channels, MAX_CHANNELS * out_channels, 4)
        self.conv2 = ChebConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels, 4)
        self.conv3 = ChebConv(int(MAX_CHANNELS/2)  * out_channels, out_channels, 4)
        self.drop  = dropout(p=0.1)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x
        

class L2ChebConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2ChebConv, self).__init__()
        self.conv1 = ChebConv(in_channels, MAX_CHANNELS * out_channels, 4)
        self.conv2 = ChebConv(MAX_CHANNELS * out_channels, out_channels, 4)
        self.drop  = dropout(p=0.1)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

## Resgatedgraphconv

class L3ResGatedGraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3ResGatedGraphConv, self).__init__()
        self.conv1 = ResGatedGraphConv(in_channels,  out_channels)
        self.conv2 = ResGatedGraphConv(in_channels,  out_channels)
        self.conv3 = ResGatedGraphConv(in_channels,  out_channels)
        self.drop  = dropout(p=0.1)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x



class L2ResGatedGraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2ResGatedGraphConv, self).__init__()
        self.conv1 = ResGatedGraphConv(in_channels,  out_channels)
        self.conv2 = ResGatedGraphConv( out_channels,  out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

## TransformerConv
class L3TransformerConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3TransformerConv, self).__init__()
        self.conv1 = TransformerConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = TransformerConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = TransformerConv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L2TransformerConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2TransformerConv, self).__init__()
        self.conv1 = TransformerConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = TransformerConv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

##AGNNConv
class L3AGNNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3AGNNConv, self).__init__()
        self.conv1 = AGNNConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = AGNNConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = AGNNConv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L2AGNNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2AGNNConv, self).__init__()
        self.conv1 = AGNNConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = AGNNConv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

##TAGConv
class L3TAGConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3TAGConv, self).__init__()
        self.conv1 = TAGConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = TAGConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = TAGConv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L2TAGConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2TAGConv, self).__init__()
        self.conv1 = TAGConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = TAGConv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x
        
## ARMAConv
class L3ARMAConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3ARMAConv, self).__init__()
        self.conv1 = ARMAConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = ARMAConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = ARMAConv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L2ARMAConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2ARMAConv, self).__init__()
        self.conv1 = ARMAConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = ARMAConv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

## SGConv
class L3SGConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3SGConv, self).__init__()
        self.conv1 = SGConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = SGConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = SGConv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L2SGConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2SGConv, self).__init__()
        self.conv1 = SGConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = SGConv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

## MFConv
class L3MFConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3MFConv, self).__init__()
        self.conv1 = MFConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = MFConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = MFConv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L2MFConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2MFConv, self).__init__()
        self.conv1 = MFConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = MFConv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

## CGConv
class L3CGConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3CGConv, self).__init__()
        self.conv1 = CGConv(in_channels,  0)
        self.conv2 = CGConv( out_channels, 0)
        self.conv3 = CGConv( out_channels, 0)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L2CGConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2CGConv, self).__init__()
        self.conv1 = CGConv(in_channels, 0)
        self.conv2 = CGConv(out_channels, 0)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

## FeaStConv
class L3FeaStConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3FeaStConv, self).__init__()
        self.conv1 = FeaStConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = FeaStConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = FeaStConv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L2FeaStConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2FeaStConv, self).__init__()
        self.conv1 = FeaStConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = FeaStConv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

## LEConv
class L3LEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3LEConv, self).__init__()
        self.conv1 = LEConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = LEConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = LEConv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L2LEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2LEConv, self).__init__()
        self.conv1 = LEConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = LEConv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

## ClusterGCNConv
class L3ClusterGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3ClusterGCNConv, self).__init__()
        self.conv1 = ClusterGCNConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = ClusterGCNConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = ClusterGCNConv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L2ClusterGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2ClusterGCNConv, self).__init__()
        self.conv1 = ClusterGCNConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = ClusterGCNConv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

## GENConv
class L3GENConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3GENConv, self).__init__()
        self.conv1 = GENConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = GENConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = GENConv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L2GENConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2GENConv, self).__init__()
        self.conv1 = GENConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = GENConv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

## PANConv
class L3PANConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3PANConv, self).__init__()
        self.conv1 = PANConv(in_channels, MAX_CHANNELS * out_channels,5)
        self.conv2 = PANConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels,5)
        self.conv3 = PANConv(int(MAX_CHANNELS/2)  * out_channels, out_channels,5)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.drop(x[0].relu())
        x = self.conv2(x, edge_index)
        x = self.drop(x[0].relu())
        x = self.conv3(x, edge_index)
        return x[0].relu()

class L2PANConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2PANConv, self).__init__()
        self.conv1 = PANConv(in_channels, MAX_CHANNELS * out_channels,5)
        self.conv2 = PANConv(MAX_CHANNELS * out_channels, out_channels,5)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.drop(x[0].relu())
        x = self.conv2(x, edge_index)
        return x[0].relu()

## FiLMConv
class L3FiLMConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3FiLMConv, self).__init__()
        self.conv1 = FiLMConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = FiLMConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = FiLMConv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L2FiLMConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2FiLMConv, self).__init__()
        self.conv1 = FiLMConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = FiLMConv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

##EGConm
class L3EGConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3EGConv, self).__init__()
        self.conv1 = EGConv(in_channels, MAX_CHANNELS * out_channels,num_heads = 1)
        self.conv2 = EGConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels,num_heads = 1)
        self.conv3 = EGConv(int(MAX_CHANNELS/2)  * out_channels, out_channels,num_heads = 1)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L2EGConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2EGConv, self).__init__()
        self.conv1 = EGConv(in_channels, MAX_CHANNELS * out_channels,num_heads = 1)
        self.conv2 = EGConv(MAX_CHANNELS * out_channels, out_channels,num_heads = 1)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

## GeneralConv
class L3GeneralConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3GeneralConv, self).__init__()
        self.conv1 = GeneralConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = GeneralConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = GeneralConv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L2GeneralConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2GeneralConv, self).__init__()
        self.conv1 = GeneralConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = GeneralConv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

## LGConv
class L3LGConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3LGConv, self).__init__()
        self.conv1 = LGConv()
        self.conv2 = LGConv()
        self.conv3 = LGConv()
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L2LGConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2LGConv, self).__init__()
        self.conv1 = LGConv()
        self.conv2 = LGConv()
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x

## GATConv
class L3GATConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L3GATConv, self).__init__()
        self.conv1 = GATConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = GATConv(MAX_CHANNELS * out_channels, int(MAX_CHANNELS/2) * out_channels)
        self.conv3 = GATConv(int(MAX_CHANNELS/2)  * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv3(x, edge_index).relu()
        return x

class L2GATConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(L2GATConv, self).__init__()
        self.conv1 = GATConv(in_channels, MAX_CHANNELS * out_channels)
        self.conv2 = GATConv(MAX_CHANNELS * out_channels, out_channels)
        self.drop  = dropout(p=0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.drop(x)
        x = self.conv2(x, edge_index).relu()
        return x


def get_encoder(encoder, in_channels = 1, out_channels = 1):

    if encoder == "L3GCNConv":
        return L3GCNConv(in_channels,out_channels)
    elif encoder == "L2GCNConv":
        return L2GCNConv(in_channels,out_channels)
    elif encoder == "L3SAGEConv":
        return L3SAGEConv(in_channels,out_channels)
    elif encoder == "L2SAGEConv":
        return L2SAGEConv(in_channels,out_channels)
    elif encoder == "L3GraphConv":
        return L3GraphConv(in_channels,out_channels)
    elif encoder == "L2GraphConv":
        return L2GraphConv(in_channels,out_channels)
    elif encoder == "L3GatedGraphConv":
        return L3GatedGraphConv(in_channels,out_channels)
    elif encoder == "L2GatedGraphConv":
        return L2GatedGraphConv(in_channels,out_channels)
    elif encoder == "L3GATv2Conv":
        return L3GATv2Conv(in_channels,out_channels)
    elif encoder == "L2GATv2Conv":
        return L2GATv2Conv(in_channels,out_channels)
    elif encoder == "L3GravNetConv":
        return L3GravNetConv(in_channels,out_channels)
    elif encoder == "L2GravNetConv":
        return L2GravNetConv(in_channels,out_channels)
    elif encoder == "L3ChebConv":
        return L3ChebConv(in_channels,out_channels)
    elif encoder == "L2ChebConv":
        return L2ChebConv(in_channels,out_channels)
    elif encoder == "L3ResGatedGraphConv":
        return L3ResGatedGraphConv(in_channels,out_channels)
    elif encoder == "L2ResGatedGraphConv":
        return L2ResGatedGraphConv(in_channels,out_channels)
    elif encoder == "L3TransformerConv":
        return L3TransformerConv(in_channels,out_channels)
    elif encoder == "L2TransformerConv":
        return L2TransformerConv(in_channels,out_channels)
    elif encoder == "L3AGNNConv":
        return L3AGNNConv(in_channels,out_channels)
    elif encoder == "L2AGNNConv":
        return L2AGNNConv(in_channels,out_channels)
    elif encoder == "L3TAGConv":
        return L3TAGConv(in_channels,out_channels)
    elif encoder == "L2TAGConv":
        return L2TAGConv(in_channels,out_channels)
    elif encoder == "L3ARMAConv":
        return L3ARMAConv(in_channels,out_channels)
    elif encoder == "L2ARMAConv":
        return L2ARMAConv(in_channels,out_channels)
    elif encoder == "L3SGConv":
        return L3SGConv(in_channels,out_channels)
    elif encoder == "L2SGConv":
        return L2SGConv(in_channels,out_channels)
    elif encoder == "L3MFConv":
        return L3MFConv(in_channels,out_channels)
    elif encoder == "L2MFConv":
        return L2MFConv(in_channels,out_channels)
    elif encoder == "L3CGConv":
        return L3CGConv(in_channels,out_channels)
    elif encoder == "L2CGConv":
        return L2CGConv(in_channels,out_channels)
    elif encoder == "L3FeaStConv":
        return L3FeaStConv(in_channels,out_channels)
    elif encoder == "L2FeaStConv":
        return L2FeaStConv(in_channels,out_channels)
    elif encoder == "L3LEConv":
        return L3LEConv(in_channels,out_channels)
    elif encoder == "L2LEConv":
        return L2LEConv(in_channels,out_channels)
    elif encoder == "L3ClusterGCNConv":
        return L3ClusterGCNConv(in_channels,out_channels)
    elif encoder == "L2ClusterGCNConv":
        return L2ClusterGCNConv(in_channels,out_channels)
    elif encoder == "L3GENConv":
        return L3GENConv(in_channels,out_channels)
    elif encoder == "L2GENConv":
        return L2GENConv(in_channels,out_channels)
    elif encoder == "L3PANConv":
        return L3PANConv(in_channels,out_channels)
    elif encoder == "L2PANConv":
        return L2PANConv(in_channels,out_channels)
    elif encoder == "L3FiLMConv":
        return L3FiLMConv(in_channels,out_channels)
    elif encoder == "L2FiLMConv":
        return L2FiLMConv(in_channels,out_channels)
    elif encoder == "L3EGConv":
        return L3EGConv(in_channels,out_channels)
    elif encoder == "L2EGConv":
        return L2EGConv(in_channels,out_channels)
    elif encoder == "L3GeneralConv":
        return L3GeneralConv(in_channels,out_channels)
    elif encoder == "L2GeneralConv":
        return L2GeneralConv(in_channels,out_channels)
    elif encoder == "L3LGConv":
        return L3LGConv(in_channels,out_channels)
    elif encoder == "L2LGConv":
        return L2LGConv(in_channels,out_channels)
    elif encoder == "L3GATConv":
        return L3GATConv(in_channels,out_channels)
    elif encoder == "L2GATConv":
        return L2GATConv(in_channels,out_channels)
    else:
        print("No encoder found")
        return False
