# add your source codes regarding the model and its layers here
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops, degree


class GCNConv(gnn.MessagePassing):
    def __init__(self, emb_dim:int):
        super().__init__(aggr='add')
        self.bias = nn.Parameter(pt.Tensor(emb_dim))
        self.reset_parameters()
    
    def reset_parameters(self):       
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr):
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, edge_attr, fill_value=.0, num_nodes=x.size(0))
        
        # Step 2: Compute normalization.
        row, col = edge_index 
        deg = degree(row, x.size(0), dtype=x.dtype) 
        deg_inv_sqrt = deg.pow(-0.5) 
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] 

        # Step 3: Start propagating messages. 
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm) + self.bias
 
        return out
    
    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j 


class HeteroGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(HeteroGCNConv, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.conv1 = GCNConv(out_channels)
        self.conv2 = GCNConv(out_channels)

    def forward(self, x, edge_index, edge_attr):
        # edge_index:[2, N-1+num_nho] edge_attr:[N-1+num_nho, 1]
        N = x.size(0)
        tai_idx = pt.stack((edge_index[0, :N-1], edge_index[1, :N-1]), dim=0)
        tai_idx = pt.cat((tai_idx, tai_idx.flip(0)), dim=1) # 两条有向边 
        nho_idx = pt.stack((edge_index[0, N-1:], edge_index[1, N-1:]), dim=0)
        nho_idx = pt.cat((nho_idx, nho_idx.flip(0)), dim=1)
        tai_attr = edge_attr[:N-1]
        tai_attr = pt.cat((tai_attr, tai_attr), dim=0)
        nho_attr = edge_attr[N-1:]
        nho_attr = pt.cat((nho_attr, nho_attr), dim=0)
        
        x = self.lin(x)
        x = F.relu(x)
        x = self.conv1(x, tai_idx, tai_attr)
        x = F.relu(x)
        x = self.conv2(x, nho_idx, nho_attr)

        return x


class ProteinGCN(nn.Module):
    def __init__(self, in_channels:int, batch_size:int, out_channels=6630) -> None:
        super(ProteinGCN, self).__init__() 
        self.batch_size =  batch_size
        self.out_channels = out_channels
        self.conv1 = HeteroGCNConv(in_channels, 128)
        self.conv2 = HeteroGCNConv(128, 256)
        self.conv3 = HeteroGCNConv(256, 512)
        self.conv4 = HeteroGCNConv(512, 1024)
        self.JK = gnn.JumpingKnowledge('cat', [128, 256, 512, 1024])
        self.lin = nn.Linear(1920, out_channels)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x1 = self.conv1(x, edge_index, edge_attr)
        x2 = self.conv2(x1, edge_index, edge_attr)
        x3 = self.conv3(x2, edge_index, edge_attr)
        x4 = self.conv4(x3, edge_index, edge_attr)
        x = self.JK([x1, x2, x3, x4])
        x = gnn.global_mean_pool(x, batch)
        x = self.lin(x)

        return x