# add your source codes regarding the model and its layers here
import torch as pt
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops, degree


class ProteinGCN(nn.Module):
    def __init__(self, in_channels:int, batch_size:int, out_channels=6630) -> None:
        super(ProteinGCN, self).__init__() 
        self.batch_size =  batch_size
        self.out_channels = out_channels
        self.model = gnn.Sequential('x, edge_index, edge_attr, batch', [
                        *self._make_layer(in_channel=in_channels, first_channel=128, 
                                          mode='double', num_layer=4, layer=gnn.GCNConv),
                        (lambda x1, x2, x3, x4: [x1, x2, x3, x4], 'x1, x2, x3, x4 -> xs'),
                        (gnn.JumpingKnowledge('cat', [128, 256, 512, 1024], num_layers=4), 'xs -> x'),
                        (nn.Dropout(p=0.5), 'x -> x'),
                        (gnn.global_mean_pool, 'x, batch -> x'),
                        nn.Linear(1920, out_channels)
                        ])

    def _make_layer(self, in_channel:int, first_channel:int, mode:str, num_layer:int, layer=gnn.GCNConv):
        assert num_layer >= 2, 'num_layer must >=2!'
        channel = first_channel
        m_list = [(layer(in_channel, channel), 'x, edge_index, edge_attr -> x1'), nn.ReLU()]
        if mode == 'same':
            for i in range(num_layer-1):
                m_list.append((layer(channel, channel), 'x%d, edge_index, edge_attr -> x%d' % (i+1, i+2)))
                m_list.append(nn.ReLU())  
        elif mode == 'double':
            for i in range(num_layer-1):
                m_list.append((layer(channel, 2*channel), 'x%d, edge_index, edge_attr -> x%d' % (i+1, i+2)))
                m_list.append(nn.ReLU())
                channel *= 2
    
        return m_list
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        return self.model(x, edge_index, edge_attr, batch)


class myGCNConv(gnn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(myGCNConv).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(pt.Tensor(out_channels))

        self.resnet_parameters()
    
    def resnet_parameters(self):       #啥意思这个函数 
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out
    

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
