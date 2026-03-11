import torch as pt
import torch.nn as nn
import torch_geometric.nn as gnn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ProteinGCN(nn.Module):
    def __init__(self, embed_dim:int=512, hidden_channels:int=256, out_channels:int=128, num_layers:int=3, num_edge_features:int=10,):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=21, embedding_dim=embed_dim, padding_idx=0,)
        # node_attr占一维
        self.gcn = gnn.GCN(in_channels=embed_dim+num_edge_features, hidden_channels=hidden_channels, 
                           num_layers=num_layers, out_channels=out_channels,)
        self.shared = nn.Sequential(nn.Linear(4*out_channels, out_channels), nn.SiLU(),)
        self.tm_head = nn.Linear(out_channels, 1,)
        self.seq_head = nn.Linear(out_channels, 1,)

        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*4, 
                                                activation='gelu', batch_first=True,)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=1)
        self.mlp = nn.Linear(embed_dim, 21)

    def embed(self, seq_mask):
        seq, mask = seq_mask
        embedding = self.emb(seq) # [batch_size, seq_len, emb_dim]
        embedding = embedding * mask.unsqueeze(-1) # mask: [batch_size, seq_len, 1]
        return embedding

    def encode_protein(self, seq, mask, graph):
        x, edge_idx, edge_attr, batch, node2seq = graph.x, graph.edge_index, graph.edge_attr, graph.batch, graph.node2seq
        emb = self.embed((seq, mask))
        B, L, D = emb.shape
        emb_flat = emb.view(-1, D)
        flat_idx = batch * L + node2seq
        node_emb = emb_flat[flat_idx]
        x = pt.cat([node_emb, x], dim=-1)
        x = self.gcn(x, edge_idx, edge_attr=edge_attr, batch=batch)
        x = gnn.global_mean_pool(x, batch)
        return x        

    def forward(self, data, mode:str='pretraining'):
        if mode == 'pretraining':
            seqs_pad, masks_pad = data
            embs = self.emb(seqs_pad) # [batch_size, seq_len, emb_dim]
            embs = self.encoder(embs, src_key_padding_mask=~masks_pad) # [batch_size, seq_len, emb_dim]
            outputs = self.mlp(embs) # [batch_size, seq_len, 21]
            return outputs
        elif mode == 'finetuning':            
            (seqs, masks, graphs), (inv_i, inv_j) = data
            prot_repr = self.encode_protein(seqs, masks, graphs)
            x_i = prot_repr[inv_i]
            x_j = prot_repr[inv_j]
            feature = pt.cat([x_i, x_j, x_i-x_j, x_i*x_j], dim=-1)
            shared = self.shared(feature)
            tm_score = self.tm_head(shared).squeeze(-1)
            seq_score = self.seq_head(shared).squeeze(-1)
            return tm_score, seq_score
        else:
            raise ValueError(f'Unknown mode: {mode}')