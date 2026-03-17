import math
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = pt.zeros(max_len, d_model)
        position = pt.arange(0, max_len, dtype=pt.float).unsqueeze(1)
        div_term = pt.exp(
            pt.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = pt.sin(position * div_term)
        pe[:, 1::2] = pt.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_len: int = 1300,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2*d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask):
        x = self.token_embedding(input_ids)
        x = self.pos_embedding(x)
        key_padding_mask = (attention_mask == 0)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled


class GINEBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.edge_encoder = (
            nn.Linear(edge_dim, hidden_dim) if edge_dim != hidden_dim else nn.Identity()
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = gnn.GINEConv(nn=mlp, train_eps=True, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_idx, edge_attr=None, batch=None):
        """
        x: [N, in_dim]
        edge_idx: [2, E]
        edge_attr: [E, edge_dim]
        """
        if edge_attr is None:
            edge_attr = x.new_zeros(edge_idx.size(1), self.edge_dim)
        edge_attr = self.edge_encoder(edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_idx, edge_attr)
            h = F.silu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = norm(x + h)  # residual
        return x


class GraphEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        gnn_dim: int = 256,
        gnn_num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 为了确保后续在gnn中维度整齐
        self.seq_encoder = nn.Embedding(num_embeddings=vocab_size, 
                                        embedding_dim=gnn_dim-node_feat_dim, padding_idx=0)
        self.gnn = GINEBackbone(
            hidden_dim=gnn_dim,
            edge_dim=edge_feat_dim,
            num_layers=gnn_num_layers,
            dropout=dropout,
        )

    def forward(self, seq, mask, graph):
        """
        seq: [B, L]
        mask: [B, L]
        graph:
          - x: [N, Fx]
          - edge_index: [2, E]
          - edge_attr: [E, Fe]
          - batch: [N]
          - node2seq: [N]
        """
        x = graph.x.float()
        edge_idx = graph.edge_index
        edge_attr = graph.edge_attr.float()
        batch = graph.batch
        node2seq = graph.node2seq.long()
        # [B, L, D]
        emb = self.seq_encoder(seq)
        emb = emb * mask.unsqueeze(-1).float()
        B, L, D = emb.shape
        if node2seq.min() < 0 or node2seq.max() >= L:
            raise ValueError(f"node2seq out of range: min={node2seq.min().item()}, max={node2seq.max().item()}, L={L}")
        emb_flat = emb.reshape(B * L, D)
        # 第 k 个节点对应到 batch[k] 这个图里的 node2seq[k] 位置
        flat_idx = batch * L + node2seq
        node_emb = emb_flat[flat_idx]  # [N, D]
        # 拼接节点属性
        x = pt.cat([node_emb, x], dim=-1)  # [N, D + Fx]
        x = self.gnn(x, edge_idx, edge_attr=edge_attr, batch=batch)
        x = gnn.global_mean_pool(x, batch)  # [B, hidden_dim]
        return x


class DualEncoderRetriever(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        query_d_model: int = 256,
        query_nhead: int = 8,
        query_num_layers: int = 1,
        gnn_dim: int = 256,
        gnn_num_layers: int = 1,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        self.query_encoder = SequenceEncoder(
            vocab_size=vocab_size,
            d_model=query_d_model,
            nhead=query_nhead,
            num_layers=query_num_layers,
            dropout=dropout,
        )
        self.cand_encoder = GraphEncoder(
            vocab_size=vocab_size,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            gnn_dim=gnn_dim,
            gnn_num_layers=gnn_num_layers,
            dropout=dropout,
        )
        self.normalize = normalize

    def encode_query(self, query_seqs, query_mask):
        q = self.query_encoder(query_seqs, query_mask)
        return q

    def encode_cand(self, cand_seqs, cand_masks, cand_graphs):
        c = self.cand_encoder(cand_seqs, cand_masks, cand_graphs)
        return c

    def encode(self, data, mode:str='query'): # 建库/检索时用
        if mode == 'query':
            query_seqs, query_masks = data['seqs_pad'], data['masks']
            emb = self.encode_query(query_seqs, query_masks)
        elif mode == 'cand':
            cand_seqs, cand_masks, cand_graphs = data['seqs_pad'], data['masks'], data['graphs']
            emb = self.encode_cand(cand_seqs, cand_masks, cand_graphs)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def forward(self, batch):
        query_seqs = batch["query_seqs"]
        query_masks = batch["query_masks"]
        cand_seqs = batch["cand_seqs"]
        cand_masks = batch["cand_masks"]
        cand_graphs = batch["cand_graphs"]
        scores = batch["scores"]
        q_emb = self.encode_query(query_seqs, query_masks)  # [B, D]
        c_emb = self.encode_cand(cand_seqs, cand_masks, cand_graphs)  # [B, D]
        cos_sim = F.cosine_similarity(q_emb, c_emb)
        target = (scores / 0.4).clamp(min=0.0, max=1.0)
        loss = F.mse_loss(cos_sim, target)
        return {
            "loss": loss,
        }