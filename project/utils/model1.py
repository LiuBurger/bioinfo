import math
from typing import Dict

import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = pt.zeros(max_len, d_model)
        position = pt.arange(0, max_len, dtype=pt.float).unsqueeze(1)
        div_term = pt.exp(pt.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = pt.sin(position * div_term)
        pe[:, 1::2] = pt.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ProjectionHead(nn.Module):
    def __init__(self, dim: int, proj_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if proj_dim is None:
            proj_dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, proj_dim),
        )
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x):
        x = self.net(x)
        x = self.norm(x)
        return x


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 1300,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
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
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim) if edge_dim != hidden_dim else nn.Identity()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            conv = gnn.GINEConv(nn=mlp, train_eps=True, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_idx, edge_attr=None):
        if edge_attr is None:
            edge_attr = x.new_zeros(edge_idx.size(1), self.edge_dim)
        edge_attr = self.edge_encoder(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_idx, edge_attr)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = norm(x + h)
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
        seq_dim = gnn_dim - node_feat_dim
        if seq_dim <= 0:
            raise ValueError(f"gnn_dim ({gnn_dim}) must be larger than node_feat_dim ({node_feat_dim}).")

        self.seq_encoder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=seq_dim,
            padding_idx=0
        )
        self.gnn = GINEBackbone(
            hidden_dim=gnn_dim,
            edge_dim=edge_feat_dim,
            num_layers=gnn_num_layers,
            dropout=dropout,
        )

    def forward(self, seq, mask, graph):
        x = graph.x.float()
        edge_idx = graph.edge_index
        edge_attr = graph.edge_attr.float()
        batch = graph.batch
        node2seq = graph.node2seq.long()

        emb = self.seq_encoder(seq)
        emb = emb * mask.unsqueeze(-1).float()
        B, L, D = emb.shape

        if node2seq.min() < 0 or node2seq.max() >= L:
            raise ValueError(
                f"node2seq out of range: min={node2seq.min().item()}, max={node2seq.max().item()}, L={L}"
            )

        emb_flat = emb.reshape(B * L, D)
        flat_idx = batch * L + node2seq
        node_emb = emb_flat[flat_idx]

        x = pt.cat([node_emb, x], dim=-1)
        x = self.gnn(x, edge_idx, edge_attr=edge_attr)
        x = gnn.global_mean_pool(x, batch)
        return x


class DualEncoderRetriever(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        query_d_model: int = 256,
        query_nhead: int = 8,
        query_num_layers: int = 3,
        gnn_dim: int = 256,
        gnn_num_layers: int = 3,
        dropout: float = 0.1,
        normalize: bool = True,
        margin: float = 0.2,
    ):
        super().__init__()
        if query_d_model != gnn_dim:
            raise ValueError('query_d_model must equal gnn_dim because query/candidate embeddings are compared directly.')
        self.embed_dim = query_d_model
        self.normalize = normalize
        self.margin = margin

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

        self.tm_head = nn.Linear(self.embed_dim, 1)
        self.seqid_head = nn.Linear(self.embed_dim, 1)

    def encode_query(self, query_seqs, query_mask):
        return self.query_encoder(query_seqs, query_mask)

    def encode_cand(self, cand_seqs, cand_masks, cand_graphs):
        return self.cand_encoder(cand_seqs, cand_masks, cand_graphs)

    def encode(self, data, mode: str = 'query'):
        if mode == 'query':
            emb = self.encode_query(data['seqs_pad'], data['masks'])
        elif mode == 'cand':
            emb = self.encode_cand(data['seqs_pad'], data['masks'], data['graphs'])
        else:
            raise ValueError("mode must be either 'query' or 'cand'.")
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def _maybe_normalize(self, x):
        return F.normalize(x, p=2, dim=-1) if self.normalize else x

    def _pair_regression(self, q_emb, c_emb):
        pair_feat = q_emb * c_emb
        tm_pred = pt.sigmoid(self.tm_head(pair_feat)).squeeze(-1)
        seqid_pred = pt.sigmoid(self.seqid_head(pair_feat)).squeeze(-1)
        return tm_pred, seqid_pred

    def forward(self, batch: Dict[str, pt.Tensor]):
        q_emb = self.encode_query(batch['query_seqs'], batch['query_masks'])
        pos_emb = self.encode_cand(batch['pos_seqs'], batch['pos_masks'], batch['pos_graphs'])
        neg_emb = self.encode_cand(batch['neg_seqs'], batch['neg_masks'], batch['neg_graphs'])

        q_rank = self._maybe_normalize(q_emb)
        pos_rank = self._maybe_normalize(pos_emb)
        neg_rank = self._maybe_normalize(neg_emb)

        pos_score = (q_rank * pos_rank).sum(dim=-1)
        neg_score = (q_rank * neg_rank).sum(dim=-1)
        bpr_loss = F.softplus(neg_score - pos_score).mean()

        tm_pred, seqid_pred = self._pair_regression(q_emb, pos_emb)
        out = {
            'bpr_loss': bpr_loss,
            'tm_pred': tm_pred,
            'seqid_pred': seqid_pred,
            'pos_score': pos_score,
            'neg_score': neg_score,
        }
        if 'tmscores' in batch:
            out['tm_l1'] = F.l1_loss(tm_pred, batch['tmscores'])
        if 'seqids' in batch:
            out['seqid_l1'] = F.l1_loss(seqid_pred, batch['seqids'])
        return out


class MultiBPROptimizer(pt.optim.AdamW):
    """代码1风格的 BPR + TM/SeqID 辅助优化器。"""

    def __init__(self, ranker: nn.Module, lr: float, weight_decay: float = 0.0, alpha: float = 0.0):
        self.alpha = alpha
        super().__init__(ranker.parameters(), lr=lr, weight_decay=weight_decay)

    def combine_loss(self, bpr_loss, tm_pred, tm_true, seqid_pred, seqid_true):
        aux_loss = pt.mean(pt.abs(tm_pred - tm_true) + pt.abs(seqid_pred - seqid_true))
        return bpr_loss + self.alpha * aux_loss, aux_loss

    def _step(
        self,
        bpr_loss: pt.Tensor,
        tmscore_pred: pt.Tensor,
        tmscore_true: pt.Tensor,
        seqid_pred: pt.Tensor,
        seqid_true: pt.Tensor,
        grad_clip: float = 1.0,
    ):
        loss, aux_loss = self.combine_loss(bpr_loss, tmscore_pred, tmscore_true, seqid_pred, seqid_true)
        self.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = pt.nn.utils.clip_grad_norm_(self.param_groups[0]['params'], grad_clip)
        self.step()
        if not isinstance(grad_norm, pt.Tensor):
            grad_norm = pt.tensor(float(grad_norm))
        return {
            'loss': loss.detach(),
            'aux_loss': aux_loss.detach(),
            'grad_norm': grad_norm.detach(),
        }


class CosineAnnealingWarmRestartsWarmup(pt.optim.lr_scheduler._LRScheduler):
    """
    代码1风格：warmup + cosine annealing + warm restart。
    以 epoch 为步长调用 scheduler.step()。
    """

    def __init__(
        self,
        optimizer,
        T_0,
        T_mult=1,
        eta_min=0.0,
        last_epoch=-1,
        warmup=0,
        decay=1.0,
    ):
        self.T_0 = int(T_0)
        self.T_mult = int(T_mult)
        self.eta_min = float(eta_min)
        self.warmup = int(warmup)
        self.decay = float(decay)
        self.cycle_length = max(1, self.T_0)
        self.cycle_start = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if self.warmup > 0 and epoch < self.warmup:
            warm_ratio = float(epoch + 1) / float(self.warmup)
            return [base_lr * warm_ratio for base_lr in self.base_lrs]

        effective_epoch = epoch - self.warmup
        cycle_len = self.cycle_length
        cycle_start = self.cycle_start
        cycle_idx = 0
        while effective_epoch >= cycle_start + cycle_len:
            cycle_start += cycle_len
            cycle_len = max(1, cycle_len * self.T_mult)
            cycle_idx += 1
        self.cycle_start = cycle_start
        self.cycle_length = cycle_len

        cycle_pos = effective_epoch - cycle_start
        decay_factor = self.decay ** cycle_idx
        lrs = []
        for base_lr in self.base_lrs:
            cur_base = base_lr * decay_factor
            lr = self.eta_min + (cur_base - self.eta_min) * (1 + math.cos(math.pi * cycle_pos / cycle_len)) / 2.0
            lrs.append(lr)
        return lrs


CosineAnnealingWarmRestarts_Warmup = CosineAnnealingWarmRestartsWarmup