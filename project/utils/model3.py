from typing import Dict

import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


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
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            conv = gnn.GINEConv(nn=mlp, train_eps=True, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = x.new_zeros(edge_index.size(1), self.edge_dim)
        edge_attr = self.edge_encoder(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr)
            h = F.silu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = norm(x + h)
        return x


class SharedGraphEncoder(nn.Module):
    """
    共享主干：
    - sequence embedding
    - graph backbone
    - graph pooling
    输出的是共享语义空间下的 pooled graph representation
    """
    def __init__(
        self,
        vocab_size: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = 'mean',
    ):
        super().__init__()
        seq_dim = hidden_dim - node_feat_dim
        if seq_dim <= 0:
            raise ValueError(
                f'hidden_dim ({hidden_dim}) must be larger than node_feat_dim ({node_feat_dim}) '
                f'to allow sequence embedding concatenation.'
            )

        self.hidden_dim = hidden_dim
        self.pooling = pooling

        self.seq_encoder = nn.Embedding(vocab_size, seq_dim, padding_idx=0)
        self.gnn = GINEBackbone(
            hidden_dim=hidden_dim,
            edge_dim=edge_feat_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, seq: pt.Tensor, mask: pt.Tensor, graph):
        x = graph.x.float()
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr.float() if getattr(graph, 'edge_attr', None) is not None else None
        batch = graph.batch

        if not hasattr(graph, 'node2seq'):
            raise ValueError('graph must contain `node2seq` for sequence-to-node alignment.')
        node2seq = graph.node2seq.long()

        emb = self.seq_encoder(seq)                      # [B, L, seq_dim]
        emb = emb * mask.unsqueeze(-1).float()
        B, L, D = emb.shape

        if node2seq.numel() > 0 and (node2seq.min() < 0 or node2seq.max() >= L):
            raise ValueError(
                f'node2seq out of range: min={node2seq.min().item()}, '
                f'max={node2seq.max().item()}, L={L}'
            )

        emb_flat = emb.reshape(B * L, D)                # [B*L, D]
        flat_idx = batch * L + node2seq                 # [num_nodes]
        node_emb = emb_flat[flat_idx]                   # [num_nodes, D]

        x = pt.cat([node_emb, x], dim=-1)               # [num_nodes, hidden_dim]
        x = self.gnn(x, edge_index, edge_attr=edge_attr)
        x = self.out_norm(x)

        if self.pooling == 'mean':
            x = gnn.global_mean_pool(x, batch)
        elif self.pooling == 'add':
            x = gnn.global_add_pool(x, batch)
        elif self.pooling == 'max':
            x = gnn.global_max_pool(x, batch)
        else:
            raise ValueError("pooling must be one of {'mean', 'add', 'max'}")

        return x


class ProjectionHead(nn.Module):
    """
    轻量 projection head：
    backbone 学共享语义空间
    projection head 学检索空间
    """
    def __init__(
        self,
        input_dim: int,
        proj_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        return self.net(x)


class DualEncoderRetriever(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        normalize: bool = True,
        temperature: float = 0.07,
        pooling: str = 'mean',
        proj_dim: int = None,   # 新增：projection 维度；默认等于 hidden_dim
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = int(proj_dim) if proj_dim is not None else hidden_dim
        self.embed_dim = self.proj_dim
        self.normalize = normalize
        self.temperature = float(temperature)

        # 共享主干：query / candidate 完全共用
        self.shared_encoder = SharedGraphEncoder(
            vocab_size=vocab_size,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
        )

        # 两个轻量头：让 query space / candidate space 有轻微可调性
        self.query_proj = ProjectionHead(
            input_dim=hidden_dim,
            proj_dim=self.proj_dim,
            dropout=dropout,
        )
        self.cand_proj = ProjectionHead(
            input_dim=hidden_dim,
            proj_dim=self.proj_dim,
            dropout=dropout,
        )

        # TM 辅助头：建议作用在 backbone 语义空间，而不是 projection 后空间
        tm_hidden = max(1, hidden_dim // 2)
        self.tm_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, tm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tm_hidden, 1),
        )

    def _maybe_normalize(self, x: pt.Tensor) -> pt.Tensor:
        return F.normalize(x, p=2, dim=-1) if self.normalize else x

    # ===== backbone features =====
    def _encode_backbone(self, seqs: pt.Tensor, masks: pt.Tensor, graphs):
        return self.shared_encoder(seqs, masks, graphs)

    def encode_query_backbone(self, query_seqs, query_masks, query_graphs):
        return self._encode_backbone(query_seqs, query_masks, query_graphs)

    def encode_cand_backbone(self, cand_seqs, cand_masks, cand_graphs):
        return self._encode_backbone(cand_seqs, cand_masks, cand_graphs)

    # ===== retrieval embeddings =====
    def encode_query(self, query_seqs, query_masks, query_graphs):
        q_base = self.encode_query_backbone(query_seqs, query_masks, query_graphs)
        q_proj = self.query_proj(q_base)
        return q_proj

    def encode_cand(self, cand_seqs, cand_masks, cand_graphs):
        c_base = self.encode_cand_backbone(cand_seqs, cand_masks, cand_graphs)
        c_proj = self.cand_proj(c_base)
        return c_proj

    def encode(self, data: Dict[str, pt.Tensor], mode: str = 'query'):
        """
        保持和你原训练/评估代码兼容：
        - mode='query' -> query tower embedding
        - mode='cand'  -> candidate tower embedding
        返回用于检索的 embedding（可选归一化）
        """
        if mode == 'query':
            emb = self.encode_query(data['seqs_pad'], data['masks'], data['graphs'])
        elif mode == 'cand':
            emb = self.encode_cand(data['seqs_pad'], data['masks'], data['graphs'])
        else:
            raise ValueError("mode must be either 'query' or 'cand'.")
        return self._maybe_normalize(emb)

    def _tm_regression(self, q_base: pt.Tensor, c_base: pt.Tensor):
        pair_feat = q_base * c_base
        tm_logit = self.tm_head(pair_feat).squeeze(-1)
        tm_prob = pt.sigmoid(tm_logit)
        return tm_logit, tm_prob

    def forward(self, batch: Dict[str, pt.Tensor]):
        # 1) 共享 backbone 输出
        q_base = self.encode_query_backbone(
            batch['query_seqs'], batch['query_masks'], batch['query_graphs']
        )
        c_base = self.encode_cand_backbone(
            batch['pos_seqs'], batch['pos_masks'], batch['pos_graphs']
        )

        # 2) 进入各自 projection head，得到检索空间 embedding
        q_emb = self.query_proj(q_base)
        pos_emb = self.cand_proj(c_base)

        q_rank = self._maybe_normalize(q_emb)
        pos_rank = self._maybe_normalize(pos_emb)

        # 3) InfoNCE
        sim_matrix = q_rank @ pos_rank.transpose(0, 1)
        logits = sim_matrix / max(self.temperature, 1e-8)
        targets = pt.arange(logits.size(0), device=logits.device)
        info_nce_loss = F.cross_entropy(logits, targets)

        pos_score = sim_matrix.diag()
        if sim_matrix.size(0) > 1:
            neg_mask = pt.eye(sim_matrix.size(0), dtype=pt.bool, device=sim_matrix.device)
            neg_score = sim_matrix.masked_fill(neg_mask, float('-inf')).max(dim=-1).values
        else:
            neg_score = pt.zeros_like(pos_score)

        # 4) 辅助 TM 头仍然看 backbone 语义空间
        tm_logit, tm_prob = self._tm_regression(q_base, c_base)

        out = {
            'info_nce_loss': info_nce_loss,
            'tm_logit': tm_logit,
            'tm_prob': tm_prob,
            'pos_score': pos_score,
            'neg_score': neg_score,
            'logits': logits,

            # 下面这些不是训练脚本必须项，但后面做调试会很有用
            'q_base': q_base,
            'c_base': c_base,
            'q_emb': q_emb,
            'pos_emb': pos_emb,
        }
        return out


class MultiTaskOptimizer(pt.optim.AdamW):
    def __init__(
        self,
        ranker: nn.Module,
        lr: float,
        weight_decay: float = 0.0,
        alpha: float = 0.0,
        tm_threshold: float = 0.6,
    ):
        self.alpha = float(alpha)
        self.tm_threshold = float(tm_threshold)
        super().__init__(ranker.parameters(), lr=lr, weight_decay=weight_decay)

    def combine_loss(self, rank_loss: pt.Tensor, tm_logit: pt.Tensor, tm_true: pt.Tensor):
        tm_label = (tm_true >= self.tm_threshold).float()
        aux_loss = F.binary_cross_entropy_with_logits(tm_logit, tm_label)
        return rank_loss + self.alpha * aux_loss, aux_loss, tm_label

    def _step(
        self,
        rank_loss: pt.Tensor,
        tm_logit: pt.Tensor,
        tm_true: pt.Tensor,
        grad_clip: float = 1.0,
    ):
        loss, aux_loss, tm_label = self.combine_loss(rank_loss, tm_logit, tm_true)
        self.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = pt.nn.utils.clip_grad_norm_(self.param_groups[0]['params'], grad_clip)
        self.step()
        if not isinstance(grad_norm, pt.Tensor):
            grad_norm = pt.tensor(float(grad_norm))
        return {
            'loss': loss.detach(),
            'aux_loss': aux_loss.detach(),
            'tm_label': tm_label.detach(),
            'grad_norm': grad_norm.detach(),
        }