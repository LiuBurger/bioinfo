from typing import Dict
import math

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


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f'RoPE head dim must be even, but got dim={dim}.')
        self.dim = dim
        self.base = float(base)
        inv_freq = 1.0 / (self.base ** (pt.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _build_cache(self, seq_len: int, device: pt.device, dtype: pt.dtype):
        pos = pt.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = pt.outer(pos, self.inv_freq.to(device=device))
        emb = pt.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        self._seq_len_cached = seq_len
        self._cos_cached = cos[None, None, :, :]
        self._sin_cached = sin[None, None, :, :]

    def forward(self, seq_len: int, device: pt.device, dtype: pt.dtype):
        need_refresh = (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        )
        if need_refresh:
            self._build_cache(seq_len=seq_len, device=device, dtype=dtype)
        return self._cos_cached[:, :, :seq_len, :], self._sin_cached[:, :, :seq_len, :]

    @staticmethod
    def rotate_half(x: pt.Tensor) -> pt.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot = pt.stack((-x_odd, x_even), dim=-1)
        return x_rot.flatten(start_dim=-2)

    def apply(self, q: pt.Tensor, k: pt.Tensor):
        _, _, seq_len, _ = q.shape
        cos, sin = self(seq_len=seq_len, device=q.device, dtype=q.dtype)
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
        return q, k


class RoPEMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f'model_dim ({model_dim}) must be divisible by num_heads ({num_heads}).')
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f'RoPE requires an even head_dim, but got head_dim={self.head_dim}. '
                f'Please choose model_dim / num_heads so that each head dimension is even.'
            )
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(model_dim, 3 * model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(dim=self.head_dim, base=rope_base)

    def forward(self, x: pt.Tensor, key_padding_mask: pt.Tensor = None) -> pt.Tensor:
        bsz, seqlen, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(bsz, seqlen, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        q, k = self.rope.apply(q, k)

        attn_mask = None
        if key_padding_mask is not None:
            # F.scaled_dot_product_attention uses bool masks where True means
            # the key position is visible and False means masked out.
            attn_mask = (~key_padding_mask)[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
        out = out.permute(0, 2, 1, 3).contiguous().view(bsz, seqlen, self.model_dim)
        return self.out_proj(out)


class RoPETransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.self_attn = RoPEMultiheadSelfAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, model_dim),
        )

    def forward(self, x: pt.Tensor, key_padding_mask: pt.Tensor = None) -> pt.Tensor:
        x = x + self.dropout1(self.self_attn(self.norm1(x), key_padding_mask=key_padding_mask))
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class SequenceTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f'model_dim ({model_dim}) must be divisible by num_heads ({num_heads}).')
        if (model_dim // num_heads) % 2 != 0:
            raise ValueError(
                f'RoPE requires an even head dimension, but model_dim // num_heads = '
                f'{model_dim // num_heads}. Please adjust model_dim or num_heads.'
            )

        self.model_dim = model_dim
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        self.embed_norm = nn.LayerNorm(model_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([
            RoPETransformerEncoderLayer(
                model_dim=model_dim,
                num_heads=num_heads,
                dim_feedforward=4 * model_dim,
                dropout=dropout,
                rope_base=rope_base,
            )
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(model_dim)

    def forward(self, seq: pt.Tensor, mask: pt.Tensor):
        _, seqlen = seq.shape
        if seqlen > self.max_seq_len:
            raise ValueError(
                f'sequence length {seqlen} exceeds max_seq_len={self.max_seq_len}. '
                f'Increase max_seq_len in model config.'
            )

        x = self.token_embed(seq)
        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        key_padding_mask = (mask == 0)
        for layer in self.encoder:
            x = layer(x, key_padding_mask=key_padding_mask)

        x = self.out_norm(x)
        x = x * mask.unsqueeze(-1).float()
        return x

    @staticmethod
    def masked_mean(x: pt.Tensor, mask: pt.Tensor) -> pt.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        return x.sum(dim=1) / denom


class SharedSeqGraphEncoder(nn.Module):
    """
    共享主干：
    1) 浅层 Transformer 先编码序列, seq_dim = hidden_dim
    2) Transformer 输出通过 node2seq 对齐到节点
    3) 与图节点特征直接拼接
    4) 经过 3 层 GINE, GNN hidden_dim = seq_dim + node_feat_dim
    5) attention pooling 得到图级表示
    返回：
    - seq_global: [B, seq_dim]
    - graph_global: [B, graph_hidden_dim]
    """
    def __init__(
        self,
        vocab_size: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 512,
        gnn_num_layers: int = 3,
        transformer_num_layers: int = 1,
        transformer_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        attn_gate_hidden: int = None,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        seq_dim = hidden_dim
        graph_hidden_dim = seq_dim + node_feat_dim

        if seq_dim % transformer_heads != 0:
            raise ValueError(
                f'seq_dim ({seq_dim}) must be divisible by transformer_heads ({transformer_heads}).'
            )

        self.seq_dim = seq_dim
        self.graph_hidden_dim = graph_hidden_dim
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim

        self.seq_encoder = SequenceTransformer(
            vocab_size=vocab_size,
            model_dim=seq_dim,
            num_heads=transformer_heads,
            num_layers=transformer_num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )
        self.gnn = GINEBackbone(
            hidden_dim=graph_hidden_dim,
            edge_dim=edge_feat_dim,
            num_layers=gnn_num_layers,
            dropout=dropout,
        )
        self.node_out_norm = nn.LayerNorm(graph_hidden_dim)

        gate_hidden = attn_gate_hidden if attn_gate_hidden is not None else max(32, graph_hidden_dim // 2)
        self.attn_pool = gnn.AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(graph_hidden_dim, gate_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden, 1),
            )
        )
        self.graph_out_norm = nn.LayerNorm(graph_hidden_dim)

    def forward(self, seq: pt.Tensor, mask: pt.Tensor, graph):
        x = graph.x.float()
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr.float() if getattr(graph, 'edge_attr', None) is not None else None
        batch = graph.batch

        if not hasattr(graph, 'node2seq'):
            raise ValueError('graph must contain `node2seq` for sequence-to-node alignment.')
        node2seq = graph.node2seq.long()

        seq_out = self.seq_encoder(seq, mask)
        seq_global = SequenceTransformer.masked_mean(seq_out, mask)

        bsz, seqlen, d_model = seq_out.shape
        if node2seq.numel() > 0 and (node2seq.min() < 0 or node2seq.max() >= seqlen):
            raise ValueError(
                f'node2seq out of range: min={node2seq.min().item()}, '
                f'max={node2seq.max().item()}, L={seqlen}'
            )

        seq_flat = seq_out.reshape(bsz * seqlen, d_model)
        flat_idx = batch * seqlen + node2seq
        node_seq_feat = seq_flat[flat_idx]

        x = pt.cat([node_seq_feat, x], dim=-1)
        x = self.gnn(x, edge_index, edge_attr=edge_attr)
        x = self.node_out_norm(x)

        graph_global = self.attn_pool(x, batch)
        graph_global = self.graph_out_norm(graph_global)
        return seq_global, graph_global


class PairRegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(4 * input_dim),
            nn.Linear(4 * input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, a: pt.Tensor, b: pt.Tensor) -> pt.Tensor:
        feat = pt.cat([a, b, a * b, (a - b).abs()], dim=-1)
        return self.net(feat).squeeze(-1)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int, dropout: float = 0.1):
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


class Protein2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 512,
        proj_dim: int = 256,
        gnn_num_layers: int = 3,
        transformer_num_layers: int = 1,
        transformer_heads: int = 8,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        normalize: bool = True,
        temperature: float = 0.07,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_dim = hidden_dim
        self.graph_hidden_dim = hidden_dim + node_feat_dim
        self.proj_dim = proj_dim
        self.embed_dim = proj_dim
        self.normalize = normalize
        self.temperature = float(temperature)

        self.shared_encoder = SharedSeqGraphEncoder(
            vocab_size=vocab_size,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            gnn_num_layers=gnn_num_layers,
            transformer_num_layers=transformer_num_layers,
            transformer_heads=transformer_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )

        self.rank_head = ProjectionHead(
            input_dim=self.graph_hidden_dim,
            proj_dim=proj_dim,
            dropout=dropout,
        )

        self.seqid_head = PairRegressionHead(
            input_dim=self.seq_dim,
            hidden_dim=max(32, self.seq_dim // 2),
            dropout=dropout,
        )

        self.tmscore_head = PairRegressionHead(
            input_dim=self.graph_hidden_dim,
            hidden_dim=max(32, self.graph_hidden_dim // 2),
            dropout=dropout,
        )

    def _maybe_normalize(self, x: pt.Tensor) -> pt.Tensor:
        return F.normalize(x, p=2, dim=-1) if self.normalize else x

    def encode_backbone(self, seqs: pt.Tensor, masks: pt.Tensor, graphs):
        return self.shared_encoder(seqs, masks, graphs)

    def encode(self, data: Dict[str, pt.Tensor]) -> pt.Tensor:
        _, graph_repr = self.encode_backbone(data['seqs_pad'], data['masks'], data['graphs'])
        emb = self.rank_head(graph_repr)
        return self._maybe_normalize(emb)

    def forward(self, batch: Dict[str, pt.Tensor]):
        q_seq, q_graph = self.encode_backbone(
            batch['query_seqs'], batch['query_masks'], batch['query_graphs']
        )
        p_seq, p_graph = self.encode_backbone(
            batch['pos_seqs'], batch['pos_masks'], batch['pos_graphs']
        )

        q_rank = self._maybe_normalize(self.rank_head(q_graph))
        p_rank = self._maybe_normalize(self.rank_head(p_graph))
        sim_matrix = q_rank @ p_rank.transpose(0, 1)
        logits = sim_matrix / max(self.temperature, 1e-8)
        targets = pt.arange(logits.size(0), device=logits.device)
        info_nce_loss = F.cross_entropy(logits, targets)

        pos_score = sim_matrix.diag()
        if sim_matrix.size(0) > 1:
            neg_mask = pt.eye(sim_matrix.size(0), dtype=pt.bool, device=sim_matrix.device)
            neg_score = sim_matrix.masked_fill(neg_mask, float('-inf')).max(dim=-1).values
        else:
            neg_score = pt.zeros_like(pos_score)

        seqid_pred = self.seqid_head(q_seq, p_seq)
        tmscore_pred = self.tmscore_head(q_graph, p_graph)

        return {
            'info_nce_loss': info_nce_loss,
            'seqid_pred': seqid_pred,
            'tmscore_pred': tmscore_pred,
            'pos_score': pos_score,
            'neg_score': neg_score,
            'logits': logits,
            'q_seq': q_seq,
            'p_seq': p_seq,
            'q_graph': q_graph,
            'p_graph': p_graph,
            'q_rank': q_rank,
            'p_rank': p_rank,
        }


class MultiTaskOptimizer(pt.optim.AdamW):
    def __init__(
        self,
        ranker: nn.Module,
        lr: float,
        weight_decay: float = 0.0,
        alpha_tmscore: float = 0.2,
        alpha_seqid: float = 0.1,
        aux_loss_type: str = 'smooth_l1',
        aux_beta: float = 0.05,
    ):
        self.alpha_tmscore = float(alpha_tmscore)
        self.alpha_seqid = float(alpha_seqid)
        self.aux_loss_type = str(aux_loss_type)
        self.aux_beta = float(aux_beta)
        super().__init__(ranker.parameters(), lr=lr, weight_decay=weight_decay)

    def _reg_loss(self, pred: pt.Tensor, target: pt.Tensor) -> pt.Tensor:
        target = target.float()
        if self.aux_loss_type == 'smooth_l1':
            return F.smooth_l1_loss(pred, target, beta=self.aux_beta)
        if self.aux_loss_type == 'mse':
            return F.mse_loss(pred, target)
        if self.aux_loss_type == 'l1':
            return F.l1_loss(pred, target)
        raise ValueError("aux_loss_type must be one of {'smooth_l1', 'mse', 'l1'}")

    def combine_loss(
        self,
        rank_loss: pt.Tensor,
        tmscore_pred: pt.Tensor,
        tmscore_true: pt.Tensor,
        seqid_pred: pt.Tensor,
        seqid_true: pt.Tensor,
    ):
        tmscore_aux = self._reg_loss(tmscore_pred, tmscore_true)
        seqid_aux = self._reg_loss(seqid_pred, seqid_true)
        total_loss = rank_loss + self.alpha_tmscore * tmscore_aux + self.alpha_seqid * seqid_aux
        return total_loss, tmscore_aux, seqid_aux

    def _step(
        self,
        rank_loss: pt.Tensor,
        tmscore_pred: pt.Tensor,
        tmscore_true: pt.Tensor,
        seqid_pred: pt.Tensor,
        seqid_true: pt.Tensor,
        grad_clip: float = 1.0,
    ):
        loss, tmscore_aux, seqid_aux = self.combine_loss(
            rank_loss,
            tmscore_pred,
            tmscore_true,
            seqid_pred,
            seqid_true,
        )
        self.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = pt.nn.utils.clip_grad_norm_(self.param_groups[0]['params'], grad_clip)
        self.step()
        if not isinstance(grad_norm, pt.Tensor):
            grad_norm = pt.tensor(float(grad_norm))
        return {
            'loss': loss.detach(),
            'tmscore_aux': tmscore_aux.detach(),
            'seqid_aux': seqid_aux.detach(),
            'grad_norm': grad_norm.detach(),
        }
