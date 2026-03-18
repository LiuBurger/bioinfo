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
        if edge_attr is None:
            edge_attr = x.new_zeros(edge_idx.size(1), self.edge_dim)
        edge_attr = self.edge_encoder(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_idx, edge_attr)
            h = F.silu(h)
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
        self.seq_encoder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=gnn_dim - node_feat_dim,
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
        x = self.gnn(x, edge_idx, edge_attr=edge_attr, batch=batch)
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
        query_num_layers: int = 1,
        gnn_dim: int = 256,
        gnn_num_layers: int = 1,
        dropout: float = 0.1,
        normalize: bool = True,
        rank_target_temperature: float = 0.10,
        rank_pred_temperature: float = 0.05,
        lambda_pull: float = 0.05,
        lambda_var: float = 0.05,
        lambda_cov: float = 0.01,
        variance_gamma: float = 1.0,
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
        self.rank_target_temperature = rank_target_temperature
        self.rank_pred_temperature = rank_pred_temperature
        self.lambda_pull = lambda_pull
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.variance_gamma = variance_gamma

    def encode_query(self, query_seqs, query_mask):
        return self.query_encoder(query_seqs, query_mask)

    def encode_cand(self, cand_seqs, cand_masks, cand_graphs):
        return self.cand_encoder(cand_seqs, cand_masks, cand_graphs)

    def encode(self, data, mode: str = 'query'):
        if mode == 'query':
            query_seqs, query_masks = data['seqs_pad'], data['masks']
            emb = self.encode_query(query_seqs, query_masks)
        elif mode == 'cand':
            cand_seqs, cand_masks, cand_graphs = data['seqs_pad'], data['masks'], data['graphs']
            emb = self.encode_cand(cand_seqs, cand_masks, cand_graphs)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    @staticmethod
    def _masked_log_softmax(logits, mask, dim=-1):
        fill_value = pt.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask, fill_value)
        return F.log_softmax(logits, dim=dim)

    def _listwise_rank_loss(self, sims, scores, mask):
        """
        sims:   [B, K]
        scores: [B, K]
        mask:   [B, K]
        """
        target_logits = scores / max(self.rank_target_temperature, 1e-8)
        target_logits = target_logits.masked_fill(~mask, pt.finfo(scores.dtype).min)
        target_prob = F.softmax(target_logits, dim=-1)

        pred_logits = sims / max(self.rank_pred_temperature, 1e-8)
        pred_log_prob = self._masked_log_softmax(pred_logits, mask, dim=-1)

        per_query_loss = -(target_prob * pred_log_prob).sum(dim=-1)
        valid_query = mask.any(dim=-1)
        return per_query_loss[valid_query].mean()

    @staticmethod
    def _weighted_pull_loss(sims, scores, mask):
        weights = scores.clamp(min=0.0) * mask.float()
        weight_sum = weights.sum(dim=-1, keepdim=True)

        uniform_weights = mask.float()
        uniform_weights = uniform_weights / uniform_weights.sum(dim=-1, keepdim=True).clamp(min=1.0)

        weights = pt.where(
            weight_sum > 0,
            weights / weight_sum.clamp(min=1e-8),
            uniform_weights,
        )

        per_query_loss = ((1.0 - sims) * weights).sum(dim=-1)
        valid_query = mask.any(dim=-1)
        return per_query_loss[valid_query].mean()

    def _variance_loss(self, z, eps=1e-4):
        if z.size(0) <= 1:
            return z.new_tensor(0.0)
        std = pt.sqrt(z.var(dim=0, unbiased=False) + eps)
        return F.relu(self.variance_gamma - std).mean()

    @staticmethod
    def _covariance_loss(z):
        if z.size(0) <= 1:
            return z.new_tensor(0.0)
        z = z - z.mean(dim=0, keepdim=True)
        n, d = z.shape
        cov = (z.T @ z) / max(n - 1, 1)
        off_diag = cov - pt.diag(pt.diag(cov))
        return off_diag.pow(2).sum() / d

    def forward(self, batch):
        query_seqs = batch["query_seqs"]
        query_masks = batch["query_masks"]
        cand_seqs = batch["cand_seqs"]
        cand_masks = batch["cand_masks"]
        cand_graphs = batch["cand_graphs"]
        scores = batch["scores"]                  # [B, K]
        pos_valid_mask = batch["pos_valid_mask"]  # [B, K]

        B, K = scores.shape

        q_emb = self.encode_query(query_seqs, query_masks)              # [B, D]
        c_emb = self.encode_cand(cand_seqs, cand_masks, cand_graphs)    # [B*K, D]
        c_emb = c_emb.view(B, K, -1)

        if self.normalize:
            q_emb = F.normalize(q_emb, p=2, dim=-1)
            c_emb = F.normalize(c_emb, p=2, dim=-1)

        sims = (q_emb.unsqueeze(1) * c_emb).sum(dim=-1)                 # [B, K]

        rank_loss = self._listwise_rank_loss(sims, scores, pos_valid_mask)
        pull_loss = self._weighted_pull_loss(sims, scores, pos_valid_mask)

        q_reg = q_emb
        c_reg = c_emb[pos_valid_mask]  # [N_valid, D]

        var_loss = self._variance_loss(q_reg) + self._variance_loss(c_reg)
        cov_loss = self._covariance_loss(q_reg) + self._covariance_loss(c_reg)

        loss = (
            rank_loss
            + self.lambda_pull * pull_loss
            + self.lambda_var * var_loss
            + self.lambda_cov * cov_loss
        )

        valid_sims = sims[pos_valid_mask]
        sim_mean = valid_sims.mean() if valid_sims.numel() > 0 else sims.new_tensor(0.0)
        sim_std = valid_sims.std(unbiased=False) if valid_sims.numel() > 1 else sims.new_tensor(0.0)

        return {
            "loss": loss,
            "rank_loss": rank_loss.detach(),
            "pull_loss": pull_loss.detach(),
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach(),
            "sim_mean": sim_mean.detach(),
            "sim_std": sim_std.detach(),
        }