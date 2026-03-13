import math
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
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
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
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
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled

    def forward_tokens(self, input_ids, attention_mask):
        """
        返回 token-level 表示，给 GraphEncoder 用
        """
        x = self.token_embedding(input_ids)
        x = self.pos_embedding(x)
        key_padding_mask = (attention_mask == 0)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x  # [B, L, D]


class GINEBackbone(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        edge_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout
        self.node_in_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_encoder = (
            nn.Linear(edge_dim, hidden_dim) if edge_dim != hidden_dim else nn.Identity()
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
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
        x = self.node_in_proj(x)
        edge_attr = self.edge_encoder(edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_idx, edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = norm(x + h)  # residual
        return x


class GraphEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        seq_d_model: int = 256,
        seq_nhead: int = 8,
        seq_num_layers: int = 2,
        seq_ffn_dim: int = 512,
        gnn_hidden_dim: int = 256,
        gnn_num_layers: int = 3,
        proj_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        # candidate sequence encoder
        self.seq_encoder = SequenceEncoder(
            vocab_size=vocab_size,
            d_model=seq_d_model,
            nhead=seq_nhead,
            num_layers=seq_num_layers,
            dim_feedforward=seq_ffn_dim,
            dropout=dropout,
        )
        gnn_input_dim = seq_d_model + node_feat_dim
        self.gnn = GINEBackbone(
            in_dim=gnn_input_dim,
            hidden_dim=gnn_hidden_dim,
            edge_dim=edge_feat_dim,
            num_layers=gnn_num_layers,
            dropout=dropout,
        )
        self.out_proj = nn.Linear(gnn_hidden_dim, proj_dim)

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
        if node2seq.min() < 0 or node2seq.max() >= L:
            raise ValueError(f"node2seq out of range: min={node2seq.min().item()}, max={node2seq.max().item()}, L={L}")
        # [B, L, D]
        emb = self.seq_encoder.forward_tokens(seq, mask)
        B, L, D = emb.shape
        emb_flat = emb.reshape(B * L, D)
        # 第 k 个节点对应到 batch[k] 这个图里的 node2seq[k] 位置
        flat_idx = batch * L + node2seq
        node_emb = emb_flat[flat_idx]  # [N, D]
        # 拼接节点属性
        x = pt.cat([node_emb, x], dim=-1)  # [N, D + Fx]
        x = self.gnn(x, edge_idx, edge_attr=edge_attr, batch=batch)
        x = gnn.global_mean_pool(x, batch)  # [B, hidden_dim]
        x = self.out_proj(x)
        return x


class DualEncoderRetriever(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        query_d_model: int = 256,
        query_nhead: int = 8,
        query_num_layers: int = 4,
        query_ffn_dim: int = 512,
        cand_seq_num_layers: int = 2,
        gnn_hidden_dim: int = 256,
        gnn_num_layers: int = 3,
        proj_dim: int = 256,
        dropout: float = 0.1,
        temperature: float = 0.07,
        normalize: bool = True,
        symmetric_loss: bool = True,
    ):
        super().__init__()
        self.query_encoder = SequenceEncoder(
            vocab_size=vocab_size,
            d_model=query_d_model,
            nhead=query_nhead,
            num_layers=query_num_layers,
            dim_feedforward=query_ffn_dim,
            dropout=dropout,
        )
        self.candidate_encoder = GraphEncoder(
            vocab_size=vocab_size,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            seq_d_model=query_d_model,
            seq_nhead=query_nhead,
            seq_num_layers=cand_seq_num_layers,
            seq_ffn_dim=query_ffn_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            proj_dim=proj_dim,
            dropout=dropout,
        )
        self.query_proj = nn.Linear(query_d_model, proj_dim)
        self.temperature = temperature
        self.normalize = normalize
        self.symmetric_loss = symmetric_loss

    def encode_query(self, query_ids, query_mask):
        q = self.query_encoder(query_ids, query_mask)
        q = self.query_proj(q)
        if self.normalize:
            q = F.normalize(q, p=2, dim=-1)
        return q

    def encode_candidate(self, cand_ids, cand_mask, candidate_graph):
        c = self.candidate_encoder(cand_ids, cand_mask, candidate_graph)
        if self.normalize:
            c = F.normalize(c, p=2, dim=-1)
        return c
    
    def encode(self, data, mode:str='lib'):
        if mode == 'lib':
            cand_ids, cand_mask, candidate_graph = data
            return self.encode_candidate(cand_ids, cand_mask, candidate_graph)
        elif mode == 'query':
            query_ids, query_mask = data
            return self.encode_query(query_ids, query_mask)
        else:
            raise ValueError(f"mode '{mode}' not exist")

    def forward(self, batch):
        """
        batch 中包含一个 mini-batch 的正样本对：
        第 i 个 query <-> 第 i 个 candidate 为正样本
        同 batch 里其他 candidate / query 自动视为负样本
        batch 应包含：
        - query_ids:      [B, Lq]
        - query_mask:     [B, Lq]
        - cand_ids:       [B, Lc]
        - cand_mask:      [B, Lc]
        - candidate_graph: PyG Batch,对应 B 个 candidate 图
        返回：
        - loss: InfoNCE 总损失
        - 以及一些监控指标
        """
        query_ids = batch["query_ids"]
        query_mask = batch["query_mask"]
        cand_ids = batch["cand_ids"]
        cand_mask = batch["cand_mask"]
        candidate_graph = batch["candidate_graph"]
        q_emb = self.encode_query(query_ids, query_mask)  # [B, D]
        c_emb = self.encode_candidate(cand_ids, cand_mask, candidate_graph)  # [B, D]
        """
        相似度计算,logits[i, j]表示：第 i 个 query 和第 j 个 candidate 的相似度
        temperature 是对 softmax 的“锐化系数”
            温度小,比如 0.07,softmax 更尖锐，更强调 hardest negatives
            温度大,softmax 更平滑
        对角线位置 logits[i, i] 是正样本
        """
        logits = pt.matmul(q_emb, c_emb.t()) / self.temperature # [B, B]
        B = logits.size(0)
        targets = pt.arange(B, device=logits.device)
        # query -> candidate 方向的 InfoNCE
        loss_q2c = F.cross_entropy(logits, targets)
        # candidate -> query 方向的 InfoNCE（可选）
        if self.symmetric_loss:
            loss_c2q = F.cross_entropy(logits.t(), targets)
            loss = 0.5 * (loss_q2c + loss_c2q) # 取平均
        else:
            loss_c2q = pt.tensor(0.0, device=logits.device)
            loss = loss_q2c
        # 计算一些训练监控指标
        with pt.no_grad():
            # query->candidate 检索准确率
            acc_q2c = (logits.argmax(dim=1) == targets).float().mean() # query->candidate 检索准确率
            # candidate->query 检索准确率
            acc_c2q = (logits.t().argmax(dim=1) == targets).float().mean() # candidate->query 检索准确率
            # 正样本平均 logit（对角线）
            pos_logit_mean = logits.diag().mean()
            # 负样本平均 logit（非对角线）
            neg_mask = ~pt.eye(B, dtype=pt.bool, device=logits.device)
            neg_logit_mean = logits.masked_select(neg_mask).mean()

        return {
            "loss": loss,                              # 用于反向传播
            "loss_q2c": loss_q2c.detach(),             # query->candidate 的损失
            "loss_c2q": loss_c2q.detach(),             # candidate->query 的损失
            "retrieval_acc_q2c": acc_q2c.detach(),     # q->c 检索准确率
            "retrieval_acc_c2q": acc_c2q.detach(),     # c->q 检索准确率
            "pos_logit_mean": pos_logit_mean.detach(), # 正样本平均相似度
            "neg_logit_mean": neg_logit_mean.detach(), # 负样本平均相似度
            "logits": logits,                          # [B, B] 相似度矩阵
        }