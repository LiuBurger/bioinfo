import math
import torch as pt
import torch.nn as nn
import torch.nn.functional as F


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


class DualEncoderRetriever(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.1,
        normalize: bool = True,
        symmetric_loss: bool = True,
    ):
        super().__init__()
        self.query_encoder = SequenceEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.candidate_encoder = SequenceEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.temperature = temperature
        self.normalize = normalize
        self.symmetric_loss = symmetric_loss        
    
    def encode(self, data, mode:str='query'):
        ids, masks = data
        if mode == 'query':
            emb = self.query_encoder(ids, masks)
        elif mode == 'candidate':
            emb = self.candidate_encoder(ids, masks)
        else:
            raise ValueError(f"mode '{mode}' not exist")
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

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
        q_emb = self.query_encoder(query_ids, query_mask)  # [B, D]
        c_emb = self.candidate_encoder(cand_ids, cand_mask)  # [B, D]
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