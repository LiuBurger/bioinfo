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
        num_layers: int = 1,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        self.encoder = SequenceEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.normalize = normalize

    def encode(self, data):
        seqs, masks = data
        emb = self.encoder(seqs, masks)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def forward(self, batch):
        query_seqs = batch["query_seqs"]
        query_mask = batch["query_mask"]
        cand_seqs = batch["cand_seqs"]
        cand_mask = batch["cand_mask"]
        scores = batch["scores"]
        q_emb = self.encode((query_seqs, query_mask))  # [B, D]
        c_emb = self.encode((cand_seqs, cand_mask))  # [B, D]
        cos_sim = F.cosine_similarity(q_emb, c_emb)
        target = (scores / 0.4).clamp(min=0.0, max=1.0)
        loss = F.mse_loss(cos_sim, target)
        return {
            "loss": loss,
        }