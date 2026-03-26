from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch


@dataclass
class SequenceRecord:
    id: str
    sequence: str
    global_idx: int | None = None


class OfficialDHRCheckpoint:
    """
    Minimal loader for the official DHR checkpoint format.

    The official repo loads the checkpoints with:
        bert1, alphabet = torch.load('dhr_qencoder.pt')
        bert2, _ = torch.load('dhr_cencoder.pt')

    So each checkpoint file is expected to contain a tuple of (ESM_model, alphabet).
    """

    def __init__(self, ckpt_dir: str, device: str = "cpu"):
        self.ckpt_dir = Path(ckpt_dir)
        self.device = torch.device(device)
        q_path = self.ckpt_dir / "dhr_qencoder.pt"
        c_path = self.ckpt_dir / "dhr_cencoder.pt"
        if not q_path.is_file():
            raise FileNotFoundError(f"Query checkpoint not found: {q_path}")
        if not c_path.is_file():
            raise FileNotFoundError(f"Candidate checkpoint not found: {c_path}")

        # weights_only=False is required because these files store Python objects,
        # not a plain state_dict.
        self.q_model, self.q_alphabet = torch.load(q_path, map_location="cpu", weights_only=False)
        self.c_model, self.c_alphabet = torch.load(c_path, map_location="cpu", weights_only=False)

        self.q_model = self.q_model.eval().to(self.device)
        self.c_model = self.c_model.eval().to(self.device)
        self.q_repr_layer = int(self.q_model.num_layers)
        self.c_repr_layer = int(self.c_model.num_layers)
        self.q_batch_converter = self.q_alphabet.get_batch_converter()
        self.c_batch_converter = self.c_alphabet.get_batch_converter()

    @torch.no_grad()
    def _encode_batch(self, records: Sequence[SequenceRecord], tower: str) -> np.ndarray:
        if tower == "query":
            batch_converter = self.q_batch_converter
            model = self.q_model
            repr_layer = self.q_repr_layer
        elif tower == "candidate":
            batch_converter = self.c_batch_converter
            model = self.c_model
            repr_layer = self.c_repr_layer
        else:
            raise ValueError("tower must be 'query' or 'candidate'.")

        data = [(rec.id, rec.sequence) for rec in records]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(self.device)
        reps = model(tokens, repr_layers=[repr_layer])["representations"][repr_layer]
        cls_emb = reps[:, 0].detach().cpu().numpy().astype(np.float32)
        return cls_emb

    @torch.no_grad()
    def encode_records(
        self,
        records: Sequence[SequenceRecord],
        tower: str,
        batch_size: int = 8,
        progress: bool = True,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if len(records) == 0:
            raise ValueError("records must not be empty.")

        try:
            from tqdm import tqdm
        except Exception:  # pragma: no cover
            tqdm = None

        iterator: Iterable[int] = range(0, len(records), batch_size)
        if progress and tqdm is not None:
            iterator = tqdm(iterator, total=(len(records) + batch_size - 1) // batch_size, desc=f"encode-{tower}")

        chunks: List[np.ndarray] = []
        for start in iterator:
            batch = records[start:start + batch_size]
            chunks.append(self._encode_batch(batch, tower=tower))

        return np.concatenate(chunks, axis=0)
