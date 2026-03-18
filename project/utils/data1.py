import h5py
import torch as pt
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch


def load(fn, mode: str = 'all'):
    with h5py.File(fn) as f:
        seq = f['node_seq'][()].astype(np.int32)
        node_idx = f['node_idx'][()]
        if mode == 'all':
            node_ss = f['node_ss'][()].astype(np.int32)
            node_rsa = f['node_rsa'][()].astype(np.float32)
            lab = f['label'][()].astype(str)
        if mode in ['all', 'part']:
            node_pos = f['node_pos'][()].astype(np.float32)
            edge_nho = f['edge_nho'][()].astype(np.int32)
            edge_idx = f['edge_idx'][()]
        if mode in ['part', 'seq']:
            lab = f['label'][()]
        f.close()

    if mode == 'all':
        return seq, node_ss, node_rsa, node_pos, node_idx, edge_nho, edge_idx, lab
    elif mode == 'part':
        return seq, node_pos, node_idx, edge_nho, edge_idx, lab
    elif mode == 'seq':
        return seq, node_idx, lab
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be one of 'all', 'part', or 'seq'.")


class ProteinDataset(Dataset):
    def __init__(self, dataset, mapping: np.ndarray = None, mode: str = 'query'):
        super().__init__()
        self.mode = mode
        if isinstance(dataset, tuple):  # raw data
            self.seq = dataset[0]
            self.lab = dataset[1]
            if mode == 'cand':
                self.graph = dataset[2]
            self.map = np.arange(len(self.lab), dtype=np.int64)
            assert len(self.seq) == len(self.lab)
        else:  # structured data
            assert mapping is not None, "Mapping must be provided for structured data."
            self.seq = dataset.seq
            self.lab = dataset.lab
            if mode == 'cand':
                self.graph = dataset.graph
            self.map = mapping
            assert np.max(self.map) < len(self.lab)

    def __getitem__(self, idx):
        idx = self.map[idx]
        data = {}
        data['seq'], data['lab'] = self.seq[idx], self.lab[idx]
        if self.mode == 'cand':
            data["graph"] = self.graph[idx]
        return data

    def __len__(self):
        return len(self.map)


class QueryHomologyDataset(Dataset):
    def __init__(self, dataset, pair_file: str = None, pdb2idx: dict = None, mapping: np.ndarray = None):
        super().__init__()
        if isinstance(dataset, ProteinDataset):
            assert pair_file is not None and pdb2idx is not None, "pair_file and pdb2idx must be provided for raw data."
            df = pd.read_csv(pair_file, names=['name1', 'name2', 'tmscore', 'seqid'], delimiter='\t')
            df['name1'] = df['name1'].str.split('/').str[-1].str.removesuffix('.pdb')
            df['name2'] = df['name2'].str.split('/').str[-1].str.removesuffix('.pdb')
            df['idx1'] = df['name1'].map(pdb2idx)
            df['idx2'] = df['name2'].map(pdb2idx)
            df = df.dropna(subset=['idx1', 'idx2'])
            df['idx1'] = df['idx1'].astype(np.int64)
            df['idx2'] = df['idx2'].astype(np.int64)

            # score = tmscore - 0.6 + min(0, 0.4 - seqid)
            df['score'] = df['tmscore'] - 0.6 + np.minimum(0.0, 0.4 - df['seqid'])

            grouped = []
            for idx1, g in df.groupby('idx1', sort=False):
                g = g.sort_values('score', ascending=False, kind='mergesort')
                grouped.append({
                    'idx1': int(idx1),
                    'idx2_list': g['idx2'].to_numpy(dtype=np.int64).tolist(),
                    'score_list': pt.from_numpy(g['score'].to_numpy(dtype=np.float32)),
                })
            self.groups = grouped
            self.map = np.arange(len(self.groups), dtype=np.int64) if mapping is None else mapping.astype(np.int64)
        else:
            assert hasattr(dataset, 'groups'), "For structured data, dataset must already contain grouped data in `dataset.groups`."
            self.groups = dataset.groups
            self.map = mapping.astype(np.int64) if mapping is not None else np.arange(len(self.groups), dtype=np.int64)
            assert np.max(self.map) < len(self.groups), "mapping index out of range."

    def __getitem__(self, idx):
        g = self.groups[self.map[idx]]
        data = {}
        data['idx1'], data['idx2_list'], data['score_list'] = g['idx1'], g['idx2_list'], g['score_list']
        return data

    def __len__(self):
        return len(self.map)


def _weighted_choice(indices, score_list, temperature=0.15):
    """在一个 index 子集内按 score 加权抽样一个。"""
    if len(indices) == 1:
        return indices[0]

    scores_np = score_list[indices].detach().cpu().numpy().astype(np.float64)
    logits = scores_np / max(temperature, 1e-8)
    logits = logits - logits.max()
    prob = np.exp(logits)
    prob_sum = prob.sum()
    if prob_sum <= 0:
        prob = np.full(len(indices), 1.0 / len(indices), dtype=np.float64)
    else:
        prob = prob / prob_sum
    chosen_local = int(np.random.choice(len(indices), p=prob))
    return indices[chosen_local]


def _build_strata(m: int, boundaries=(4, 16, 64)):
    """
    m: 当前 query 可用候选数（已截断到 top_m）
    boundaries=(4,16,64) -> [0,4), [4,16), [16,64), [64,m)
    """
    strata = []
    start = 0
    for b in boundaries:
        end = min(b, m)
        if end > start:
            strata.append(list(range(start, end)))
        start = end
    if start < m:
        strata.append(list(range(start, m)))
    return strata


def _sample_positive_indices(
    score_list,
    m: int,
    num_pos: int,
    mode: str = 'stratified',
    temperature: float = 0.15,
    boundaries=(4, 16, 64),
):
    """
    从前 m 个候选中抽 num_pos 个局部下标。
    默认 stratified:
      - 每层尽量抽 1 个
      - 剩余名额从所有未选样本里继续按权重抽
    """
    if m <= 0:
        raise ValueError("m must be positive.")
    actual_num = min(num_pos, m)

    if mode == 'top':
        chosen = list(range(actual_num))
    elif mode == 'random':
        chosen = np.random.choice(m, size=actual_num, replace=False).tolist()
    elif mode == 'weight':
        pool = list(range(m))
        chosen = []
        while len(chosen) < actual_num and len(pool) > 0:
            idx = _weighted_choice(np.array(pool, dtype=np.int64), score_list, temperature=temperature)
            chosen.append(int(idx))
            pool.remove(int(idx))
    elif mode == 'stratified':
        strata = _build_strata(m, boundaries=boundaries)
        chosen = []

        # 第一轮：每层先抽一个
        for stratum in strata:
            if len(chosen) >= actual_num:
                break
            idx = _weighted_choice(np.array(stratum, dtype=np.int64), score_list, temperature=temperature)
            chosen.append(int(idx))

        # 第二轮：剩余名额从所有未选样本继续抽
        if len(chosen) < actual_num:
            remaining = [i for i in range(m) if i not in chosen]
            while len(chosen) < actual_num and len(remaining) > 0:
                idx = _weighted_choice(np.array(remaining, dtype=np.int64), score_list, temperature=temperature)
                chosen.append(int(idx))
                remaining.remove(int(idx))
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be one of 'top', 'random', 'weight', 'stratified'.")

    chosen.sort(key=lambda i: float(score_list[i]), reverse=True)
    return chosen


def collate_fun_train(
    protein_dataset,
    mode: str = 'stratified',
    top_m: int = 64,
    temperature: float = 0.15,
    num_pos: int = 6,
    strata_boundaries=(4, 16, 64),
):
    """
    输出:
        query_seqs:      [B, Lq]
        query_masks:     [B, Lq]
        cand_seqs:       [B*K, Lc]
        cand_masks:      [B*K, Lc]
        cand_graphs:     Batch over B*K candidate graphs
        scores:          [B, K]
        pos_valid_mask:  [B, K]
    """
    def collate_fn(batch):
        query_seqs = []
        cand_seqs = []
        cand_graphs = []
        scores = []
        pos_valid_mask = []

        for data in batch:
            idx1, idx2_list, score_list = data['idx1'], data['idx2_list'], data['score_list']
            q_seq = protein_dataset[idx1]['seq']
            query_seqs.append(q_seq)

            m = min(top_m, len(idx2_list)) if top_m is not None else len(idx2_list)
            if m <= 0:
                raise ValueError(f"No candidates found for query idx={idx1}")

            chosen_local_idx = _sample_positive_indices(
                score_list=score_list,
                m=m,
                num_pos=num_pos,
                mode=mode,
                temperature=temperature,
                boundaries=strata_boundaries,
            )

            per_query_scores = []
            per_query_valid = []

            pad_src_local_idx = chosen_local_idx[0]

            for local_idx in chosen_local_idx:
                cand_idx = int(idx2_list[local_idx])
                cand_item = protein_dataset[cand_idx]
                cand_seqs.append(cand_item['seq'])
                cand_graphs.append(cand_item['graph'])
                per_query_scores.append(float(score_list[local_idx]))
                per_query_valid.append(True)

            while len(per_query_scores) < num_pos:
                cand_idx = int(idx2_list[pad_src_local_idx])
                cand_item = protein_dataset[cand_idx]
                cand_seqs.append(cand_item['seq'])
                cand_graphs.append(cand_item['graph'])
                per_query_scores.append(0.0)
                per_query_valid.append(False)

            scores.append(per_query_scores)
            pos_valid_mask.append(per_query_valid)

        query_seqs = pad_sequence(query_seqs, batch_first=True)
        cand_seqs = pad_sequence(cand_seqs, batch_first=True)
        query_mask = (query_seqs != 0).long()
        cand_mask = (cand_seqs != 0).long()
        cand_graphs = Batch.from_data_list(cand_graphs)
        scores = pt.tensor(scores, dtype=pt.float32)
        pos_valid_mask = pt.tensor(pos_valid_mask, dtype=pt.bool)

        return {
            "query_seqs": query_seqs,
            "query_masks": query_mask,
            "cand_seqs": cand_seqs,
            "cand_masks": cand_mask,
            "cand_graphs": cand_graphs,
            "scores": scores,
            "pos_valid_mask": pos_valid_mask,
            "num_pos": num_pos,
        }

    return collate_fn


def collate_fun_emb(mode: str = 'query'):
    def collate_fun(batch):
        seqs_pad = []
        data = {}
        for b in batch:
            seqs_pad.append(b['seq'])
        seqs_pad = pad_sequence(seqs_pad, batch_first=True)
        masks = (seqs_pad != 0).long()
        data['seqs_pad'] = seqs_pad
        data['masks'] = masks
        if mode == 'cand':
            graphs = []
            for b in batch:
                graphs.append(b['graph'])
            graphs = Batch.from_data_list(graphs)
            data['graphs'] = graphs
        return data

    return collate_fun