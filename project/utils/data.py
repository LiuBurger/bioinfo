import h5py
import random
import torch as pt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
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
    if mode == 'all':
        return seq, node_ss, node_rsa, node_pos, node_idx, edge_nho, edge_idx, lab
    if mode == 'part':
        return seq, node_pos, node_idx, edge_nho, edge_idx, lab
    if mode == 'seq':
        return seq, node_idx, lab
    raise ValueError(f"Invalid mode: {mode}. Must be one of 'all', 'part', or 'seq'.")


class ProteinDataset(Dataset):
    def __init__(self, dataset, mapping: np.ndarray = None, mode: str = 'graph'):
        super().__init__()
        self.mode = mode
        if isinstance(dataset, tuple):
            self.seq = dataset[0]
            self.lab = dataset[1]
            self.graph = dataset[2]
            self.map = np.arange(len(self.lab), dtype=np.int64)
            assert len(self.seq) == len(self.lab) == len(self.graph)
        else:
            assert mapping is not None or hasattr(dataset, 'lab'), "dataset must provide `lab`."
            self.seq = dataset.seq
            self.lab = dataset.lab
            self.graph = dataset.graph
            self.map = mapping if mapping is not None else np.arange(len(self.lab), dtype=np.int64)
            assert np.max(self.map) < len(self.lab)

    def __getitem__(self, idx):
        idx = self.map[idx]
        return {
            'seq': self.seq[idx],
            'lab': self.lab[idx],
            'graph': self.graph[idx],
        }

    def __len__(self):
        return len(self.map)


class QueryHomologyDataset(Dataset):
    def __init__(self, dataset=None, pair_file: str = None, pdb2idx: dict = None, mapping: np.ndarray = None):
        super().__init__()
        if dataset is None:
            assert pair_file is not None and pdb2idx is not None, 'pair_file and pdb2idx must be provided for raw data.'
            df = pd.read_csv(pair_file, names=['name1', 'name2', 'tmscore', 'seqid'], delimiter='\t')
            df['name1'] = df['name1'].str.split('/').str[-1].str.removesuffix('.pdb')
            df['name2'] = df['name2'].str.split('/').str[-1].str.removesuffix('.pdb')
            df['idx1'] = df['name1'].map(pdb2idx)
            df['idx2'] = df['name2'].map(pdb2idx)
            df = df.dropna(subset=['idx1', 'idx2'])
            df['idx1'] = df['idx1'].astype(np.int64)
            df['idx2'] = df['idx2'].astype(np.int64)

            grouped = []
            for idx1, g in df.groupby('idx1', sort=False):
                grouped.append({
                    'idx1': int(idx1),
                    'idx2_list': g['idx2'].to_numpy(dtype=np.int64).tolist(),
                    'tmscore_list': pt.from_numpy(g['tmscore'].to_numpy(dtype=np.float32)),
                    'seqid_list': pt.from_numpy(g['seqid'].to_numpy(dtype=np.float32)),
                })
            self.groups = grouped
            self.map = np.arange(len(self.groups), dtype=np.int64) if mapping is None else mapping.astype(np.int64)
        elif isinstance(dataset, QueryHomologyDataset):
            self.groups = dataset.groups
            self.map = mapping.astype(np.int64) if mapping is not None else np.arange(len(self.groups), dtype=np.int64)
            assert np.max(self.map) < len(self.groups), 'mapping index out of range.'
        else:
            raise TypeError('dataset must be None or QueryHomologyDataset')

    def __getitem__(self, idx):
        g = self.groups[self.map[idx]]
        return {
            'idx1': g['idx1'],
            'idx2_list': g['idx2_list'],
            'tmscore_list': g['tmscore_list'],
            'seqid_list': g['seqid_list'],
        }

    def __len__(self):
        return len(self.map)


def collate_fun_train(protein_dataset: ProteinDataset, positive_top_ratio: float = 0.3):
    """
    每个 query 抽一个正样本；batch 内其余正样本自动构成 in-batch negatives。
    监督同时返回：
    - seqids: 用于 SEQID 回归
    - tmscores: 用于 tmscore 回归
    - remote_scores: 仅用于日志或兼容保留
    """
    positive_top_ratio = float(max(0.0, min(1.0, positive_top_ratio)))

    def collate_fn(batch):
        query_seqs, pos_seqs = [], []
        query_graphs, pos_graphs = [], []
        tmscores, seqids, remote_scores = [], [], []

        for data in batch:
            idx1 = data['idx1']
            idx2_list = data['idx2_list']
            tmscore_list = data['tmscore_list']
            seqid_list = data['seqid_list']

            query_item = protein_dataset[idx1]
            query_seqs.append(query_item['seq'])
            query_graphs.append(query_item['graph'])

            score_list = tmscore_list - 0.6 + pt.minimum(0.4 - seqid_list, pt.zeros_like(seqid_list))
            if len(idx2_list) == 1 or positive_top_ratio <= 0.0:
                idx2_idx = int(pt.argmax(score_list).item())
            else:
                topm = max(1, int(np.ceil(len(idx2_list) * positive_top_ratio)))
                top_idx = pt.topk(score_list, k=topm).indices
                rand_pos = random.randrange(topm)
                idx2_idx = int(top_idx[rand_pos].item())

            pos_idx = idx2_list[idx2_idx]
            pos_item = protein_dataset[pos_idx]
            pos_seqs.append(pos_item['seq'])
            pos_graphs.append(pos_item['graph'])

            tm_value = float(tmscore_list[idx2_idx])
            seqid_value = float(seqid_list[idx2_idx])
            remote_value = tm_value - 0.6 + min(0.4 - seqid_value, 0.0)

            tmscores.append(tm_value)
            seqids.append(seqid_value)
            remote_scores.append(remote_value)

        query_seqs = pad_sequence(query_seqs, batch_first=True)
        query_masks = (query_seqs != 0).long()
        pos_seqs = pad_sequence(pos_seqs, batch_first=True)
        pos_masks = (pos_seqs != 0).long()

        return {
            'query_seqs': query_seqs,
            'query_masks': query_masks,
            'query_graphs': Batch.from_data_list(query_graphs),
            'pos_seqs': pos_seqs,
            'pos_masks': pos_masks,
            'pos_graphs': Batch.from_data_list(pos_graphs),
            'tmscores': pt.tensor(tmscores, dtype=pt.float32),
            'seqids': pt.tensor(seqids, dtype=pt.float32),
            'remote_scores': pt.tensor(remote_scores, dtype=pt.float32),
        }

    return collate_fn


def collate_fun_emb(mode: str = 'query', protein_dataset: ProteinDataset = None):
    """
    注意：虽然模型端已经统一成单一 encode 函数，这里的数据拼装仍然保留
    query / candidate 区分，因为两者的 dataset 结构不同：

    - query: batch 元素来自 QueryHomologyDataset，需要通过 idx1 去 protein_dataset 取真实样本
    - cand : batch 元素直接来自 ProteinDataset，可直接取 seq / graph
    """
    if mode == 'query':
        assert protein_dataset is not None, 'As query, ProteinDataset must be provided.'

    def collate_fn(batch):
        seqs = []
        graphs = []

        if mode == 'query':
            for b in batch:
                item = protein_dataset[b['idx1']]
                seqs.append(item['seq'])
                graphs.append(item['graph'])
        elif mode == 'cand':
            for b in batch:
                seqs.append(b['seq'])
                graphs.append(b['graph'])
        else:
            raise ValueError("mode must be either 'query' or 'cand'.")

        seqs_pad = pad_sequence(seqs, batch_first=True)
        masks = (seqs_pad != 0).long()
        return {
            'seqs_pad': seqs_pad,
            'masks': masks,
            'graphs': Batch.from_data_list(graphs),
        }

    return collate_fn