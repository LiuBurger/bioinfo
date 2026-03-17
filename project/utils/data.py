import h5py
import random
import torch as pt
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch


def load(fn, mode:str='all'):
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
    def __init__(self, dataset, mapping:np.ndarray=None, mode:str='query'):
        super().__init__()
        self.mode = mode
        if isinstance(dataset, tuple): # raw data
            self.seq = dataset[0]
            self.lab = dataset[1]
            if mode == 'cand':
                self.graph = dataset[2]
            self.map = np.arange(len(self.lab), dtype=np.int64) # 恒等映射
            assert len(self.seq) == len(self.lab)
        else: # structured data
            assert mapping is not None, "Mapping must be provided for structured data."
            self.seq = dataset.seq
            self.lab = dataset.lab
            if mode == 'cand':
                self.graph = dataset.graph
            self.map = mapping
            assert np.max(self.map) < len(self.lab)
    # self.map旨在维护一个data子集的映射，数据仍然是全部数据

    def __getitem__(self, idx):
        idx = self.map[idx]
        data = {}
        data['seq'], data['lab'] = self.seq[idx], self.lab[idx]
        if self.mode == 'cand':
            data["graph"] = self.graph[idx]
        return data

    def __len__(self):
        return len(self.map) #子集的大小是map的大小


class QueryHomologyDataset(Dataset):
    def __init__(self, dataset, pair_file:str=None, pdb2idx:dict=None, mapping:np.ndarray=None):
        super().__init__()
        if isinstance(dataset, ProteinDataset): # raw data
            assert pair_file is not None and pdb2idx is not None, "pair_file and pdb2idx must be provided for raw data."
            df = pd.read_csv(pair_file, names=['name1', 'name2', 'tmscore', 'seqid'], delimiter='	')
            df['name1'] = df['name1'].str.split('/').str[-1].str.removesuffix('.pdb')
            df['name2'] = df['name2'].str.split('/').str[-1].str.removesuffix('.pdb')
            df['idx1'] = df['name1'].map(pdb2idx)
            df['idx2'] = df['name2'].map(pdb2idx)
            df = df.dropna(subset=['idx1', 'idx2'])
            df['idx1'] = df['idx1'].astype(np.int64)
            df['idx2'] = df['idx2'].astype(np.int64)
            # score = tmscore - 0.6 + min(0, 0.4 - seqid)
            df['score'] = df['tmscore'] - 0.6 + np.minimum(0.0, 0.4 - df['seqid'])
            # 先按 idx1 分组，再按 score 降序排列
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
        else: # structured data
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


def collate_fun_train(protein_dataset, mode:str='weight', top_m:int=16, temperature:float=0.05):
    """
    batch item:
        idx1 : query index
        idx2_list : list[candidate index]
        score_list : tensor
    """
    def collate_fn(batch):
        query_seqs, cand_seqs, cand_graphs, scores = [], [], [], []
        for data in batch:
            idx1, idx2_list, score_list = data['idx1'], data['idx2_list'], data['score_list']
            q_seq = protein_dataset[idx1]['seq']
            query_seqs.append(q_seq)

            m = min(top_m, len(idx2_list)) if top_m is not None else len(idx2_list)
            if m <= 0:
                raise ValueError(f"No candidates found for query idx={idx1}")

            if mode == 'random':
                idx2_idx = random.randint(0, m - 1)
            elif mode == 'weight':
                scores_np = score_list[:m].detach().cpu().numpy().astype(np.float64)
                logits = scores_np / max(temperature, 1e-8)
                logits = logits - logits.max()
                prob = np.exp(logits)
                prob = prob / prob.sum()
                idx2_idx = int(np.random.choice(m, p=prob))
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be one of 'random' or 'weight'.")

            score = float(score_list[idx2_idx])
            scores.append(score)
            cand_idx = int(idx2_list[idx2_idx])
            c_seq = protein_dataset[cand_idx]['seq']
            cand_seqs.append(c_seq)
            c_graph = protein_dataset[cand_idx]['graph']
            cand_graphs.append(c_graph)

        query_seqs = pad_sequence(query_seqs, batch_first=True)
        cand_seqs = pad_sequence(cand_seqs, batch_first=True)
        query_mask = (query_seqs != 0).long()
        cand_mask = (cand_seqs != 0).long()
        cand_graphs = Batch.from_data_list(cand_graphs)
        scores = pt.tensor(scores, dtype=pt.float32)
        return {
            "query_seqs": query_seqs,
            "query_masks": query_mask,
            "cand_seqs": cand_seqs,
            "cand_masks": cand_mask,
            "cand_graphs": cand_graphs,
            "scores": scores,
        }

    return collate_fn


# def collate_fun_emb(batch):
#     seqs_pad = []
#     for seq, _ in batch: # seq, graph, lab
#         seqs_pad.append(seq)
#     seqs_pad = pad_sequence(seqs_pad, batch_first=True)
#     masks = (seqs_pad != 0).long()
#     return seqs_pad, masks


def collate_fun_emb(mode:str='query'):
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