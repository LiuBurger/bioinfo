import h5py
import random
import torch as pt
import numpy as np
from torch_geometric.data import Batch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


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
    def __init__(self, dataset, mapping:np.ndarray=None):
        super().__init__()
        if isinstance(dataset, tuple): # raw data
            self.seq = dataset[0]           
            self.graph = dataset[1]
            self.lab = dataset[2]
            self.map = np.arange(len(self.lab), dtype=np.int64) # 恒等映射 
            assert len(self.seq) == len(self.graph) == len(self.lab)
        else: # structured data
            assert mapping is not None, "Mapping must be provided for structured data."
            self.seq = dataset.seq
            self.graph = dataset.graph
            self.lab = dataset.lab
            self.map = mapping
            assert np.max(self.map) < len(self.lab)
    # self.map旨在维护一个data子集的映射，数据仍然是全部数据   

    def __getitem__(self, idx):
        idx = self.map[idx]
        seq, graph, lab = self.seq[idx], self.graph[idx], self.lab[idx]
        return seq, graph, lab

    def __len__(self):  
        return len(self.map) #子集的大小是map的大小
    

class QueryHomologyDataset(Dataset):
    def __init__(self, dataset, pair_file:str=None, pdb2idx:dict=None, mapping:np.ndarray=None):
        super().__init__()
        if isinstance(dataset, ProteinDataset): # raw data
            assert pair_file is not None and pdb2idx is not None, "pair_file and pdb2idx must be provided for raw data."
            df = pd.read_csv(pair_file, names=['name1', 'name2', 'tmscore', 'seqid'], delimiter='\t')
            df['name1'] = df['name1'].str.split('/').str[-1].str.removesuffix('.pdb')
            df['name2'] = df['name2'].str.split('/').str[-1].str.removesuffix('.pdb')
            df['idx1'] = df['name1'].map(pdb2idx)
            df['idx2'] = df['name2'].map(pdb2idx)
            df = df.dropna(subset=['idx1', 'idx2'])
            assert len(df) > 0, "No valid protein pairs found!"
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
                    'idx2': g['idx2'].to_numpy(dtype=np.int64).tolist(),
                    'score': pt.from_numpy(g['score'].to_numpy(dtype=np.float32)),
                })
            self.groups = grouped
            self.map = np.arange(len(self.groups), dtype=np.int64) if mapping is None else mapping.astype(np.int64)
        else: # structured data
            assert hasattr(dataset, 'groups'), \
                "For structured data, dataset must already contain grouped data in `dataset.groups`."
            self.groups = dataset.groups
            self.map = mapping.astype(np.int64) if mapping is not None else np.arange(len(self.groups), dtype=np.int64)
            assert np.max(self.map) < len(self.groups), "mapping index out of range."
    
    def __getitem__(self, idx):
        g = self.groups[self.map[idx]]
        #          0           1        2     
        return g['idx1'], g['idx2'], g['score']

    def __len__(self):
        return len(self.map)


def collate_fun_train(protein_dataset, positive_strategy: str = "weighted",):
    assert positive_strategy in ["top1", "random", "weighted"]
    def collate_fun(batch):
        query_seqs = []
        cand_seqs = []
        cand_graphs = []
        q_indices = []
        c_indices = []
        pos_score = []

        for item in batch:
            idx1, idx2_list, score_list = item
            if len(idx2_list) == 0:
                continue
            if positive_strategy == "top1":
                pos_j = 0
            elif positive_strategy == "random":
                pos_j = random.randint(0, len(idx2_list) - 1)
            else:
                prob = score_list.float().clamp(min=0)
                if prob.sum() > 0:
                    prob = prob / prob.sum()
                    pos_j = pt.multinomial(prob, 1).item()
                else:
                    pos_j = random.randint(0, len(idx2_list) - 1)

            idx2 = idx2_list[pos_j]
            # ProteinDataset(mode='graph') -> (seq, graph, lab)
            q_seq, _, _ = protein_dataset[idx1]
            c_seq, c_graph, _ = protein_dataset[idx2]
            query_seqs.append(q_seq)
            cand_seqs.append(c_seq)
            cand_graphs.append(c_graph)
            q_indices.append(idx1)
            c_indices.append(idx2)
            pos_score.append(score_list[pos_j].float())

        if len(query_seqs) == 0:
            raise RuntimeError("Empty batch after positive pair selection.")

        query_ids = pad_sequence(query_seqs, batch_first=True, padding_value=0)
        query_mask = (query_ids != 0).long()
        cand_ids = pad_sequence(cand_seqs, batch_first=True, padding_value=0)
        cand_mask = (cand_ids != 0).long()
        cand_graph_batch = Batch.from_data_list(cand_graphs)

        return {
            "query_ids": query_ids,
            "query_mask": query_mask,
            "cand_ids": cand_ids,
            "cand_mask": cand_mask,
            "candidate_graph": cand_graph_batch,
            "q_idx": pt.tensor(q_indices, dtype=pt.long),
            "c_idx": pt.tensor(c_indices, dtype=pt.long),
            "score": pt.stack(pos_score),
        }
    return collate_fun


def collate_fun_emb(mode:str='lib'):
    def collate_fun(batch):
        seqs_pad, masks, graphs = [], [], []
        max_len = max(len(seq) for seq, _, _ in batch)
        for seq, graph, _ in batch: # seq, graph, lab
            pad_len = max_len - len(seq)
            seqs_pad.append(F.pad(seq, (0, pad_len), value=0))
            masks.append(F.pad(pt.ones(len(seq), dtype=pt.bool), (0, pad_len), value=False))
            graphs.append(graph)
        seqs_pad = pt.stack(seqs_pad, dim=0)
        masks = pt.stack(masks, dim=0)
        graphs = Batch.from_data_list(graphs)
        if mode == 'lib':
            return seqs_pad, masks, graphs
        elif mode == 'query':
            return seqs_pad, masks
        else:
            raise ValueError(f"mode '{mode}' not exist")
    return collate_fun