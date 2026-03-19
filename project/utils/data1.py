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
    def __init__(self, dataset=None, pair_file:str=None, pdb2idx:dict=None, mapping:np.ndarray=None):
        super().__init__()
        if dataset is None: # raw data
            assert pair_file is not None and pdb2idx is not None, "pair_file and pdb2idx must be provided for raw data."
            df = pd.read_csv(pair_file, names=['name1', 'name2', 'tmscore', 'seqid'], delimiter='\t')
            df['name1'] = df['name1'].str.split('/').str[-1].str.removesuffix('.pdb')
            df['name2'] = df['name2'].str.split('/').str[-1].str.removesuffix('.pdb')
            df['idx1'] = df['name1'].map(pdb2idx)
            df['idx2'] = df['name2'].map(pdb2idx)
            df = df.dropna(subset=['idx1', 'idx2'])
            df['idx1'] = df['idx1'].astype(np.int64)
            df['idx2'] = df['idx2'].astype(np.int64)
            # 按 idx1 分组
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
        elif isinstance(dataset, QueryHomologyDataset): # structured data
            assert hasattr(dataset, 'groups'), "For structured data, dataset must already contain grouped data in `dataset.groups`."
            self.groups = dataset.groups
            self.map = mapping.astype(np.int64) if mapping is not None else np.arange(len(self.groups), dtype=np.int64)
            assert np.max(self.map) < len(self.groups), "mapping index out of range."

    def __getitem__(self, idx):
        g = self.groups[self.map[idx]]
        data = {'idx1':g['idx1'], 'idx2_list':g['idx2_list'], 'tmscore_list':g['tmscore_list'], 'seqid_list':g['seqid_list']}
        return data

    def __len__(self):
        return len(self.map)


def collate_fun_train(protein_dataset:ProteinDataset):
    N = len(protein_dataset)
    def collate_fn(batch):
        query_seqs, pos_seqs, pos_graphs, tmscores, seqids, neg_seqs, neg_graphs = [], [], [], [], [], [], []
        for data in batch:
            idx1, idx2_list, tmscore_list, seqid_list = data['idx1'], data['idx2_list'], data['tmscore_list'], data['seqid_list']
            q_seq = protein_dataset[idx1]['seq']
            query_seqs.append(q_seq)         

            idx2_idx = random.randint(0, len(idx2_list)-1)
            pos_seq = protein_dataset[idx2_list[idx2_idx]]['seq']
            pos_graph = protein_dataset[idx2_list[idx2_idx]]['graph']
            tmscore = float(tmscore_list[idx2_idx])
            seqid = float(seqid_list[idx2_idx])
            
            pos_seqs.append(pos_seq)
            pos_graphs.append(pos_graph)
            tmscores.append(tmscore)
            seqids.append(seqid)
            
            neg_idx = random.randint(0, N-1)
            while neg_idx in idx2_list or neg_idx == idx1:
                neg_idx = random.randint(0, N-1)
            neg_seq = protein_dataset[neg_idx]['seq']
            neg_graph = protein_dataset[neg_idx]['graph']
            neg_seqs.append(neg_seq)
            neg_graphs.append(neg_graph)

        query_seqs = pad_sequence(query_seqs, batch_first=True)
        query_masks = (query_seqs != 0).long()
        pos_seqs = pad_sequence(pos_seqs, batch_first=True)
        pos_masks = (pos_seqs != 0).long()
        pos_graphs = Batch.from_data_list(pos_graphs)        
        tmscores = pt.tensor(tmscores, dtype=pt.float32)
        seqids = pt.tensor(seqids, dtype=pt.float32)

        neg_seqs = pad_sequence(neg_seqs, batch_first=True)
        neg_masks = (neg_seqs != 0).long()
        neg_graphs = Batch.from_data_list(neg_graphs)
        
        return {
            "query_seqs": query_seqs,
            "query_masks": query_masks,
            "pos_seqs": pos_seqs,
            "pos_masks": pos_masks,
            "pos_graphs": pos_graphs,
            "tmscores": tmscores,
            "seqids": seqids,
            "neg_seqs": neg_seqs,
            "neg_masks": neg_masks,
            "neg_graphs": neg_graphs,
        }

    return collate_fn


def collate_fun_emb(mode:str='query', protein_dataset:ProteinDataset=None):
    if mode == 'query':
        assert protein_dataset is not None, f"As query, ProteinDataset must be provided"
    def collate_fun(batch):
        seqs_pad = []
        data = {}
        if mode == 'query': # 这里的batch是QueryHomologyDataset
            for b in batch:
                seqs_pad.append(protein_dataset[b['idx1']]['seq'])
        elif mode == 'cand':
            for b in batch: # 这里的batch是ProteinDataset
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