import h5py
import torch as pt
import numpy as np
from torch_geometric.data import Batch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F


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
    

class ProteinPairDataset(Dataset):
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
            self.idx1 = df['idx1'].values.astype(np.int64)
            assert len(self.idx1) > 0, "No valid protein pairs found!"
            self.idx2 = df['idx2'].values.astype(np.int64)
            self.tm = pt.from_numpy(df['tmscore'].values)
            self.seqid = pt.from_numpy(df['seqid'].values)
            assert len(self.idx1) == len(self.idx2) == len(self.tm) == len(self.seqid), "Inconsistent lengths in protein pair data!"
            self.map = np.arange(len(self.idx1), dtype=np.int64) # 恒等映射 
        else: # structured data
            assert mapping is not None, "Mapping must be provided for structured data."
            self.idx1 = dataset.idx1
            self.idx2 = dataset.idx2
            self.tm = dataset.tm
            self.seqid = dataset.seqid
            self.map = mapping
            assert np.max(self.map) < len(self.idx1)
    
    def __getitem__(self, idx):
        idx = self.map[idx]
        return self.idx1[idx], self.idx2[idx], self.tm[idx], self.seqid[idx]

    def __len__(self):
        return len(self.map)
    

def pair_collate_fun(protein_dataset: ProteinDataset):
    def collate_fun(batch): 
        idx1, idx2, tm, seqid = zip(*batch)
        idx1 = pt.tensor(idx1, dtype=pt.long)
        idx2 = pt.tensor(idx2, dtype=pt.long)
        score = pt.stack([pt.stack([t, s]) for t, s in zip(tm, seqid)], dim=0) # shape: (batch_size, 2)

        all_idx = pt.cat([idx1, idx2], dim=0)
        unique_idx, inverse_idx = pt.unique(all_idx, return_inverse=True)
        inv_i = inverse_idx[:len(idx1)]
        inv_j = inverse_idx[len(idx1):]

        seqs, graphs = [], []
        for pidx in unique_idx.tolist():
            seq, graph, _ = protein_dataset.__getitem__(pidx)
            seqs.append(seq)
            graphs.append(graph)
        
        max_len = max(len(seq) for seq in seqs)
        seqs_pad, masks = [], []
        for seq in seqs:
            pad_len = max_len - len(seq)
            seqs_pad.append(F.pad(seq, (0, pad_len), value=0))
            masks.append(F.pad(pt.ones(len(seq), dtype=pt.bool), (0, pad_len), value=False))
        seqs_pad = pt.stack(seqs_pad, dim=0)
        masks = pt.stack(masks, dim=0)
        graphs = Batch.from_data_list(graphs)
        return (seqs_pad, masks, graphs), (inv_i, inv_j), score
    return collate_fun