import h5py
import torch as pt
import numpy as np
from torch_geometric.data import Batch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F


def load(fn):
    with h5py.File(fn) as f:
        seq = f['node_seq'][()].astype(np.int32)
        node_pos = f['node_pos'][()].astype(np.float32)
        node_idx = f['node_idx'][()]
        edge_nho = f['edge_nho'][()].astype(np.int32)
        edge_idx = f['edge_idx'][()]
        lab = f['label'][()].astype(str)
    return seq, node_pos, node_idx,edge_nho, edge_idx, lab


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
    
    def get(self, idx):
        seq, graph, lab = self.seq[idx], self.graph[idx], self.lab[idx]
        return seq, graph, lab

    def __getitem__(self, idx):
        idx_ = self.map[idx]
        return self.get(idx_)

    def __len__(self):  
        return len(self.map) #子集的大小是map的大小
    

class ProteinPairDataset(Dataset):
    def __init__(self, dataset, pair_file:str=None, pdb2idx:dict=None, mapping:np.ndarray=None):
        super().__init__()
        self.data_cache = {}
        if isinstance(dataset, ProteinDataset): # raw data
            assert pair_file is not None and pdb2idx is not None, "pair_file and pdb2idx must be provided for raw data."
            self.protein = dataset
            df = pd.read_csv(pair_file, names=['name1', 'name2', 'tmscore', 'seqid'], delimiter='\t')
            df['name1'] = df['name1'].str.split('/').str[-1].str.removesuffix('.pdb')
            df['name2'] = df['name2'].str.split('/').str[-1].str.removesuffix('.pdb')
            df['idx1'] = df['name1'].map(pdb2idx).astype(np.int64)
            df['idx2'] = df['name2'].map(pdb2idx).astype(np.int64)
            df = df.dropna(subset=['idx1', 'idx2'])
            self.idx1 = df['idx1'].values
            assert len(self.idx1) > 0, "No valid protein pairs found!"
            self.idx2 = df['idx2'].values
            self.tm = pt.from_numpy(df['tmscore'].values)
            self.seqid = pt.from_numpy(df['seqid'].values)
            assert len(self.idx1) == len(self.idx2) == len(self.tm) == len(self.seqid), "Inconsistent lengths in protein pair data!"
            self.map = np.arange(len(self.idx1), dtype=np.int64) # 恒等映射 
        else: # structured data
            assert mapping is not None, "Mapping must be provided for structured data."
            self.protein = dataset.protein
            self.idx1 = dataset.idx1
            self.idx2 = dataset.idx2
            self.tm = dataset.tm
            self.seqid = dataset.seqid
            self.map = mapping
            assert np.max(self.map) < len(self.idx1)

    def get(self, idx):
        i, j, tm, seqid = self.idx1[idx], self.idx2[idx], self.tm[idx], self.seqid[idx]
        seq_i, graph_i, _ = self.protein.get(int(i)) # 从np.ndarray索引转换为int
        seq_j, graph_j, _ = self.protein.get(int(j))
        score = pt.tensor([tm, seqid], dtype=pt.float32)
        return (seq_i, graph_i), (seq_j, graph_j), score
    
    def __getitem__(self, idx):
        idx_ = self.map[idx]
        if idx_ not in self.data_cache:
            self.data_cache[idx_] = self.get(idx_)
        return self.data_cache[idx_]

    def __len__(self):
        return len(self.map)
    

def pair_collate_fn(batch):
    seqs_graphs_i, seqs_graphs_j, scores = zip(*batch)
    seqs_i, graphs_i = zip(*seqs_graphs_i)
    seqs_j, graphs_j = zip(*seqs_graphs_j)
    graphs_i = Batch.from_data_list(graphs_i)
    graphs_j = Batch.from_data_list(graphs_j)
    seqs_list_i, masks_i = [], []
    seqs_list_j, masks_j = [], []
    max_len_i = max([len(seq) for seq in seqs_i])
    max_len_j = max([len(seq) for seq in seqs_j])
    for seq in seqs_i:
        pad_num = max_len_i - len(seq)
        seqs_list_i.append(F.pad(seq, (0, pad_num)))
        masks_i.append(F.pad(pt.ones_like(seq), (0, pad_num)))
    for seq in seqs_j:
        pad_num = max_len_j - len(seq)
        seqs_list_j.append(F.pad(seq, (0, pad_num)))
        masks_j.append(F.pad(pt.ones_like(seq), (0, pad_num)))
    seqs_list_i = pt.stack(seqs_list_i)
    seqs_list_j = pt.stack(seqs_list_j)
    masks_i = pt.stack(masks_i)
    masks_j = pt.stack(masks_j)
    return (seqs_list_i, masks_i, graphs_i), (seqs_list_j, masks_j, graphs_j), pt.stack(scores, dim=0)