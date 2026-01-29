import h5py
import torch as pt
from copy import deepcopy
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
import pandas as pd


def load(fn):
    with h5py.File(fn) as f:
        seq = f['node_seq'][()]
        node_pos = f['node_pos'][()]
        node_idx = f['node_idx'][()]
        edge_nho = f['edge_nho'][()]
        edge_idx = f['edge_idx'][()]
        lab = f['label'][()].astype(str)
    return seq, node_pos, node_idx,edge_nho, edge_idx, lab


class ProteinDataset(Dataset):
    def __init__(self, dataset, mapping=None):
        super().__init__()
        if isinstance(dataset, tuple): # raw data
            self.seq = pt.from_numpy(dataset[0])
            self.node_pos = pt.from_numpy(dataset[1])
            self.node_idx = pt.from_numpy(dataset[2])
            self.edge_nho = pt.from_numpy(dataset[3])
            self.edge_idx = pt.from_numpy(dataset[4])
            self.lab = dataset[5].astype(str)
            self.map = pt.arange(len(self.lab), dtype=pt.int32) # 恒等映射 
            assert len(self.seq) == self.node_idx[-1] # 
            assert len(self.lab) == len(self.node_idx) - 1 == len(self.edge_idx) - 1        
        else: # structured data
            self.seq = dataset.seq
            self.node_pos = dataset.node_pos
            self.node_idx = dataset.node_idx
            self.edge_nho = dataset.edge_nho
            self.edge_idx = dataset.edge_idx
            self.lab = dataset.lab
            self.map = mapping
            assert self.map is not None
            assert pt.max(self.map) < len(self.lab)
    # self.map旨在维护一个data子集的映射，数据仍然是全部数据   
    
    def __getitem__(self, idx):
        idx_ = self.map[idx]
        seq = self.seq[self.node_idx[idx_] : self.node_idx[idx_+1]]
        len_seq = len(seq)
        # shape:(len_seq,5,3) 0:N, 1:α, 2:C, 3:O, 4:β 
        node_pos = self.node_pos[self.node_idx[idx_] : self.node_idx[idx_+1]] 
        # N-α-β这个夹角反映了氨基酸的空间结构，在化学键确定的前提下，N和β之间的距离就能反应角度 
        node_attr = pt.sqrt(pt.sum((node_pos[:,0] - node_pos[:,4])**2, dim=1))
        # 连接关系(肽键), 单向, 后续在embeddingBlock中搞双向
        edge_tai = pt.stack((pt.arange(0, len_seq-1), pt.arange(1, len_seq)), dim=0)
        # 连接关系(氢键), 单向
        edge_nho = pt.stack((self.edge_nho[0][self.edge_idx[idx_] : self.edge_idx[idx_+1]],
                             self.edge_nho[1][self.edge_idx[idx_] : self.edge_idx[idx_+1]]), dim=0)
        edge_idx = pt.cat((edge_tai, edge_nho), dim=1)
        # 边长 
        # 肽键：羧基碳接氨基氮
        tai_len = pt.sqrt(pt.sum((node_pos[edge_tai[0][:], 2] - node_pos[edge_tai[1][:], 0])**2, dim=1))
        # 氢键：氨基的氢和羧基的氧之间吸引产生，用氨基的氮坐标代替氢坐标 
        nho_len = pt.sqrt(pt.sum((node_pos[edge_nho[0][:], 0] - node_pos[edge_nho[1][:], 3])**2, dim=1))
        edge_len = pt.cat((tai_len, nho_len), dim=0)        
        # 标签
        lab = self.lab[idx_]
        graph = Data(x=node_attr, edge_index=edge_idx, edge_attr=edge_len)
        return deepcopy(graph), deepcopy(seq), deepcopy(lab)
        
    def __len__(self):  
        return len(self.map) #子集的大小是map的大小
    

class ProteinPairDataset(Dataset):
    def __init__(self, dataset, pair_file:str, pdb2idx:dict=None):
        super().__init__()
        if isinstance(dataset, ProteinDataset): # raw data
            self.protein = dataset
            df = pd.read_csv(pair_file, names=['name1', 'name2', 'tmscore', 'seqid'], delimiter='\t')
            df['name1'] = df['name1'].str.split('/').str[-1].str.removesuffix('.pdb')
            df['name2'] = df['name2'].str.split('/').str[-1].str.removesuffix('.pdb')
            df['idx1'] = df['name1'].map(pdb2idx)
            df['idx2'] = df['name2'].map(pdb2idx)
            df = df.dropna(subset=['idx1', 'idx2'])
            self.idx1 = pt.from_numpy(df['idx1'].values)
            assert len(self.idx1) > 0, "No valid protein pairs found!"
            self.idx2 = pt.from_numpy(df['idx2'].values)
            self.tm = pt.from_numpy(df['tmscore'].values)
            self.seqid = pt.from_numpy(df['seqid'].values)
            assert len(self.idx1) == len(self.idx2) == len(self.tm) == len(self.seqid), "Inconsistent lengths in protein pair data!"
            self.map = pt.arange(len(self.idx1), dtype=pt.int32) # 恒等映射 
        else: # structured data
            self.protein = dataset.protein
            self.idx1 = dataset.idx1
            self.idx2 = dataset.idx2
            self.tm = dataset.tm
            self.seqid = dataset.seqid
            self.map = dataset.map
            assert self.map is not None
            assert pt.max(self.map) < len(self.idx1)

    def __getitem__(self, idx):
        idx = self.map[idx]
        i, j, tm, seqid = self.idx1[idx], self.idx2[idx], self.tm[idx], self.seqid[idx]
        graph_i, seq_i, _ = self.protein.__getitem__(i)
        graph_j, seq_j, _ = self.protein.__getitem__(j)
        label = pt.tensor([tm, seqid], dtype=pt.float32)
        return deepcopy((graph_i, seq_i)), deepcopy((graph_j, seq_j)), deepcopy(label)
    
    def __len__(self):
        return len(self.map)
    

def pair_collate_fn(batch):
    graphs_i, seqs_i, graphs_j, seqs_j, labels = zip(*batch)
    batch_i = Batch.from_data_list(graphs_i)
    batch_j = Batch.from_data_list(graphs_j)
    return (batch_i, seqs_i), (batch_j, seqs_j), pt.stack(labels, dim=0)