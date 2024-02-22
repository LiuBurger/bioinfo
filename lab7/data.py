# add your source codes regarding the data fllow here
import h5py
import torch as pt
from torch_geometric.data import Data, Dataset


def load(fn):
    with h5py.File(fn) as f:
        seq = f['node_seq'][()]
        node_pos = f['node_pos'][()]
        node_idx = f['node_idx'][()]
        edge_nho = f['edge_nho'][()]
        edge_idx = f['edge_idx'][()]
        lab = f['label'][()]
    return seq, node_pos, node_idx,edge_nho, edge_idx, lab


class ProteinDataset(Dataset):
    def __init__(self, dataset, mapping=None):
        super(ProteinDataset, self).__init__()
        if isinstance(dataset, tuple): # raw data
            self.seq = pt.tensor(dataset[0], dtype=pt.int32)
            self.node_pos = pt.tensor(dataset[1], dtype=pt.float32)
            self.node_idx = pt.tensor(dataset[2])
            self.edge_nho = pt.tensor(dataset[3], dtype=pt.int32)
            self.edge_idx = pt.tensor(dataset[4])
            self.lab = pt.tensor(dataset[5])
            self.map = pt.arange(len(self.lab), dtype=pt.int32) # 恒等映射 
            assert len(self.seq) == self.node_idx[-1]
            assert len(self.lab) == len(self.node_idx) - 1
            assert len(self.lab) == len(self.edge_idx) - 1
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
    
    def get(self, idx):
        idx_ = self.map[idx]
        seq = self.seq[self.node_idx[idx_] : self.node_idx[idx_+1]]
        len_seq = len(seq)
        seq_ = pt.zeros(len_seq, 21, dtype=pt.float32)
        seq_[pt.arange(len_seq) , seq[:]-1] = 1.0
        # shape:(len_seq,5,3) 0:N, 1:α, 2:C, 3:O, 4:β 
        node_pos = self.node_pos[self.node_idx[idx_] : self.node_idx[idx_+1]] 
        # N-α-β这个夹角反映了氨基酸的空间结构，在化学键确定的前提下，N和β之间的距离就能反应角度 
        seq_[:, 20] = pt.sqrt(pt.sum((node_pos[:,0] - node_pos[:,4])**2, dim=1))
        
        # 连接关系(氢键) 
        edge_nho = pt.stack((self.edge_nho[0][self.edge_idx[idx_] : self.edge_idx[idx_+1]],
                             self.edge_nho[1][self.edge_idx[idx_] : self.edge_idx[idx_+1]]), dim=0)
        # 连接关系(肽键)
        edge_tai = pt.stack((pt.arange(0, len_seq-1), pt.arange(1, len_seq)), dim=0)
        # 连接关系
        edge_idx = pt.cat((edge_tai, edge_nho), dim=1)
        # 边长 
        nho_attr = pt.empty(edge_nho.shape[1])
        # 氢键：氨基的氢和羧基的氧之间吸引产生，用氨基的氮坐标代替氢坐标 
        nho_attr[:] = pt.sqrt(pt.sum((node_pos[edge_nho[0][:], 0] - node_pos[edge_nho[1][:], 3])**2, dim=1))
        tai_attr = pt.empty(edge_tai.shape[1])
        # 羧基碳接氨基氮
        tai_attr[:] = pt.sqrt(pt.sum((node_pos[edge_tai[0][:], 2] - node_pos[edge_tai[1][:], 0])**2, dim=1))
        edge_attr = pt.cat((tai_attr, nho_attr))[:, None]
        
        lab = self.lab[idx_]
        one_hot = pt.zeros(6630, dtype=pt.float32)
        one_hot[lab[0]] = 1.0
        graph = Data(x=seq_, edge_index=edge_idx, edge_attr=edge_attr, y=one_hot)

        return graph
        

    def len(self):  
        return len(self.map) #子集的大小是map的大小