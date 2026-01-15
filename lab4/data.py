# add your source codes regarding the data fllow here
import h5py
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset


def load(fn):
    with h5py.File(fn) as f:
        seq = f['node_seq'][()]
        idx = f['node_idx'][()]
        lab = f['label'][()]
        f.close()
    return seq, idx, lab


class ProteinDataset(Dataset):
    def __init__(self, dataset, mapping=None):
        super(ProteinDataset, self).__init__()
        if isinstance(dataset, tuple):
            self.seq = dataset[0]
            self.idx = dataset[1]
            self.lab = dataset[2]
            self.map = np.arange(len(self.lab), dtype=np.int64)
            assert self.idx[-1] == len(self.seq)
            assert len(self.idx) == len(self.lab)+1
        else:
            self.seq = dataset.seq
            self.idx = dataset.idx
            self.lab = dataset.lab
            self.map = mapping
            assert self.map is not None
            assert np.max(self.map) < len(self.lab)

    def __getitem__(self, idx):
        idx_ = self.map[idx]
        seq = self.seq[self.idx[idx_]:self.idx[idx_+1]]
        seq = np.concatenate([[21], seq, [22]])  # add start and end token
        lab = self.lab[idx_]
        lab_ = np.zeros(6630)
        lab_[lab[0]] = 1 # one-hot
        return deepcopy(seq), deepcopy(lab_)

    def __len__(self):
        return len(self.map)


def collate_fn(batch, bucketsize=1024*6):
    seq = [i[0] for i in batch] + [[0]*(bucketsize-1)]
    seq = np.concatenate(seq)
    seq = seq[:len(seq)//bucketsize*bucketsize]
    ptr = [0]+[len(i[0]) for i in batch]
    ptr = np.append(np.cumsum(ptr), [len(seq)])
    lab = [i[1] for i in batch]
    lab = np.array(lab)
    return seq.astype(np.int16), ptr.astype(np.int32), lab.astype(np.int16)
