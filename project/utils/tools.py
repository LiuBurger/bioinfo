import numpy as np
import torch as pt
import torch.nn as nn
from torch.utils.data import DataLoader
import faiss
from datetime import datetime
from os.path import exists


def gen_embeddings(model:nn.Module, loader:DataLoader, gpu:int):
    model.eval()
    embs = []
    with pt.no_grad():
        for seq_pad, masks in loader:
            data = [d.to(gpu) for d in (seq_pad, masks)]
            emb = model.embed(tuple(data)).detach().cpu().numpy()
            embs.append(emb)
    pt.cuda.empty_cache()
    embs = np.concatenate(embs, axis=0)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


def faiss_idx(embs_lib:np.ndarray, gpu:int):
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(embs_lib.shape[1])
    index_flat = faiss.index_cpu_to_gpu(res, gpu, index_flat)
    index_flat.add(embs_lib)
    return index_flat


def build_idx(embs_lib:np.ndarray, embs_test:np.ndarray, gpu:int, topk:int=200, keep_index:bool=False):
    assert embs_lib.shape[1] == embs_test.shape[1], 'Dimension not match'
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(embs_lib.shape[1])
    index_flat = faiss.index_cpu_to_gpu(res, gpu, index_flat)
    index_flat.add(embs_lib)
    time_start = datetime.now()
    Distance, I = index_flat.search(embs_test, topk)
    print('Searching time: ', datetime.now()-time_start)
    pt.cuda.empty_cache()
    if keep_index:
        return I, Distance, index_flat
    else:
        index_flat.reset()
        return I, Distance


def save_model(model:nn.Module, model_name:str, epoch:int):
    if not exists(f'./model/{model_name}_epoch{epoch+1}.pth'):
        pt.save(model.state_dict(), f'./model/{model_name}_epoch{epoch+1}.pth')
    else:
        print('Model already exists!')