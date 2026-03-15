import os
import torch as pt
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.data import ProteinDataset, QueryHomologyDataset, collate_fun_train, collate_fun_emb
from utils.model import DualEncoderRetriever
from utils.tools import gen_embeddings, build_idx, calculate_remote_homology_score, save_model


@dataclass
class TrainConfig:
    epochs: int = 10
    gpu: int = 8  
    batch_size: int = 64
    num_workers: int = 6
    model_name: str = "dual_encoder"
    lr: float = 1e-3
    pdb_root: str = "../../data/pdb"
    tmalign_path: str = "./TMalign"
    topk: int = 12
    tmalign_reference: int = 1
    tmalign_workers: int = None


class ContrastiveTrainer:
    def __init__(self, model, config: TrainConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.log_file = open(f"{config.model_name}.txt", "w", encoding="utf-8")
        self.max_top1, self.max_topk = float("-inf"), float("-inf")
        # DualEncoderRetriever 前向传播会直接返回 InfoNCE loss 
        self.optimizer = pt.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = pt.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs, eta_min=config.lr/100)

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()

    def train(self, train_loader: DataLoader, test_loader: DataLoader, lib_loader: DataLoader, test_set, datalib,):
        try:
            for epoch in range(self.config.epochs):
                print(f"======================= Train epoch {epoch + 1} =======================")
                self.log_file.write(f"\nEpoch {epoch + 1} Training\n")
                self.model.train()
                for i, batch in enumerate(tqdm(train_loader, unit="batch")):
                    # 解析 collate_fn 传回来的 Dict，将其扔到 GPU
                    batch_gpu = {}
                    for k, v in batch.items():
                        if hasattr(v, 'to'):
                            batch_gpu[k] = v.to(self.config.gpu, non_blocking=True)
                        else:
                            batch_gpu[k] = v

                    output = self.model(batch_gpu)
                    loss = output["loss"]
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if (i + 1) % 100 == 0:
                        msg = f"Epoch [{epoch + 1}/{self.config.epochs}], Step [{i+1}]\n \
                                Train Loss: {loss:.4f}\n\
                                q2c loss: {output['loss_q2c']:.4f}\n\
                                c2q loss: {output['loss_c2q']:.4f}\n\
                                retrieval_acc_q2c: {output['retrieval_acc_q2c']:.4f}\n\
                                retrieval_acc_c2q: {output['retrieval_acc_c2q']:.4f}"
                        print(msg)
                        self.log_file.write(msg + "\n")
                        train_loss = []
                self.scheduler.step()
                print(f"======================= Test epoch {epoch + 1} =======================")
                self.log_file.write(f"\nEpoch {epoch + 1} Testing\n")
                pt.cuda.empty_cache()
                embs_lib = gen_embeddings(self.model, lib_loader, self.config.gpu, mode='candidate')
                embs_test = gen_embeddings(self.model, test_loader, self.config.gpu, mode='query')       
                I, _ = build_idx(embs_lib, embs_test)
                sorted_results, avg_top1_score, avg_topk_score = calculate_remote_homology_score(
                    query=test_set, database=datalib, idx=I,
                    k=self.config.topk,
                    pdb_root=self.config.pdb_root,
                    tmalign_path=self.config.tmalign_path,
                    reference=self.config.tmalign_reference,
                    num_workers=self.config.tmalign_workers,
                )
                msg = f"Average Top-1 Remote homologous score: {avg_top1_score:.6f}\n\
                        Average Top-{self.config.topk} Remote homologous score: {avg_topk_score:.6f}"
                print(msg)
                self.log_file.write(msg + "\n")
                if avg_top1_score > self.max_top1 or avg_topk_score > self.max_topk:
                    self.max_top1 = avg_top1_score
                    self.max_topk = avg_topk_score
                    save_model(self.model, self.config.model_name, epoch)
                print("======================================================================")
        finally:
            self.close()



if __name__ == "__main__":
    config = TrainConfig()
    pt.cuda.set_device(config.gpu)
    """# 1. 加载数据
    datalib(ProteinDataset)->libloader(shuffle=False)
        |->query_homo_data(QueryHomologyDataset)
        |       |->train_loader(shuffle=True)
        |
        |->test_set(ProteinDataset)
                |->test_loader(shuffle=False)
    """
    print('loading data')
    # (seq_list, lab_list)
    data = pt.load(f'./data/seq_shorter_than_1300_sorted.pt', weights_only=False)
    datalib = ProteinDataset(data)
    lib_map = np.arange(len(datalib), dtype=np.int64)
    print('number of proteins:', len(lib_map))
    
    pdb2idx = {data[1][i]: i for i in range(len(data[1]))} 
    query_homo_data = QueryHomologyDataset(datalib, './data/tmalign.out', pdb2idx)
    print('Query_i homology_ij pair num:', len(query_homo_data))
    # train_test_split 会随机打乱索引，我们需要重新对 test_map 进行升序排序
    _, test_map = train_test_split(lib_map, test_size=1024, random_state=42)
    test_map = np.sort(test_map)
    test_set = ProteinDataset(datalib, test_map)

    batch_size = config.batch_size
    libloader = DataLoader(datalib, batch_size=batch_size//2, shuffle=False, num_workers=config.num_workers,
                           collate_fn=collate_fun_emb, pin_memory=True, drop_last=False)
    train_loader = DataLoader(query_homo_data, batch_size=batch_size, shuffle=True, num_workers=config.num_workers,
                              collate_fn=collate_fun_train(datalib), pin_memory=True, drop_last=True)                     
    test_loader = DataLoader(test_set, batch_size=batch_size//2, shuffle=False, num_workers=config.num_workers,
                             collate_fn=collate_fun_emb, pin_memory=True, drop_last=False)

    model = DualEncoderRetriever(
        vocab_size=21,          # 氨基酸种类+1
    ).cuda(config.gpu)

    trainer = ContrastiveTrainer(model, config)
    trainer.train(train_loader=train_loader, test_loader=test_loader, lib_loader=libloader, test_set=test_set, datalib=datalib,)
    pt.cuda.empty_cache()